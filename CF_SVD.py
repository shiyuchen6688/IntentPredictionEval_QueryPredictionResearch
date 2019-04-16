from __future__ import division
import sys, operator
import os
import time
import QueryRecommender as QR
from bitmap import BitMap
import math
import heapq
import TupleIntent as ti
import ParseConfigFile as parseConfig
from ParseConfigFile import getConfig
import ConcurrentSessions
import ParseResultsToExcel
import multiprocessing
from multiprocessing.pool import ThreadPool
from multiprocessing import Array
from multiprocessing import Queue
import LSTM_RNN_Parallel
import argparse
from sklearn.decomposition import NMF
import CFCosineSim_Parallel


def initSingularity(configDict):
    intentSessionFile = QR.fetchIntentFileFromConfigDict(configDict)
    episodeResponseTimeDictName = getConfig(configDict['OUTPUT_DIR']) + "/ResponseTimeDict_" + configDict[
        'ALGORITHM'] + configDict['INTENT_REP'] + "_" + \
                                       configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                                           'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict[
                                           'EPISODE_IN_QUERIES'] + ".pickle"
    outputIntentFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputFileShortTermIntent_" + configDict[
        'ALGORITHM'] + configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                                    'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']
    numEpisodes = 0
    queryKeysSetAside = []
    episodeResponseTime = {}
    sessionLengthDict = ConcurrentSessions.countQueries(getConfig(configDict['QUERYSESSIONS']))
    try:
        os.remove(outputIntentFileName)
    except OSError:
        pass
    # manager = multiprocessing.Manager()
    sessionStreamDict = {}
    resultDict = {}
    keyOrder = []
    with open(intentSessionFile) as f:
        for line in f:
            (sessID, queryID, curQueryIntent, sessionStreamDict) = QR.updateSessionDict(line, configDict,
                                                                                             sessionStreamDict)
            keyOrder.append(str(sessID) + "," + str(queryID))
    f.close()
    matrix = []  # this will be an array of arrays
    queryVocab = {}  # key is index and val is sessID,queryID
    sessAdjList = {}  # key is sess index and val is a list of query vocab indices
    startEpisode = time.time()
    leftFactorMatrix = None
    rightFactorMatrix = None
    return (intentSessionFile, episodeResponseTimeDictName, outputIntentFileName, numEpisodes, queryKeysSetAside, episodeResponseTime, sessionLengthDict, sessionStreamDict,
            resultDict, keyOrder, matrix, queryVocab, sessAdjList, startEpisode, leftFactorMatrix, rightFactorMatrix)

def createMatrix(matrix, sessAdjList, queryVocab):
    # based on svdObj.sessAdjList and svdObj.queryVocab
    if len(matrix) > 0:
        del matrix
        matrix = []
    sortedSessKeys = sessAdjList.keys()
    sortedSessKeys.sort()
    for sessID in sortedSessKeys:
        queryVocabIDs = sessAdjList[sessID]
        rowEntry = []
        for queryVocabID in queryVocab.keys():
            if queryVocabID in queryVocabIDs:
                rowEntry.append(1.0)
            else:
                rowEntry.append(0.0)
        matrix.append(rowEntry)
    # lastRow represents an empty cushion entry for sessIDs that are yet to come -- queries which belong to new sessions will use it for prediction
    rowEntry = [0.0] * len(queryVocab)
    matrix.append(rowEntry)
    return (matrix, sortedSessKeys)

def updateQueryVocabSessAdjList(queryKeysSetAside, sessionStreamDict, queryVocab, sessAdjList):
    distinctQueries = []
    for sessQueryID in queryKeysSetAside:
        if LSTM_RNN_Parallel.findIfQueryInside(sessQueryID, sessionStreamDict, queryVocab.values(), distinctQueries) == "False":
            distinctQueries.append(sessQueryID)
            key = len(queryVocab)
            queryVocab[key] = sessQueryID
            sessID = int(sessQueryID.split(",")[0])
            if sessID not in sessAdjList:
                sessAdjList[sessID] = []
            sessAdjList[sessID].append(key)
    return (queryVocab, sessAdjList)

def updateResultsToExcel(configDict, episodeResponseTimeDictName, outputIntentFileName):
    accThres = float(configDict['ACCURACY_THRESHOLD'])
    QR.evaluateQualityPredictions(outputIntentFileName, configDict, accThres,
                                  configDict['ALGORITHM'])
    print "--Completed Quality Evaluation for accThres:" + str(accThres)
    QR.evaluateTimePredictions(episodeResponseTimeDictName, configDict,
                               configDict['ALGORITHM'])

    outputEvalQualityFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputEvalQualityShortTermIntent_" + configDict[
        'ALGORITHM'] + "_" + configDict['INTENT_REP'] + "_" + configDict[
                                    'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                                configDict['EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres)
    outputExcelQuality = getConfig(configDict['OUTPUT_DIR']) + "/OutputExcelQuality_" + configDict['ALGORITHM'] + "_" + configDict['INTENT_REP'] + "_" + configDict[
                             'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict[
                             'EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres) + ".xlsx"
    ParseResultsToExcel.parseQualityFileWithEpisodeRep(outputEvalQualityFileName, outputExcelQuality, configDict)

    outputEvalTimeFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputEvalTimeShortTermIntent_" + configDict[
        'ALGORITHM'] + "_" + configDict['INTENT_REP'] + "_" + configDict[
                                 'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                             configDict['EPISODE_IN_QUERIES']
    outputExcelTimeEval = getConfig(configDict['OUTPUT_DIR']) + "/OutputExcelTime_" + configDict['ALGORITHM'] + "_" + configDict['INTENT_REP'] + "_" + configDict[
                              'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                          configDict['EPISODE_IN_QUERIES'] + ".xlsx"
    ParseResultsToExcel.parseTimeFile(outputEvalTimeFileName, outputExcelTimeEval)
    return (outputIntentFileName, episodeResponseTimeDictName)

def factorizeMatrix(queryVocab, matrix):
    print "Factorization using SVD_NMF_sklearn or NIMFA"
    latentFactors = min(int(configDict['SVD_LATENT_DIMS']), int(0.1 * len(queryVocab)))
    if latentFactors == 0 and len(queryVocab) > 2 and len(queryVocab) < 10:
        latentFactors = 2
    model = NMF(n_components=latentFactors, init='nndsvdar', solver='mu') # multiplicative update solver, cd for coordinate descent
    leftFactorMatrix = model.fit_transform(matrix)
    rightFactorMatrix = model.components_
    return (leftFactorMatrix, rightFactorMatrix)

def getProductElement(rowArr, colArr):
    pdt = 0.0
    assert len(rowArr) == len(colArr)
    for i in range(len(rowArr)):
        pdt += float(rowArr[i] * colArr[i])
    return pdt

def completeMatrix(matrix, leftFactorMatrix, rightFactorMatrix, configDict):
    pool = multiprocessing.Pool(processes=int(configDict['SVD_THREADS']))
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 1.0:
                matrix[i][j] = -1.0 # to exclude earlier queries from a session from being recommended for the same session
    for i in range(len(leftFactorMatrix)):
        rowArr = leftFactorMatrix[i]
        for j in range(len(rightFactorMatrix[0])):
            colArr = []
            for k in range(len(rightFactorMatrix)):
                colArr.append(rightFactorMatrix[k][j])
            if matrix[i][j] == 0.0:
                matrix[i][j]=pool.apply_async(getProductElement, rowArr, colArr)
    return matrix

def predictTopKIntents(threadID, matrix, queryVocab, sortedSessKeys, sessID, configDict):
    if sessID in sortedSessKeys:
        matchingRowIndex = sortedSessKeys.index(sessID)
    else:
        matchingRowIndex = len(sortedSessKeys) - 1 # last row accommodates for new session as it inits with all 0s
    sessRow = matrix[matchingRowIndex]
    topK = int(configDict['TOP_K'])
    topKIndices = zip(*heapq.nlargest(topK, enumerate(sessRow), key=operator.itemgetter(1)))[0]
    # other alternatives: sorted(range(len(a)), key=lambda i: a[i])[-topK:]
    # sorted(range(len(a)), key=lambda i: a[i], reverse=True)[:topK]
    # zip(*sorted(enumerate(a), key=operator.itemgetter(1)))[0][-topK:]
    topKSessQueryIndices = []
    for topKIndex in topKIndices:
        topKSessQueryIndices.append(queryVocab[topKIndex])
    return topKSessQueryIndices


def predictTopKIntentsPerThread((threadID, t_lo, t_hi, keyOrder, matrix, resList, queryVocab, sortedSessKeys, sessionStreamDictKeys, configDict)):
    for i in range(t_lo, t_hi+1):
        sessQueryID = keyOrder[i]
        sessID = int(sessQueryID.split(",")[0])
        queryID = int(sessQueryID.split(",")[1])
        #curQueryIntent = sessionStreamDict[sessQueryID]
        #if queryID < sessionLengthDict[sessID]-1:
        if str(sessID) + "," + str(queryID + 1) in sessionStreamDictKeys:
            topKSessQueryIndices = predictTopKIntents(threadID, matrix, queryVocab, sortedSessKeys, sessID, configDict)
            for sessQueryID in topKSessQueryIndices:
                #print "Length of sample: "+str(len(sessionSampleDict[int(sessQueryID.split(",")[0])]))
                if sessQueryID not in sessionStreamDictKeys:
                    print "sessQueryID: "+sessQueryID+" not in sessionStreamDict !!"
                    sys.exit(0)
            #print "ThreadID: "+str(threadID)+", computed Top-K="+str(len(topKSessQueryIndices))+\
            #      " Candidates sessID: " + str(sessID) + ", queryID: " + str(queryID)
            if topKSessQueryIndices is not None:
                resList.append((sessID, queryID, topKSessQueryIndices))
    QR.writeToPickleFile(
        getConfig(configDict['PICKLE_TEMP_OUTPUT_DIR']) + "SVDResList_" + str(threadID) + ".pickle", resList)
    return resList

def predictIntentsWithoutCurrentBatch(lo, hi, keyOrder, matrix, resultDict, queryVocab, sortedSessKeys, sessionStreamDictKeys, configDict):
    print "Prediction parallelized"
    numThreads = min(int(configDict['SVD_THREADS']), hi - lo + 1)
    numKeysPerThread = int(float(hi - lo + 1) / float(numThreads))
    # threads = {}
    t_loHiDict = {}
    t_hi = lo - 1
    for threadID in range(numThreads):
        t_lo = t_hi + 1
        if threadID == numThreads - 1:
            t_hi = hi
        else:
            t_hi = t_lo + numKeysPerThread - 1
        t_loHiDict[threadID] = (t_lo, t_hi)
        resultDict[threadID] = list()
        # print "Set tuple boundaries for Threads"
    #sortedSessKeys = svdObj.sessAdjList.keys().sort()
    if numThreads == 1:
        resultDict[0] = predictTopKIntentsPerThread((0, lo, hi, keyOrder, matrix, resultDict[0], queryVocab, sortedSessKeys, sessionStreamDictKeys, configDict))
    elif numThreads > 1:
        #sharedMtx = svdObj.matrix
        #manager = multiprocessing.Manager()
        #sharedMtx = manager.list()
        #for row in svdObj.matrix:
            #sharedMtx.append(row)
        pool = multiprocessing.Pool()
        argsList = []
        for threadID in range(numThreads):
            (t_lo, t_hi) = t_loHiDict[threadID]
            argsList.append((threadID, t_lo, t_hi, keyOrder, matrix, resultDict[threadID], queryVocab, sortedSessKeys, sessionStreamDictKeys, configDict))
            #threads[i] = threading.Thread(target=predictTopKIntentsPerThread, args=(i, t_lo, t_hi, keyOrder, resList, sessionDict, sessionSampleDict, sessionStreamDict, sessionLengthDict, configDict))
            #threads[i].start()
        pool.map(predictTopKIntentsPerThread, argsList)
        pool.close()
        pool.join()
        for threadID in range(numThreads):
            resultDict[threadID] = QR.readFromPickleFile(getConfig(configDict['PICKLE_TEMP_OUTPUT_DIR']) + "SVDResList_" + str(threadID) + ".pickle")
    return resultDict

def trainTestBatchWise(configDict, episodeResponseTimeDictName, outputIntentFileName, numEpisodes, queryKeysSetAside,
     episodeResponseTime, sessionStreamDict,
     resultDict, keyOrder, matrix, queryVocab, sessAdjList, startEpisode):
    batchSize = int(configDict['EPISODE_IN_QUERIES'])
    lo = 0
    hi = -1
    assert configDict['INCLUDE_CUR_SESS'] == "False"  # you never recommend queries from current session coz it is the most similar to the query you have
    sortedSessKeys = None
    while hi < len(keyOrder) - 1:
        lo = hi + 1
        if len(keyOrder) - lo < batchSize:
            batchSize = len(keyOrder) - lo
        hi = lo + batchSize - 1
        elapsedAppendTime = 0.0
        # test first for each query in the batch if the classifier is not None
        print "Starting prediction in Episode " + str(numEpisodes) + ", lo: " + str(lo) + ", hi: " + str(
            hi) + ", len(keyOrder): " + str(len(keyOrder))
        if len(sessAdjList) > 1 and len(queryVocab) > 2: # unless at least two rows hard to recommend
            resultDict = predictIntentsWithoutCurrentBatch(lo, hi, keyOrder, matrix, resultDict, queryVocab, sortedSessKeys, sessionStreamDict.keys(), configDict)
        print "Starting training in Episode " + str(numEpisodes)
        startTrainTime = time.time()
        queryKeysSetAside = CFCosineSim_Parallel.updateQueriesSetAside(lo, hi, keyOrder, queryKeysSetAside)
        (queryVocab, sessAdjList) = updateQueryVocabSessAdjList(queryKeysSetAside, sessionStreamDict, queryVocab, sessAdjList)
        if len(queryVocab) > 2:
            (matrix, sortedSessKeys) = createMatrix(matrix, sessAdjList, queryVocab)
            (leftFactorMatrix, rightFactorMatrix) = factorizeMatrix(queryVocab, matrix)
            matrix = completeMatrix(matrix, leftFactorMatrix, rightFactorMatrix, configDict)
        assert configDict['SVD_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL' or configDict[
                                                                                  'SVD_INCREMENTAL_OR_FULL_TRAIN'] == 'FULL'
        # we have empty queryKeysSetAside because we want to incrementally train the CF at the end of each episode
        if configDict['SVD_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL':
            del queryKeysSetAside
            queryKeysSetAside = []
        # we record the times including train and test
        numEpisodes += 1
        if len(resultDict) > 0:
            elapsedAppendTime = CFCosineSim_Parallel.appendResultsToFile(sessionStreamDict, resultDict, elapsedAppendTime, numEpisodes,
                                                    outputIntentFileName, configDict, -1)
            (episodeResponseTimeDictName, episodeResponseTime, startEpisode, elapsedAppendTime) = QR.updateResponseTime(
                episodeResponseTimeDictName, episodeResponseTime, numEpisodes, startEpisode, elapsedAppendTime)
            resultDict = LSTM_RNN_Parallel.clear(resultDict)
        totalTrainTime = float(time.time() - startTrainTime)
        print "Total Train Time: " + str(totalTrainTime)
    updateResultsToExcel(configDict, episodeResponseTimeDictName, outputIntentFileName)


def runSVD(configDict):
    assert configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY'
    (intentSessionFile, episodeResponseTimeDictName, outputIntentFileName, numEpisodes, queryKeysSetAside,
     episodeResponseTime, sessionLengthDict, sessionStreamDict,
     resultDict, keyOrder, matrix, queryVocab, sessAdjList, startEpisode, leftFactorMatrix, rightFactorMatrix) = initSingularity(configDict)
    trainTestBatchWise(configDict, episodeResponseTimeDictName, outputIntentFileName, numEpisodes, queryKeysSetAside,
     episodeResponseTime, sessionStreamDict,
     resultDict, keyOrder, matrix, queryVocab, sessAdjList, startEpisode)

if __name__ == "__main__":
    # configDict = parseConfig.parseConfigFile("configFile.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="Config parameters file", type=str, required=True)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    runSVD(configDict)
