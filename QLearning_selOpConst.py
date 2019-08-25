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
import CF_SVD_selOpConst
import argparse
from sklearn.decomposition import NMF
import CFCosineSim_Parallel
import random

class Q_Obj:
    def __init__(self, configDict):
        self.configDict = configDict
        self.intentSessionFile = QR.fetchIntentFileFromConfigDict(configDict)
        self.episodeResponseTimeDictName = getConfig(configDict['OUTPUT_DIR']) + "/ResponseTimeDict_" + configDict[
            'ALGORITHM'] + "_" + configDict['QL_BOOLEAN_NUMERIC_REWARD'] + "_" + configDict['INTENT_REP'] + "_" + \
                                      configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                                          'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict[
                                          'EPISODE_IN_QUERIES'] + ".pickle"
        self.outputIntentFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputFileShortTermIntent_" + configDict[
            'ALGORITHM'] + "_" + configDict['QL_BOOLEAN_NUMERIC_REWARD'] + "_" + configDict['INTENT_REP'] + "_" + \
                                    configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                                   'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']
        self.numEpisodes = 0
        self.queryKeysSetAside = []
        self.episodeResponseTime = {}
        self.sessionLengthDict = ConcurrentSessions.countQueries(getConfig(configDict['QUERYSESSIONS']))
        try:
            os.remove(self.outputIntentFileName)
        except OSError:
            pass
        self.manager = multiprocessing.Manager()
        self.sessionStreamDict = self.manager.dict()
        self.resultDict = {}
        self.keyOrder = []
        with open(self.intentSessionFile) as f:
            for line in f:
                (sessID, queryID, curQueryIntent, self.sessionStreamDict) = QR.updateSessionDict(line, self.configDict,
                                                                                                 self.sessionStreamDict)
                self.keyOrder.append(str(sessID) + "," + str(queryID))
        f.close()
        self.qTable = {} # this will be a dictionary
        self.queryVocabValOrder = []  # list of SHA keys
        self.queryVocab = {} # key is SHA of bitvector and value is sessID,queryID
        #self.sessionDict = {} # key is sess index and val is a list of query vocab indices
        self.sessionDict = {} # key is sess index and val is a list of query indices
        self.learningRate = float(configDict['QL_LEARNING_RATE'])
        self.decayRate = float(configDict['QL_DECAY_RATE'])
        self.startEpisode = time.time()

def findMostSimilarQuery(sessQueryID, queryVocabValOrder, sessionStreamDict):
    maxCosineSim = 0.0
    maxSimSessQueryID = None
    for oldSessQueryID in queryVocabValOrder:
        if oldSessQueryID == sessQueryID:
            return (1.0, oldSessQueryID)
        #if oldSessQueryID in sessionStreamDict:
        cosineSim = CFCosineSim_Parallel.computeBitCosineSimilarity(sessionStreamDict[oldSessQueryID], sessionStreamDict[sessQueryID])
        if cosineSim >= 1.0:
            return (1.0, oldSessQueryID)
        elif cosineSim >= maxCosineSim:
            maxCosineSim = cosineSim
            maxSimSessQueryID = oldSessQueryID
    return (maxCosineSim, maxSimSessQueryID)

def updateQTableDims(distinctSessQueryID, qObj):
    if distinctSessQueryID not in qObj.qTable:
        qObj.qTable[distinctSessQueryID] = [0.0] * len(qObj.queryVocabValOrder)
    for sessQueryID in qObj.queryVocabValOrder:
        if sessQueryID not in qObj.qTable:
            qObj.qTable[sessQueryID] = [0.0] * len(qObj.queryVocabValOrder)
        qValues = qObj.qTable[sessQueryID]
        while len(qValues) < len(qObj.queryVocabValOrder):
            qValues.append(0.0)
    return

def invokeBellmanUpdate(curSessQueryID, nextKeyIndex, qObj, rewVal):
    nextSessQueryID = qObj.queryVocabValOrder[nextKeyIndex]
    maxNextQVal = max(qObj.qTable[nextSessQueryID])
    qObj.qTable[curSessQueryID][nextKeyIndex] = qObj.qTable[curSessQueryID][nextKeyIndex] * (1-qObj.learningRate) + \
                                                qObj.learningRate * (rewVal + qObj.decayRate * maxNextQVal)
    return

def updateQValues(prevDistinctSessQueryID, curSessQueryID, qObj):
    keyIndex = qObj.queryVocabValOrder.index(curSessQueryID)
    invokeBellmanUpdate(prevDistinctSessQueryID, keyIndex, qObj, 1.0)
    return

def updateQTable(curSessQueryID, prevSessQueryID, qObj):
    assert qObj.configDict['QTABLE_MEM_DISK'] == 'MEM' or qObj.configDict['QTABLE_MEM_DISK'] == 'DISK'
    if qObj.configDict['QTABLE_MEM_DISK'] == 'MEM':
            prevDistinctSessQueryID= findIfQueryInside(prevSessQueryID, qObj)
            updateQValues(prevDistinctSessQueryID, curSessQueryID, qObj)
    return

def findIfQueryInside(sessQueryID, qObj):
    hexDigestKey = CF_SVD_selOpConst.computeHexDigest(qObj.sessionStreamDict[sessQueryID])
    try:
        oldSessQueryID = qObj.queryVocab[hexDigestKey]
        return oldSessQueryID
    except:
        qObj.queryVocab[hexDigestKey] = sessQueryID
        qObj.queryVocabValOrder.append(sessQueryID)
        updateQTableDims(sessQueryID, qObj)
        return sessQueryID

def updateQueryVocabQTable(qObj):
    for sessQueryID in qObj.queryKeysSetAside:
        retDistinctSessQueryID = findIfQueryInside(sessQueryID, qObj)
        sessID = int(sessQueryID.split(",")[0])
        queryID = int(sessQueryID.split(",")[1])
        if queryID - 1 >= 0:
            prevSessQueryID = str(sessID) + "," + str(queryID - 1)
            updateQTable(retDistinctSessQueryID, prevSessQueryID, qObj)
    return

def printQTable(qTable, queryVocabValOrder):
    assert len(qTable)==len(queryVocabValOrder)
    for key in queryVocabValOrder:
        line = str(key)+":"
        line += str(qTable[key])+"\n"
        print line
    return

def assignReward(startDistinctSessQueryID, endDistinctSessQueryID, qObj):
    assert qObj.configDict['QL_BOOLEAN_NUMERIC_REWARD'] == 'BOOLEAN' or qObj.configDict[
                                                                            'QL_BOOLEAN_NUMERIC_REWARD'] == 'NUMERIC'
    startSessID = int(startDistinctSessQueryID.split(",")[0])
    startQueryID = int(startDistinctSessQueryID.split(",")[1])
    endSessID = int(endDistinctSessQueryID.split(",")[0])
    endQueryID = int(endDistinctSessQueryID.split(",")[1])
    rewVal = 0.0
    if startSessID == endSessID and endQueryID == startQueryID + 1:
        rewVal = 1.0
    else:
        idealSuccSessQueryID = str(startSessID) + "," + str(startQueryID+1)
        try:
            if qObj.configDict['QL_BOOLEAN_NUMERIC_REWARD'] == 'BOOLEAN' and \
                            LSTM_RNN_Parallel.compareBitMaps(qObj.sessionStreamDict[endDistinctSessQueryID],
                                                             qObj.sessionStreamDict[idealSuccSessQueryID]) == "True":
                rewVal = 1.0
        except:
            pass # if successor query not present as curQuery marks the end of session
        if qObj.configDict['QL_BOOLEAN_NUMERIC_REWARD'] == 'NUMERIC':
            try:
                rewVal = CFCosineSim_Parallel.computeBitCosineSimilarity(qObj.sessionStreamDict[endDistinctSessQueryID],
                                                                         qObj.sessionStreamDict[idealSuccSessQueryID])
            except:
                pass # if successor query not present as curQuery marks the end of session
    return rewVal

def refineQTableUsingBellmanUpdate(qObj):
    assert len(qObj.queryVocab) == len(qObj.queryVocabValOrder)
    print "Number of distinct queries: "+str(len(qObj.queryVocab))+", #cells in QTable: "+str(int(len(qObj.queryVocab)*len(qObj.queryVocab)))
    #print "Expected number of refinement iterations: max("+str(len(qObj.queryVocab))+","+str(int(configDict['QL_REFINE_ITERS']))+")"
    #numRefineIters = max(len(qObj.queryVocab), int(configDict['QL_REFINE_ITERS']))
    # if len(qObj.queryVocab) * len(qObj.queryVocab)/10 <= int(configDict['QL_REFINE_ITERS']):
    numRefineIters = int(configDict['QL_REFINE_ITERS'])
    print "Expected number of refinement iterations: " + str(numRefineIters)
    #else:
        #numRefineIters = min(len(qObj.queryVocab) * len(qObj.queryVocab) / 100, int(configDict['QL_REFINE_ITERS']))
    for i in range(numRefineIters):
        if i%100 == 0:
            print "Refining using Bellman update, Iteration:"+str(i)
        # pick a random start and end sessQueryID pair within the vocabulary in queryVocabValOrder
        startSessQueryIndex = random.randrange(len(qObj.queryVocabValOrder))
        endSessQueryIndex = random.randrange(len(qObj.queryVocabValOrder))
        startDistinctSessQueryID = qObj.queryVocabValOrder[startSessQueryIndex]
        endDistinctSessQueryID = qObj.queryVocabValOrder[endSessQueryIndex]
        rewVal = assignReward(startDistinctSessQueryID, endDistinctSessQueryID, qObj)
        invokeBellmanUpdate(startDistinctSessQueryID, endSessQueryIndex, qObj, rewVal)
    return

def predictTopKIntents(threadID, qTable, queryVocabValOrder, sessQueryID, sessionStreamDict, configDict):
    #print "Inside ThreadID:"+str(threadID)
    (maxCosineSim, maxSimSessQueryID) = findMostSimilarQuery(sessQueryID, queryVocabValOrder, sessionStreamDict)
    qValues = qTable[maxSimSessQueryID]
    topK = int(configDict['TOP_K'])
    topKIndices = zip(*heapq.nlargest(topK, enumerate(qValues), key=operator.itemgetter(1)))[0]
    topKSessQueryIndices = []
    for topKIndex in topKIndices:
        topKSessQueryIndices.append(queryVocabValOrder[topKIndex])
    #print "maxSimSessQueryID: "+str(maxSimSessQueryID)+", topKIndices: "+str(topKIndices)+", topKSessQueryIndices: "+str(topKSessQueryIndices)
    return topKSessQueryIndices

def predictTopKIntentsPerThread((threadID, t_lo, t_hi, keyOrder, qTable, resList, queryVocabValOrder, sessionStreamDict, configDict)):
    #printQTable(qTable, queryVocabValOrder)
    #print "QueryVocabValOrder:"+str(queryVocabValOrder)
    for i in range(t_lo, t_hi+1):
        sessQueryID = keyOrder[i]
        sessID = int(sessQueryID.split(",")[0])
        queryID = int(sessQueryID.split(",")[1])
        #curQueryIntent = sessionStreamDict[sessQueryID]
        #if queryID < sessionLengthDict[sessID]-1:
        if str(sessID) + "," + str(queryID + 1) in sessionStreamDict:
            topKSessQueryIndices = predictTopKIntents(threadID, qTable, queryVocabValOrder, sessQueryID, sessionStreamDict, configDict)
            for sessQueryID in topKSessQueryIndices:
                #print "Length of sample: "+str(len(sessionSampleDict[int(sessQueryID.split(",")[0])]))
                if sessQueryID not in sessionStreamDict:
                    print "sessQueryID: "+sessQueryID+" not in sessionStreamDict !!"
                    sys.exit(0)
            #print "ThreadID: "+str(threadID)+", computed Top-K="+str(len(topKSessQueryIndices))+\
            #      " Candidates sessID: " + str(sessID) + ", queryID: " + str(queryID)
            if topKSessQueryIndices is not None:
                resList.append((sessID, queryID, topKSessQueryIndices))
    QR.writeToPickleFile(
        getConfig(configDict['PICKLE_TEMP_OUTPUT_DIR']) + "QLResList_" + str(threadID) + ".pickle", resList)
    return resList

def predictIntentsWithoutCurrentBatch(lo, hi, qObj, keyOrder):
    numThreads = min(int(configDict['QL_THREADS']), hi - lo + 1)
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
        qObj.resultDict[threadID] = list()
        # print "Set tuple boundaries for Threads"
    # sortedSessKeys = svdObj.sessAdjList.keys().sort()
    if numThreads == 1:
        qObj.resultDict[0] = predictTopKIntentsPerThread((0, lo, hi, keyOrder, qObj.qTable,
                                                            qObj.resultDict[0], qObj.queryVocabValOrder,
                                                            qObj.sessionStreamDict,
                                                            qObj.configDict))
    elif numThreads > 1:
        sharedTable = qObj.manager.dict()
        for key in qObj.qTable:
            sharedTable[key]=qObj.qTable[key]
        pool = multiprocessing.Pool()
        argsList = []
        for threadID in range(numThreads):
            (t_lo, t_hi) = t_loHiDict[threadID]
            argsList.append((threadID, t_lo, t_hi, keyOrder, sharedTable, qObj.resultDict[threadID],
                             qObj.queryVocabValOrder, qObj.sessionStreamDict, qObj.configDict))
            # threads[i] = threading.Thread(target=predictTopKIntentsPerThread, args=(i, t_lo, t_hi, keyOrder, resList, sessionDict, sessionSampleDict, sessionStreamDict, sessionLengthDict, configDict))
            # threads[i].start()
        pool.map(predictTopKIntentsPerThread, argsList)
        pool.close()
        pool.join()
        for threadID in range(numThreads):
            qObj.resultDict[threadID] = QR.readFromPickleFile(
                getConfig(configDict['PICKLE_TEMP_OUTPUT_DIR']) + "QLResList_" + str(threadID) + ".pickle")
        del sharedTable
    #print "len(resultDict): " + str(len(qObj.resultDict))
    return qObj.resultDict

def saveModelToFile(qObj):
    QR.writeToPickleFile(
        getConfig(configDict['OUTPUT_DIR']) + configDict['QL_BOOLEAN_NUMERIC_REWARD'] + "_QTable.pickle", qObj.qTable)
    QR.writeToPickleFile(
        getConfig(configDict['OUTPUT_DIR']) + configDict['QL_BOOLEAN_NUMERIC_REWARD'] + "_QLQueryVocab.pickle", qObj.queryVocab)
    QR.writeToPickleFile(
        getConfig(configDict['OUTPUT_DIR']) + configDict['QL_BOOLEAN_NUMERIC_REWARD'] + "_QLQueryVocabValOrder.pickle",
        qObj.queryVocabValOrder)
    return

def updateResultsToExcel(configDict, episodeResponseTimeDictName, outputIntentFileName):
    accThres = float(configDict['ACCURACY_THRESHOLD'])
    QR.evaluateQualityPredictions(outputIntentFileName, configDict, accThres,
                                  configDict['ALGORITHM']  + "_" + configDict['QL_BOOLEAN_NUMERIC_REWARD'])
    print "--Completed Quality Evaluation for accThres:" + str(accThres)
    QR.evaluateTimePredictions(episodeResponseTimeDictName, configDict,
                               configDict['ALGORITHM'] + "_" + configDict['QL_BOOLEAN_NUMERIC_REWARD'])

    outputEvalQualityFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputEvalQualityShortTermIntent_" + configDict[
        'ALGORITHM'] + "_" + configDict['QL_BOOLEAN_NUMERIC_REWARD'] + "_" + configDict['INTENT_REP'] + "_" + configDict[
                                    'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                                configDict['EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres)
    outputExcelQuality = getConfig(configDict['OUTPUT_DIR']) + "/OutputExcelQuality_" + configDict['ALGORITHM'] \
                         + "_" + configDict['QL_BOOLEAN_NUMERIC_REWARD'] + "_" + configDict['INTENT_REP'] + "_" + configDict[
                             'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict[
                             'EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres) + ".xlsx"
    ParseResultsToExcel.parseQualityFileWithEpisodeRep(outputEvalQualityFileName, outputExcelQuality, configDict)

    outputEvalTimeFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputEvalTimeShortTermIntent_" + configDict[
        'ALGORITHM'] + "_" + configDict['QL_BOOLEAN_NUMERIC_REWARD'] + "_" + configDict['INTENT_REP'] + "_" + configDict[
                                 'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                             configDict['EPISODE_IN_QUERIES']
    outputExcelTimeEval = getConfig(configDict['OUTPUT_DIR']) + "/OutputExcelTime_" + configDict['ALGORITHM']  + "_" +\
                          configDict['QL_BOOLEAN_NUMERIC_REWARD'] + "_" + configDict['INTENT_REP'] + "_" + configDict[
                              'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                          configDict['EPISODE_IN_QUERIES'] + ".xlsx"
    ParseResultsToExcel.parseTimeFile(outputEvalTimeFileName, outputExcelTimeEval)
    return (outputIntentFileName, episodeResponseTimeDictName)

def trainTestBatchWise(qObj):
    batchSize = int(qObj.configDict['EPISODE_IN_QUERIES'])
    lo = 0
    hi = -1
    #assert qObj.configDict['INCLUDE_CUR_SESS'] == "False"
    while hi < len(qObj.keyOrder) - 1:
        lo = hi + 1
        if len(qObj.keyOrder) - lo < batchSize:
            batchSize = len(qObj.keyOrder) - lo
        hi = lo + batchSize - 1
        elapsedAppendTime = 0.0
        # test first for each query in the batch if the classifier is not None
        print "Starting prediction in Episode " + str(qObj.numEpisodes) + ", lo: " + str(lo) + ", hi: " + str(
            hi) + ", len(keyOrder): " + str(len(qObj.keyOrder))
        if len(qObj.queryVocab) > 2:  # unless at least two rows hard to recommend
            qObj.resultDict = predictIntentsWithoutCurrentBatch(lo, hi, qObj, qObj.keyOrder)
        print "Starting training in Episode " + str(qObj.numEpisodes)
        startTrainTime = time.time()
        (qObj.sessionDict, qObj.queryKeysSetAside) = LSTM_RNN_Parallel.updateGlobalSessionDict(lo, hi, qObj.keyOrder,
                                                                              qObj.queryKeysSetAside, qObj.sessionDict)
        updateQueryVocabQTable(qObj)
        if len(qObj.queryVocab) > 2:
            refineQTableUsingBellmanUpdate(qObj)
            saveModelToFile(qObj)
            #printQTable(qObj.qTable, qObj.queryVocabValOrder) # only enabled for debugging purposes
        totalTrainTime = float(time.time() - startTrainTime)
        print "Total Train Time: " + str(totalTrainTime)
        assert qObj.configDict['QL_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL' or qObj.configDict[
                                                                                          'QL_INCREMENTAL_OR_FULL_TRAIN'] == 'FULL'
        # we have empty queryKeysSetAside because we want to incrementally train the CF at the end of each episode
        if qObj.configDict['QL_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL':
            del qObj.queryKeysSetAside
            qObj.queryKeysSetAside = []
        # we record the times including train and test
        qObj.numEpisodes += 1
        if len(qObj.resultDict) > 0:
            elapsedAppendTime = CFCosineSim_Parallel.appendResultsToFile(qObj.sessionStreamDict, qObj.resultDict,
                                                                         elapsedAppendTime, qObj.numEpisodes,
                                                                         qObj.outputIntentFileName, qObj.configDict,
                                                                         -1)
            (qObj.episodeResponseTimeDictName, qObj.episodeResponseTime, qObj.startEpisode,
             qObj.elapsedAppendTime) = QR.updateResponseTime(
                qObj.episodeResponseTimeDictName, qObj.episodeResponseTime, qObj.numEpisodes, qObj.startEpisode,
                elapsedAppendTime)
            qObj.resultDict = LSTM_RNN_Parallel.clear(qObj.resultDict)
    updateResultsToExcel(qObj.configDict, qObj.episodeResponseTimeDictName, qObj.outputIntentFileName)

def loadModel(qObj):
    qObj.queryVocabValOrder = QR.readFromPickleFile(getConfig(configDict['OUTPUT_DIR']) + configDict['QL_BOOLEAN_NUMERIC_REWARD'] + "_QLQueryVocabValOrder.pickle")
    qObj.queryVocab = QR.readFromPickleFile(getConfig(configDict['OUTPUT_DIR']) + configDict['QL_BOOLEAN_NUMERIC_REWARD'] + "_QLQueryVocab.pickle")
    qObj.qTable = QR.readFromPickleFile(getConfig(configDict['OUTPUT_DIR']) + configDict['QL_BOOLEAN_NUMERIC_REWARD'] + "_QTable.pickle")
    print "Loaded len(queryVocabValOrder): "+str(len(qObj.queryVocabValOrder))+"len(queryVocab): "+str(len(qObj.queryVocab))+", len(qObj.qTable): "+str(len(qObj.qTable))
    notInQT = 0
    notInSessionStreamDict = 0
    for key in qObj.queryVocabValOrder:
        if key not in qObj.qTable:
            print "key: "+key+" not in qObj.qTable"
            notInQT += 1
        if key not in qObj.sessionStreamDict:
            print "key: "+key+" not in qObj.sessionStreamDict"
            notInSessionStreamDict += 1
    print "notInQT: "+str(notInQT)+", notInSessionStreamDict: "+str(notInSessionStreamDict)
    return

def trainEpisodicModelSustenance(episodicTraining, trainKeyOrder, qObj):
    numTrainEpisodes = 0
    assert episodicTraining == 'True' or episodicTraining == 'False'
    if episodicTraining == 'True':
        batchSize = int(qObj.configDict['EPISODE_IN_QUERIES'])
    elif episodicTraining == 'False':
        batchSize = len(trainKeyOrder)
    lo = 0
    hi = -1
    # assert qObj.configDict['INCLUDE_CUR_SESS'] == "False"
    while hi < len(trainKeyOrder) - 1:
        lo = hi + 1
        if len(trainKeyOrder) - lo < batchSize:
            batchSize = len(trainKeyOrder) - lo
        hi = lo + batchSize - 1
        print "Starting training in Episode " + str(numTrainEpisodes)
        startTrainTime = time.time()
        if configDict['QL_SUSTENANCE_LOAD_EXISTING_MODEL'] == 'False':
            (qObj.sessionDict, qObj.queryKeysSetAside) = LSTM_RNN_Parallel.updateGlobalSessionDict(lo, hi, qObj.keyOrder,
                                                                                                   qObj.queryKeysSetAside,
                                                                                                   qObj.sessionDict)
            updateQueryVocabQTable(qObj)
            if len(qObj.queryVocab) > 2:
                refineQTableUsingBellmanUpdate(qObj)
                saveModelToFile(qObj)
                # printQTable(qObj.qTable, qObj.queryVocab) # only enabled for debugging purposes
        totalTrainTime = float(time.time() - startTrainTime)
        print "Total Train Time: " + str(totalTrainTime)
        assert qObj.configDict['QL_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL' or qObj.configDict[
                                                                                       'QL_INCREMENTAL_OR_FULL_TRAIN'] == 'FULL'
        # we have empty queryKeysSetAside because we want to incrementally train the CF at the end of each episode
        if qObj.configDict['QL_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL':
            del qObj.queryKeysSetAside
            qObj.queryKeysSetAside = []
        numTrainEpisodes += 1
    return

def trainModelSustenance(trainKeyOrder, qObj):
    assert configDict['QL_SUSTENANCE_LOAD_EXISTING_MODEL'] == 'True' or configDict[
                                                                            'QL_SUSTENANCE_LOAD_EXISTING_MODEL'] == 'False'
    if configDict['QL_SUSTENANCE_LOAD_EXISTING_MODEL'] == 'False':
        episodicTraining = 'True'
        trainEpisodicModelSustenance(episodicTraining, trainKeyOrder, qObj)
    elif configDict['QL_SUSTENANCE_LOAD_EXISTING_MODEL'] == 'True':
        loadModel(qObj)
    return

def testModelSustenance(testKeyOrder, qObj):
    batchSize = int(qObj.configDict['EPISODE_IN_QUERIES'])
    lo = 0
    hi = -1
    # assert qObj.configDict['INCLUDE_CUR_SESS'] == "False"
    while hi < len(testKeyOrder) - 1:
        lo = hi + 1
        if len(testKeyOrder) - lo < batchSize:
            batchSize = len(testKeyOrder) - lo
        hi = lo + batchSize - 1
        elapsedAppendTime = 0.0
        # test first for each query in the batch if the classifier is not None
        print "Starting prediction in Episode " + str(qObj.numEpisodes) + ", lo: " + str(lo) + ", hi: " + str(
            hi) + ", len(testKeyOrder): " + str(len(testKeyOrder))+ ", len(queryVocab): " +str(len(qObj.queryVocab))
        if len(qObj.queryVocab) > 2:  # unless at least two rows hard to recommend
            qObj.resultDict = predictIntentsWithoutCurrentBatch(lo, hi, qObj, testKeyOrder)
            # we record the times including train and test
            qObj.numEpisodes += 1
            if len(qObj.resultDict) > 0:
                print "appending results"
                elapsedAppendTime = CFCosineSim_Parallel.appendResultsToFile(qObj.sessionStreamDict, qObj.resultDict,
                                                                             elapsedAppendTime, qObj.numEpisodes,
                                                                             qObj.outputIntentFileName, qObj.configDict,
                                                                             -1)
                (qObj.episodeResponseTimeDictName, qObj.episodeResponseTime, qObj.startEpisode,
                 qObj.elapsedAppendTime) = QR.updateResponseTime(
                    qObj.episodeResponseTimeDictName, qObj.episodeResponseTime, qObj.numEpisodes, qObj.startEpisode,
                    elapsedAppendTime)
                qObj.resultDict = LSTM_RNN_Parallel.clear(qObj.resultDict)
    updateResultsToExcel(qObj.configDict, qObj.episodeResponseTimeDictName, qObj.outputIntentFileName)
    return

def evalSustenance(qObj):
    (trainKeyOrder, testKeyOrder) = LSTM_RNN_Parallel.splitIntoTrainTestSets(qObj.keyOrder, qObj.configDict)
    sustStartTrainTime = time.time()
    trainModelSustenance(trainKeyOrder, qObj)
    sustTotalTrainTime = float(time.time() - sustStartTrainTime)
    print "Sustenace Train Time: "+str(sustTotalTrainTime)
    testModelSustenance(testKeyOrder, qObj)

def runQLearning(configDict):
    assert configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY'
    assert configDict['ALGORITHM'] == 'QLEARNING'
    qObj = Q_Obj(configDict)
    assert configDict['QL_SUSTENANCE'] == 'True' or configDict['QL_SUSTENANCE'] == 'False'
    if configDict['QL_SUSTENANCE'] == 'False':
        trainTestBatchWise(qObj)
    elif configDict['QL_SUSTENANCE'] == 'True':
        evalSustenance(qObj)

if __name__ == "__main__":
    # configDict = parseConfig.parseConfigFile("configFile.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="Config parameters file", type=str, required=True)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    runQLearning(configDict)

'''
def findDistinctQueryAllArgs(sessQueryID, queryVocab, sessionStreamDict):
    for oldSessQueryID in queryVocab:
        if oldSessQueryID == sessQueryID:
            return oldSessQueryID
        elif LSTM_RNN_Parallel.compareBitMaps(sessionStreamDict[oldSessQueryID], sessionStreamDict[sessQueryID]) == "True":
            return oldSessQueryID
    return None

def findDistinctQuery(sessQueryID, qObj):
    return findDistinctQueryAllArgs(sessQueryID, qObj.queryVocab, qObj.sessionStreamDict)
'''