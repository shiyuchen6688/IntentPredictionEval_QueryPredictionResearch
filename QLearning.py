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
import CF_SVD
import argparse
from sklearn.decomposition import NMF
import CFCosineSim_Parallel
import random

class Q_Obj:
    def __init__(self, configDict):
        self.configDict = configDict
        self.intentSessionFile = QR.fetchIntentFileFromConfigDict(configDict)
        self.episodeResponseTimeDictName = getConfig(configDict['OUTPUT_DIR']) + "/ResponseTimeDict_" + configDict[
            'ALGORITHM'] + "_" + configDict['INTENT_REP'] + "_" + \
                                      configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                                          'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict[
                                          'EPISODE_IN_QUERIES'] + ".pickle"
        self.outputIntentFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputFileShortTermIntent_" + configDict[
            'ALGORITHM'] + "_" + configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
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
        self.queryVocab = []  # list of (sessID,queryID)
        self.sessionDict = {} # key is sess index and val is a list of query vocab indices
        self.learningRate = float(configDict['QL_LEARNING_RATE'])
        self.decayRate = float(configDict['QL_DECAY_RATE'])
        self.startEpisode = time.time()

def findMostSimilarQuery(sessQueryID, queryVocab, sessionStreamDict):
    maxCosineSim = 0.0
    maxSimSessQueryID = None
    for oldSessQueryID in queryVocab:
        if oldSessQueryID == sessQueryID:
            return oldSessQueryID
        cosineSim = CFCosineSim_Parallel.computeBitCosineSimilarity(sessionStreamDict[oldSessQueryID], sessionStreamDict[sessQueryID])
        if cosineSim >= 1.0:
            return (1.0, oldSessQueryID)
        elif cosineSim > maxCosineSim:
            maxCosineSim = cosineSim
            maxSimSessQueryID = oldSessQueryID
    return (maxCosineSim, maxSimSessQueryID)

def findDistinctQueryAllArgs(sessQueryID, queryVocab, sessionStreamDict):
    for oldSessQueryID in queryVocab:
        if oldSessQueryID == sessQueryID:
            return oldSessQueryID
        elif LSTM_RNN_Parallel.compareBitMaps(sessionStreamDict[oldSessQueryID], sessionStreamDict[sessQueryID]) == "True":
            return oldSessQueryID
    return None

def findDistinctQuery(sessQueryID, qObj):
    return findDistinctQueryAllArgs(sessQueryID, qObj.queryVocab, qObj.sessionStreamDict)

def updateQTableDims(distinctSessQueryID, qObj):
    if distinctSessQueryID not in qObj.qTable:
        qObj.qTable[distinctSessQueryID] = [0.0] * len(qObj.queryVocab)
    for sessQueryID in qObj.queryVocab:
        if sessQueryID not in qObj.qTable:
            qObj.qTable[sessQueryID] = [0.0] * len(qObj.queryVocab)
        qValues = qObj.qTable[sessQueryID]
        while len(qValues) < len(qObj.queryVocab):
            qValues.append(0.0)
    return

def invokeBellmanUpdate(curSessQueryID, nextKeyIndex, qObj, rewVal):
    nextSessQueryID = qObj.queryVocab[nextKeyIndex]
    maxNextQVal = max(qObj.qTable[nextSessQueryID])
    qObj.qTable[curSessQueryID][nextKeyIndex] = qObj.qTable[curSessQueryID][nextKeyIndex] * (1-qObj.learningRate) + \
                                                qObj.learningRate * (rewVal + qObj.decayRate * maxNextQVal)
    return

def updateQValues(prevDistinctSessQueryID, curSessQueryID, qObj):
    keyIndex = qObj.queryVocab.index(curSessQueryID)
    invokeBellmanUpdate(prevDistinctSessQueryID, keyIndex, qObj, 1.0)
    return

def updateQTable(curSessQueryID, prevSessQueryID, qObj):
    assert qObj.configDict['QTABLE_MEM_DISK'] == 'MEM' or qObj.configDict['QTABLE_MEM_DISK'] == 'DISK'
    if qObj.configDict['QTABLE_MEM_DISK'] == 'MEM':
            prevDistinctSessQueryID= findDistinctQuery(prevSessQueryID, qObj)
            if prevDistinctSessQueryID is not None:
                #updateQTableDims(prevDistinctSessQueryID, qObj)
                updateQValues(prevDistinctSessQueryID, curSessQueryID, qObj)
    return

def findIfQueryInside(sessQueryID, sessionStreamDict, queryVocab, distinctQueries):
    for oldSessQueryID in distinctQueries:
        if LSTM_RNN_Parallel.compareBitMaps(sessionStreamDict[oldSessQueryID], sessionStreamDict[sessQueryID]) == "True":
            return oldSessQueryID
    for oldSessQueryID in queryVocab:
        if LSTM_RNN_Parallel.compareBitMaps(sessionStreamDict[oldSessQueryID], sessionStreamDict[sessQueryID]) == "True":
            return oldSessQueryID
    return None

def updateQueryVocabQTable(qObj):
    distinctQueries = []
    for sessQueryID in qObj.queryKeysSetAside:
        retDistinctSessQueryID = findIfQueryInside(sessQueryID, qObj.sessionStreamDict, qObj.queryVocab, distinctQueries)
        if retDistinctSessQueryID is None:
            distinctQueries.append(sessQueryID)
            qObj.queryVocab.append(sessQueryID)
            retDistinctSessQueryID = sessQueryID
            updateQTableDims(retDistinctSessQueryID, qObj)
        sessID = int(sessQueryID.split(",")[0])
        queryID = int(sessQueryID.split(",")[1])
        if queryID - 1 >= 0:
            prevSessQueryID = str(sessID) + "," + str(queryID - 1)
            updateQTable(retDistinctSessQueryID, prevSessQueryID, qObj)
    return

def refineQTableUsingBellmanUpdate(qObj):
    for i in range(int(configDict['QL_REFINE_ITERS'])):
        print "Refining using Bellman update, Iteration "+str(i)
        # pick a random start and end sessQueryID pair within the vocabulary in sessionDict
        startSessID = random.choice(qObj.sessionDict.keys())
        startQueryID = random.randint(0, qObj.sessionDict[startSessID])
        endSessID = random.choice(qObj.sessionDict.keys())
        endQueryID = random.randint(0, qObj.sessionDict[endSessID])
        startSessQueryID = str(startSessID)+","+str(startQueryID)
        endSessQueryID = str(endSessID)+","+str(endQueryID)
        if startSessID == endSessID and endQueryID == startQueryID+1:
            rewVal = 1.0
        else:
            rewVal = 0.0
        startDistinctSessQueryID = findDistinctQuery(startSessQueryID, qObj)
        endDistinctSessQueryID = findDistinctQuery(endSessQueryID, qObj)
        endSessQueryIndex = qObj.queryVocab.index(endDistinctSessQueryID)
        invokeBellmanUpdate(startDistinctSessQueryID, endSessQueryIndex, qObj, rewVal)
    return

def predictTopKIntents(threadID, qTable, queryVocab, sessQueryID, sessionStreamDict, configDict):
    (maxCosineSim, maxSimSessQueryID) = findMostSimilarQuery(sessQueryID, queryVocab, sessionStreamDict)
    qValues = qTable[maxSimSessQueryID]
    topK = int(configDict['TOP_K'])
    topKIndices = zip(*heapq.nlargest(topK, enumerate(qValues), key=operator.itemgetter(1)))[0]
    topKSessQueryIndices = []
    for topKIndex in topKIndices:
        topKSessQueryIndices.append(queryVocab[topKIndex])
    return topKSessQueryIndices

def predictTopKIntentsPerThread((threadID, t_lo, t_hi, keyOrder, qTable, resList, queryVocab, sessionStreamDict, configDict)):
    for i in range(t_lo, t_hi+1):
        sessQueryID = keyOrder[i]
        sessID = int(sessQueryID.split(",")[0])
        queryID = int(sessQueryID.split(",")[1])
        #curQueryIntent = sessionStreamDict[sessQueryID]
        #if queryID < sessionLengthDict[sessID]-1:
        if str(sessID) + "," + str(queryID + 1) in sessionStreamDict:
            topKSessQueryIndices = predictTopKIntents(threadID, qTable, queryVocab, sessQueryID, sessionStreamDict, configDict)
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

def predictIntentsWithoutCurrentBatch(lo, hi, qObj):
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
        qObj.resultDict[0] = predictTopKIntentsPerThread((0, lo, hi, qObj.keyOrder, qObj.qTable,
                                                            qObj.resultDict[0], qObj.queryVocab,
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
            argsList.append((threadID, t_lo, t_hi, qObj.keyOrder, sharedTable, qObj.resultDict[threadID],
                             qObj.queryVocab, qObj.sessionStreamDict, qObj.configDict))
            # threads[i] = threading.Thread(target=predictTopKIntentsPerThread, args=(i, t_lo, t_hi, keyOrder, resList, sessionDict, sessionSampleDict, sessionStreamDict, sessionLengthDict, configDict))
            # threads[i].start()
        pool.map(predictTopKIntentsPerThread, argsList)
        pool.close()
        pool.join()
        for threadID in range(numThreads):
            qObj.resultDict[threadID] = QR.readFromPickleFile(
                getConfig(configDict['PICKLE_TEMP_OUTPUT_DIR']) + "QLResList_" + str(threadID) + ".pickle")
        del sharedTable
    return qObj.resultDict

def saveModelToFile(qObj):
    QR.writeToPickleFile(
        getConfig(configDict['OUTPUT_DIR']) + "QTable.pickle", qObj.qTable)
    QR.writeToPickleFile(
        getConfig(configDict['OUTPUT_DIR']) + "QLQueryVocab.pickle", qObj.queryVocab)
    return

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
            qObj.resultDict = predictIntentsWithoutCurrentBatch(lo, hi, qObj)
        print "Starting training in Episode " + str(qObj.numEpisodes)
        startTrainTime = time.time()
        (qObj.sessionDict, qObj.queryKeysSetAside) = LSTM_RNN_Parallel.updateGlobalSessionDict(lo, hi, qObj.keyOrder,
                                                                              qObj.queryKeysSetAside, qObj.sessionDict)
        updateQueryVocabQTable(qObj)
        if len(qObj.queryVocab) > 2:
            refineQTableUsingBellmanUpdate(qObj)
            saveModelToFile(qObj)
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
    CF_SVD.updateResultsToExcel(qObj.configDict, qObj.episodeResponseTimeDictName, qObj.outputIntentFileName)

def runQLearning(configDict):
    assert configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY'
    assert configDict['ALGORITHM'] == 'QLEARNING'
    qObj = Q_Obj(configDict)
    trainTestBatchWise(qObj)

if __name__ == "__main__":
    # configDict = parseConfig.parseConfigFile("configFile.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="Config parameters file", type=str, required=True)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    runQLearning(configDict)