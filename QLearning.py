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

def findQueryIndex(sessQueryID, qObj):
    for oldSessQueryID in qObj.queryVocab:
        if oldSessQueryID == sessQueryID:
            return oldSessQueryID
        elif LSTM_RNN_Parallel.compareBitMaps(qObj.sessionStreamDict[oldSessQueryID], qObj.sessionStreamDict[sessQueryID]) == "True":
            return oldSessQueryID
    return None

def updateQTableDims(prevKey, qObj):
    if prevKey not in qObj.qTable:
        qObj.qTable[prevKey] = [0.0] * len(qObj.queryVocab)
    for sessQueryID in qObj.qTable:
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

def updateQValues(prevKey, curSessQueryID, qObj):
    keyIndex = qObj.queryVocab.index(curSessQueryID)
    invokeBellmanUpdate(prevKey, keyIndex, qObj, 1.0)
    return

def updateQTable(curSessQueryID, prevSessQueryID, qObj):
    assert qObj.configDict['QTABLE_MEM_DISK'] == 'MEM' or qObj.configDict['QTABLE_MEM_DISK'] == 'DISK'
    if qObj.configDict['QTABLE_MEM_DISK'] == 'MEM':
            prevKey = findQueryIndex(prevSessQueryID, qObj)
            if prevKey is not None:
                updateQTableDims(prevKey, qObj)
                updateQValues(prevKey, curSessQueryID, qObj)
    return

def findIfQueryInside(sessQueryID, sessionStreamDict, queryVocab, distinctQueries):
    for oldSessQueryID in distinctQueries:
        if LSTM_RNN_Parallel.compareBitMaps(sessionStreamDict[oldSessQueryID], sessionStreamDict[sessQueryID]) == "True":
            return oldSessQueryID
    for oldSessQueryID in range(len(queryVocab)):
        if LSTM_RNN_Parallel.compareBitMaps(sessionStreamDict[oldSessQueryID], sessionStreamDict[sessQueryID]) == "True":
            return oldSessQueryID
    return None

def updateQueryVocabQTable(qObj):
    distinctQueries = []
    for sessQueryID in qObj.queryKeysSetAside:
        retDistinctSessQueryID = findIfQueryInside(sessQueryID, qObj.sessionStreamDict, qObj.queryVocab, distinctQueries)
        if retDistinctSessQueryID is None:
            distinctQueries.append(sessQueryID)
            key = len(qObj.queryVocab)
            qObj.queryVocab[key] = sessQueryID
            retDistinctSessQueryID = sessQueryID
        sessID = int(sessQueryID.split(",")[0])
        queryID = int(sessQueryID.split(",")[1])
        if queryID - 1 >= 0:
            prevSessQueryID = str(sessID) + "," + str(queryID - 1)
            updateQTable(retDistinctSessQueryID, prevSessQueryID, qObj)
    return

def refineQTableUsingBellmanUpdate(qObj):
    

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
        (qObj.queryKeysSetAside, qObj.sessionDict) = LSTM_RNN_Parallel.updateGlobalSessionDict(lo, hi, qObj.keyOrder,
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