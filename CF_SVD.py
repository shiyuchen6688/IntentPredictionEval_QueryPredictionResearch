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

class SVD_Obj:
    def __init__(self, configDict):
        self.configDict = configDict
        self.intentSessionFile = QR.fetchIntentFileFromConfigDict(configDict)
        self.episodeResponseTimeDictName = getConfig(configDict['OUTPUT_DIR']) + "/ResponseTimeDict_" + configDict[
            'ALGORITHM'] + configDict['INTENT_REP'] + "_" + \
                                      configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                                          'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict[
                                          'EPISODE_IN_QUERIES'] + ".pickle"
        self.outputIntentFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputFileShortTermIntent_" + configDict[
            'ALGORITHM'] + configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                                   'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']
        self.numEpisodes = 0
        self.queryKeysSetAside = []
        self.episodeResponseTime = {}
        self.sessionLengthDict = ConcurrentSessions.countQueries(getConfig(configDict['QUERYSESSIONS']))
        try:
            os.remove(self.outputIntentFileName)
        except OSError:
            pass
        self.sessionStreamDict = {}
        self.resultDict = {}
        self.keyOrder = []
        with open(self.intentSessionFile) as f:
            for line in f:
                (sessID, queryID, curQueryIntent, self.sessionStreamDict) = QR.updateSessionDict(line, self.configDict,
                                                                                                 self.sessionStreamDict)
                self.keyOrder.append(str(sessID) + "," + str(queryID))
        f.close()
        self.matrix = [] # this will be an array of arrays
        self.queryVocab = {}  # key is index and val is sessID,queryID
        self.sessAdjList = {} # key is sess index and val is a list of query vocab indices
        self.startEpisode = time.time()
        self.leftFactorMatrix = None
        self.rightFactorMatrix = None

def createMatrix(svdObj):
    # based on svdObj.sessAdjList and svdObj.queryVocab
    if len(svdObj.matrix) > 0:
        del svdObj.matrix
        svdObj.matrix = []
    sortedSessKeys = svdObj.sessAdjList.keys().sort()
    for sessID in sortedSessKeys:
        queryVocabIDs = svdObj.sessAdjList[sessID]
        rowEntry = []
        for queryVocabID in svdObj.queryVocab.keys():
            if queryVocabID in queryVocabIDs:
                rowEntry.append(1.0)
            else:
                rowEntry.append(0.0)
        svdObj.matrix.append(rowEntry)
    # lastRow represents an empty cushion entry for sessIDs that are yet to come -- queries which belong to new sessions will use it for prediction
    rowEntry = [0.0] * len(svdObj.queryVocab)
    svdObj.matrix.append(rowEntry)
    return

def predictIntentsWithoutCurrentBatch(svdObj, lo, hi):
    print "Prediction parallelized"

def updateQueryVocabSessAdjList(svdObj):
    distinctQueries = []
    for sessQueryID in svdObj.queryKeysSetAside:
        if LSTM_RNN_Parallel.findIfQueryInside(sessQueryID, svdObj.sessionStreamDict, svdObj.queryVocab.values(), distinctQueries) == "False":
            distinctQueries.append(sessQueryID)
            key = len(svdObj.queryVocab)
            svdObj.queryVocab[key] = sessQueryID
            sessID = int(sessQueryID.split(",")[0])
            if sessID not in svdObj.sessAdjList:
                svdObj.sessAdjList[sessID] = []
            svdObj.sessAdjList[sessID].append(key)
    return

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

def factorizeMatrix(svdObj):
    print "Factorization using SVD_NMF_sklearn or NIMFA"
    latentFactors = min(int(configDict['SVD_LATENT_DIMS']), int(0.1 * len(svdObj.queryVocab)))
    if latentFactors == 0 and len(svdObj.queryVocab) > 2 and len(svdObj.queryVocab) < 10:
        latentFactors = 2
    model = NMF(n_components=latentFactors, init='nndsvd', solver='mu') # multiplicative update solver, cd for coordinate descent
    model.leftFactorMatrix = model.fit_transform(svdObj.matrix)
    model.rightFactorMatrix = model.components_

def trainTestBatchWise(svdObj):
    batchSize = int(configDict['EPISODE_IN_QUERIES'])
    lo = 0
    hi = -1
    assert configDict['INCLUDE_CUR_SESS'] == "False"  # you never recommend queries from current session coz it is the most similar to the query you have
    while hi < len(svdObj.keyOrder) - 1:
        lo = hi + 1
        if len(svdObj.keyOrder) - lo < batchSize:
            batchSize = len(svdObj.keyOrder) - lo
        hi = lo + batchSize - 1
        elapsedAppendTime = 0.0
        # test first for each query in the batch if the classifier is not None
        print "Starting prediction in Episode " + str(svdObj.numEpisodes) + ", lo: " + str(lo) + ", hi: " + str(
            hi) + ", len(keyOrder): " + str(len(svdObj.keyOrder))
        if len(svdObj.sessAdjList) > 1: # unless at least two rows hard to recommend
            svdObj.resultDict = predictIntentsWithoutCurrentBatch(svdObj, lo, hi)
        print "Starting training in Episode " + str(svdObj.numEpisodes)
        startTrainTime = time.time()
        svdObj.queryKeysSetAside = CFCosineSim_Parallel.updateQueriesSetAside(lo, hi, svdObj.keyOrder, svdObj.queryKeysSetAside)
        updateQueryVocabSessAdjList(svdObj)
        createMatrix(svdObj)
        factorizeMatrix(svdObj)
        assert svdObj.configDict['SVD_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL' or svdObj.configDict[
                                                                                  'SVD_INCREMENTAL_OR_FULL_TRAIN'] == 'FULL'
        # we have empty queryKeysSetAside because we want to incrementally train the CF at the end of each episode
        if svdObj.configDict['SVD_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL':
            del svdObj.queryKeysSetAside
            svdObj.queryKeysSetAside = []
        # we record the times including train and test
        svdObj.numEpisodes += 1
        if len(svdObj.resultDict) > 0:
            elapsedAppendTime = CFCosineSim_Parallel.appendResultsToFile(svdObj.sessionStreamDict, svdObj.resultDict, elapsedAppendTime, svdObj.numEpisodes,
                                                    svdObj.outputIntentFileName, svdObj.configDict, -1)
            (svdObj.episodeResponseTimeDictName, svdObj.episodeResponseTime, svdObj.startEpisode, svdObj.elapsedAppendTime) = QR.updateResponseTime(
                svdObj.episodeResponseTimeDictName, svdObj.episodeResponseTime, svdObj.numEpisodes, svdObj.startEpisode, elapsedAppendTime)
            svdObj.resultDict = LSTM_RNN_Parallel.clear(svdObj.resultDict)
        totalTrainTime = float(time.time() - startTrainTime)
        print "Total Train Time: " + str(totalTrainTime)
    updateResultsToExcel(svdObj.configDict, svdObj.episodeResponseTimeDictName, svdObj.outputIntentFileName)


def runSVD(configDict):
    assert configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY'
    svdObj = SVD_Obj(configDict)
    trainTestBatchWise(svdObj)

if __name__ == "__main__":
    # configDict = parseConfig.parseConfigFile("configFile.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="Config parameters file", type=str, required=True)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    runSVD(configDict)
