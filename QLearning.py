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
# import LSTM_RNN_Parallel
# import CF_SVD
import argparse
from sklearn.decomposition import NMF
# import CFCosineSim_Parallel
import random

class Q_Obj:
    def __init__(self, configDict):
        # configurations
        self.configDict = configDict
        # TODO: confirm
        
        # input file name
        # Documents/DataExploration-Research/BusTracker/InputOutput/MincBitFragmentIntentSessions
        # self.intentSessionFile = QR.fetchIntentFileFromConfigDict(configDict)
        self.intentSessionFile = fetchIntentFileFromConfigDict(configDict)
        # file path name of respone time dict
        self.episodeResponseTimeDictName = getConfig(configDict['OUTPUT_DIR']) + "/ResponseTimeDict_" + configDict[
            'ALGORITHM'] + "_" + configDict['INTENT_REP'] + "_" + \
                                      configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                                          'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict[
                                          'EPISODE_IN_QUERIES'] + ".pickle"
        # file path for output                                
        self.outputIntentFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputFileShortTermIntent_" + configDict[
            'ALGORITHM'] + "_" + configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                                   'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']
        self.numEpisodes = 0
        self.queryKeysSetAside = []
        self.episodeResponseTime = {}
        # configDict['QUERYSESSIONS']: file path to queries seperated for each session
        # self.sessionLengthDict stores count (value) of query of each session (key)
        self.sessionLengthDict = ConcurrentSessions.countQueries(getConfig(configDict['QUERYSESSIONS']))

        # if output file exist, remove it
        try:
            os.remove(self.outputIntentFileName)
        except OSError:
            pass
        # multiprocessing is a package that supports spawning processes using an API similar to the threading module
        # A manager object returned by Manager() controls a server process which holds Python objects and allows other processes to manipulate them using proxies.
        # Doc here: https://docs.python.org/3/library/multiprocessing.html
        self.manager = multiprocessing.Manager()
        # this dict can be manipulated by different process
        self.sessionStreamDict = self.manager.dict()
        self.resultDict = {}
        # array of "sessID,queryID" for each query in each session
        self.keyOrder = []
        with open(self.intentSessionFile) as f:
            for line in f:
                # for each line in f
                # get sessID, queryID, curQueryIntent, updated sessionStreamDict
                # add BitMap object of current query in sessionStreamDict
                # key is: sessID, queryID
                (sessID, queryID, curQueryIntent, self.sessionStreamDict) = QR.updateSessionDict(line, self.configDict, self.sessionStreamDict)
                # append current key to self.KeyOrder                                                                                 
                self.keyOrder.append(str(sessID) + "," + str(queryID))
        f.close()
        # dictionary that stores qTable
        # key is distinct queries' sessQueryID
        self.qTable = {} # this will be a dictionary

        self.queryVocab = []  # list of (sessID,queryID)
        self.sessionDict = {} # key is sess index and val is a list of query vocab indices
        # get learning rate and decay Rate base on configuration file
        self.learningRate = float(configDict['QL_LEARNING_RATE'])
        self.decayRate = float(configDict['QL_DECAY_RATE'])
        # record start episode time, didn't see it being used any where else
        self.startEpisode = time.time()

def findMostSimilarQuery(sessQueryID, queryVocab, sessionStreamDict):
    maxCosineSim = 0.0
    maxSimSessQueryID = None
    for oldSessQueryID in queryVocab:
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

# Helper functions copied over from other files
def fetchIntentFileFromConfigDict(configDict):
    # INTENT_REP means intended representation

    # tuple based representation
    if configDict['INTENT_REP'] == 'TUPLE':
        intentSessionFile = getConfig(configDict['TUPLEINTENTSESSIONS'])
    # fragment bit based or weighted fragemmnt representation
    # all configuration file are using bit: BIT_OR_WEIGHTED=BIT
    elif configDict['INTENT_REP'] == 'FRAGMENT' and configDict['BIT_OR_WEIGHTED'] == 'BIT':
        # if predicting query or table
        # TODO: not exatly sure what is the difference
        if configDict['RNN_PREDICT_QUERY_OR_TABLE'] == 'TABLE':
            intentSessionFile = getConfig(configDict['BIT_FRAGMENT_TABLE_INTENT_SESSIONS'])
        else:
            # If predicting query
            # file path: 
            # Documents/DataExploration-Research/BusTracker/InputOutput/MincBitFragmentIntentSessions
            intentSessionFile = getConfig(configDict['BIT_FRAGMENT_INTENT_SESSIONS'])

    # weighted fragment based
    elif configDict['INTENT_REP'] == 'FRAGMENT' and configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
        intentSessionFile = getConfig(configDict['WEIGHTED_FRAGMENT_INTENT_SESSIONS'])
    elif configDict['INTENT_REP'] == 'QUERY':
        intentSessionFile = getConfig(configDict['QUERY_INTENT_SESSIONS'])
    else:
        print("ConfigDict['INTENT_REP'] must either be TUPLE or FRAGMENT or QUERY !!")
        sys.exit(0)
    return intentSessionFile

# LSTM_RNN_Parallel.compareBitMaps
def compareBitMaps(bitMap1, bitMap2):
    set1 = set(bitMap1.nonzero())
    set2 = set(bitMap2.nonzero())
    if set1 == set2:
        return "True"
    return "False"

# LSTM_RNN_Parallel.clear
def clear(resultDict):
    keys = resultDict.keys()
    for resKey in keys:
        del resultDict[resKey]
    return resultDict

# LSTM_RNN_Parallel.updateGlobalSessionDict
def updateGlobalSessionDict(lo, hi, keyOrder, queryKeysSetAside, sessionDictGlobal):
    """
    For each key in the range [lo, hi] (which is a batch)
    keyOrder[cur] is sessQueryId of the key with index cur

    Output:
    queryKeysSetAside: stores session ID and query ID for all queries in keyOrder[lo, hi]
    sessionDictGlobal: stores session to query mapping for the lastest queryID of that session
    """
    cur = lo
    while(cur<hi+1):
        sessQueryID = keyOrder[cur]
        queryKeysSetAside.append(sessQueryID)
        sessID = int(sessQueryID.split(",")[0])
        queryID = int(sessQueryID.split(",")[1])
        sessionDictGlobal[sessID]= queryID # key is sessID and value is the latest queryID
        cur+=1
    #print("updated Global Session Dict")
    return (sessionDictGlobal, queryKeysSetAside)


# LSTM_RNN_Parallel.splitIntoTrainTestSets
def splitIntoTrainTestSets(keyOrder, configDict):
    keyCount = 0
    trainKeyOrder = []
    testKeyOrder = []
    for key in keyOrder:
        if keyCount <= int(configDict['RNN_SUSTENANCE_TRAIN_LIMIT']):
            trainKeyOrder.append(key)
        else:
            testKeyOrder.append(key)
        keyCount+=1
    return (trainKeyOrder, testKeyOrder)


# CF_SVD.updateResultsToExcel
def updateResultsToExcel(configDict, episodeResponseTimeDictName, outputIntentFileName):
    accThres = float(configDict['ACCURACY_THRESHOLD'])
    QR.evaluateQualityPredictions(outputIntentFileName, configDict, accThres,
                                  configDict['ALGORITHM'])
    print("--Completed Quality Evaluation for accThres:" + str(accThres))
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

# End of helper functions copied over from other files

def findDistinctQueryAllArgs(sessQueryID, queryVocab, sessionStreamDict):
    for oldSessQueryID in queryVocab:
        if oldSessQueryID == sessQueryID:
            return oldSessQueryID
        # elif LSTM_RNN_Parallel.compareBitMaps(sessionStreamDict[oldSessQueryID], sessionStreamDict[sessQueryID]) == "True":
        elif compareBitMaps(sessionStreamDict[oldSessQueryID], sessionStreamDict[sessQueryID]) == "True":
            return oldSessQueryID
    return None

def findDistinctQuery(sessQueryID, qObj):
    """
    find sessQueryID of any previous ID with same query as
    prevSessQueryID, None if not found
    """
    return findDistinctQueryAllArgs(sessQueryID, qObj.queryVocab, qObj.sessionStreamDict)

def updateQTableDims(distinctSessQueryID, qObj):
    """
    add 1 dimention, which is the new query distinctSessQueryID to original qTable
    """
    # add a new row
    if distinctSessQueryID not in qObj.qTable:
        # initial Q value for that new dimension is a row of 0
        qObj.qTable[distinctSessQueryID] = [0.0] * len(qObj.queryVocab)
    
    # add a new column
    for sessQueryID in qObj.queryVocab:
        # if any sessQueryID in queeyVocab is not in qTable
        # add a row for them too
        if sessQueryID not in qObj.qTable:
            qObj.qTable[sessQueryID] = [0.0] * len(qObj.queryVocab)
        
        # get the rows of q value for each row/state in qTable
        # if that row's length < queryVocab, we need to add new
        # column(s) to that row, most of the time just 1 column
        # added for the newly added dimension distinctSessQueryID
        qValues = qObj.qTable[sessQueryID]
        while len(qValues) < len(qObj.queryVocab):
            qValues.append(0.0)
    return

def invokeBellmanUpdate(curSessQueryID, nextKeyIndex, qObj, rewVal):
    """
    Update base on BellmanUpdate equation:
    new q(s, a) = 
    (1 - learning rate) * old q(s, a) + 
    (learning rate) * (immediate return + decay rate * (max q value over all action a' at state s' q(s', a'))

    where q(s, a) is q value at state s picking action a
    """
    nextSessQueryID = qObj.queryVocab[nextKeyIndex]
    maxNextQVal = max(qObj.qTable[nextSessQueryID]) # second part of equation
    # the qTable (didct) is indexed first by sessQueryID (key) that represent the current state (previously we call this prevDistinctSessQueryID)
    # then the result is an array and we need to index the array by a numeric index which is nextKeyIndex (previously called current keyIndex)
    # this step is just updating the qvalue base on the bellman udpate equation
    qObj.qTable[curSessQueryID][nextKeyIndex] = qObj.qTable[curSessQueryID][nextKeyIndex] * (1-qObj.learningRate) + \
                                                qObj.learningRate * (rewVal + qObj.decayRate * maxNextQVal)
    return

def updateQValues(prevDistinctSessQueryID, curSessQueryID, qObj):
    """
    update Q value by calling invokeBellmanUpdate
    """
    # find index of curSessQueryID in qObj.queryVocab (this is a list)
    keyIndex = qObj.queryVocab.index(curSessQueryID)
    # update Q value base on bellman update equation
    invokeBellmanUpdate(prevDistinctSessQueryID, keyIndex, qObj, 1.0)
    return

def updateQTable(curSessQueryID, prevSessQueryID, qObj):
    """
    when we saw a pair of (prevSessQueryID, curSessQueryID)
    we can update the Qtable base on this pair
    """
    assert qObj.configDict['QTABLE_MEM_DISK'] == 'MEM' or qObj.configDict['QTABLE_MEM_DISK'] == 'DISK'
    # devide if we are going to store the qTable on disk or memory
    # I think the current model only deals with in-memory qtable
    if qObj.configDict['QTABLE_MEM_DISK'] == 'MEM':
            # find sessQueryID of any previous ID with same query as
            # prevSessQueryID, None if not found
            prevDistinctSessQueryID= findDistinctQuery(prevSessQueryID, qObj)
            # if previous query exist, then we update Q value
            if prevDistinctSessQueryID is not None:
                #updateQTableDims(prevDistinctSessQueryID, qObj)
                # udate Q value for Qrew_prevDistinctSessQueryID_curSessQueryID
                updateQValues(prevDistinctSessQueryID, curSessQueryID, qObj)
    return

def findIfQueryInside(sessQueryID, sessionStreamDict, queryVocab, distinctQueries):
    """
    Check if given query sessQueryID is already in distinctQueries or queryVocab
    """
    # check if current query already exist in distinctQueries
    for oldSessQueryID in distinctQueries:
        if compareBitMaps(sessionStreamDict[oldSessQueryID], sessionStreamDict[sessQueryID]) == "True":
            return oldSessQueryID
    # check if current query already exist in queryVocab
    for oldSessQueryID in queryVocab:
        if compareBitMaps(sessionStreamDict[oldSessQueryID], sessionStreamDict[sessQueryID]) == "True":
            return oldSessQueryID
    # return None if this is the first time seeing this query
    return None

def updateQueryVocabQTable(qObj):
    """
    """
    distinctQueries = []
    # for each session query id in qObj.queryKeysSetAside
    # which contains all queries of this episode
    for sessQueryID in qObj.queryKeysSetAside:
        retDistinctSessQueryID = findIfQueryInside(sessQueryID, qObj.sessionStreamDict, qObj.queryVocab, distinctQueries)
        # if first time seeing this query
        if retDistinctSessQueryID is None:
            # add sessQueryID tp distinctQueries and qObj.queryVocab
            distinctQueries.append(sessQueryID)
            qObj.queryVocab.append(sessQueryID)
            retDistinctSessQueryID = sessQueryID
            # update Q tables dimension because we just saw a brand new query
            updateQTableDims(retDistinctSessQueryID, qObj)
        
        # parse and get sessID and queryID
        sessID = int(sessQueryID.split(",")[0])
        queryID = int(sessQueryID.split(",")[1])
        
        # if this is not the first query of that session
        if queryID - 1 >= 0:
            # get sessQueryID of the query prior to the current query
            # now we get a pair of query (prevSessQueryID, sessQueryID)
            prevSessQueryID = str(sessID) + "," + str(queryID - 1)
            updateQTable(retDistinctSessQueryID, prevSessQueryID, qObj)
    return distinctQueries

def printQTable(qTable, queryVocab):
    assert len(qTable)==len(queryVocab)
    for key in queryVocab:
        line = str(key)+":"
        line += str(qTable[key])+"\n"
        print(line)
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
                            compareBitMaps(qObj.sessionStreamDict[endDistinctSessQueryID],
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
    print("Number of distinct queries: "+str(len(qObj.queryVocab))+", #cells in QTable: "+str(int(len(qObj.queryVocab)*len(qObj.queryVocab))))
    #print("Expected number of refinement iterations: max("+str(len(qObj.queryVocab))+","+str(int(configDict['QL_REFINE_ITERS']))+")"
    #numRefineIters = max(len(qObj.queryVocab), int(configDict['QL_REFINE_ITERS']))
    # if len(qObj.queryVocab) * len(qObj.queryVocab)/10 <= int(configDict['QL_REFINE_ITERS']):
    numRefineIters = int(configDict['QL_REFINE_ITERS'])
    print("Expected number of refinement iterations: " + str(numRefineIters))
    #else:
        #numRefineIters = min(len(qObj.queryVocab) * len(qObj.queryVocab) / 100, int(configDict['QL_REFINE_ITERS']))
    for i in range(numRefineIters):
        # print message every 100 iteration
        if i%100 == 0:
            print("Refining using Bellman update, Iteration:"+str(i))
        # pick a random start and end sessQueryID pair within the vocabulary in queryVocab
        startSessQueryIndex = random.randrange(len(qObj.queryVocab))
        endSessQueryIndex = random.randrange(len(qObj.queryVocab))
        startDistinctSessQueryID = qObj.queryVocab[startSessQueryIndex]
        endDistinctSessQueryID = qObj.queryVocab[endSessQueryIndex]
        rewVal = assignReward(startDistinctSessQueryID, endDistinctSessQueryID, qObj)
        invokeBellmanUpdate(startDistinctSessQueryID, endSessQueryIndex, qObj, rewVal)
    return

def predictTopKIntents(threadID, qTable, queryVocab, sessQueryID, sessionStreamDict, configDict):
    #print("Inside ThreadID:"+str(threadID))
    (maxCosineSim, maxSimSessQueryID) = findMostSimilarQuery(sessQueryID, queryVocab, sessionStreamDict)
    qValues = qTable[maxSimSessQueryID]
    topK = int(configDict['TOP_K'])
    topKIndices = zip(*heapq.nlargest(topK, enumerate(qValues), key=operator.itemgetter(1)))[0]
    topKSessQueryIndices = []
    for topKIndex in topKIndices:
        topKSessQueryIndices.append(queryVocab[topKIndex])
    #print("maxSimSessQueryID: "+str(maxSimSessQueryID)+", topKIndices: "+str(topKIndices)+", topKSessQueryIndices: "+str(topKSessQueryIndices))
    return topKSessQueryIndices

def predictTopKIntentsPerThread(threadID, t_lo, t_hi, keyOrder, qTable, resList, queryVocab, sessionStreamDict, configDict):
    #printQTable(qTable, queryVocab)
    #print("QueryVocab:"+str(queryVocab))
    for i in range(t_lo, t_hi+1):
        sessQueryID = keyOrder[i]
        sessID = int(sessQueryID.split(",")[0])
        queryID = int(sessQueryID.split(",")[1])
        #curQueryIntent = sessionStreamDict[sessQueryID]
        #if queryID < sessionLengthDict[sessID]-1:
        if str(sessID) + "," + str(queryID + 1) in sessionStreamDict:
            topKSessQueryIndices = predictTopKIntents(threadID, qTable, queryVocab, sessQueryID, sessionStreamDict, configDict)
            for sessQueryID in topKSessQueryIndices:
                #print("Length of sample: "+str(len(sessionSampleDict[int(sessQueryID.split(",")[0])])))
                if sessQueryID not in sessionStreamDict:
                    print("sessQueryID: "+sessQueryID+" not in sessionStreamDict !!")
                    sys.exit(0)
            #print("ThreadID: "+str(threadID)+", computed Top-K="+str(len(topKSessQueryIndices))+\
            #      " Candidates sessID: " + str(sessID) + ", queryID: " + str(queryID))
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
        # print("Set tuple boundaries for Threads")
    # sortedSessKeys = svdObj.sessAdjList.keys().sort()
    if numThreads == 1:
        qObj.resultDict[0] = predictTopKIntentsPerThread((0, lo, hi, keyOrder, qObj.qTable,
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
            argsList.append((threadID, t_lo, t_hi, keyOrder, sharedTable, qObj.resultDict[threadID],
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
    #print("len(resultDict): " + str(len(qObj.resultDict)))
    return qObj.resultDict

def saveModelToFile(qObj):
    QR.writeToPickleFile(
        getConfig(configDict['OUTPUT_DIR']) + configDict['QL_BOOLEAN_NUMERIC_REWARD'] + "_QTable.pickle", qObj.qTable)
    QR.writeToPickleFile(
        getConfig(configDict['OUTPUT_DIR']) + configDict['QL_BOOLEAN_NUMERIC_REWARD'] + "_QLQueryVocab.pickle", qObj.queryVocab)
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
        print("Starting prediction in Episode " + str(qObj.numEpisodes) + ", lo: " + str(lo) + ", hi: " + str(
            hi) + ", len(keyOrder): " + str(len(qObj.keyOrder)))
        if len(qObj.queryVocab) > 2:  # unless at least two rows hard to recommend
            qObj.resultDict = predictIntentsWithoutCurrentBatch(lo, hi, qObj, qObj.keyOrder)
        print("Starting training in Episode " + str(qObj.numEpisodes))
        startTrainTime = time.time()
        (qObj.sessionDict, qObj.queryKeysSetAside) = updateGlobalSessionDict(lo, hi, qObj.keyOrder,
                                                                              qObj.queryKeysSetAside, qObj.sessionDict)
        updateQueryVocabQTable(qObj)

        if len(qObj.queryVocab) > 2:
            refineQTableUsingBellmanUpdate(qObj)
            saveModelToFile(qObj)
            #printQTable(qObj.qTable, qObj.queryVocab) # only enabled for debugging purposes
        totalTrainTime = float(time.time() - startTrainTime)
        print("Total Train Time: " + str(totalTrainTime))
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
            qObj.resultDict = clear(qObj.resultDict)
    updateResultsToExcel(qObj.configDict, qObj.episodeResponseTimeDictName, qObj.outputIntentFileName)

def loadModel(qObj):
    qObj.queryVocab = QR.readFromPickleFile(getConfig(configDict['OUTPUT_DIR']) + configDict['QL_BOOLEAN_NUMERIC_REWARD'] + "_QLQueryVocab.pickle")
    qObj.qTable = QR.readFromPickleFile(getConfig(configDict['OUTPUT_DIR']) + configDict['QL_BOOLEAN_NUMERIC_REWARD'] + "_QTable.pickle")
    print("Loaded len(queryVocab): "+str(len(qObj.queryVocab))+", len(qObj.qTable): "+str(len(qObj.qTable)))
    notInQT = 0
    notInSessionStreamDict = 0
    for key in qObj.queryVocab:
        if key not in qObj.qTable:
            print("key: "+key+" not in qObj.qTable")
            notInQT += 1
        if key not in qObj.sessionStreamDict:
            print("key: "+key+" not in qObj.sessionStreamDict")
            notInSessionStreamDict += 1
    print("notInQT: "+str(notInQT)+", notInSessionStreamDict: "+str(notInSessionStreamDict))
    return

def trainEpisodicModelSustenance(episodicTraining, trainKeyOrder, qObj):
    """
    Train episodic QLearning model
    """
    numTrainEpisodes = 0
    assert episodicTraining == 'True' or episodicTraining == 'False'
    if episodicTraining == 'True':
        # each episode contain 1000 in BusTracker configuration
        batchSize = int(qObj.configDict['EPISODE_IN_QUERIES'])
    elif episodicTraining == 'False':
        batchSize = len(trainKeyOrder)

    lo = 0
    hi = -1
    # assert qObj.configDict['INCLUDE_CUR_SESS'] == "False"
    # separate into episode of 1000 query each
    while hi < len(trainKeyOrder) - 1:
        lo = hi + 1
        # if not enough data left for last batch, shrink batchSize
        if len(trainKeyOrder) - lo < batchSize:
            batchSize = len(trainKeyOrder) - lo
        # update lo, hi to point to region of this batch
        hi = lo + batchSize - 1
        print("Starting training in Episode " + str(numTrainEpisodes))
        # record start train time
        startTrainTime = time.time()
        # train model if we are loading existing model
        if configDict['QL_SUSTENANCE_LOAD_EXISTING_MODEL'] == 'False':
            # qObj.queryKeysSetAside and qObj.sessionDict are being updated
            (qObj.sessionDict, qObj.queryKeysSetAside) = updateGlobalSessionDict(lo, hi, qObj.keyOrder,
                                                                                                   qObj.queryKeysSetAside,
                                                                                                   qObj.sessionDict)
            # update Q table
            updateQueryVocabQTable(qObj)

            if len(qObj.queryVocab) > 2:
                refineQTableUsingBellmanUpdate(qObj)
                saveModelToFile(qObj)
                # printQTable(qObj.qTable, qObj.queryVocab) # only enabled for debugging purposes
        # calculate total training time
        totalTrainTime = float(time.time() - startTrainTime)
        print("Total Train Time: " + str(totalTrainTime))
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
    # retrain a new model                                                                     
    if configDict['QL_SUSTENANCE_LOAD_EXISTING_MODEL'] == 'False':
        episodicTraining = 'True'
        trainEpisodicModelSustenance(episodicTraining, trainKeyOrder, qObj)
    # allow us to load existing models here
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
        print("Starting prediction in Episode " + str(qObj.numEpisodes) + ", lo: " + str(lo) + ", hi: " + str(
            hi) + ", len(testKeyOrder): " + str(len(testKeyOrder))+ ", len(queryVocab): " +str(len(qObj.queryVocab)))
        if len(qObj.queryVocab) > 2:  # unless at least two rows hard to recommend
            qObj.resultDict = predictIntentsWithoutCurrentBatch(lo, hi, qObj, testKeyOrder)
            # we record the times including train and test
            qObj.numEpisodes += 1
            if len(qObj.resultDict) > 0:
                print("appending results")
                elapsedAppendTime = CFCosineSim_Parallel.appendResultsToFile(qObj.sessionStreamDict, qObj.resultDict,
                                                                             elapsedAppendTime, qObj.numEpisodes,
                                                                             qObj.outputIntentFileName, qObj.configDict,
                                                                             -1)
                (qObj.episodeResponseTimeDictName, qObj.episodeResponseTime, qObj.startEpisode,
                 qObj.elapsedAppendTime) = QR.updateResponseTime(
                    qObj.episodeResponseTimeDictName, qObj.episodeResponseTime, qObj.numEpisodes, qObj.startEpisode,
                    elapsedAppendTime)
                qObj.resultDict = clear(qObj.resultDict)
    updateResultsToExcel(qObj.configDict, qObj.episodeResponseTimeDictName, qObj.outputIntentFileName)
    return

def evalSustenance(qObj):
    # split to train test split
    # qObj.keyOrder stores sessionId,queryId
    (trainKeyOrder, testKeyOrder) = splitIntoTrainTestSets(qObj.keyOrder, qObj.configDict)
    # record start time
    sustStartTrainTime = time.time()
    # train model
    trainModelSustenance(trainKeyOrder, qObj)
    # get total train time by subtracting
    sustTotalTrainTime = float(time.time() - sustStartTrainTime)
    print("Sustenace Train Time: "+str(sustTotalTrainTime))
    # test model
    testModelSustenance(testKeyOrder, qObj)

def runQLearning(configDict):
    assert configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY'
    assert configDict['ALGORITHM'] == 'QLEARNING'
    
    # construct QLearning Q_Obj by calling the constructor
    qObj = Q_Obj(configDict)


    assert configDict['QL_SUSTENANCE'] == 'True' or configDict['QL_SUSTENANCE'] == 'False'



    if configDict['QL_SUSTENANCE'] == 'False':
        # train by episode, prequential test then train
        trainTestBatchWise(qObj)
    elif configDict['QL_SUSTENANCE'] == 'True':
        # train test split
        evalSustenance(qObj)

if __name__ == "__main__":
    # configDict = parseConfig.parseConfigFile("configFile.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="Config parameters file", type=str, required=True)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    runQLearning(configDict)