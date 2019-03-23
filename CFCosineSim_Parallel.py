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

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

def OR(sessionSummary, curQueryIntent, configDict):
    if configDict['INTENT_REP'] == 'TUPLE' or configDict['INTENT_REP'] == 'FRAGMENT' or configDict['INTENT_REP'] == 'QUERY':
        assert sessionSummary.size() == curQueryIntent.size()
    idealSize = min(sessionSummary.size(), curQueryIntent.size())
    for i in range(idealSize):
        if curQueryIntent.test(i):
            sessionSummary.set(i)
    return sessionSummary

def ADD(sessionSummary, curQueryIntent, configDict):
    queryTokens = curQueryIntent.split(";")
    sessTokens = sessionSummary.split(";")
    if configDict['INTENT_REP'] == 'TUPLE' or configDict['INTENT_REP'] == 'FRAGMENT' or configDict['INTENT_REP'] == 'QUERY':
        assert len(queryTokens) == len(sessTokens)
    idealSize = min(len(queryTokens), len(sessTokens))
    for i in range(idealSize):
        sessTokens[i] = float(sessTokens[i])+float(queryTokens[i])
    sessionSummary = QR.normalizeWeightedVector(';'.join(sessTokens))
    return sessionSummary

def fetchCurSessSummary(curQueryIntent, sessionSummaries, sessID, configDict):
    if sessID in sessionSummaries:
        curSessSummary = sessionSummaries[sessID]  # bitmap returned
    else:
        curSessSummary = createEntrySimilarTo(curQueryIntent, configDict)
    return curSessSummary

def computePredSessSummary(curQueryIntent, sessionSummaries, sessID, configDict):
    alpha = 0.5  # fixed does not change so no problem hardcoding
    predSessSummary = []
    if sessID in sessionSummaries:
        curSessSummary = sessionSummaries[sessID] #predSessSummary is a list coz it will consist of weights and floats, but curSessSummary is either a bitmap or a string separated by ;s
    else:
        curSessSummary = createEntrySimilarTo(curQueryIntent, configDict)
    if configDict['BIT_OR_WEIGHTED'] == 'BIT':
        for i in range(curSessSummary.size()):
            if curSessSummary.test(i):
                predSessSummary.append(alpha)
            else:
                predSessSummary.append(0)
    elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
        curSessionTokens = curSessSummary.split(";")
        for i in range(len(curSessionTokens)):
            predSessSummary.append(float(curSessionTokens[i] * alpha))
    for index in sessionSummaries:
        if index != sessID:
            oldSessionSummary = sessionSummaries[index]
            if configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
                cosineSim = computeWeightedCosineSimilarity(curSessSummary, oldSessionSummary, ";", configDict)
                idealSize = min(len(predSessSummary), len(oldSessionSummary.split(";")))
            elif configDict['BIT_OR_WEIGHTED'] == 'BIT':
                cosineSim = computeBitCosineSimilarity(curSessSummary, oldSessionSummary)
                idealSize = min(len(predSessSummary), oldSessionSummary.size())
            if configDict['INTENT_REP'] == 'TUPLE' or configDict['INTENT_REP'] == 'FRAGMENT' or configDict['INTENT_REP'] == 'QUERY':
                if configDict['BIT_OR_WEIGHTED'] == 'BIT':
                    assert len(predSessSummary) == oldSessionSummary.size()
                elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
                    assert len(predSessSummary) == len(oldSessionSummary.split(";"))
            for i in range(idealSize):
                if configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
                    predSessSummary[i] = predSessSummary[i]+ (1-alpha)*cosineSim*oldSessionSummary[i]
                elif configDict['BIT_OR_WEIGHTED'] == 'BIT' and oldSessionSummary.test(i):
                    predSessSummary[i] = predSessSummary[i] + (1-alpha)*cosineSim*1.0
    return predSessSummary

def createEntrySimilarTo(curQueryIntent, configDict):
    if configDict['BIT_OR_WEIGHTED']=='BIT':
        sessSumEntry = BitMap.fromstring(curQueryIntent.tostring())
    elif configDict['BIT_OR_WEIGHTED']=='WEIGHTED':
        sessSumEntry = curQueryIntent
    return sessSumEntry

def refineSessionSummaries(sessID, configDict, curQueryIntent, sessionSummaries):
    if sessID in sessionSummaries:
        if configDict['BIT_OR_WEIGHTED']=='BIT':
            sessionSummaries[sessID] = OR(sessionSummaries[sessID],curQueryIntent, configDict)
        elif configDict['BIT_OR_WEIGHTED']=='WEIGHTED':
            sessionSummaries[sessID] = ADD(sessionSummaries[sessID],curQueryIntent, configDict)
    else:
        sessionSummaries[sessID] = createEntrySimilarTo(curQueryIntent, configDict)
    return sessionSummaries

def computeBitCosineSimilarity(curSessionSummary, oldSessionSummary):
    nonzeroDimsCurSess = curSessionSummary.nonzero() # set of all 1-bit dimensions in curQueryIntent
    nonzeroDimsOldSess = oldSessionSummary.nonzero() # set of all 1-bit dimensions in sessionSummary
    numSetBitsIntersect = len(list(set(nonzeroDimsCurSess) & set(nonzeroDimsOldSess)))  # number of overlapping one bit dimensions
    l2NormProduct = math.sqrt(len(nonzeroDimsCurSess)) * math.sqrt(len(nonzeroDimsOldSess))
    cosineSim = float(numSetBitsIntersect)/l2NormProduct
    #assert cosineSim >=0 and cosineSim < 1.1
    return cosineSim

def computeListBitCosineSimilarityPredictOnlyOptimized(predSessSummary, oldSessionSummary, configDict):
    #if configDict['INTENT_REP'] == 'TUPLE' or configDict['INTENT_REP'] == 'FRAGMENT' or configDict['INTENT_REP'] == 'QUERY':
        #assert(len(predSessSummary))==oldSessionSummary.size()
    #idealSize = min(len(predSessSummary), oldSessionSummary.size())
    numerator = 0.0
    setDims = oldSessionSummary.nonzero()
    #No need to compute L2-norm for predSess because it is the same for all vectors being compared
    for i in setDims:
        #assert oldSessionSummary.test(i)
        numerator += float(predSessSummary[i])
    #if oldSessionSummary.count() == 0:
        #print "L2NormSquares cannot be zero !!"
        #sys.exit(0)
    cosineSim = numerator / math.sqrt(oldSessionSummary.count())
    return cosineSim


def computeListBitCosineSimilarity(predSessSummary, oldSessionSummary, configDict):
    if configDict['INTENT_REP'] == 'TUPLE' or configDict['INTENT_REP'] == 'FRAGMENT' or configDict['INTENT_REP'] == 'QUERY':
        assert(len(predSessSummary))==oldSessionSummary.size()
    idealSize = min(len(predSessSummary), oldSessionSummary.size())
    numerator = 0.0
    l2NormPredSess = 0.0
    l2NormOldSess = 0.0
    for i in range(len(predSessSummary)):
        l2NormPredSess += float(predSessSummary[i] * predSessSummary[i])
    for i in range(oldSessionSummary.size()):
        if oldSessionSummary.test(i):
            l2NormOldSess += float(1.0 * 1.0)
    for i in range(idealSize):
        predSessDim = predSessSummary[i]
        if oldSessionSummary.test(i):
            numerator += float(predSessDim * 1.0)
    if l2NormOldSess == 0 or l2NormPredSess == 0:
        print "L2NormSquares cannot be zero !!"
        sys.exit(0)
    cosineSim = numerator / (math.sqrt(l2NormPredSess) * math.sqrt(l2NormOldSess))
    return cosineSim

def computeWeightedCosineSimilarity(curSessionSummary, oldSessionSummary, delimiter, configDict):
    curSessDims = curSessionSummary.split(delimiter)
    oldSessDims = oldSessionSummary.split(delimiter)
    if configDict['INTENT_REP'] == 'TUPLE' or configDict['INTENT_REP'] == 'FRAGMENT' or configDict['INTENT_REP'] == 'QUERY':
        assert len(curSessDims) == len(oldSessDims)
    idealSize = min(len(curSessDims), len(oldSessDims))
    numerator = 0.0
    l2NormQuery = 0.0
    l2NormSession = 0.0
    for i in range(len(curSessDims)):
        l2NormQuery = l2NormQuery + float(curSessDims[i] * curSessDims[i])
    for i in range(len(oldSessDims)):
        l2NormSession = l2NormSession + float(oldSessDims[i] * oldSessDims[i])
    for i in range(idealSize):
        numerator = numerator + float(curSessDims[i] * oldSessDims[i])
    if l2NormQuery == 0 or l2NormSession == 0:
        print "L2NormSquares cannot be zero !!"
        sys.exit(0)
    cosineSim = numerator / (math.sqrt(l2NormQuery) * math.sqrt(l2NormSession))
    return cosineSim

def computeListWeightedCosineSimilarity(predSessSummary, oldSessionSummary, delimiter, configDict):
    oldSessDims = oldSessionSummary.split(delimiter)
    if configDict['INTENT_REP'] == 'TUPLE' or configDict['INTENT_REP'] == 'FRAGMENT' or configDict['INTENT_REP'] == 'QUERY':
        assert len(predSessSummary) == len(oldSessDims)
    idealSize = min(len(predSessSummary), len(oldSessDims))
    numerator = 0.0
    l2NormQuery = 0.0
    l2NormSession = 0.0
    for i in range(len(predSessSummary)):
        l2NormQuery = l2NormQuery + float(predSessSummary[i] * predSessSummary[i])
    for i in range(len(oldSessDims)):
        l2NormSession = l2NormSession + float(oldSessDims[i] * oldSessDims[i])
    for i in range(idealSize):
        numerator = numerator + float(predSessSummary[i] * oldSessDims[i])
    if l2NormQuery == 0 or l2NormSession == 0:
        print "L2NormSquares cannot be zero !!"
        sys.exit(0)
    cosineSim = numerator / (math.sqrt(l2NormQuery) * math.sqrt(l2NormSession))
    return cosineSim

def findTopKSessIndex(topCosineSim, cosineSimDict, topKSessindices):
    if topCosineSim not in cosineSimDict:
        print "cosineSimilarity not found in the dictionary !!"
        sys.exit(0)
    for sessIndex in cosineSimDict[topCosineSim]:
        if sessIndex not in topKSessindices:
            return sessIndex

def popTopKfromHeap(configDict, minheap, cosineSimDict):
    topKIndices = []
    numElemToPop = int(configDict['TOP_K'])
    if len(minheap) < numElemToPop:
        numElemToPop = len(minheap)
    #print "len(minheap): "+str(len(minheap))+", numElemToPop: "+str(numElemToPop)
    while len(topKIndices) < numElemToPop and len(minheap)>0:
        topCosineSim = 0 - (heapq.heappop(minheap))  # negated to get back the item
        topKIndex = findTopKSessIndex(topCosineSim, cosineSimDict, topKIndices)
        if topKIndex is not None:
            topKIndices.append(topKIndex)
    return (minheap, topKIndices)

def insertIntoMinSessHeap(minheap, cosineSim, cosineSimDict, insertKey):
    heapq.heappush(minheap, -cosineSim)  # insert -ve cosineSim
    if cosineSim not in cosineSimDict:
        cosineSimDict[cosineSim] = list()
    cosineSimDict[cosineSim].append(insertKey)
    return (minheap, cosineSimDict)

def insertIntoMinQueryHeap(minheap, sessionSampleDict, sessionStreamDict, configDict, cosineSimDict, curSessSummary, topKSessIndex):
    for sessQueryIndex in sessionSampleDict[topKSessIndex]:
        elem = sessionStreamDict[sessQueryIndex]
        assert configDict['BIT_OR_WEIGHTED'] == 'BIT'
        cosineSim = computeBitCosineSimilarity(curSessSummary, elem)
        #assert cosineSim >= 0 and cosineSim <= 1
        heapq.heappush(minheap, -cosineSim)  # insert -ve cosineSim
        if cosineSim not in cosineSimDict:
            cosineSimDict[cosineSim] = list()
        cosineSimDict[cosineSim].append(sessQueryIndex)
    return (minheap, cosineSimDict)

def computeSessSimilaritySingleThread(sessionSummaries, curSessSummary):
    sessSimDict = {}
    for sessID in sessionSummaries:
        prevSessSummary = sessionSummaries[sessID]
        sessSim = computeBitCosineSimilarity(curSessSummary, prevSessSummary)
        #assert sessSim >=0 and sessSim <=1
        sessSimDict[sessID] = sessSim
    return sessSimDict

def computeSessSimilarityMultiThread((threadID, subThreadID, sessPartition, sessionSummaries, curSessSummary, configDict)):
    sessSimDict = {}
    for sessID in sessPartition:
        prevSessSummary = sessionSummaries[sessID]
        sessSim = computeBitCosineSimilarity(curSessSummary, prevSessSummary)
        sessSimDict[sessID] = sessSim
    QR.writeToPickleFile(
        getConfig(configDict['PICKLE_TEMP_OUTPUT_DIR']) + "CFSessSimDict_" + str(threadID) + "_" + str(subThreadID)+ ".pickle", sessSimDict)
    return

def partitionSessionsAmongSubThreads(numSubThreads, sessionSummaries, curSessID):
    #numSessPerThread = int(len(sessionSummaries) / numSubThreads)
    # round robin assignment of queries to threads
    sessPartitions = {}
    for i in range(numSubThreads):
        sessPartitions[i] = []
    sessCount = 0
    for sessID in sessionSummaries:
        if sessID != curSessID:
            sessCount += 1
            subThreadID = sessCount % numSubThreads
            sessPartitions[subThreadID].append(sessID)
    return sessPartitions

def concatenateLocalDicts(localCosineSimDicts, cosineSimDict):
    for subThreadID in localCosineSimDicts:
        for sessID in localCosineSimDicts[subThreadID]:
            cosineSimDict[sessID] = localCosineSimDicts[subThreadID][sessID]
    return cosineSimDict

def predictTopKIntents(threadID, curQueryIntent, sessionSummaries, sessionSampleDict, sessionStreamDict, sessID, configDict):
    # python supports for min-heap not max-heap so negate items and insert into min-heap
    curSessSummary = fetchCurSessSummary(curQueryIntent, sessionSummaries, sessID, configDict)
    minheap = []
    sessSimDict = {}
    # compute cosine similarity in parallel between curSessSummary and all the sessions from sessionSummaries
    numSubThreads = min(int(configDict['CF_SUB_THREADS']), len(sessionSummaries))

    if numSubThreads == 1:
        sessSimDict = computeSessSimilaritySingleThread(sessionSummaries, curSessSummary)
    else:
        manager = multiprocessing.Manager()
        sharedSessSummaryDict = manager.dict()
        for sessID in sessionSummaries:
            sharedSessSummaryDict[sessID] = sessionSummaries[sessID]
        sessPartitions = partitionSessionsAmongSubThreads(numSubThreads, sessionSummaries, sessID)
        pool = multiprocessing.Pool()
        argsList = []
        localSessSimDicts = {}
        for subThreadID in range(numSubThreads):
            argsList.append((threadID, subThreadID, sessPartitions[subThreadID], sharedSessSummaryDict, curSessSummary, configDict))
            # threads[i] = threading.Thread(target=predictTopKIntentsPerThread, args=(i, t_lo, t_hi, keyOrder, resList, sessionDict, sessionSampleDict, sessionStreamDict, sessionLengthDict, configDict))
            # threads[i].start()
        pool.map(computeSessSimilarityMultiThread, argsList)
        pool.close()
        pool.join()
        for subThreadID in range(numSubThreads):
            localSessSimDicts[subThreadID] = QR.readFromPickleFile(
                getConfig(configDict['PICKLE_TEMP_OUTPUT_DIR']) + "CFSessSimDict_" + str(threadID) + "_" + str(
                    subThreadID) + ".pickle")
        sessSimDict = concatenateLocalDicts(localSessSimDicts, sessSimDict)
    #sorted_csd = sorted(sessSimDict.items(), key=operator.itemgetter(1), reverse=True)
    cosineSimDict = {}
    for sessIndex in sessSimDict: # exclude the current session
        if sessIndex != sessID:
            (minheap, cosineSimDict) = insertIntoMinSessHeap(minheap, sessSimDict[sessIndex], cosineSimDict, sessIndex)
    if len(minheap) > 0:
        (minheap, topKSessIndices) = popTopKfromHeap(configDict, minheap, cosineSimDict)
        #print "ThreadID: "+str(threadID)+", Found Top-K Sessions"
    else:
        return (None, None)

    del minheap
    minheap = []
    del cosineSimDict
    cosineSimDict = {}
    topKSessQueryIndices = None
    for topKSessIndex in topKSessIndices:
        (minheap, cosineSimDict) = insertIntoMinQueryHeap(minheap, sessionSampleDict, sessionStreamDict, configDict, cosineSimDict, curSessSummary, topKSessIndex)
    if len(minheap) > 0:
        (minheap, topKSessQueryIndices) = popTopKfromHeap(configDict, minheap, cosineSimDict)
        #print "ThreadID: "+str(threadID)+", Found Top-K Queries"
    '''
    topKPredictedIntents = []
    for topKSessQueryIndex in topKSessQueryIndices:
        topKSessIndex = int(topKSessQueryIndex.split(",")[0])
        topKQueryIndex = int(topKSessQueryIndex.split(",")[1])
        topKIntent = sessionDict[topKSessIndex][topKQueryIndex]
        topKPredictedIntents.append(topKIntent)
    '''
    return topKSessQueryIndices

def saveModel(configDict, sessionSummaries):
    sessionSummaryFile =  getConfig(configDict['OUTPUT_DIR']) + "/SessionSummaries_" + configDict[
        'ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF'] + "_" + configDict['INTENT_REP'] + "_" + \
                                  configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                                      'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES'] + ".pickle"
    QR.writeToPickleFile(sessionSummaryFile, sessionSummaries)
    return

def refineSessionSummariesForAllQueriesSetAside(queryKeysSetAside, configDict, sessionSummaries, sessionStreamDict):
    for key in queryKeysSetAside:
        sessID = int(key.split(",")[0])
        queryID = int(key.split(",")[1])
        curQueryIntent = sessionStreamDict[key]
        sessionSummaries = refineSessionSummaries(sessID, configDict, curQueryIntent, sessionSummaries)
    saveModel(configDict, sessionSummaries)
    return sessionSummaries

def runCFCosineSimKFoldExp(configDict):
    intentSessionFile = QR.fetchIntentFileFromConfigDict(configDict)
    kFoldOutputIntentFiles = []
    kFoldEpisodeResponseTimeDicts = []
    avgTrainTime = []
    avgTestTime = []
    algoName = configDict['ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF']
    for foldID in range(int(configDict['KFOLD'])):
        outputIntentFileName = getConfig(configDict['KFOLD_OUTPUT_DIR']) + "/OutputFileShortTermIntent_" + configDict['ALGORITHM'] + "_" + \
                                  configDict['CF_COSINESIM_MF'] + "_" + \
                                  configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + \
                                  configDict['TOP_K'] + "_FOLD_" + str(foldID)
        episodeResponseTimeDictName = getConfig(configDict['KFOLD_OUTPUT_DIR']) + "/ResponseTimeDict_" + configDict['ALGORITHM'] + "_" + \
                                      configDict['CF_COSINESIM_MF'] + "_" + \
                                      configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + \
                                      configDict['TOP_K'] + "_FOLD_" + str(foldID) + ".pickle"
        trainIntentSessionFile = getConfig(configDict['KFOLD_INPUT_DIR'])+intentSessionFile.split("/")[len(intentSessionFile.split("/"))-1]+"_TRAIN_FOLD_"+str(foldID)
        testIntentSessionFile = getConfig(configDict['KFOLD_INPUT_DIR']) + intentSessionFile.split("/")[len(intentSessionFile.split("/")) - 1] + "_TEST_FOLD_" + str(foldID)
        (sessionSummaries, sessionDict, sessionLengthDict, sessionStreamDict, keyOrder, episodeResponseTime) = initCFCosineSimOneFold(trainIntentSessionFile, configDict)
        startTrain = time.time()
        (sessionDict, sessionSummaries) = refineSessionSummariesForAllQueriesSetAside(keyOrder, configDict, sessionDict, sessionSummaries, sessionStreamDict)
        trainTime = float(time.time() - startTrain)
        avgTrainTime.append(trainTime)
        (testSessionSummaries, testSessionDict, sessionLengthDict, testSessionStreamDict, testKeyOrder, testEpisodeResponseTime) = initCFCosineSimOneFold(testIntentSessionFile, configDict)
        startTest = time.time()
        testCFCosineSim(foldID, testIntentSessionFile, outputIntentFileName, sessionDict, sessionSummaries, sessionLengthDict, testSessionStreamDict, testEpisodeResponseTime, episodeResponseTimeDictName, configDict)
        testTime = float(time.time() - startTest)
        avgTestTime.append(testTime)
        kFoldOutputIntentFiles.append(outputIntentFileName)
        kFoldEpisodeResponseTimeDicts.append(episodeResponseTimeDictName)
        (avgTrainTimeFN, avgTestTimeFN) = QR.writeKFoldTrainTestTimesToPickleFiles(avgTrainTime, avgTestTime, algoName, configDict)
    QR.avgKFoldTimeAndQualityPlots(kFoldOutputIntentFiles,kFoldEpisodeResponseTimeDicts, avgTrainTimeFN, avgTestTimeFN, algoName, configDict)
    return

def testCFCosineSim(foldID, testIntentSessionFile, outputIntentFileName, sessionDict, sessionSummaries, sessionLengthDict, sessionStreamDict, episodeResponseTime, episodeResponseTimeDictName, configDict):
    try:
        os.remove(outputIntentFileName)
    except OSError:
        pass
    numEpisodes = 1
    startEpisode = time.time()
    prevSessID = -1
    elapsedAppendTime = 0.0
    with open(testIntentSessionFile) as f:
        for line in f:
            (sessID, queryID, curQueryIntent) = QR.retrieveSessIDQueryIDIntent(line, configDict)
            # we need to delete previous test session entries from the summary
            if prevSessID!=sessID:
                if prevSessID in sessionDict:
                    assert prevSessID in sessionSummaries
                    del sessionDict[prevSessID]
                    del sessionSummaries[prevSessID]
                    (episodeResponseTime, startEpisode, elapsedAppendTime) = QR.updateResponseTime(episodeResponseTime,
                                                                                                   numEpisodes,
                                                                                                   startEpisode,
                                                                                                   elapsedAppendTime)
                    numEpisodes += 1  # here numEpisodes is analogous to numSessions
                prevSessID = sessID

            queryKeysSetAside = []
            queryKeysSetAside.append(str(sessID)+","+str(queryID))
            (sessionDict, sessionSummaries) = refineSessionSummariesForAllQueriesSetAside(queryKeysSetAside, configDict,
                                                                                          sessionSummaries,
                                                                                          sessionStreamDict)
            (topKSessQueryIndices, topKPredictedIntents) = predictTopKIntents(sessionSummaries, sessionDict, sessID,
                                                                      curQueryIntent, configDict)
            if queryID+1 >= int(sessionLengthDict[sessID]):
                continue

            nextQueryIntent = sessionStreamDict[str(sessID) + "," + str(queryID + 1)]
            elapsedAppendTime += QR.appendPredictedIntentsToFile(topKSessQueryIndices, topKPredictedIntents,
                                                        sessID, queryID, nextQueryIntent, numEpisodes,
                                                        configDict, outputIntentFileName, foldID)
        (episodeResponseTime, startEpisode, elapsedAppendTime) = QR.updateResponseTime(episodeResponseTime,
                                                                                       numEpisodes,
                                                                                       startEpisode,
                                                                                       elapsedAppendTime) # last session
        QR.writeToPickleFile(episodeResponseTimeDictName, episodeResponseTime)
    f.close()
    return episodeResponseTimeDictName

def initCFCosineSimOneFold(trainIntentSessionFile, configDict):
    sessionSummaries = {}  # key is sessionID and value is summary
    sessionDict = {}  # key is session ID and value is a list of query intent vectors; no need to store the query itself
    sessionStreamDict = {}
    keyOrder = []
    episodeResponseTime = {}
    sessionLengthDict = ConcurrentSessions.countQueries(getConfig(configDict['QUERYSESSIONS']))
    with open(trainIntentSessionFile) as f:
        for line in f:
            (sessID, queryID, curQueryIntent, sessionStreamDict) = QR.updateSessionDict(line, configDict,
                                                                                        sessionStreamDict)
            keyOrder.append(str(sessID) + "," + str(queryID))
    f.close()
    return (sessionSummaries, sessionDict, sessionLengthDict, sessionStreamDict, keyOrder, episodeResponseTime)

def initCFCosineSimSingularity(configDict):
    intentSessionFile = QR.fetchIntentFileFromConfigDict(configDict)
    episodeResponseTimeDictName = getConfig(configDict['OUTPUT_DIR']) + "/ResponseTimeDict_" + configDict[
        'ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF'] + "_" + configDict['INTENT_REP'] + "_" + \
                                  configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                                      'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES'] + ".pickle"
    outputIntentFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputFileShortTermIntent_" + configDict[
        'ALGORITHM'] + "_" + \
                           configDict['CF_COSINESIM_MF'] + "_" + \
                           configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                               'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']
    sessionSummaries = {}  # key is sessionID and value is summary
    sessionSampleDict = {} # key is sessionID and value is a list of sampled intent vectors
    numEpisodes = 0
    queryKeysSetAside = []
    episodeResponseTime = {}
    #sessionLengthDict = ConcurrentSessions.countQueries(getConfig(configDict['QUERYSESSIONS']))
    try:
        os.remove(outputIntentFileName)
    except OSError:
        pass
    if int(configDict['CF_THREADS'])>1:
        manager = multiprocessing.Manager()
        sessionStreamDict = manager.dict()
    else:
        sessionStreamDict = {}
    resultDict = {}
    keyOrder = []
    with open(intentSessionFile) as f:
        for line in f:
            (sessID, queryID, curQueryIntent, sessionStreamDict) = QR.updateSessionDict(line, configDict,
                                                                                        sessionStreamDict)
            keyOrder.append(str(sessID) + "," + str(queryID))
    f.close()
    startEpisode = time.time()
    return (sessionSummaries, sessionSampleDict, queryKeysSetAside, resultDict, sessionStreamDict, numEpisodes,
     episodeResponseTimeDictName, episodeResponseTime, keyOrder, startEpisode, outputIntentFileName)

def updateSessionHistory(distinctQueriesSessWise, sessionSampleDict, configDict):
    sampleFrac = float(configDict['CF_SAMPLING_FRACTION'])
    for sessID in sessionSampleDict:
        distinctSessCount = len(distinctQueriesSessWise[sessID])
        count = int(float(distinctSessCount) * sampleFrac)
        if count == 0:
            count = 1
        if count > 0:
            batchSize = int(len(distinctQueriesSessWise[sessID]) / count)
            if batchSize == 0:
                batchSize = 1
            curIndex = 0
            covered = 0
            while covered < count and curIndex < len(distinctQueriesSessWise[sessID]):
                sessionSampleDict[sessID].append(distinctQueriesSessWise[sessID][curIndex])
                curIndex += batchSize
                covered += 1
    return sessionSampleDict


def updateSampledQueryDictHistory(configDict, sessionSampleDict, queryKeysSetAside, sessionStreamDict):
    distinctQueriesSessWise = {} # key is sessID and value is a list of distinct keys
    for sessQueryID in queryKeysSetAside:
        sessID = int(sessQueryID.split(",")[0])
        if sessID not in distinctQueriesSessWise:
            distinctQueriesSessWise[sessID] = []
        if sessID not in sessionSampleDict:
            sessionSampleDict[sessID] = []
        if LSTM_RNN_Parallel.findIfQueryInside(sessQueryID, sessionStreamDict, sessionSampleDict[sessID],
                                               distinctQueriesSessWise[sessID]) == "False":
            distinctQueriesSessWise[sessID].append(sessQueryID)
    sessionSampleDict = updateSessionHistory(distinctQueriesSessWise, sessionSampleDict, configDict)
    del distinctQueriesSessWise
    return sessionSampleDict

def updateResultsToExcel(configDict, episodeResponseTimeDictName, outputIntentFileName):
    accThres = float(configDict['ACCURACY_THRESHOLD'])
    QR.evaluateQualityPredictions(outputIntentFileName, configDict, accThres,
                                  configDict['ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF'])
    print "--Completed Quality Evaluation for accThres:" + str(accThres)
    QR.evaluateTimePredictions(episodeResponseTimeDictName, configDict,
                               configDict['ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF'])

    outputEvalQualityFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputEvalQualityShortTermIntent_" + configDict[
        'ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF'] + "_" + configDict['INTENT_REP'] + "_" + configDict[
                                    'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                                configDict['EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres)
    outputExcelQuality = getConfig(configDict['OUTPUT_DIR']) + "/OutputExcelQuality_" + configDict['ALGORITHM'] + "_" + \
                         configDict['CF_COSINESIM_MF'] + "_" + configDict['INTENT_REP'] + "_" + configDict[
                             'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict[
                             'EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres) + ".xlsx"
    ParseResultsToExcel.parseQualityFileWithEpisodeRep(outputEvalQualityFileName, outputExcelQuality, configDict)

    outputEvalTimeFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputEvalTimeShortTermIntent_" + configDict[
        'ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF'] + "_" + configDict['INTENT_REP'] + "_" + configDict[
                                 'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                             configDict['EPISODE_IN_QUERIES']
    outputExcelTimeEval = getConfig(configDict['OUTPUT_DIR']) + "/OutputExcelTime_" + configDict['ALGORITHM'] + "_" + \
                          configDict['CF_COSINESIM_MF'] + "_" + configDict['INTENT_REP'] + "_" + configDict[
                              'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                          configDict['EPISODE_IN_QUERIES'] + ".xlsx"
    ParseResultsToExcel.parseTimeFile(outputEvalTimeFileName, outputExcelTimeEval)
    return (outputIntentFileName, episodeResponseTimeDictName)

def predictTopKIntentsPerThread((threadID, t_lo, t_hi, keyOrder, resList, sessionSummaries, sessionSampleDict, sessionStreamDict, configDict)):
    for i in range(t_lo, t_hi+1):
        sessQueryID = keyOrder[i]
        sessID = int(sessQueryID.split(",")[0])
        queryID = int(sessQueryID.split(",")[1])
        curQueryIntent = sessionStreamDict[sessQueryID]
        #if queryID < sessionLengthDict[sessID]-1:
        if str(sessID) + "," + str(queryID + 1) in sessionStreamDict:
            topKSessQueryIndices = predictTopKIntents(threadID, curQueryIntent, sessionSummaries, sessionSampleDict, sessionStreamDict,
                                                                              sessID, configDict)
            for sessQueryID in topKSessQueryIndices:
                if sessQueryID not in sessionStreamDict:
                    print "sessQueryID: "+sessQueryID+" not in sessionStreamDict !!"
                    sys.exit(0)
            print "ThreadID: "+str(threadID)+", computed Top-K="+str(len(topKSessQueryIndices))+" Candidates sessID: " + str(sessID) + ", queryID: " + str(queryID)
            if topKSessQueryIndices is not None:
                resList.append((sessID, queryID, topKSessQueryIndices))
    QR.writeToPickleFile(
        getConfig(configDict['PICKLE_TEMP_OUTPUT_DIR']) + "CFCosineSimResList_" + str(threadID) + ".pickle", resList)
    return resList


def predictIntentsWithoutCurrentBatch(lo, hi, keyOrder, resultDict, sessionSummaries, sessionSampleDict, sessionStreamDict, configDict):
    numThreads = min(int(configDict['CF_THREADS']), hi-lo+1)
    numKeysPerThread = int(float(hi - lo + 1) / float(numThreads))
    #threads = {}
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
    if numThreads == 1:
        predictTopKIntentsPerThread((0, lo, hi, keyOrder, resultDict[0], sessionSummaries, sessionSampleDict, sessionStreamDict, configDict))
    elif numThreads > 1:
        if int(configDict['CF_SUB_THREADS']) == 1:
            pool = multiprocessing.Pool()
        if int(configDict['CF_SUB_THREADS']) > 1:
            pool = ThreadPool()
        argsList = []
        for threadID in range(numThreads):
            (t_lo, t_hi) = t_loHiDict[threadID]
            argsList.append((threadID, t_lo, t_hi, keyOrder, resultDict[threadID], sessionSummaries, sessionSampleDict, sessionStreamDict, configDict))
            #threads[i] = threading.Thread(target=predictTopKIntentsPerThread, args=(i, t_lo, t_hi, keyOrder, resList, sessionDict, sessionSampleDict, sessionStreamDict, sessionLengthDict, configDict))
            #threads[i].start()
        pool.map(predictTopKIntentsPerThread, argsList)
        pool.close()
        pool.join()
        for threadID in range(numThreads):
            resultDict[threadID] = QR.readFromPickleFile(
                getConfig(configDict['PICKLE_TEMP_OUTPUT_DIR']) + "CFCosineSimResList_" + str(threadID) + ".pickle")
    return resultDict

def appendResultsToFile(sessionStreamDict, resultDict, elapsedAppendTime, numEpisodes, outputIntentFileName, configDict, foldID):
    for threadID in resultDict:
        for i in range(len(resultDict[threadID])):
            (sessID, queryID, topKSessQueryIDs) = resultDict[threadID][i]
            nextQueryIntent = sessionStreamDict[str(sessID)+","+str(queryID+1)]
            topKPredictedIntents = []
            for sessQueryID in topKSessQueryIDs:
                topKPredictedIntents.append(sessionStreamDict[sessQueryID])
            elapsedAppendTime += QR.appendPredictedRNNIntentToFile(sessID, queryID, topKPredictedIntents,
                                                                   nextQueryIntent, numEpisodes,
                                                                   outputIntentFileName, configDict, foldID)
    return elapsedAppendTime


def updateQueriesSetAside(lo, hi, keyOrder, queryKeysSetAside):
    cur = lo
    while(cur<hi+1):
        sessQueryID = keyOrder[cur]
        queryKeysSetAside.append(sessQueryID)
        cur+=1
    return queryKeysSetAside


def trainTestBatchWise(sessionSummaries, sessionSampleDict, queryKeysSetAside, resultDict, sessionStreamDict, numEpisodes,
     episodeResponseTimeDictName, episodeResponseTime, keyOrder, startEpisode, outputIntentFileName):
    batchSize = int(configDict['EPISODE_IN_QUERIES'])
    lo = 0
    hi = -1
    assert configDict['INCLUDE_CUR_SESS'] == "False"  # you never recommend queries from current session coz it is the most similar to the query you have
    while hi < len(keyOrder) - 1:
        lo = hi + 1
        if len(keyOrder) - lo < batchSize:
            batchSize = len(keyOrder) - lo
        hi = lo + batchSize - 1
        elapsedAppendTime = 0.0

        # test first for each query in the batch if the classifier is not None
        print "Starting prediction in Episode " + str(numEpisodes) + ", lo: " + str(lo) + ", hi: " + str(
            hi) + ", len(keyOrder): " + str(len(keyOrder))
        # model is the sessionSummaries
        if len(sessionSummaries) > 0:
            # predict queries for the batch
            resultDict = predictIntentsWithoutCurrentBatch(lo, hi, keyOrder, resultDict, sessionSummaries, sessionSampleDict, sessionStreamDict, configDict)
        print "Starting training in Episode " + str(numEpisodes)
        # update SessionDictGlobal and train with the new batch
        queryKeysSetAside = updateQueriesSetAside(lo, hi, keyOrder, queryKeysSetAside)
        sessionSampleDict = updateSampledQueryDictHistory(configDict, sessionSampleDict, queryKeysSetAside, sessionStreamDict)
        # -- Refinement and prediction is done at every query, episode update alone is done at end of the episode --
        sessionSummaries = refineSessionSummariesForAllQueriesSetAside(queryKeysSetAside, configDict, sessionSummaries, sessionStreamDict)
        assert configDict['CF_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL' or configDict['CF_INCREMENTAL_OR_FULL_TRAIN'] == 'FULL'
        # we have empty queryKeysSetAside because we want to incrementally train the CF at the end of each episode
        if configDict['CF_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL':
            del queryKeysSetAside
            queryKeysSetAside = []
        # we record the times including train and test
        numEpisodes += 1
        if len(resultDict) > 0:
            elapsedAppendTime = appendResultsToFile(sessionStreamDict, resultDict, elapsedAppendTime, numEpisodes, outputIntentFileName, configDict, -1)
            (episodeResponseTimeDictName, episodeResponseTime, startEpisode, elapsedAppendTime) = QR.updateResponseTime(episodeResponseTimeDictName, episodeResponseTime, numEpisodes, startEpisode, elapsedAppendTime)
            resultDict = LSTM_RNN_Parallel.clear(resultDict)
    updateResultsToExcel(configDict, episodeResponseTimeDictName, outputIntentFileName)


def runCFCosineSimSingularityExp(configDict):
    (sessionSummaries, sessionSampleDict, queryKeysSetAside, resultDict, sessionStreamDict, numEpisodes,
     episodeResponseTimeDictName, episodeResponseTime, keyOrder, startEpisode, outputIntentFileName) = initCFCosineSimSingularity(configDict)
    trainTestBatchWise(sessionSummaries, sessionSampleDict, queryKeysSetAside, resultDict, sessionStreamDict, numEpisodes,
     episodeResponseTimeDictName, episodeResponseTime, keyOrder, startEpisode, outputIntentFileName)



def runCFCosineSim(configDict):
    if configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY':
        runCFCosineSimSingularityExp(configDict)
    elif configDict['SINGULARITY_OR_KFOLD'] == 'KFOLD':
        runCFCosineSimKFoldExp(configDict)

if __name__ == "__main__":
    #configDict = parseConfig.parseConfigFile("configFile.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="Config parameters file", type=str, required=True)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    runCFCosineSim(configDict)



'''
def findLatestIntentPredictedSoFar(sessID, queryID, topKPredictedIntentDict, topKSessQueryIndicesDict, sessionLengthDict):
    curSessID = sessID
    while curSessID >= 0:
        if curSessID in topKPredictedIntentDict:
            if curSessID == sessID:
                curQueryID = queryID-1 # u shd start with the prev queryID to check if there was a prediction made at that query
            else:
                curQueryID = sessionLengthDict[sessID]-1 # last index in a session is count-1
            while curQueryID >= 0:
                if curQueryID in topKPredictedIntentDict:
                    return (topKSessQueryIndicesDict[curSessID][curQueryID], topKPredictedIntentDict[curSessID][curQueryID])
                curQueryID = curQueryID-1
        curSessID = curSessID - 1
    print "Could not find sessID, queryID !!"
    sys.exit(0)

def insertIntoTopKDict(sessID, queryID, topKIndices, topKIndicesDict):
    if sessID in topKIndicesDict:
        if queryID in topKIndicesDict:
            print "raise error sessID queryID already exists !!"
            sys.exit(0)
    else:
        topKIndicesDict[sessID] = {}
    topKIndicesDict[sessID][queryID] = topKIndices
    return topKIndicesDict

    for key in keyOrder:
        sessID = int(key.split(",")[0])
        queryID = int(key.split(",")[1])
        curQueryIntent = sessionStreamDict[key]
        if sessID > 0:
            debug = True
        # Here we are putting together the predictedIntent from previous step and the actualIntent from the current query, so that it will be easier for evaluation
        elapsedAppendTime = 0.0
        queryKeysSetAside.append(key)
        numQueries += 1
        # -- Refinement and prediction is done at every query, episode update alone is done at end of the episode --
        (sessionDict, sessionSummaries) = refineSessionSummariesForAllQueriesSetAside(queryKeysSetAside, configDict, sessionDict, sessionSummaries, sessionStreamDict)
        del queryKeysSetAside
        queryKeysSetAside = []
        if len(sessionSummaries)>1 and sessID in sessionSummaries and queryID < sessionLengthDict[sessID]-1: # because we do not predict intent for last query in a session
            (topKSessQueryIndices,topKPredictedIntents) = predictTopKIntents(sessionSummaries, sessionDict, sessID, curQueryIntent, configDict)
            nextQueryIntent = sessionStreamDict[str(sessID)+","+str(queryID+1)]
            elapsedAppendTime = QR.appendPredictedIntentsToFile(topKSessQueryIndices, topKPredictedIntents,
                                                                sessID, queryID, nextQueryIntent, numEpisodes,
                                                                configDict, outputIntentFileName, -1) # foldID does not exist for singularity exps so -1
        if numQueries % int(configDict['EPISODE_IN_QUERIES']) == 0:
            numEpisodes += 1
            (episodeResponseTime, startEpisode, elapsedAppendTime) = QR.updateResponseTime(episodeResponseTime, numEpisodes, startEpisode, elapsedAppendTime)
    episodeResponseTimeDictName = getConfig(configDict['OUTPUT_DIR']) + "/ResponseTimeDict_" +configDict['ALGORITHM']+"_"+configDict['CF_COSINESIM_MF']+"_"+\
                                  configDict['INTENT_REP']+"_"+configDict['BIT_OR_WEIGHTED']+"_TOP_K_"+configDict['TOP_K']+"_EPISODE_IN_QUERIES_"+configDict['EPISODE_IN_QUERIES']+ ".pickle"
    QR.writeToPickleFile(episodeResponseTimeDictName, episodeResponseTime)
    accThres=float(configDict['ACCURACY_THRESHOLD'])
    QR.evaluateQualityPredictions(outputIntentFileName, configDict, accThres, configDict['ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF'])
    print "--Completed Quality Evaluation for accThres:" + str(accThres)
    QR.evaluateTimePredictions(episodeResponseTimeDictName, configDict, configDict['ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF'])

    outputEvalQualityFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputEvalQualityShortTermIntent_" + configDict['ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF']+ "_" + configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres)
    outputExcelQuality = getConfig(configDict['OUTPUT_DIR']) + "/OutputExcelQuality_" + configDict['ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF'] + "_" + configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres) + ".xlsx"
    ParseResultsToExcel.parseQualityFileWithEpisodeRep(outputEvalQualityFileName, outputExcelQuality, configDict)

    outputEvalTimeFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputEvalTimeShortTermIntent_" + configDict['ALGORITHM'] + "_" +configDict['CF_COSINESIM_MF']+ "_" + configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']
    outputExcelTimeEval = getConfig(configDict['OUTPUT_DIR']) + "/OutputExcelTime_" + configDict['ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF']+ "_" +configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES'] + ".xlsx"
    ParseResultsToExcel.parseTimeFile(outputEvalTimeFileName, outputExcelTimeEval)
    return (outputIntentFileName, episodeResponseTimeDictName)
    
def predictTopKIntents(threadID, curQueryIntent, sessionSummaries, sessionSampleDict, sessionStreamDict, sessID, configDict):
    # python supports for min-heap not max-heap so negate items and insert into min-heap
    predSessSummary = computePredSessSummary(curQueryIntent, sessionSummaries, sessID, configDict)
    minheap = []
    cosineSimDict = {}
    for sessIndex in sessionSummaries: # exclude the current session
        if sessIndex != sessID:
            (minheap, cosineSimDict) = insertIntoMinSessHeap(minheap, sessionSummaries, sessIndex, configDict, cosineSimDict, predSessSummary, sessIndex)
    if len(minheap) > 0:
        (minheap, topKSessIndices) = popTopKfromHeap(configDict, minheap, cosineSimDict)
        print "ThreadID: "+str(threadID)+", Found Top-K Sessions"
    else:
        return (None, None)

    del minheap
    minheap = []
    del cosineSimDict
    cosineSimDict = {}
    topKSessQueryIndices = None
    for topKSessIndex in topKSessIndices:
        (minheap, cosineSimDict) = insertIntoMinQueryHeap(minheap, sessionSampleDict, sessionStreamDict, configDict, cosineSimDict, predSessSummary, topKSessIndex)
    if len(minheap) > 0:
        (minheap, topKSessQueryIndices) = popTopKfromHeap(configDict, minheap, cosineSimDict)
        print "ThreadID: "+str(threadID)+", Found Top-K Queries"
    return topKSessQueryIndices
    
def insertIntoMinSessHeap(minheap, elemList, elemIndex, configDict, cosineSimDict, predSessSummary, insertKey):
    elem = elemList[elemIndex]
    cosineSim = 0.0
    assert configDict['BIT_OR_WEIGHTED'] == 'BIT' or configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED'
    if configDict['BIT_OR_WEIGHTED'] == 'BIT':
        cosineSim = computeListBitCosineSimilarity(predSessSummary, elem, configDict)
    elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
        cosineSim = computeListWeightedCosineSimilarity(predSessSummary, elem, ";", configDict)
    heapq.heappush(minheap, -cosineSim)  # insert -ve cosineSim
    if cosineSim not in cosineSimDict:
        cosineSimDict[cosineSim] = list()
    cosineSimDict[cosineSim].append(insertKey)
    return (minheap, cosineSimDict)
'''