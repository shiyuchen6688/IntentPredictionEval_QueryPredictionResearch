from __future__ import division
import sys
import os
import time
import QueryRecommender as QR
from bitmap import BitMap
import math
import heapq
import TupleIntent as ti
import ParseConfigFile as parseConfig
import ConcurrentSessions
import ParseResultsToExcel

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

def computePredSessSummary(sessionSummaries, sessID, configDict):
    alpha = 0.5  # fixed does not change so no problem hardcoding
    predSessSummary = []
    curSessSummary = sessionSummaries[sessID] #predSessSummary is a list coz it will consist of weights and floats, but curSessSummary is either a bitmap or a string separated by ;s
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

def refineSessionSummaries(sessID, configDict, curQueryIntent, sessionSummaries, sessionDict):
    if sessID in sessionDict:
        sessionDict[sessID].append(curQueryIntent)
    else:
        sessionDict[sessID] = []
        sessionDict[sessID].append(curQueryIntent)
    if sessID in sessionSummaries:
        if configDict['BIT_OR_WEIGHTED']=='BIT':
            sessionSummaries[sessID] = OR(sessionSummaries[sessID],curQueryIntent, configDict)
        elif configDict['BIT_OR_WEIGHTED']=='WEIGHTED':
            sessionSummaries[sessID] = ADD(sessionSummaries[sessID],curQueryIntent, configDict)
    else:
        sessionSummaries[sessID] = createEntrySimilarTo(curQueryIntent, configDict)
    return (sessionDict, sessionSummaries)

def computeBitCosineSimilarity(curSessionSummary, oldSessionSummary):
    nonzeroDimsCurSess = curSessionSummary.nonzero() # set of all 1-bit dimensions in curQueryIntent
    nonzeroDimsOldSess = oldSessionSummary.nonzero() # set of all 1-bit dimensions in sessionSummary
    numSetBitsIntersect = len(list(set(nonzeroDimsCurSess) & set(nonzeroDimsOldSess)))  # number of overlapping one bit dimensions
    l2NormProduct = math.sqrt(len(nonzeroDimsCurSess)) * math.sqrt(len(nonzeroDimsOldSess))
    cosineSim = float(numSetBitsIntersect)/l2NormProduct
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
    for i in range(numElemToPop):
        topCosineSim = 0 - (heapq.heappop(minheap))  # negated to get back the item
        topKIndex = findTopKSessIndex(topCosineSim, cosineSimDict, topKIndices)
        topKIndices.append(topKIndex)
    return (minheap, topKIndices)

def insertIntoMinHeap(minheap, elemList, elemIndex, configDict, cosineSimDict, predSessSummary, insertKey):
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

def predictTopKIntents(sessionSummaries, sessionDict, sessID, curQueryIntent, configDict):
    # python supports for min-heap not max-heap so negate items and insert into min-heap
    predSessSummary = computePredSessSummary(sessionSummaries, sessID, configDict)
    minheap = []
    cosineSimDict = {}
    for sessIndex in sessionSummaries: # exclude the current session
        if sessIndex != sessID:
            (minheap, cosineSimDict) = insertIntoMinHeap(minheap, sessionSummaries, sessIndex, configDict, cosineSimDict, predSessSummary, sessIndex)
    if len(minheap) > 0:
        (minheap, topKSessIndices) = popTopKfromHeap(configDict, minheap, cosineSimDict)
    else:
        return (None, None)

    del minheap
    minheap = []
    del cosineSimDict
    cosineSimDict = {}
    topKSessQueryIndices = None
    for topKSessIndex in topKSessIndices:
        for queryIndex in range(len(sessionDict[topKSessIndex])):
            (minheap, cosineSimDict) = insertIntoMinHeap(minheap, sessionDict[topKSessIndex], queryIndex, configDict, cosineSimDict, predSessSummary, str(topKSessIndex)+","+str(queryIndex))
    if len(minheap) > 0:
        (minheap, topKSessQueryIndices) = popTopKfromHeap(configDict, minheap, cosineSimDict)

    topKPredictedIntents = []
    for topKSessQueryIndex in topKSessQueryIndices:
        topKSessIndex = int(topKSessQueryIndex.split(",")[0])
        topKQueryIndex = int(topKSessQueryIndex.split(",")[1])
        topKIntent = sessionDict[topKSessIndex][topKQueryIndex]
        topKPredictedIntents.append(topKIntent)
    return (topKSessQueryIndices,topKPredictedIntents)


def refineSessionSummariesForAllQueriesSetAside(queryKeysSetAside, configDict, sessionDict, sessionSummaries, sessionStreamDict):
    for key in queryKeysSetAside:
        sessID = int(key.split(",")[0])
        queryID = int(key.split(",")[1])
        curQueryIntent = sessionStreamDict[key]
        (sessionDict, sessionSummaries) = refineSessionSummaries(sessID, configDict, curQueryIntent, sessionSummaries, sessionDict)
    return (sessionDict, sessionSummaries)


def plotAllFoldQualityTime(kFoldOutputIntentFiles, kFoldEpisodeResponseTimeDicts, configDict):
    QR.computeAvgFoldAccuracy(kFoldOutputIntentFiles, configDict)
    QR.computeAvgFoldTime(kFoldEpisodeResponseTimeDicts, configDict)
    accThresList = []
    accThresList.append(float(configDict['ACCURACY_THRESHOLD']))
    for accThres in accThresList:
        QR.evaluateQualityPredictions(outputIntentFileName, configDict, accThres,
                                      configDict['ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF'])
        print "--Completed Quality Evaluation for accThres:" + str(accThres)
    QR.evaluateTimePredictions(episodeResponseTimeDictName, configDict,
                               configDict['ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF'])

def runCFCosineSimKFoldExp(configDict):
    intentSessionFile = QR.fetchIntentFileFromConfigDict(configDict)
    kFoldOutputIntentFiles = []
    kFoldEpisodeResponseTimeDicts = []
    for foldID in range(int(configDict['KFOLD'])):
        outputIntentFileName = configDict['KFOLD_OUTPUT_DIR'] + "/OutputFileShortTermIntent_" + configDict[
            'ALGORITHM'] + "_" + \
                                  configDict['CF_COSINESIM_MF'] + "_" + \
                                  configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + \
                                  configDict['TOP_K'] + "_FOLD_" + str(foldID)
        episodeResponseTimeDictName = configDict['KFOLD_OUTPUT_DIR'] + "/ResponseTimeDict_" + configDict[
            'ALGORITHM'] + "_" + \
                                      configDict['CF_COSINESIM_MF'] + "_" + \
                                      configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + \
                                      configDict['TOP_K'] + "_FOLD_" + str(foldID) + ".pickle"
        trainIntentSessionFile = configDict['KFOLD_INPUT_DIR']+intentSessionFile.split("/")[len(intentSessionFile.split("/"))-1]+"_TRAIN_FOLD_"+str(foldID)
        testIntentSessionFile = configDict['KFOLD_INPUT_DIR'] + intentSessionFile.split("/")[len(intentSessionFile.split("/")) - 1] + "_TEST_FOLD_" + str(foldID)
        (sessionSummaries, sessionDict, sessionStreamDict, keyOrder, episodeResponseTime) = initCFCosineSimOneFold(trainIntentSessionFile, configDict)
        startTrain = time.time()
        (sessionDict, sessionSummaries) = refineSessionSummariesForAllQueriesSetAside(keyOrder, configDict, sessionDict, sessionSummaries, sessionStreamDict)
        trainTime = float(time.time() - startTrain)
        startTest = time.time()
        testCFCosineSim(testIntentSessionFile, outputIntentFileName, sessionDict, sessionSummaries, sessionStreamDict, episodeResponseTime, episodeResponseTimeDictName, configDict)
        testTime = float(time.time() - startTest)
        kFoldOutputIntentFiles.append(outputIntentFileName)
        kFoldEpisodeResponseTimeDicts.append(episodeResponseTimeDictName)
    plotAllFoldQualityTime(kFoldOutputIntentFiles, kFoldEpisodeResponseTimeDicts, configDict)
    return

def testCFCosineSim(testIntentSessionFile, outputIntentFileName, sessionDict, sessionSummaries, sessionStreamDict, episodeResponseTime, episodeResponseTimeDictName, configDict):
    try:
        os.remove(outputIntentFileName)
    except OSError:
        pass
    numEpisodes = 0
    startEpisode = time.time()
    prevSessID = -1
    with open(testIntentSessionFile) as f:
        for line in f:
            numEpisodes += 1
            (sessID, queryID, curQueryIntent) = QR.retrieveSessIDQueryIDIntent(line, configDict)
            # we need to delete previous test session entries from the summary
            if prevSessID!=sessID:
                if prevSessID in sessionDict:
                    assert prevSessID in sessionSummaries
                    del sessionDict[prevSessID]
                    del sessionSummaries[prevSessID]
                prevSessID = sessID

            queryKeysSetAside = []
            queryKeysSetAside.append(str(sessID)+","+str(queryID))
            (sessionDict, sessionSummaries) = refineSessionSummariesForAllQueriesSetAside(queryKeysSetAside, configDict,
                                                                                          sessionDict, sessionSummaries,
                                                                                          sessionStreamDict)
            (topKSessQueryIndices, topKPredictedIntents) = predictTopKIntents(sessionSummaries, sessionDict, sessID,
                                                                      curQueryIntent, configDict)
            nextQueryIntent = sessionStreamDict[str(sessID) + "," + str(queryID + 1)]
            elapsedAppendTime = QR.appendPredictedIntentsToFile(topKSessQueryIndices, topKPredictedIntents,
                                                        sessID, queryID, nextQueryIntent, numEpisodes,
                                                        configDict, outputIntentFileName)
            (episodeResponseTime, startEpisode, elapsedAppendTime) = QR.updateResponseTime(episodeResponseTime,
                                                                                           numEpisodes, startEpisode,
                                                                                           elapsedAppendTime)
        QR.writeToPickleFile(episodeResponseTimeDictName, episodeResponseTime)

    f.close()
    return episodeResponseTimeDictName

def initCFCosineSimOneFold(intentSessionFile, configDict):
    sessionSummaries = {}  # key is sessionID and value is summary
    sessionDict = {}  # key is session ID and value is a list of query intent vectors; no need to store the query itself
    sessionStreamDict = {}
    keyOrder = []
    episodeResponseTime = {}
    with open(intentSessionFile) as f:
        for line in f:
            (sessID, queryID, curQueryIntent, sessionStreamDict) = QR.updateSessionDict(line, configDict,
                                                                                        sessionStreamDict)
            keyOrder.append(str(sessID) + "," + str(queryID))
    f.close()
    return (sessionSummaries, sessionDict, sessionStreamDict, keyOrder, episodeResponseTime)

def initCFCosineSimSingularity(intentSessionFile, outputIntentFileName, configDict):
    sessionSummaries = {}  # key is sessionID and value is summary
    sessionDict = {}  # key is session ID and value is a list of query intent vectors; no need to store the query itself
    numEpisodes = 0
    queryKeysSetAside = []
    episodeResponseTime = {}

    sessionLengthDict = ConcurrentSessions.countQueries(configDict['QUERYSESSIONS'])
    try:
        os.remove(outputIntentFileName)
    except OSError:
        pass
    numQueries = 0
    sessionStreamDict = {}
    keyOrder = []
    with open(intentSessionFile) as f:
        for line in f:
            (sessID, queryID, curQueryIntent, sessionStreamDict) = QR.updateSessionDict(line, configDict,
                                                                                        sessionStreamDict)
            keyOrder.append(str(sessID) + "," + str(queryID))
    f.close()
    startEpisode = time.time()
    return (sessionSummaries, sessionDict, sessionLengthDict, sessionStreamDict, numEpisodes, queryKeysSetAside, episodeResponseTime, numQueries, keyOrder, startEpisode)

def runCFCosineSimSingularityExp(configDict):
    intentSessionFile = QR.fetchIntentFileFromConfigDict(configDict)
    outputIntentFileName = configDict['OUTPUT_DIR'] + "/OutputFileShortTermIntent_" + configDict['ALGORITHM'] + "_" + \
                           configDict['CF_COSINESIM_MF'] + "_" + \
                           configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                               'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']
    (sessionSummaries, sessionDict, sessionLengthDict, sessionStreamDict, numEpisodes, queryKeysSetAside, episodeResponseTime, numQueries, keyOrder, startEpisode) = initCFCosineSimSingularity(intentSessionFile, outputIntentFileName, configDict)
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
        # -- Refinement is done only at the end of episode, prediction could be done outside but no use for CF and response time update also happens at one shot --
        if numQueries % int(configDict['EPISODE_IN_QUERIES']) == 0:
            numEpisodes += 1
            (sessionDict, sessionSummaries) = refineSessionSummariesForAllQueriesSetAside(queryKeysSetAside, configDict, sessionDict, sessionSummaries, sessionStreamDict)
            del queryKeysSetAside
            queryKeysSetAside = []
            if len(sessionSummaries)>1 and sessID in sessionSummaries and queryID < sessionLengthDict[sessID]-1: # because we do not predict intent for last query in a session
                (topKSessQueryIndices,topKPredictedIntents) = predictTopKIntents(sessionSummaries, sessionDict, sessID, curQueryIntent, configDict)
                nextQueryIntent = sessionStreamDict[str(sessID)+","+str(queryID+1)]
                elapsedAppendTime = QR.appendPredictedIntentsToFile(topKSessQueryIndices, topKPredictedIntents,
                                                                    sessID, queryID, nextQueryIntent, numEpisodes,
                                                                    configDict, outputIntentFileName)
            (episodeResponseTime, startEpisode, elapsedAppendTime) = QR.updateResponseTime(episodeResponseTime, numEpisodes, startEpisode, elapsedAppendTime)
    episodeResponseTimeDictName = configDict['OUTPUT_DIR'] + "/ResponseTimeDict_" +configDict['ALGORITHM']+"_"+configDict['CF_COSINESIM_MF']+"_"+\
                                  configDict['INTENT_REP']+"_"+configDict['BIT_OR_WEIGHTED']+"_TOP_K_"+configDict['TOP_K']+"_EPISODE_IN_QUERIES_"+configDict['EPISODE_IN_QUERIES']+ ".pickle"
    QR.writeToPickleFile(episodeResponseTimeDictName, episodeResponseTime)
    accThresList = []
    accThresList.append(float(configDict['ACCURACY_THRESHOLD']))
    for accThres in accThresList:
        QR.evaluateQualityPredictions(outputIntentFileName, configDict, accThres,
                                      configDict['ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF'])
        print "--Completed Quality Evaluation for accThres:" + str(accThres)
    QR.evaluateTimePredictions(episodeResponseTimeDictName, configDict,
                               configDict['ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF'])

    for accThres in accThresList:
        outputEvalQualityFileName = configDict['OUTPUT_DIR'] + "/OutputEvalQualityShortTermIntent_" + configDict[
            'ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF']+ "_" + configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + \
                                    configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict[
                                        'EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres)
        outputExcelQuality = configDict['OUTPUT_DIR'] + "/OutputExcelQuality_" + configDict['ALGORITHM'] + "_" + \
                             configDict['CF_COSINESIM_MF'] + "_" + configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                                 'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict[
                                 'EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres) + ".xlsx"
        ParseResultsToExcel.parseQualityFileCFCosineSim(outputEvalQualityFileName, outputExcelQuality, configDict)

    outputEvalTimeFileName = configDict['OUTPUT_DIR'] + "/OutputEvalTimeShortTermIntent_" + configDict[
        'ALGORITHM'] + "_" +configDict['CF_COSINESIM_MF']+ "_" + configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                                 'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']
    outputExcelTimeEval = configDict['OUTPUT_DIR'] + "/OutputExcelTime_" + configDict['ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF']+ "_" +configDict[
        'INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                          configDict['EPISODE_IN_QUERIES'] + ".xlsx"
    ParseResultsToExcel.parseTimeFile(outputEvalTimeFileName, outputExcelTimeEval)
    return (outputIntentFileName, episodeResponseTimeDictName)

def runCFCosineSim(configDict):
    if configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY':
        runCFCosineSimSingularityExp(configDict)
    elif configDict['SINGULARITY_OR_KFOLD'] == 'KFOLD':
        runCFCosineSimKFoldExp(configDict)

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("configFile.txt")
    (outputIntentFileName, episodeResponseTimeDictName) = runCFCosineSim(configDict)



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
'''