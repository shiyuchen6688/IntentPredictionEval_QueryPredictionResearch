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

def OR(sessionSummary, curQueryIntent, configDict):
    if configDict['INTENT_REP'] == 'TUPLE' or configDict['INTENT_REP'] == 'FRAGMENT':
        assert sessionSummary.size() == curQueryIntent.size()
    idealSize = min(sessionSummary.size(), curQueryIntent.size())
    for i in range(idealSize):
        if curQueryIntent.test(i):
            sessionSummary.set(i)
    return sessionSummary

def ADD(sessionSummary, curQueryIntent, configDict):
    queryTokens = curQueryIntent.split(";")
    sessTokens = sessionSummary.split(";")
    if configDict['INTENT_REP'] == 'TUPLE' or configDict['INTENT_REP'] == 'FRAGMENT':
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
            if configDict['INTENT_REP'] == 'TUPLE' or configDict['INTENT_REP'] == 'FRAGMENT':
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
    predSessSummary = computePredSessSummary(sessionSummaries, sessID, configDict)
    return (predSessSummary, sessionDict, sessionSummaries)

def computeBitCosineSimilarity(curSessionSummary, oldSessionSummary):
    nonzeroDimsCurSess = curSessionSummary.nonzero() # set of all 1-bit dimensions in curQueryIntent
    nonzeroDimsOldSess = oldSessionSummary.nonzero() # set of all 1-bit dimensions in sessionSummary
    numSetBitsIntersect = len(list(set(nonzeroDimsCurSess) & set(nonzeroDimsOldSess)))  # number of overlapping one bit dimensions
    l2NormProduct = math.sqrt(len(nonzeroDimsCurSess)) * math.sqrt(len(nonzeroDimsOldSess))
    cosineSim = float(numSetBitsIntersect)/l2NormProduct
    return cosineSim

def computeListBitCosineSimilarity(predSessSummary, oldSessionSummary, configDict):
    if configDict['INTENT_REP'] == 'TUPLE' or configDict['INTENT_REP'] == 'FRAGMENT':
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
    if configDict['INTENT_REP'] == 'TUPLE' or configDict['INTENT_REP'] == 'FRAGMENT':
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
    if configDict['INTENT_REP'] == 'TUPLE' or configDict['INTENT_REP'] == 'FRAGMENT':
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

def predictTopKIntents(sessionSummaries, sessionDict, sessID, predSessSummary, curQueryIntent, configDict):
    # python supports for min-heap not max-heap so negate items and insert into min-heap
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


def refineSessionSummariesForAllQueriesSetAside(queryLinesSetAside, configDict, sessionDict, sessionSummaries):
    predSessSummary = None
    for line in queryLinesSetAside:
        (sessID, queryID, curQueryIntent) = QR.retrieveSessIDQueryIDIntent(line, configDict)
        (predSessSummary, sessionDict, sessionSummaries) = refineSessionSummaries(sessID, configDict, curQueryIntent, sessionSummaries, sessionDict)
    return (predSessSummary, sessionDict, sessionSummaries)


def runCFCosineSim(intentSessionFile, configDict):
    sessionSummaries = {} # key is sessionID and value is summary
    sessionDict = {} # key is session ID and value is a list of query intent vectors; no need to store the query itself
    numEpisodes = 0
    queryLinesSetAside = []
    episodeResponseTime = {}
    startEpisode = time.time()
    predSessSummary = None
    outputIntentFileName = configDict['OUTPUT_DIR']+"/OutputFileShortTermIntent_"+configDict['INTENT_REP']+"_"+configDict['BIT_OR_WEIGHTED']+"_TOP_K_"+configDict['TOP_K']+"_EPISODE_IN_QUERIES_"+configDict['EPISODE_IN_QUERIES']
    try:
        os.remove(outputIntentFileName)
    except OSError:
        pass
    numQueries = 0
    with open(intentSessionFile) as f:
        topKPredictedIntents = None
        topKSessQueryIndices = None
        for line in f:
            (sessID, queryID, curQueryIntent) = QR.retrieveSessIDQueryIDIntent(line, configDict)
            if sessID > 0:
                debug = True
            # Here we are putting together the predictedIntent from previous step and the actualIntent from the current query, so that it will be easier for evaluation
            elapsedAppendTime = 0.0
            if topKPredictedIntents is not None:
                elapsedAppendTime = QR.appendPredictedIntentsToFile(topKSessQueryIndices, topKPredictedIntents, sessID, queryID, curQueryIntent, numEpisodes,
                                             configDict, outputIntentFileName)
            numQueries += 1
            queryLinesSetAside.append(line)
            # -- Refinement is done only at the end of episode, prediction could be done outside but no use for CF and response time update also happens at one shot --
            if numQueries % int(configDict['EPISODE_IN_QUERIES']) == 0:
                numEpisodes += 1
                (predSessSummary,sessionDict, sessionSummaries) = refineSessionSummariesForAllQueriesSetAside(queryLinesSetAside, configDict, sessionDict, sessionSummaries)
                del queryLinesSetAside
                queryLinesSetAside = []
                if len(sessionSummaries)>0 and sessID in sessionSummaries:
                    (topKSessQueryIndices,topKPredictedIntents) = predictTopKIntents(sessionSummaries, sessionDict, sessID, predSessSummary, curQueryIntent, configDict)
                else:
                    topKPredictedIntents = None
                    topKSessQueryIndices = None
                (episodeResponseTime, startEpisode) = QR.updateResponseTime(episodeResponseTime, numEpisodes, startEpisode, elapsedAppendTime)
    episodeResponseTimeDictName = configDict['OUTPUT_DIR'] + "/ResponseTimeDict_" +configDict['INTENT_REP']+"_"+configDict['BIT_OR_WEIGHTED']+"_TOP_K_"+configDict['TOP_K']+"_EPISODE_IN_QUERIES_"+configDict['EPISODE_IN_QUERIES']+ ".pickle"
    QR.writeToPickleFile(episodeResponseTimeDictName, episodeResponseTime)
    return (outputIntentFileName, episodeResponseTimeDictName)

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("configFile.txt")
    if configDict['INTENT_REP']=='TUPLE':
        intentSessionFile = configDict['TUPLEINTENTSESSIONS']
    elif configDict['INTENT_REP']=='FRAGMENT' and configDict['BIT_OR_WEIGHTED']=='BIT':
        intentSessionFile = configDict['BIT_FRAGMENT_INTENT_SESSIONS']
    elif configDict['INTENT_REP']=='FRAGMENT' and configDict['BIT_OR_WEIGHTED']=='WEIGHTED':
        intentSessionFile = configDict['WEIGHTED_FRAGMENT_INTENT_SESSIONS']
    elif configDict['INTENT_REP']=='QUERY':
        intentSessionFile = configDict['QUERY_INTENT_SESSIONS']
    else:
        print "ConfigDict['INTENT_REP'] must either be TUPLE or FRAGMENT or QUERY !!"
        sys.exit(0)
    runCFCosineSim(intentSessionFile, configDict)
