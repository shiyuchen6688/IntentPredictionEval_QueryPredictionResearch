import sys
import os
import time
import QueryRecommender as QR
from bitmap import BitMap
from __future__ import division
import math
import heapq

def OR(sessionSummary, curQueryIntent):
    assert sessionSummary.size() == curQueryIntent.size()
    for i in range(sessionSummary.size()):
        if curQueryIntent.test(i):
            sessionSummary.set(i)
    return sessionSummary

def normalizeWeightedVector(curQueryIntent):
    tokens = curQueryIntent.split(";")
    total = 0.0
    for token in tokens:
        total = total+float(token)
    normalizedVector = []
    for token in tokens:
        normalizedVector.append(float(token)/total)
    res = ';'.join(normalizedVector)
    return res

def ADD(sessionSummary, curQueryIntent):
    queryTokens = curQueryIntent.split(";")
    sessTokens = sessionSummary.split(";")
    assert len(queryTokens) == len(sessTokens)
    for i in range(len(queryTokens)):
        sessTokens[i] = sessTokens[i]+queryTokens[i]
    sessionSummary = normalizeWeightedVector(';'.join(sessTokens))
    return sessionSummary

def refineSessionSummaries(sessID, configDict, curQueryIntent, sessionSummaries, sessionDict):
    if sessID in sessionDict:
        sessionDict[sessID].append(curQueryIntent)
    else:
        sessionDict[sessID] = [].append(curQueryIntent)
    if sessID in sessionSummaries:
        if configDict['BIT_OR_WEIGHTED']=='BIT':
            sessionSummaries[sessID] = OR(sessionSummaries[sessID],curQueryIntent)
        elif configDict['BIT_OR_WEIGHTED']=='WEIGHTED':
            sessionSummaries[sessID] = ADD(sessionSummaries[sessID],curQueryIntent)
    else:
        sessionSummaries[sessID] = curQueryIntent
    return (sessionDict, sessionSummaries)

def computeBitCosineSimilarity(curSessionSummary, oldSessionSummary):
    nonzeroDimsCurSess = curSessionSummary.nonzero() # set of all 1-bit dimensions in curQueryIntent
    nonzeroDimsOldSess = oldSessionSummary.nonzero() # set of all 1-bit dimensions in sessionSummary
    numSetBitsIntersect = len(list(set(nonzeroDimsCurSess) & set(nonzeroDimsOldSess)))  # number of overlapping one bit dimensions
    l2NormProduct = math.sqrt(len(nonzeroDimsCurSess)) * math.sqrt(len(nonzeroDimsOldSess))
    cosineSim = float(numSetBitsIntersect)/l2NormProduct
    return cosineSim

def computeWeightedCosineSimilarity(curSessionSummary, oldSessionSummary):
    curSessDims = curSessionSummary.split(";")
    oldSessDims = oldSessionSummary.split(";")
    assert len(curSessDims) == len(oldSessDims)
    numerator = 0.0
    l2NormQuery = 0.0
    l2NormSession = 0.0
    for i in range(len(curSessDims)):
        numerator = numerator + float(curSessDims[i] * oldSessDims[i])
        l2NormQuery = l2NormQuery + float(curSessDims[i]*curSessDims[i])
        l2NormSession = l2NormSession + float(oldSessDims[i] * oldSessDims[i])
    cosineSim = numerator / (math.sqrt(l2NormQuery) * math.sqrt(l2NormSession))
    return cosineSim

def findTopKSessIndex(topCosineSim, cosineSimDict, topKSessindices):
    if topCosineSim not in cosineSimDict:
        print "cosineSimilarity not found in the dictionary !!"
        sys.exit(0)
    for sessIndex in cosineSimDict[topCosineSim]:
        if sessIndex not in topKSessindices:
            return sessIndex

def predictTopKIntents(sessionSummaries, sessID, curQueryIntent, configDict):
    # python supports for min-heap not max-heap so negate items and insert into min-heap
    minheap = []
    curSessionSummary = sessionSummaries[sessID]
    cosineSimDict = {}
    for sessIndex in range(len(sessionSummaries)-1): # exclude the current session
        oldSessionSummary = sessionSummaries[sessIndex]
        if configDict['BIT_OR_WEIGHTED']=='BIT':
            cosineSim = computeBitCosineSimilarity(curSessionSummary, oldSessionSummary)
        elif configDict['BIT_OR_WEIGHTED']=='WEIGHTED':
            cosineSim = computeWeightedCosineSimilarity(curSessionSummary, oldSessionSummary)
        heapq.heappush(minheap, -cosineSim)  # insert -ve cosineSim
        if cosineSim not in cosineSimDict:
            cosineSimDict[cosineSim] = list()
        cosineSimDict[cosineSim].append(sessIndex)

    topKSessIndices = []
    for i in range(int(cosineSim['TOP_K'])):
        topCosineSim = 0-(heapq.heappop(minheap)) # negated to get back the item
        topKSessIndex = findTopKSessIndex(topCosineSim, cosineSimDict, topKSessIndices)
        topKSessIndices.append(topKSessIndex)

def checkEpisodeCompletion(startEpisode, configDict):
    timeElapsed = time.time() - startEpisode
    if timeElapsed > configDict['EPISODE_IN_SECONDS']:
        startEpisode = time.time()
        return (True, startEpisode)
    else:
        return (False, startEpisode)

def retrieveSessIDQueryIDIntent(line, configDict):
    tokens = line.split(";")
    sessQueryName = tokens[0]
    sessID = int(sessQueryName.split(",")[0].split(" ")[1])
    queryID = int(sessQueryName.split(",")[1].split(" ")[1]) - 1  # coz queryID starts from 1 instead of 0
    curQueryIntent = ';'.join(tokens[2:])
    if ";" not in curQueryIntent and configDict['BIT_OR_WEIGHTED'] == 'BIT':
        curQueryIntent = BitMap.fromstring(curQueryIntent)
    else:
        curQueryIntent = normalizeWeightedVector(curQueryIntent)
    return (sessID, queryID, curQueryIntent)

def refineSessionSummariesForAllQueriesSetAside(queryLinesSetAside, configDict, sessionDict):
    for line in queryLinesSetAside:
        (sessID, queryID, curQueryIntent) = retrieveSessIDQueryIDIntent(line, configDict)
        (sessionDict, sessionSummaries) = refineSessionSummaries(sessID, configDict, curQueryIntent, sessionSummaries, sessionDict)
    return (sessionDict, sessionSummaries)

def runCFCosineSim(intentSessionFile, configDict):
    sessionSummaries = {} # key is sessionID and value is summary
    sessionDict = {} # key is session ID and value is a list of query intent vectors; no need to store the query itself
    numEpisodes = 0
    queryLinesSetAside = []
    startEpisode = time.time()
    with open(intentSessionFile) as f:
        for line in f:
            (episodeDone, startEpisode) = checkEpisodeCompletion(startEpisode, configDict)
            if episodeDone:
                numEpisodes += 1
                sessionSummaries = refineSessionSummariesForAllQueriesSetAside(queryLinesSetAside, configDict, sessionDict)
            else:
                queryLinesSetAside.append(line)
            (sessID, queryID, curQueryIntent) = retrieveSessIDQueryIDIntent(line, configDict)
            if len(sessionSummaries)>0:
                topKPredictedIntents = predictTopKIntents(sessionSummaries, sessID, curQueryIntent, configDict)
            else:
                topKPredictedIntents = None