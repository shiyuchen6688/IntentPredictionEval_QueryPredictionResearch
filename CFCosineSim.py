from __future__ import division
import sys
import os
import time
import QueryRecommender as QR
from bitmap import BitMap
import math
import heapq
import TupleIntent as ti

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
                cosineSim = computeWeightedCosineSimilarity(curSessSummary, oldSessionSummary, ";")
            elif configDict['BIT_OR_WEIGHTED'] == 'BIT':
                cosineSim = computeBitCosineSimilarity(curSessSummary, oldSessionSummary)
            for i in range(len(predSessSummary)):
                if configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
                    predSessSummary[i] = predSessSummary[i]+ (1-alpha)*cosineSim*oldSessionSummary[i]
                elif configDict['BIT_OR_WEIGHTED'] == 'BIT' and oldSessionSummary.test(i):
                    predSessSummary[i] = predSessSummary[i] + (1-alpha)*cosineSim*1.0
    return predSessSummary

def refineSessionSummaries(sessID, configDict, curQueryIntent, sessionSummaries, sessionDict):
    if sessID in sessionDict:
        sessionDict[sessID].append(curQueryIntent)
    else:
        sessionDict[sessID] = []
        sessionDict[sessID].append(curQueryIntent)
    if sessID in sessionSummaries:
        if configDict['BIT_OR_WEIGHTED']=='BIT':
            sessionSummaries[sessID] = OR(sessionSummaries[sessID],curQueryIntent)
        elif configDict['BIT_OR_WEIGHTED']=='WEIGHTED':
            sessionSummaries[sessID] = ADD(sessionSummaries[sessID],curQueryIntent)
    else:
        sessionSummaries[sessID] = curQueryIntent
    predSessSummary = computePredSessSummary(sessionSummaries, sessID, configDict)
    return (predSessSummary, sessionDict, sessionSummaries)

def computeBitCosineSimilarity(curSessionSummary, oldSessionSummary):
    nonzeroDimsCurSess = curSessionSummary.nonzero() # set of all 1-bit dimensions in curQueryIntent
    nonzeroDimsOldSess = oldSessionSummary.nonzero() # set of all 1-bit dimensions in sessionSummary
    numSetBitsIntersect = len(list(set(nonzeroDimsCurSess) & set(nonzeroDimsOldSess)))  # number of overlapping one bit dimensions
    l2NormProduct = math.sqrt(len(nonzeroDimsCurSess)) * math.sqrt(len(nonzeroDimsOldSess))
    cosineSim = float(numSetBitsIntersect)/l2NormProduct
    return cosineSim

def computeListBitCosineSimilarity(predSessSummary, oldSessionSummary):
    assert(len(predSessSummary))==oldSessionSummary.size()
    numerator = 0.0
    l2NormPredSess = 0.0
    l2NormOldSess = 0.0
    for i in range(len(predSessSummary)):
        predSessDim = predSessSummary[i]
        if oldSessionSummary.test(i):
            numerator += float(predSessDim * 1.0)
            l2NormOldSess += float(1.0 * 1.0)
        l2NormPredSess += float(predSessDim * predSessDim)
    cosineSim = numerator / (math.sqrt(l2NormPredSess) * math.sqrt(l2NormOldSess))
    return cosineSim

def computeWeightedCosineSimilarity(curSessionSummary, oldSessionSummary, delimiter):
    curSessDims = curSessionSummary.split(delimiter)
    oldSessDims = oldSessionSummary.split(delimiter)
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

def computeListWeightedCosineSimilarity(predSessSummary, oldSessionSummary, delimiter):
    oldSessDims = oldSessionSummary.split(delimiter)
    assert len(predSessSummary) == len(oldSessDims)
    numerator = 0.0
    l2NormQuery = 0.0
    l2NormSession = 0.0
    for i in range(len(predSessSummary)):
        numerator = numerator + float(predSessSummary[i] * oldSessDims[i])
        l2NormQuery = l2NormQuery + float(predSessSummary[i] * predSessSummary[i])
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
        cosineSim = computeListBitCosineSimilarity(predSessSummary, elem)
    elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
        cosineSim = computeListWeightedCosineSimilarity(predSessSummary, elem, ";")
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
        return None

    del minheap
    minheap = []
    del cosineSimDict
    cosineSimDict = {}
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
    return topKPredictedIntents

def retrieveSessIDQueryIDIntent(line, configDict):
    tokens = line.strip().split(";")
    sessQueryName = tokens[0]
    sessID = int(sessQueryName.split(", ")[0].split(" ")[1])
    queryID = int(sessQueryName.split(", ")[1].split(" ")[1]) - 1  # coz queryID starts from 1 instead of 0
    curQueryIntent = ';'.join(tokens[2:])
    if ";" not in curQueryIntent and configDict['BIT_OR_WEIGHTED'] == 'BIT':
        curQueryIntent = BitMap.fromstring(curQueryIntent)
    else:
        curQueryIntent = normalizeWeightedVector(curQueryIntent)
    return (sessID, queryID, curQueryIntent)

def refineSessionSummariesForAllQueriesSetAside(queryLinesSetAside, configDict, sessionDict, sessionSummaries):
    predSessSummary = None
    for line in queryLinesSetAside:
        (sessID, queryID, curQueryIntent) = retrieveSessIDQueryIDIntent(line, configDict)
        (predSessSummary, sessionDict, sessionSummaries) = refineSessionSummaries(sessID, configDict, curQueryIntent, sessionSummaries, sessionDict)
    return (predSessSummary, sessionDict, sessionSummaries)

def appendPredictedIntentsToFile(topKPredictedIntents, sessID, queryID, curQueryIntent, numEpochs, configDict, outputIntentFileName):
    startAppendTime = time.time()
    output_str = "Session:"+str(sessID)+";Query:"+str(queryID)+";#Epochs:"+str(numEpochs)+";CurQueryIntent:"
    if configDict['BIT_OR_WEIGHTED'] == 'BIT':
        output_str += curQueryIntent.tostring()
    elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
        if ";" in curQueryIntent:
            curQueryIntent.replace(";",",")
        output_str += curQueryIntent
    for k in range(len(topKPredictedIntents)):
        output_str += ";TOP_" +str(k)+"_PREDICTED_INTENT:"
        if configDict['BIT_OR_WEIGHTED'] == 'BIT':
            output_str += curQueryIntent.tostring()
        elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
            output_str += curQueryIntent.replace(";",",")
    ti.appendToFile(outputIntentFileName, output_str)
    print "Predicted "+str(len(topKPredictedIntents))+" query intent vectors for Session "+str(sessID)+", Query "+str(queryID)
    elapsedAppendTime = float(time.time()-startAppendTime)
    return elapsedAppendTime

def updateResponseTime(epochResponseTime, numEpochs, startEpoch, elapsedAppendTime):
    epochResponseTime[numEpochs] = float(time.time()-startEpoch) - elapsedAppendTime # we exclude the time consumed by appending predicted intents to the output intent file
    startEpoch = time.time()
    return (epochResponseTime, startEpoch)

def runCFCosineSim(intentSessionFile, configDict):
    sessionSummaries = {} # key is sessionID and value is summary
    sessionDict = {} # key is session ID and value is a list of query intent vectors; no need to store the query itself
    numEpochs = 0
    queryLinesSetAside = []
    epochResponseTime = {}
    startEpoch = time.time()
    predSessSummary = None
    outputIntentFileName = configDict['OUTPUT_DIR']+"/OutputFileShortTermIntent_"+configDict['INTENT_REP']+"_"+configDict['BIT_OR_WEIGHTED']+"_TOP_K_"+configDict['TOP_K']+"_EPOCH_IN_QUERIES_"+configDict['EPISODE_IN_SECONDS']
    try:
        os.remove(outputIntentFileName)
    except OSError:
        pass
    numQueries = 0
    with open(intentSessionFile) as f:
        topKPredictedIntents = None
        for line in f:
            (sessID, queryID, curQueryIntent) = retrieveSessIDQueryIDIntent(line, configDict)
            # Here we are putting together the predictedIntent from previous step and the actualIntent from the current query, so that it will be easier for evaluation
            elapsedAppendTime = 0.0
            if topKPredictedIntents is not None:
                elapsedAppendTime = appendPredictedIntentsToFile(topKPredictedIntents, sessID, queryID, curQueryIntent, numEpochs,
                                             configDict, outputIntentFileName)
            numQueries += 1
            queryLinesSetAside.append(line)
            if numQueries % int(configDict['EPOCH_IN_QUERIES']) == 0:
                numEpochs += 1
                (predSessSummary,sessionDict, sessionSummaries) = refineSessionSummariesForAllQueriesSetAside(queryLinesSetAside, configDict, sessionDict, sessionSummaries)
                del queryLinesSetAside
                queryLinesSetAside = []

            if len(sessionSummaries)>0 and sessID in sessionSummaries:
                topKPredictedIntents = predictTopKIntents(sessionSummaries, sessionDict, sessID, predSessSummary, curQueryIntent, configDict)
            else:
                topKPredictedIntents = None
            if numQueries % int(configDict['EPOCH_IN_QUERIES']) == 0:
                (epochResponseTime, startEpoch) = updateResponseTime(epochResponseTime, numEpochs, startEpoch, elapsedAppendTime)
    return (outputIntentFileName, epochResponseTime)