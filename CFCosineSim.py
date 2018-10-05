import sys
import os
import time
import QueryRecommender as QR
from bitmap import BitMap
from __future__ import division
import math

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
    return sessionSummaries

def computeBitCosineSimilarity(curQueryIntent, sessionSummary):
    nonzeroDimsQuery = curQueryIntent.nonzero() # set of all 1-bit dimensions in curQueryIntent
    nonzeroDimsSess = sessionSummary.nonzero() # set of all 1-bit dimensions in sessionSummary
    numSetBitsIntersect = len(list(set(nonzeroDimsQuery) & set(nonzeroDimsSess)))  # number of overlapping one bit dimensions
    l2NormProduct = math.sqrt(len(nonzeroDimsQuery)) * math.sqrt(len(nonzeroDimsSess))
    cosineSim = float(numSetBitsIntersect)/l2NormProduct
    return cosineSim

def computeWeightedCosineSimilarity(curQueryIntent, sessionSummary):
    queryDims = curQueryIntent.split(";")
    sessDims = sessionSummary.split(";")
    assert len(queryDims) == len(sessDims)
    numerator = 0.0
    l2NormQuery = 0.0
    l2NormSession = 0.0
    for i in range(len(queryDims)):
        numerator = numerator + float(queryDims[i] * sessDims[i])
        l2NormQuery = l2NormQuery + float(queryDims[i]*queryDims[i])
        l2NormSession = l2NormSession + float(sessDims[i] * sessDims[i])
    cosineSim = numerator / (math.sqrt(l2NormQuery) * math.sqrt(l2NormSession))
    return cosineSim

def predictTopKIntents(sessionSummaries, curQueryIntent, configDict):
    maxSessSim = 0.0
    for sessIndex in sessionSummaries:
        sessionSummary = sessionSummaries[sessIndex]
        if configDict['BIT_OR_WEIGHTED']=='BIT':
            cosineSim = computeBitCosineSimilarity(curQueryIntent, sessionSummary)
        elif configDict['BIT_OR_WEIGHTED']=='WEIGHTED':
            cosineSim = computeWeightedCosineSimilarity(curQueryIntent, sessionSummary)
        if cosineSim > maxSessSim:
            maxSessSim = cosineSim

def runCFCosineSim(intentSessionFile, configDict):
    sessionSummaries = {} # key is sessionID and value is summary
    sessionDict = {} # key is session ID and value is a list of query intent vectors; no need to store the query itself
    predictedIntents = [] # list of top-K predictions
    with open(intentSessionFile) as f:
        for line in f:
            tokens = line.split(";")
            sessQueryName = tokens[0]
            sessID = int(sessQueryName.split(",")[0].split(" ")[1])
            queryID = int(sessQueryName.split(",")[1].split(" ")[1])-1 # coz queryID starts from 1 instead of 0
            curQueryIntent = ';'.join(tokens[2:])
            if ";" not in curQueryIntent and configDict['BIT_OR_WEIGHTED']=='BIT':
                curQueryIntent = BitMap.fromstring(curQueryIntent)
            else:
                curQueryIntent = normalizeWeightedVector(curQueryIntent)
            sessionSummaries = refineSessionSummaries(sessID, configDict, curQueryIntent, sessionSummaries, sessionDict)
            topKPredictedIntents = predictTopKIntents(sessionSummaries, curQueryIntent, configDict)