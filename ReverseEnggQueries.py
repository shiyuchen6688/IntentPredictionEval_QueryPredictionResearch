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
import ParseResultsToExcel
import ConcurrentSessions
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
import CFCosineSim
import argparse
from ParseConfigFile import getConfig
import threading
import copy
import multiprocessing
from multiprocessing.pool import ThreadPool
from multiprocessing import Array
import CreateSQLFromIntentVec

class SchemaDicts:
    def __init__(self, tableDict, tableOrderDict, colDict, joinPredDict, joinPredBitPosDict):
        self.tableDict = tableDict
        self.tableOrderDict = tableOrderDict
        self.colDict = colDict
        self.joinPredDict = joinPredDict
        self.joinPredBitPosDict = joinPredBitPosDict

        self.queryTypeBitMapSize = 4  # select, insert, update, delete
        self.limitBitMapSize = 1
        self.tableBitMapSize = estimateTableBitMapSize(self)
        self.allColumnsSize = estimateAllColumnCount(self)
        self.joinPredicatesBitMapSize = estimateJoinPredicatesBitMapSize(self)

        # the following requires careful order mapping
        self.queryTypeStartBitIndex = 0
        self.tableStartBitIndex = self.queryTypeStartBitIndex + self.queryTypeBitMapSize
        self.projectionStartBitIndex = self.tableStartBitIndex + self.tableBitMapSize
        self.avgStartBitIndex = self.projectionStartBitIndex + self.allColumnsSize
        self.minStartBitIndex = self.avgStartBitIndex + self.allColumnsSize
        self.maxStartBitIndex = self.minStartBitIndex + self.allColumnsSize
        self.sumStartBitIndex = self.maxStartBitIndex + self.allColumnsSize
        self.countStartBitIndex = self.sumStartBitIndex + self.allColumnsSize
        self.selectionStartBitIndex = self.countStartBitIndex + self.allColumnsSize
        self.groupByStartBitIndex = self.selectionStartBitIndex + self.allColumnsSize
        self.orderByStartBitIndex = self.groupByStartBitIndex + self.allColumnsSize
        self.havingStartBitIndex = self.orderByStartBitIndex + self.allColumnsSize
        self.limitStartBitIndex = self.havingStartBitIndex + self.allColumnsSize
        self.joinPredicatesStartBitIndex = self.limitStartBitIndex + self.limitBitMapSize
        self.allOpSize = self.queryTypeBitMapSize + self.tableBitMapSize + self.allColumnsSize * 10 + self.limitBitMapSize + self.joinPredicatesBitMapSize
        # the following populates the map which can look up from bits to maps and from maps to bits
        self.forwardMapBitsToOps = {}
        self.backwardMapOpsToBits = {}
        (self.forwardMapBitsToOps, self.backwardMapOpsToBits) = populateBiDirectionalLookupMap(self)

def populateQueryType(schemaDicts):
    schemaDicts.forwardMapBitsToOps[0] = "select;querytype"
    schemaDicts.backwardMapOpsToBits["select;querytype"] = 0
    schemaDicts.forwardMapBitsToOps[1] = "update;querytype"
    schemaDicts.backwardMapOpsToBits["update;querytype"] = 1
    schemaDicts.forwardMapBitsToOps[2] = "insert;querytype"
    schemaDicts.backwardMapOpsToBits["insert;querytype"] = 2
    schemaDicts.forwardMapBitsToOps[3] = "delete;querytype"
    schemaDicts.backwardMapOpsToBits["delete;querytype"] = 3
    return schemaDicts

def populateTables(schemaDicts):
    indexToSet = schemaDicts.tableStartBitIndex
    for i in range(0, len(schemaDicts.tableOrderDict)):
        schemaDicts.forwardMapBitsToOps[indexToSet] = schemaDicts.tableOrderDict[i]+";table"
        schemaDicts.backwardMapOpsToBits[schemaDicts.tableOrderDict[i]+";table"] = indexToSet
        indexToSet += 1
    assert indexToSet == schemaDicts.tableStartBitIndex + schemaDicts.tableBitMapSize
    return schemaDicts

def populateColsForOp(opString, schemaDicts):
    assert opString == "project" or opString == "avg" or opString == "min" or opString == "max" or opString == "sum" \
           or opString == "count" or opString == "select" or opString == "groupby" \
           or opString == "orderby" or opString == "having"
    if opString == "project":
        startBitIndex = schemaDicts.projectionStartBitIndex
    elif opString == "avg":
        startBitIndex = schemaDicts.avgStartBitIndex
    elif opString == "min":
        startBitIndex = schemaDicts.minStartBitIndex
    elif opString == "max":
        startBitIndex = schemaDicts.maxStartBitIndex
    elif opString == "sum":
        startBitIndex = schemaDicts.sumStartBitIndex
    elif opString == "count":
        startBitIndex = schemaDicts.countStartBitIndex
    elif opString == "select":
        startBitIndex = schemaDicts.selectionStartBitIndex
    elif opString == "groupby":
        startBitIndex = schemaDicts.groupByStartBitIndex
    elif opString == "orderby":
        startBitIndex = schemaDicts.orderByStartBitIndex
    elif opString == "having":
        startBitIndex = schemaDicts.havingStartBitIndex
    else:
        print "ColError !!"
    indexToSet = startBitIndex
    for tableIndex in range(len(schemaDicts.tableOrderDict)):
        tableName = schemaDicts.tableOrderDict[tableIndex]
        colList = schemaDicts.colDict[tableName]
        for col in colList:
            schemaDicts.forwardMapBitsToOps[indexToSet] = tableName+"."+col+";"+opString
            schemaDicts.backwardMapOpsToBits[tableName+"."+col+";"+opString] = indexToSet
            indexToSet+=1
    assert indexToSet == startBitIndex + schemaDicts.allColumnsSize
    return schemaDicts

def populateLimit(schemaDicts):
    schemaDicts.forwardMapBitsToOps[schemaDicts.limitStartBitIndex] = "limit"
    schemaDicts.backwardMapOpsToBits["limit"] = schemaDicts.limitStartBitIndex
    return schemaDicts

def populateJoinPreds(schemaDicts):
    opString = "join"
    for tablePairIndex in schemaDicts.joinPredBitPosDict:
        startEndBitPos = schemaDicts.joinPredBitPosDict[tablePairIndex]
        startBitPos = startEndBitPos[0]+schemaDicts.joinPredicatesStartBitIndex
        endBitPos = startEndBitPos[1]+schemaDicts.joinPredicatesStartBitIndex
        for indexToSet in range(startBitPos, endBitPos):
            joinColPair = schemaDicts.joinPredDict[tablePairIndex][indexToSet-startBitPos]
            joinStrToAppend = tablePairIndex.split(",")[0] + "." + joinColPair.split(",")[0]+ "," + tablePairIndex.split(",")[1] + "." + joinColPair.split(",")[1]
            if indexToSet in schemaDicts.forwardMapBitsToOps:
                print "Already exists "+str(indexToSet)+" :"+schemaDicts.forwardMapBitsToOps[indexToSet]
            schemaDicts.forwardMapBitsToOps[indexToSet] = joinStrToAppend + ";" + opString
            schemaDicts.backwardMapOpsToBits[joinStrToAppend + ";" + opString] = indexToSet
    return schemaDicts

def populateBiDirectionalLookupMap(schemaDicts):
    schemaDicts = populateQueryType(schemaDicts)
    schemaDicts = populateTables(schemaDicts)
    schemaDicts = populateColsForOp("project", schemaDicts)
    schemaDicts = populateColsForOp("avg", schemaDicts)
    schemaDicts = populateColsForOp("min", schemaDicts)
    schemaDicts = populateColsForOp("max", schemaDicts)
    schemaDicts = populateColsForOp("sum", schemaDicts)
    schemaDicts = populateColsForOp("count", schemaDicts)
    schemaDicts = populateColsForOp("select", schemaDicts)
    schemaDicts = populateColsForOp("groupby", schemaDicts)
    schemaDicts = populateColsForOp("orderby", schemaDicts)
    schemaDicts = populateColsForOp("having", schemaDicts)
    schemaDicts = populateLimit(schemaDicts)
    schemaDicts = populateJoinPreds(schemaDicts)
    #print len(schemaDicts.forwardMapBitsToOps)
    #print len(schemaDicts.backwardMapOpsToBits)
    #print schemaDicts.allOpSize - len(schemaDicts.joinPredDict)
    assert len(schemaDicts.forwardMapBitsToOps) == len(schemaDicts.backwardMapOpsToBits)
    assert len(schemaDicts.forwardMapBitsToOps) == schemaDicts.allOpSize - len(schemaDicts.joinPredDict)
    return (schemaDicts.forwardMapBitsToOps, schemaDicts.backwardMapOpsToBits)

def estimateTableBitMapSize(schemaDicts):
    tableDict = schemaDicts.tableDict
    return len(tableDict)

def estimateAllColumnCount(schemaDicts):
    colCount = 0
    for tableName in schemaDicts.tableDict:
        colCount += len(schemaDicts.colDict[tableName])
    return colCount

def estimateJoinPredicatesBitMapSize(schemaDicts):
    #joinPredDict = schemaDicts.joinPredDict
    joinPredBitPosDict = schemaDicts.joinPredBitPosDict
    joinPredBitCount = 0
    #joinPredCount = 0
    for tabPair in joinPredBitPosDict:
        #joinPredCount += len(joinPredDict[tabPair]) + 1
        joinPredBitCount += joinPredBitPosDict[tabPair][1] - joinPredBitPosDict[tabPair][0] + 1
    return joinPredBitCount

def pruneUnImportantDimensions(predictedY, weightThreshold):
    newPredictedY = []
    minY = min(predictedY)
    maxY = max(predictedY)
    for y in predictedY:
        newY = float(y-minY)/float(maxY-minY) # normalize each dimension to lie between 0 and 1
        if newY < float(weightThreshold):
            newY = 0.0
        newPredictedY.append(newY)
    return newPredictedY

def readTableDict(fn):
    tableDict = {}
    tableOrderDict = {}
    with open(fn) as f:
        for line in f:
            tokens = line.strip().split(":")
            tableDict[tokens[0]] = int(tokens[1])
            tableOrderDict[int(tokens[1])] = tokens[0]
    return (tableDict, tableOrderDict)

def readColDict(fn):
    colDict = {}
    with open(fn) as f:
        for line in f:
            tokens = line.strip().split(":")
            key = tokens[0]
            val = tokens[1].replace("[","").replace("]","").replace("'","")
            columns = val.split(", ")
            colDict[key] = columns
    return colDict

def readJoinPredDict(fn):
    joinPredDict = {}
    with open(fn) as f:
        for line in f:
            tokens = line.strip().split(":")
            key = tokens[0]
            val = tokens[1].replace("[", "").replace("]", "").replace("'", "")
            columns = val.split(", ")
            joinPredDict[key] = columns
    return joinPredDict

def readJoinPredBitPosDict(fn):
    joinPredBitPosDict = {}
    with open(fn) as f:
        for line in f:
            tokens = line.strip().split(":")
            key = tokens[0]
            startEndBitPos = [int(x) for x in tokens[1].split(",")]
            joinPredBitPosDict[key]=startEndBitPos
    return joinPredBitPosDict

def checkSanity(joinPredDict, joinPredBitPosDict):
    joinPredCount = 0
    joinPredBitPosCount = 0
    for key in joinPredDict:
        joinPredCount += len(joinPredDict[key])
        joinPredBitPosCount += joinPredBitPosDict[key][1] - joinPredBitPosDict[key][0]
    assert len(joinPredDict) == len(joinPredBitPosDict)
    assert joinPredCount == joinPredBitPosCount
    #print "joinPredCount: "+str(joinPredCount)+", joinPredBitPosCount: "+str(joinPredBitPosCount)

def readJoinColDicts(joinPredFile, joinPredBitPosFile):
    joinPredDict = readJoinPredDict(joinPredFile)
    joinPredBitPosDict = readJoinPredBitPosDict(joinPredBitPosFile)
    checkSanity(joinPredDict, joinPredBitPosDict)
    return (joinPredDict, joinPredBitPosDict)

def readSchemaDicts(configDict):
    (tableDict, tableOrderDict) = readTableDict(getConfig(configDict['MINC_TABLES']))
    colDict = readColDict(getConfig(configDict['MINC_COLS']))
    (joinPredDict, joinPredBitPosDict) = readJoinColDicts(getConfig(configDict['MINC_JOIN_PREDS']), getConfig(configDict['MINC_JOIN_PRED_BIT_POS']))
    schemaDicts = SchemaDicts(tableDict, tableOrderDict, colDict, joinPredDict, joinPredBitPosDict)
    return schemaDicts

def topKThres(configDict):
    thresholds = []
    thresList = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    assert int(configDict['TOP_K']) > 0 and int(configDict['TOP_K']) < 10
    for i in range(int(configDict['TOP_K'])):
        thresholds.append(thresList[i])
    # special case:
    if int(configDict['TOP_K']) == 3:
        thresholds = [0.8, 0.6, 0.4]
    return thresholds

def refineIntent(threadID, topKCandidateVector, schemaDicts, configDict):
    # Step 1: regenerate the query ops from the topKCandidateVector
    intentObj = CreateSQLFromIntentVec.regenerateSQL(topKCandidateVector, schemaDicts)
    # Step 2: refine SQL violations
    CreateSQLFromIntentVec.fixSQLViolations(intentObj, precOrRecallFavor="recall")

def predictTopKNovelIntents(threadID, predictedY, schemaDicts, configDict):
    topKPredictedIntents = []
    #schemaDicts = readSchemaDicts(configDict)
    thresholds = topKThres(configDict)
    for threshold in thresholds:
        topKCandidateVector = pruneUnImportantDimensions(predictedY, threshold)
        topKNovelIntent = refineIntent(threadID, topKCandidateVector, schemaDicts, configDict)
        topKPredictedIntents.append(topKNovelIntent)
    return topKPredictedIntents

if __name__ == "__main__":
    #configDict = parseConfig.parseConfigFile("configFile.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="Config parameters file", type=str, required=True)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    schemaDicts = readSchemaDicts(configDict)