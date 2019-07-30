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
    schemaDicts.forwardMapBitsToOps[schemaDicts.limitStartBitIndex] = ";limit"
    schemaDicts.backwardMapOpsToBits[";limit"] = schemaDicts.limitStartBitIndex
    return schemaDicts

def populateJoinPreds(schemaDicts):
    opString = "join"
    for tablePairIndex in schemaDicts.joinPredBitPosDict:
        startEndBitPos = schemaDicts.joinPredBitPosDict[tablePairIndex]
        startBitPos = startEndBitPos[0]+schemaDicts.joinPredicatesStartBitIndex
        endBitPos = startEndBitPos[1]+schemaDicts.joinPredicatesStartBitIndex
        for indexToSet in range(startBitPos, endBitPos+1):
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
    #print schemaDicts.allOpSize
    assert len(schemaDicts.forwardMapBitsToOps) == len(schemaDicts.backwardMapOpsToBits)
    assert len(schemaDicts.forwardMapBitsToOps) == schemaDicts.allOpSize
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

def employWeightThreshold(predictedY, schemaDicts, weightThreshold):
    startBit = len(predictedY) - schemaDicts.allOpSize
    predictedY = predictedY[startBit:len(predictedY)]
    newPredictedY = []
    minY = min(predictedY)
    maxY = max(predictedY)
    denom = float(maxY - minY)
    for y in predictedY:
        newY = float(y)
        if denom > 0:
            newY = float(y - minY) / float(maxY - minY)  # normalize each dimension to lie between 0 and 1
        if newY < float(weightThreshold):
            newY = 0
        else:
            newY = 1
        newPredictedY.append(newY)
    bitMapStr = ''.join(str(e) for e in newPredictedY)
    bitMap = BitMap.fromstring(bitMapStr)
    return bitMap

def pruneUnImportantDimensions(predictedY, weightThreshold):
    newPredictedY = []
    minY = min(predictedY)
    maxY = max(predictedY)
    denom = float(maxY - minY)
    for y in predictedY:
        newY = float(y)
        if denom > 0:
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
        joinPredBitPosCount += joinPredBitPosDict[key][1] - joinPredBitPosDict[key][0] + 1
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

def refineIntentForQuery(threadID, topKCandidateVector, schemaDicts, precOrRecallFavor, configDict, curIntentBitVec):
    # Step 1: regenerate the query ops from the topKCandidateVector
    # print "-----------Original SQL----------------"
    predictedIntentObj = CreateSQLFromIntentVec.regenerateSQL(topKCandidateVector, schemaDicts)
    curIntentObj = None
    if configDict['RNN_DEFAULT_CUR_QUERY'] == 'True' and curIntentBitVec is not None:
        curIntentObj = CreateSQLFromIntentVec.regenerateSQL(curIntentBitVec, schemaDicts)
    # Step 2: refine SQL violations
    intentObj = CreateSQLFromIntentVec.fixSQLViolations(predictedIntentObj, precOrRecallFavor, curIntentObj)
    # print "-----------Refined SQL-----------------"
    # intentObj = CreateSQLFromIntentVec.regenerateSQL(intentObj.intentBitVec, schemaDicts)
    return intentObj

def refineIntentForTable(threadID, topKCandidateVector, schemaDicts, precOrRecallFavor, configDict, curIntentBitVec):
    predictedIntentObj = CreateSQLFromIntentVec.regenerateSQLTable(topKCandidateVector, curIntentBitVec, schemaDicts, configDict)
    return predictedIntentObj

def refineIntent(threadID, topKCandidateVector, schemaDicts, precOrRecallFavor, configDict, curIntentBitVec):
    assert configDict['RNN_PREDICT_QUERY_OR_TABLE'] == 'QUERY' or configDict['RNN_PREDICT_QUERY_OR_TABLE'] =='TABLE'
    if configDict['RNN_PREDICT_QUERY_OR_TABLE'] == 'QUERY':
        intentObj = refineIntentForQuery(threadID, topKCandidateVector, schemaDicts, precOrRecallFavor, configDict, curIntentBitVec)
    elif configDict['RNN_PREDICT_QUERY_OR_TABLE'] =='TABLE':
        intentObj = refineIntentForTable(threadID, topKCandidateVector, schemaDicts, precOrRecallFavor, configDict, curIntentBitVec)
    return intentObj.intentBitVec

def predictTopKNovelIntentsSingleThread(threadID, predictedY, schemaDicts, configDict, curIntentBitVec):
    topKPredictedIntents = []
    #schemaDicts = readSchemaDicts(configDict)
    thresholds = topKThres(configDict)
    precOrRecallFavor = configDict['RNN_PREC_OR_RECALL_FAVOR']
    for threshold in thresholds:
        topKCandidateVector = employWeightThreshold(predictedY, schemaDicts, threshold)
        if int(configDict['TOP_K']) == 3 and threshold == 0.8:
            precOrRecallFavor = "recall"
        elif int(configDict['TOP_K']) == 3 and threshold == 0.6:
            precOrRecallFavor = "recall"
        elif int(configDict['TOP_K']) == 3 and threshold == 0.4:
            precOrRecallFavor = "recall"
        topKNovelIntent = refineIntent(threadID, topKCandidateVector, schemaDicts, precOrRecallFavor, configDict, curIntentBitVec)
        topKPredictedIntents.append(topKNovelIntent)
    return topKPredictedIntents

def predictTopKNovelIntentsProcess((threadID, predictedY, schemaDicts, configDict, curIntentBitVec)):
    topKPredictedIntents = predictTopKNovelIntentsSingleThread(threadID, predictedY, schemaDicts, configDict, curIntentBitVec)
    QR.writeToPickleFile(getConfig(configDict['PICKLE_TEMP_OUTPUT_DIR']) + "localTopKDict_" + str(threadID) + ".pickle", topKPredictedIntents)
    return

def predictTopKNovelIntents(threadID, predictedY, schemaDicts, configDict, curIntentBitVec):
    if int(configDict['RNN_SUB_THREADS']) == 0:
        return predictTopKNovelIntentsSingleThread(threadID, predictedY, schemaDicts, configDict, curIntentBitVec)
    else:
        pool = multiprocessing.Pool()
        argList = []
        argList.append((threadID, predictedY, schemaDicts, configDict, curIntentBitVec))
        pool.map(predictTopKNovelIntentsProcess, argList)
        pool.close()
        pool.join()
        topKPredictedIntents = QR.readFromPickleFile(getConfig(configDict['PICKLE_TEMP_OUTPUT_DIR']) + "localTopKDict_" + str(threadID) + ".pickle")
        return topKPredictedIntents

def createAndRefineIntentsForMockQueries(schemaDicts, configDict):
    intentObj = CreateSQLFromIntentVec.SQLForBitMapIntent(schemaDicts, None, None)
    intentObj.intentBitVec = BitMap(schemaDicts.allOpSize)
    #intentObj.queryType = "select"
    intentObj.queryType = None
    if intentObj.queryType is not None:
        bitToSet = schemaDicts.backwardMapOpsToBits["select;querytype"]
        CreateSQLFromIntentVec.setBit(bitToSet, intentObj)
    #intentObj.tables = ['jos_menu', 'jos_components']
    intentObj.tables = []
    for tableName in intentObj.tables:
        bitToSet = schemaDicts.backwardMapOpsToBits[tableName+";table"]
        CreateSQLFromIntentVec.setBit(bitToSet, intentObj)
    #intentObj.projCols = ['jos_components.option', 'jos_components.id']
    intentObj.projCols = []
    for projCol in intentObj.projCols:
        bitToSet = schemaDicts.backwardMapOpsToBits[projCol + ";project"]
        CreateSQLFromIntentVec.setBit(bitToSet, intentObj)
    intentObj.avgCols = [] # placeholder for future
    for avgCol in intentObj.avgCols:
        bitToSet = schemaDicts.backwardMapOpsToBits[avgCol + ";avg"]
        CreateSQLFromIntentVec.setBit(bitToSet, intentObj)
    #intentObj.minCols = ['jos_components.id']  # placeholder for future
    intentObj.minCols = []
    for minCol in intentObj.minCols:
        bitToSet = schemaDicts.backwardMapOpsToBits[minCol + ";min"]
        CreateSQLFromIntentVec.setBit(bitToSet, intentObj)
    intentObj.maxCols = []  # placeholder for future
    for maxCol in intentObj.maxCols:
        bitToSet = schemaDicts.backwardMapOpsToBits[maxCol + ";max"]
        CreateSQLFromIntentVec.setBit(bitToSet, intentObj)
    intentObj.sumCols = []  # placeholder for future
    for sumCol in intentObj.sumCols:
        bitToSet = schemaDicts.backwardMapOpsToBits[sumCol + ";sum"]
        CreateSQLFromIntentVec.setBit(bitToSet, intentObj)
    intentObj.countCols = []  # placeholder for future
    for countCol in intentObj.countCols:
        bitToSet = schemaDicts.backwardMapOpsToBits[countCol + ";count"]
        CreateSQLFromIntentVec.setBit(bitToSet, intentObj)
    #intentObj.selCols = ['jos_menu.published']  # placeholder for future
    intentObj.selCols =[]
    for selCol in intentObj.selCols:
        bitToSet = schemaDicts.backwardMapOpsToBits[selCol + ";select"]
        CreateSQLFromIntentVec.setBit(bitToSet, intentObj)
    #intentObj.groupByCols = ['jos_menu.sublevel']  # placeholder for future
    intentObj.groupByCols = []
    for groupByCol in intentObj.groupByCols:
        bitToSet = schemaDicts.backwardMapOpsToBits[groupByCol + ";groupby"]
        CreateSQLFromIntentVec.setBit(bitToSet, intentObj)
    #intentObj.orderByCols = ['jos_menu.ordering', 'jos_menu.sublevel', 'jos_menu.parent']  # placeholder for future
    intentObj.orderByCols = []
    for orderByCol in intentObj.orderByCols:
        bitToSet = schemaDicts.backwardMapOpsToBits[orderByCol + ";orderby"]
        CreateSQLFromIntentVec.setBit(bitToSet, intentObj)
    #intentObj.havingCols = ['jos_menu.lft']  # placeholder for future
    intentObj.havingCols = []
    for havingCol in intentObj.havingCols:
        bitToSet = schemaDicts.backwardMapOpsToBits[havingCol + ";having"]
        CreateSQLFromIntentVec.setBit(bitToSet, intentObj)
    intentObj.limit = None
    if intentObj.limit is not None:
        bitToSet = schemaDicts.backwardMapOpsToBits[";limit"]
        CreateSQLFromIntentVec.setBit(bitToSet, intentObj)
    #intentObj.joinPreds = ['jos_menu.params,jos_components.params', 'jos_menu.ordering,jos_components.ordering', 'jos_menu.componentid,jos_components.id', 'jos_menu.name,jos_components.name']
    intentObj.joinPreds = []
    for joinPred in intentObj.joinPreds:
        bitToSet = schemaDicts.backwardMapOpsToBits[joinPred + ";join"]
        CreateSQLFromIntentVec.setBit(bitToSet, intentObj)
    refineIntentForQuery(0, intentObj.intentBitVec, schemaDicts, "recall", configDict, None)


if __name__ == "__main__":
    #configDict = parseConfig.parseConfigFile("configFile.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="Config parameters file", type=str, required=True)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    schemaDicts = readSchemaDicts(configDict)
    #intentObjDict = CreateSQLFromIntentVec.readIntentObjectsFromFile("/Users/postgres/Documents/DataExploration-Research/MINC/InputOutput/tempVector")
    #refineIntent(0, BitMap.fromstring(intentObjDict['intentVector']), schemaDicts, configDict['RNN_PREC_OR_RECALL_FAVOR'], configDict)
    createAndRefineIntentsForMockQueries(schemaDicts, configDict)