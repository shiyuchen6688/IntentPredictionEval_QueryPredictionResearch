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
import ReverseEnggQueries

class SQLForIntent:
    def __init__(self, schemaDicts, intentVec):
        self.schemaDicts = schemaDicts
        self.intentVec = intentVec
        self.queryTypeBitMapSize = 4  # select, insert, update, delete
        self.limitBitMapSize = 1
        self.tableBitMapSize = estimateTableBitMapSize(schemaDicts)
        self.allColumnsSize = estimateAllColumnCount(schemaDicts)
        self.joinPredicatesBitMapSize = estimateJoinPredicatesBitMapSize(schemaDicts)

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

        self.queryTypeBitMap = self.intentVec[self.queryTypeStartBitIndex:self.queryTypeStartBitIndex + self.queryTypeBitMapSize]
        self.tableBitMap = self.intentVec[self.tableStartBitIndex:self.tableStartBitIndex + self.tableBitMapSize]
        self.projectionBitMap = self.intentVec[self.projectionStartBitIndex:self.projectionStartBitIndex + self.allColumnsSize]
        self.avgBitMap = self.intentVec[self.avgStartBitIndex:self.avgStartBitIndex + self.allColumnsSize]
        self.minBitMap = self.intentVec[self.minStartBitIndex:self.minStartBitIndex + self.allColumnsSize]
        self.maxBitMap = self.intentVec[self.maxStartBitIndex:self.maxStartBitIndex + self.allColumnsSize]
        self.sumBitMap = self.intentVec[self.sumStartBitIndex:self.sumStartBitIndex + self.allColumnsSize]
        self.countBitMap = self.intentVec[self.countStartBitIndex:self.countStartBitIndex + self.allColumnsSize]
        self.selectionBitMap = self.intentVec[self.selectionStartBitIndex:self.selectionStartBitIndex + self.allColumnsSize]
        self.groupByBitMap = self.intentVec[self.groupByStartBitIndex:self.groupByStartBitIndex + self.allColumnsSize]
        self.orderByBitMap = self.intentVec[self.orderByStartBitIndex:self.orderByStartBitIndex + self.allColumnsSize]
        self.havingBitMap = self.intentVec[self.havingStartBitIndex:self.havingStartBitIndex + self.allColumnsSize]
        self.limitBitMap = self.intentVec[self.limitStartBitIndex:self.limitStartBitIndex + self.limitBitMapSize]
        self.joinPredicatesBitMap = self.intentVec[self.joinPredicatesStartBitIndex:self.joinPredicatesStartBitIndex + self.joinPredicatesBitMapSize]

        self.queryType = None
        self.tables = []
        self.projCols = []
        self.avgCols = []
        self.minCols = []
        self.maxCols = []
        self.sumCols = []
        self.countCols = []
        self.selCols = []
        self.groupByCols = []
        self.orderByCols = []
        self.havingCols = []
        self.limit = None
        self.joinPreds = []


def readIntentObjectsFromFile(intentFileName):
    intentObjDict = {}
    with open(intentFileName) as f:
        for line in f:
            tokens = line.strip().split(":")
            assert len(tokens) == 2
            intentObjDict[tokens[0]] =tokens[1]
    return intentObjDict

def initIntentObj(schemaDicts, intentVec):
    intentObj = SQLForIntent(schemaDicts, intentVec)
    return intentObj

def estimateTableBitMapSize(schemaDicts):
    tableDict = schemaDicts.tableDict
    return len(tableDict)

def estimateAllColumnCount(schemaDicts):
    colCount = 0
    for tableName in schemaDicts.tableDict:
        colCount += len(schemaDicts.colDict[tableName])
    return colCount

def estimateJoinPredicatesBitMapSize(schemaDicts):
    joinPredDict = schemaDicts.joinPredDict
    joinPredCount = 0
    for tabPair in joinPredDict:
        joinPredCount += len(joinPredDict[tabPair])+1 # accounting for the one extra bit
    return joinPredCount

def assertSize(intentObj):
    # estAllColumnsSize = estGroupByBitMapSize = estOrderByBitMapSize = estProjectionBitMapSize = estHavingBitMapSize =
    # estMinBitMapSize = estMaxBitMapSize = estAvgBitMapSize = estSumBitMapSize = estCountBitMapSize
    allOpSize = intentObj.queryTypeBitMapSize + intentObj.tableBitMapSize + intentObj.allColumnsSize * 9 + intentObj.limitBitMapSize + intentObj.joinPredicatesBitMapSize
    print "estAllOpSize: "+str(allOpSize)+", len(intentVec): "+str(len(intentObj.intentVec))
    return

def assertIntentOpObjects(intentObj, intentObjDict):
    assertSize(intentObj)
    sameObjs = 1
    if(intentObj.queryTypeBitMap != intentObjDict['queryTypeBitMap'] or intentObj.tableBitMap != intentObjDict['TableBitMap']
       or intentObj.groupByBitMap != intentObjDict['GroupByBitMap'] or intentObj.orderByBitMap != intentObjDict['OrderByBitMap']
       or intentObj.projectionBitMap != intentObjDict['ProjectionBitMap'] or intentObj.havingBitMap != intentObjDict['HavingBitMap']
       or intentObj.joinPredicatesBitMap != intentObjDict['JoinPredicatesBitMap'] or intentObj.limitBitMap != intentObjDict['LimitBitMap']
       or intentObj.minBitMap != intentObjDict['MINBitMap'] or intentObj.maxBitMap != intentObjDict['MAXBitMap']
       or intentObj.avgBitMap != intentObjDict['AVGBitMap'] or intentObj.sumBitMap != intentObjDict['SUMBitMap']
       or intentObj.countBitMap != intentObjDict['COUNTBitMap'] or intentObj.selectionBitMap != intentObjDict['SelectionBitMap']):
        sameObjs = 0
    print "Assertion outcome: "+str(sameObjs)
    return sameObjs

def populateQueryType(intentObj):
    assert len(intentObj.queryTypeBitMap) == 4 # select, update, insert, delete
    if int(intentObj.queryTypeBitMap[0]) == 1:
        intentObj.queryType = "select"
    elif int(intentObj.queryTypeBitMap[1]) == 1:
        intentObj.queryType = "update"
    elif int(intentObj.queryTypeBitMap[2]) == 1:
        intentObj.queryType = "insert"
    elif int(intentObj.queryTypeBitMap[3]) == 1:
        intentObj.queryType = "delete"
    return intentObj

def searchForTable(tableDict, index):
    for table in tableDict:
        val = tableDict[table]
        if int(val) == index:
            return table
    return None

def populateTables(intentObj):
    assert len(intentObj.tables) == 0
    for i in range(len(intentObj.tableBitMap)):
        if int(intentObj.tableBitMap[i]) == 1:
            tableName = intentObj.schemaDicts.tableOrderDict[i]
            assert tableName is not None
            intentObj.tables.append(tableName)
    return intentObj

def populateLimit(intentObj):
    assert len(intentObj.limitBitMap)==1
    if int(intentObj.limitBitMap[0])==1:
        intentObj.limit = "Limit"
    return intentObj

def populateColsForOp(opCols, opBitMap, intentObj):
    assert len(opCols) == 0
    bitMapIndex=0
    for tableIndex in range(len(intentObj.schemaDicts.tableOrderDict)):
        tableName = intentObj.schemaDicts.tableOrderDict[tableIndex]
        colList = intentObj.schemaDicts.colDict[tableName]
        for col in colList:
            if int(opBitMap[bitMapIndex]) == 1:
                opCols.append(tableName+"."+col)
            bitMapIndex+=1
    return opCols

def populateJoinPreds(intentObj):
    assert len(intentObj.joinPreds) == 0
    for tablePairIndex in intentObj.schemaDicts.joinPredBitPosDict:
        startEndBitPos = intentObj.schemaDicts.joinPredBitPosDict[tablePairIndex]
        startBitPos = startEndBitPos[0]
        endBitPos = startEndBitPos[1]
        for i in range(startBitPos, endBitPos):
            if int(intentObj.joinPredicatesBitMap[i])==1:
                joinColPair = intentObj.schemaDicts.joinPredDict[tablePairIndex][i-startBitPos]
                joinStrToAppend = tablePairIndex.split(",")[0] + "." + joinColPair.split(",")[0]+ "," + tablePairIndex.split(",")[1] + "." + joinColPair.split(",")[1]
                intentObj.joinPreds.append(joinStrToAppend)
    return intentObj

def createSQLFromIntent(intentObj):
    intentObj = populateQueryType(intentObj)
    intentObj = populateTables(intentObj)
    intentObj.projCols = populateColsForOp(intentObj.projCols, intentObj.projectionBitMap, intentObj)
    intentObj.avgCols = populateColsForOp(intentObj.avgCols, intentObj.projectionBitMap, intentObj)
    intentObj.minCols = populateColsForOp(intentObj.minCols, intentObj.minBitMap, intentObj)
    intentObj.maxCols = populateColsForOp(intentObj.maxCols, intentObj.maxBitMap, intentObj)
    intentObj.sumCols = populateColsForOp(intentObj.sumCols, intentObj.sumBitMap, intentObj)
    intentObj.countCols = populateColsForOp(intentObj.countCols, intentObj.countBitMap, intentObj)
    intentObj.selCols = populateColsForOp(intentObj.selCols, intentObj.selectionBitMap, intentObj)
    intentObj.groupByCols = populateColsForOp(intentObj.groupByCols, intentObj.groupByBitMap, intentObj)
    intentObj.orderByCols = populateColsForOp(intentObj.orderByCols, intentObj.orderByBitMap, intentObj)
    intentObj.havingCols = populateColsForOp(intentObj.havingCols, intentObj.havingBitMap, intentObj)
    intentObj = populateLimit(intentObj)
    intentObj = populateJoinPreds(intentObj)
    return intentObj

def printSQLOps(intentObj):
    print "Query Type: "+intentObj.queryType
    print "Tables: "+str(intentObj.tables)
    print "Projected Columns: "+str(intentObj.projCols)
    print "AVG Columns: "+str(intentObj.avgCols)
    print "MIN Columns: "+str(intentObj.minCols)
    print "MAX Columns: " + str(intentObj.maxCols)
    print "SUM Columns: " + str(intentObj.sumCols)
    print "COUNT Columns: " + str(intentObj.countCols)
    print "SEL Columns: " + str(intentObj.selCols)
    print "GROUP BY Columns: " + str(intentObj.groupByCols)
    print "ORDER BY Columns: " + str(intentObj.orderByCols)
    print "HAVING Columns: " + str(intentObj.havingCols)
    print "Limit: " + str(intentObj.limit)
    print "JOIN PRED ColPairs: "+ str(intentObj.joinPreds)

def createSQLFromIntentSanityCheck(schemaDicts, intentObjDict):
    intentObj = initIntentObj(schemaDicts, intentObjDict['intentVector'])
    assertSize(intentObj)
    assertIntentOpObjects(intentObj, intentObjDict)
    createSQLFromIntent(intentObj)
    printSQLOps(intentObj)


if __name__ == "__main__":
    #configDict = parseConfig.parseConfigFile("configFile.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="Config parameters file", type=str, required=True)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    schemaDicts = ReverseEnggQueries.readSchemaDicts(configDict)
    intentObjDict = readIntentObjectsFromFile("/Users/postgres/Documents/DataExploration-Research/MINC/InputOutput/tempVector")
    createSQLFromIntentSanityCheck(schemaDicts, intentObjDict)
    #createSQLFromIntent(schemaDicts, intentObjDict['intentVector'])