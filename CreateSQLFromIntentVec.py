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

class SQLForIntentStr:
    def __init__(self, schemaDicts, intentVec):
        self.schemaDicts = schemaDicts
        self.intentVec = intentVec
        # following populates the required sub-bitmaps
        self.queryTypeBitMap = self.intentVec[schemaDicts.queryTypeStartBitIndex:schemaDicts.queryTypeStartBitIndex + schemaDicts.queryTypeBitMapSize]
        self.tableBitMap = self.intentVec[schemaDicts.tableStartBitIndex:schemaDicts.tableStartBitIndex + schemaDicts.tableBitMapSize]
        self.projectionBitMap = self.intentVec[schemaDicts.projectionStartBitIndex:schemaDicts.projectionStartBitIndex + schemaDicts.allColumnsSize]
        self.avgBitMap = self.intentVec[schemaDicts.avgStartBitIndex:schemaDicts.avgStartBitIndex + schemaDicts.allColumnsSize]
        self.minBitMap = self.intentVec[schemaDicts.minStartBitIndex:schemaDicts.minStartBitIndex + schemaDicts.allColumnsSize]
        self.maxBitMap = self.intentVec[schemaDicts.maxStartBitIndex:schemaDicts.maxStartBitIndex + schemaDicts.allColumnsSize]
        self.sumBitMap = self.intentVec[schemaDicts.sumStartBitIndex:schemaDicts.sumStartBitIndex + schemaDicts.allColumnsSize]
        self.countBitMap = self.intentVec[schemaDicts.countStartBitIndex:schemaDicts.countStartBitIndex + schemaDicts.allColumnsSize]
        self.selectionBitMap = self.intentVec[schemaDicts.selectionStartBitIndex:schemaDicts.selectionStartBitIndex + schemaDicts.allColumnsSize]
        self.groupByBitMap = self.intentVec[schemaDicts.groupByStartBitIndex:schemaDicts.groupByStartBitIndex + schemaDicts.allColumnsSize]
        self.orderByBitMap = self.intentVec[schemaDicts.orderByStartBitIndex:schemaDicts.orderByStartBitIndex + schemaDicts.allColumnsSize]
        self.havingBitMap = self.intentVec[schemaDicts.havingStartBitIndex:schemaDicts.havingStartBitIndex + schemaDicts.allColumnsSize]
        self.limitBitMap = self.intentVec[schemaDicts.limitStartBitIndex:schemaDicts.limitStartBitIndex + schemaDicts.limitBitMapSize]
        self.joinPredicatesBitMap = self.intentVec[schemaDicts.joinPredicatesStartBitIndex:schemaDicts.joinPredicatesStartBitIndex + schemaDicts.joinPredicatesBitMapSize]

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

def initIntentStrObj(schemaDicts, intentVec):
    intentObj = SQLForIntentStr(schemaDicts, intentVec)
    return intentObj

def assertSize(intentObj):
    # estAllColumnsSize = estGroupByBitMapSize = estOrderByBitMapSize = estProjectionBitMapSize = estHavingBitMapSize =
    # estMinBitMapSize = estMaxBitMapSize = estAvgBitMapSize = estSumBitMapSize = estCountBitMapSize
    print "estAllOpSize: "+str(intentObj.schemaDicts.allOpSize)+", len(intentVec): "+str(len(intentObj.intentVec))
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

def populateQueryTypeStr(intentObj):
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

def populateTablesStr(intentObj):
    assert len(intentObj.tables) == 0
    for i in range(len(intentObj.tableBitMap)):
        if int(intentObj.tableBitMap[i]) == 1:
            tableName = intentObj.schemaDicts.tableOrderDict[i]
            assert tableName is not None
            intentObj.tables.append(tableName)
    return intentObj

def populateLimitStr(intentObj):
    assert len(intentObj.limitBitMap)==1
    if int(intentObj.limitBitMap[0])==1:
        intentObj.limit = "Limit"
    return intentObj

def populateColsForOpStr(opCols, opBitMap, intentObj):
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

def populateJoinPredsStr(intentObj):
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

def createSQLFromIntentString(intentObj):
    intentObj = populateQueryTypeStr(intentObj)
    intentObj = populateTablesStr(intentObj)
    intentObj.projCols = populateColsForOpStr(intentObj.projCols, intentObj.projectionBitMap, intentObj)
    intentObj.avgCols = populateColsForOpStr(intentObj.avgCols, intentObj.projectionBitMap, intentObj)
    intentObj.minCols = populateColsForOpStr(intentObj.minCols, intentObj.minBitMap, intentObj)
    intentObj.maxCols = populateColsForOpStr(intentObj.maxCols, intentObj.maxBitMap, intentObj)
    intentObj.sumCols = populateColsForOpStr(intentObj.sumCols, intentObj.sumBitMap, intentObj)
    intentObj.countCols = populateColsForOpStr(intentObj.countCols, intentObj.countBitMap, intentObj)
    intentObj.selCols = populateColsForOpStr(intentObj.selCols, intentObj.selectionBitMap, intentObj)
    intentObj.groupByCols = populateColsForOpStr(intentObj.groupByCols, intentObj.groupByBitMap, intentObj)
    intentObj.orderByCols = populateColsForOpStr(intentObj.orderByCols, intentObj.orderByBitMap, intentObj)
    intentObj.havingCols = populateColsForOpStr(intentObj.havingCols, intentObj.havingBitMap, intentObj)
    intentObj = populateLimitStr(intentObj)
    intentObj = populateJoinPredsStr(intentObj)
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

def checkBitMapWorking(intentObjDict):
    bitStr = "0100000001110001000"
    b = BitMap.fromstring(bitStr)
    if(b.size()>len(bitStr)):
        startPos = b.size() % 8
        bitStr = b.tostring()[startPos:b.size()]
    print bitStr
    print b.nonzero()
    print b.size()
    print len(bitStr)
    print "Length of intentObjDict['intentVector']: " + str(len(intentObjDict['intentVector']))
    bitmap = BitMap.fromstring(intentObjDict['intentVector'])
    print "Length of bitmap.tostring(): " + str(len(bitmap.tostring()))

def createSQLFromIntentStrSanityCheck(schemaDicts, intentObjDict):
    intentObj = initIntentStrObj(schemaDicts, intentObjDict['intentVector'])
    assertIntentOpObjects(intentObj, intentObjDict)
    createSQLFromIntentString(intentObj)
    printSQLOps(intentObj)

def checkOpToPopulate(newSetBitPos, intentObj):
    if newSetBitPos >= intentObj.schemaDicts.joinPredicatesStartBitIndex:
        return "join"
    elif newSetBitPos >= intentObj.schemaDicts.limitStartBitIndex:
        return "limit"
    elif newSetBitPos >= intentObj.schemaDicts.havingStartBitIndex:
        return "having"
    elif newSetBitPos >= intentObj.schemaDicts.orderByStartBitIndex:
        return "orderby"
    elif newSetBitPos >= intentObj.schemaDicts.groupByStartBitIndex:
        return "groupby"
    elif newSetBitPos >= intentObj.schemaDicts.selectionStartBitIndex:
        return "select"
    elif newSetBitPos >= intentObj.schemaDicts.countStartBitIndex:
        return "count"
    elif newSetBitPos >= intentObj.schemaDicts.sumStartBitIndex:
        return "sum"
    elif newSetBitPos >= intentObj.schemaDicts.maxStartBitIndex:
        return "max"
    elif newSetBitPos >= intentObj.schemaDicts.minStartBitIndex:
        return "min"
    elif newSetBitPos >= intentObj.schemaDicts.avgStartBitIndex:
        return "avg"
    elif newSetBitPos >= intentObj.schemaDicts.projectionStartBitIndex:
        return "project"
    elif newSetBitPos >= intentObj.schemaDicts.tableStartBitIndex:
        return "table"
    elif newSetBitPos >= intentObj.schemaDicts.queryTypeStartBitIndex:
        return "querytype"
    else:
        print "not possible !!"

def populateOps(intentObj, opsToPopulate):
    for opToPopulate in opsToPopulate:
        if opToPopulate == "querytype":
            intentObj = populateQueryTypeStr(intentObj)
        elif opToPopulate == "table":
            intentObj = populateTablesStr(intentObj)
        elif opToPopulate == "project":
            intentObj.projCols = populateColsForOpStr(intentObj.projCols, intentObj.projectionBitMap, intentObj)
        elif opToPopulate == "avg":
            intentObj.avgCols = populateColsForOpStr(intentObj.avgCols, intentObj.projectionBitMap, intentObj)
        elif opToPopulate == "min":
            intentObj.minCols = populateColsForOpStr(intentObj.minCols, intentObj.minBitMap, intentObj)
        elif opToPopulate == "max":
            intentObj.maxCols = populateColsForOpStr(intentObj.maxCols, intentObj.maxBitMap, intentObj)
        elif opToPopulate == "sum":
            intentObj.sumCols = populateColsForOpStr(intentObj.sumCols, intentObj.sumBitMap, intentObj)
        elif opToPopulate == "count":
            intentObj.countCols = populateColsForOpStr(intentObj.countCols, intentObj.countBitMap, intentObj)
        elif opToPopulate == "select":
            intentObj.selCols = populateColsForOpStr(intentObj.selCols, intentObj.selectionBitMap, intentObj)
        elif opToPopulate == "groupby":
            intentObj.groupByCols = populateColsForOpStr(intentObj.groupByCols, intentObj.groupByBitMap, intentObj)
        elif opToPopulate == "orderby":
            intentObj.orderByCols = populateColsForOpStr(intentObj.orderByCols, intentObj.orderByBitMap, intentObj)
        elif opToPopulate == "having":
            intentObj.havingCols = populateColsForOpStr(intentObj.havingCols, intentObj.havingBitMap, intentObj)
        elif opToPopulate == "limit":
            intentObj = populateLimitStr(intentObj)
        elif opToPopulate == "join":
            intentObj = populateJoinPredsStr(intentObj)
    return intentObj

def createSQLFromIntentStringBitPos(intentObj, newSetBitPosList):
    opsToPopulate = set()
    for newSetBitPos in newSetBitPosList:
        opToPopulate = checkOpToPopulate(newSetBitPos, intentObj)
        opsToPopulate.add(opToPopulate)
    populateOps(intentObj, opsToPopulate)
    printSQLOps(intentObj)

def createSQLFromIntentBitMapSanityCheck(schemaDicts, intentObjDict):
    checkBitMapWorking(intentObjDict)
    intentBitMap = BitMap.fromstring(intentObjDict['intentVector'])
    intentStr = intentBitMap.tostring()
    if intentBitMap.size() > schemaDicts.allOpSize:
        startBit = intentBitMap.size() - schemaDicts.allOpSize
        intentStr = intentStr[startBit:intentBitMap.size()]
    assert intentStr == intentObjDict['intentVector']
    intentObj = initIntentStrObj(schemaDicts, intentStr)
    assertIntentOpObjects(intentObj, intentObjDict)
    setBitPosList = intentBitMap.nonzero()
    newSetBitPosList = []
    for bitPos in setBitPosList:
        newBitPos = schemaDicts.allOpSize - 1 - bitPos
        newSetBitPosList.append(newBitPos)
    createSQLFromIntentStringBitPos(intentObj, newSetBitPosList)


if __name__ == "__main__":
    #configDict = parseConfig.parseConfigFile("configFile.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="Config parameters file", type=str, required=True)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    schemaDicts = ReverseEnggQueries.readSchemaDicts(configDict)
    intentObjDict = readIntentObjectsFromFile("/Users/postgres/Documents/DataExploration-Research/MINC/InputOutput/tempVector")
    #createSQLFromIntentStrSanityCheck(schemaDicts, intentObjDict)
    createSQLFromIntentBitMapSanityCheck(schemaDicts, intentObjDict)
    #createSQLFromIntent(schemaDicts, intentObjDict['intentVector'])