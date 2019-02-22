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

def pruneUnImportantDimensions(predictedY, configDict):
    newPredictedY = []
    minY = min(predictedY)
    maxY = max(predictedY)
    for y in predictedY:
        newY = float(y-minY)/float(maxY-minY)
        if newY < float(configDict['RNN_WEIGHT_VECTOR_THRESHOLD']):
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
    print "joinPredCount: "+str(joinPredCount)+", joinPredBitPosCount: "+str(joinPredBitPosCount)

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

def regenerateQuery(threadID, predictedY, configDict, curSessID, curQueryID, sessionDictCurThread, sessionStreamDict):
    topKPredictedIntents = []
    schemaDicts = readSchemaDicts(configDict)
    return topKPredictedIntents

if __name__ == "__main__":
    #configDict = parseConfig.parseConfigFile("configFile.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="Config parameters file", type=str, required=True)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    schemaDicts = readSchemaDicts(configDict)