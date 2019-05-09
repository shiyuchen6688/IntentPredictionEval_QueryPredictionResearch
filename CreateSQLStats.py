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
import argparse
from ParseConfigFile import getConfig
import threading
import copy
import multiprocessing
from multiprocessing.pool import ThreadPool
from multiprocessing import Array
import ReverseEnggQueries
import CreateSQLFromIntentVec


def increment(key, dictName):
    if key not in dictName:
        dictName[key] = 1
    else:
        dictName[key] = dictName[key] + 1
    return dictName

def checkTableAddedOrDropped(curTableSet, nextTableSet):
    if set(curTableSet).issubset(set(nextTableSet)) and len(nextTableSet) > len(curTableSet):
        return "numTableAddedOnly"
    elif set(nextTableSet).issubset(set(curTableSet)) and len(nextTableSet) < len(curTableSet):
        return "numTableDroppedOnly"

def checkTableOverlaps(curTableSet, nextTableSet):
    overlap = list(set(curTableSet).intersection(set(nextTableSet)))
    if len(overlap) == 0:
        return "numTableSetChangedCompletely"


def computeTableStats(intentObjDict):
    statsDict = {}
    # numTransitions, numOneTableToSeveralTables, numSeveralTablesToOneTable, numOneTableToOneTable,
    # numSameTableToSameTable, numTableAddedOnly, numTableDroppedOnly, numTableSetNotChanged,
    # numTableSetChangedCompletely, numTableSetChangedWithOverlaps
    for sessQueryID in intentObjDict:
        sessID = int(sessQueryID.split(",")[0])
        queryID = int(sessQueryID.split(",")[1])
        nextQueryID = queryID + 1
        nextSessQueryID = str(sessID)+","+str(nextQueryID)
        if nextSessQueryID in intentObjDict:
            statsDict = increment('numTransitions', statsDict)
            nextTableSet = intentObjDict[nextSessQueryID].tables
            curTableSet = intentObjDict[sessQueryID].tables
            if len(curTableSet) == 1 and len(nextTableSet) == 1:
                statsDict = increment("numOneTableToOneTable", statsDict)
                if curTableSet[0] == nextTableSet[0]:
                    statsDict = increment("numSameTableToSameTable", statsDict)
            elif len(curTableSet) == 1 and len(nextTableSet) > 1:
                statsDict = increment("numOneTableToSeveralTables", statsDict)
            elif len(curTableSet) > 1 and len(nextTableSet) == 1:
                statsDict = increment("numSeveralTablesToOneTable", statsDict)
            if set(curTableSet) == set(nextTableSet):
                statsDict = increment("numTableSetNotChanged", statsDict)
            key = checkTableAddedOrDropped(curTableSet, nextTableSet)
            statsDict = increment(key, statsDict)
            key = checkTableOverlaps(curTableSet, nextTableSet)
            statsDict = increment(key, statsDict)
            statsDict['numTableSetChangedWithOverlaps'] = statsDict['numTransitions'] - (statsDict['numTableSetChangedCompletely'] + statsDict['numTableSetNotChanged'])
    return statsDict

def writeDictToFile(statsDict, outputSQLStats):
    for key in statsDict:
        val = statsDict[key]
        outputStr = str(key)+"="+str(val)+"\n"
        ti.appendToFile(outputSQLStats, outputStr)

def computeStats(configDict, intentObjDict, outputSQLStats):
    QR.deleteIfExists(outputSQLStats)
    statsDict = None
    assert configDict['RNN_PREDICT_QUERY_OR_TABLE'] == 'QUERY' or configDict['RNN_PREDICT_QUERY_OR_TABLE'] == 'TABLE'
    if configDict['RNN_PREDICT_QUERY_OR_TABLE'] == 'TABLE':
        statsDict = computeTableStats(intentObjDict)
    writeDictToFile(statsDict, outputSQLStats)
    return

def createIntentObjDict(schemaDicts, intentFileName):
    intentObjDict = {} # key is sessQueryID and val is intentObj
    assert configDict['RNN_PREDICT_QUERY_OR_TABLE'] == 'QUERY' or configDict['RNN_PREDICT_QUERY_OR_TABLE'] == 'TABLE'
    with open(intentFileName) as f:
        for line in f:
            tokens = line.strip().split(";")
            assert len(tokens) == 3
            sessQueryID = tokens[0]
            sessID = int(sessQueryID.split(", ")[0].split(" ")[1])
            queryID = int(sessQueryID.split(", ")[1].split(" ")[1]) - 1
            sessQueryID = str(sessID)+","+str(queryID)
            curIntentBitMap = BitMap.fromstring(tokens[2])
            if configDict['RNN_PREDICT_QUERY_OR_TABLE'] == 'QUERY':
                intentObj = CreateSQLFromIntentVec.regenerateSQL(curIntentBitMap, schemaDicts)
            elif configDict['RNN_PREDICT_QUERY_OR_TABLE'] == 'TABLE':
                intentObj = CreateSQLFromIntentVec.regenerateSQLTable(curIntentBitMap, None, schemaDicts, configDict)
            intentObjDict[sessQueryID] = intentObj
    return intentObjDict


def createSQLStatsFromConfigDict(configDict, args):
    if args.intent is not None:
        intentFileName = args.intent
    else:
        intentFileName = QR.fetchIntentFileFromConfigDict(configDict)
    if args.conc is not None:
        concSessFile = args.conc
    else:
        concSessFile = getConfig(configDict['CONCURRENT_QUERY_SESSIONS'])
    if args.output is not None:
        outputSQLStats = args.output
    else:
        outputSQLStats = getConfig(configDict['OUTPUT_DIR']) + "/outputSQLStats"
    #curQueryDict = readFromConcurrentFile(concSessFile)
    schemaDicts = ReverseEnggQueries.readSchemaDicts(configDict)
    intentObjDict = createIntentObjDict(schemaDicts, intentFileName)
    computeStats(configDict, intentObjDict, outputSQLStats)
    return

if __name__ == "__main__":
    #configDict = parseConfig.parseConfigFile("configFile.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="Config parameters file", type=str, required=True)
    parser.add_argument("-intent", help="intent file", type=str, required=False)
    parser.add_argument("-conc", help="concurrent session file", type=str, required=False)
    parser.add_argument("-output", help="output sql stats file", type=str, required=False)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    createSQLStatsFromConfigDict(configDict, args)