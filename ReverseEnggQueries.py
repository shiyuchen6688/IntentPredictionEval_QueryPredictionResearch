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
    with open(fn) as f:
        for line in f:
            tokens = line.strip().split(":")
            tableDict[tokens[0]] = int(tokens[1])
    return tableDict

def readColDict(fn):
    colDict = {}
    with open(fn) as f:
        for line in f:
            tokens = line.strip().split(":")
            key = tokens[0]
            val = tokens[1].replace("[","").replace("]","").replace("'","")
            columns = val.split(",")
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
    tableDict = readTableDict(getConfig(configDict['MINC_TABLES']))
    colDict = readColDict(getConfig(configDict['MINC_COLS']))
    (joinPredDict, joinPredBitPosDict) = readJoinColDicts(getConfig(configDict['MINC_JOIN_PREDS']), getConfig(configDict['MINC_JOIN_PRED_BIT_POS']))

def regenerateQuery(threadID, predictedY, configDict, curSessID, curQueryID, sessionDictCurThread, sampledQueryHistory, sessionStreamDict):
    topKPredictedIntents = []
    readSchemaDicts(configDict)
    return topKPredictedIntents

if __name__ == "__main__":
    #configDict = parseConfig.parseConfigFile("configFile.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="Config parameters file", type=str, required=True)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    readSchemaDicts(configDict)