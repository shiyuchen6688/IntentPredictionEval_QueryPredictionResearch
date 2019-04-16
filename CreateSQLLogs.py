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

def readFromConcurrentFile(concSessFile):
    # Note that query IDs start in the file from 1 but in the outputIntent, query ID starts from 0: so Decrement by 1
    curQueryDict = {}
    try:
        with open(concSessFile) as f:
            for line in f:
                tokens = line.strip().split(";")
                sessQueryID = tokens[0]
                sessID = int(sessQueryID.split(", ")[0].split(" ")[1])
                queryID = int(sessQueryID.split(", ")[1].split(" ")[1]) - 1
                curQuery = tokens[1]
                sessQueryID = "Session:"+str(sessID)+";"+"Query:"+str(queryID)
                assert sessQueryID not in curQueryDict
                curQueryDict[sessQueryID] = curQuery
    except:
        print "cannot read line !!"
        sys.exit(0)
    return curQueryDict

def readFromOutputEvalFile(outputEvalQualityFileName):
    outputEvalDict = {}
    with open(outputEvalQualityFileName) as f:
        for line in f:
            tokens = line.strip().split(";")
            outputEvalDict[tokens[0]+";"+tokens[1]] = ";".join(tokens[2:])
    return outputEvalDict

def procPredictedIntents(configDict, schemaDicts, curQueryDict, outputEvalDict, outputIntentFileName, outputSQLLog):
    QR.deleteIfExists(outputSQLLog)
    assert configDict['RNN_PREDICT_QUERY_OR_TABLE'] == 'QUERY' or configDict['RNN_PREDICT_QUERY_OR_TABLE'] == 'TABLE'
    with open(outputIntentFileName) as f:
        for line in f:
            tokens = line.strip().split(";")
            #assert len(tokens) == 4 + int(configDict['TOP_K'])
            sessQueryID = tokens[0]+";"+tokens[1]
            outputSQLStr = "-----------------------------------------\n"
            outputSQLStr += outputEvalDict[sessQueryID]+"\n" # prints the metrics first
            outputSQLStr += "Current Query: "+curQueryDict[sessQueryID]+"\n"
            nextQueryID = "Query:"+str(int(tokens[1].split(":")[1]) + 1)
            outputSQLStr += "Next Query: "+curQueryDict[tokens[0]+";"+nextQueryID]+"\n"
            actualIntent = BitMap.fromstring(tokens[3].split(":")[1])
            if configDict['RNN_PREDICT_QUERY_OR_TABLE'] == 'QUERY':
                actualIntentObj = CreateSQLFromIntentVec.regenerateSQL(actualIntent, schemaDicts)
                outputSQLStr += "Actual SQL Ops:\n" + CreateSQLFromIntentVec.createSQLString(actualIntentObj)
            elif configDict['RNN_PREDICT_QUERY_OR_TABLE'] == 'TABLE':
                actualIntentObj = CreateSQLFromIntentVec.regenerateSQLTable(actualIntent, None, schemaDicts, configDict)
                outputSQLStr += "Actual SQL Ops:\n" + CreateSQLFromIntentVec.createSQLStringForTable(actualIntentObj)
            for i in range(4, len(tokens)):
                predictedIntent = BitMap.fromstring(tokens[i].split(":")[1])
                relIndex = i - 4
                if configDict['RNN_PREDICT_QUERY_OR_TABLE'] == 'QUERY':
                    predictedIntentObj = CreateSQLFromIntentVec.regenerateSQL(predictedIntent, schemaDicts)
                    outputSQLStr += "Predicted SQL Ops " + str(
                        relIndex) + ":\n" + CreateSQLFromIntentVec.createSQLString(predictedIntentObj)
                elif configDict['RNN_PREDICT_QUERY_OR_TABLE'] == 'TABLE':
                    predictedIntentObj = CreateSQLFromIntentVec.regenerateSQLTable(predictedIntent, None, schemaDicts, configDict)
                    outputSQLStr += "Predicted SQL Ops " + str(
                        relIndex) + ":\n" + CreateSQLFromIntentVec.createSQLStringForTable(predictedIntentObj)
            ti.appendToFile(outputSQLLog, outputSQLStr)
    return

def createSQLLogsFromConfigDict(configDict, args):
    accThres = float(configDict['ACCURACY_THRESHOLD'])
    if args.intent is not None:
        outputIntentFileName = args.intent
    elif configDict['ALGORITHM'] == 'RNN':
        outputIntentFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputFileShortTermIntent_" + \
                               configDict['ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"] + "_" + \
                               configDict['INTENT_REP'] + "_" + \
                               configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                               configDict['EPISODE_IN_QUERIES']
    elif configDict['ALGORITHM'] == 'CF':
        outputIntentFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputFileShortTermIntent_" + configDict[
        'ALGORITHM'] + "_" + \
                           configDict['CF_COSINESIM_MF'] + "_" + \
                           configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                               'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']
    elif configDict['ALGORITHM'] == 'SVD':
        outputIntentFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputFileShortTermIntent_" + configDict[
            'ALGORITHM'] + "_" + configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + \
        configDict[
            'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']
    if args.eval is not None:
        outputEvalQualityFileName = args.eval
    elif configDict['ALGORITHM'] == 'RNN':
        outputEvalQualityFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputEvalQualityShortTermIntent_" + configDict[
            'ALGORITHM'] + "_" + configDict['RNN_BACKPROP_LSTM_GRU'] + "_" + configDict['INTENT_REP'] + "_" + configDict[
                                        'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                                    configDict['EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres)
    elif configDict['ALGORITHM'] == 'CF':
        outputEvalQualityFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputEvalQualityShortTermIntent_" + configDict[
        'ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF'] + "_" + configDict['INTENT_REP'] + "_" + configDict[
                                    'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                                configDict['EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres)
    elif configDict['ALGORITHM'] == 'SVD':
        outputEvalQualityFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputEvalQualityShortTermIntent_" + \
                                    configDict[
                                        'ALGORITHM'] + "_" + configDict['INTENT_REP'] + "_" + configDict[
                                        'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                                    configDict['EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres)
    if args.conc is not None:
        concSessFile = args.conc
    else:
        concSessFile = getConfig(configDict['CONCURRENT_QUERY_SESSIONS'])
    if args.output is not None:
        outputSQLLog = args.output
    else:
        outputSQLLog = getConfig(configDict['OUTPUT_DIR']) + "/outputSQLLog"
    curQueryDict = readFromConcurrentFile(concSessFile)
    outputEvalDict = readFromOutputEvalFile(outputEvalQualityFileName)
    schemaDicts = ReverseEnggQueries.readSchemaDicts(configDict)
    procPredictedIntents(configDict, schemaDicts, curQueryDict, outputEvalDict, outputIntentFileName, outputSQLLog)
    return

if __name__ == "__main__":
    #configDict = parseConfig.parseConfigFile("configFile.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="Config parameters file", type=str, required=True)
    parser.add_argument("-intent", help="intent output file", type=str, required=False)
    parser.add_argument("-eval", help="eval quality file", type=str, required=False)
    parser.add_argument("-conc", help="concurrent session file", type=str, required=False)
    parser.add_argument("-output", help="output sql log file", type=str, required=False)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    createSQLLogsFromConfigDict(configDict, args)