from __future__ import division
import sys
import os
import time
import QueryExecution as QExec
from bitmap import BitMap
import CFCosineSim
import TupleIntent as ti
import ParseConfigFile as parseConfig
from ParseConfigFile import getConfig
import pickle
import argparse
from pandas import DataFrame
from openpyxl import load_workbook
import pandas as pd
import ConcurrentSessions

def countQueries(inputFile):
    sessionLenHist = {} # key is length and value is number of sessions
    with open(inputFile) as f:
        for line in f:
            sessTokens = line.strip().split(";")
            # sessionIndices need to be noted that they are neither sequential nor complete. For instance session 15 or 16 does not exist.
            sessIndex = int(sessTokens[0].split(" ")[1])
            key = line.count(";")-1 #line ends with a semicolon but has the first token as session name which is ignored
            try:
                sessionLenHist[key] = sessionLenHist[key]+1
            except:
                sessionLenHist[key] = 1
    return sessionLenHist

def plotSessionLengths(configDict):
    sessionLenHist = countQueries(getConfig(configDict['QUERYSESSIONS']))
    sessKeys = []
    sessVals = []
    for key in sorted(sessionLenHist):
        sessKeys.append(key)
        sessVals.append(sessionLenHist[key])
    df = DataFrame(
        {'sessLength': sessKeys, 'count': sessVals})
    outputSessLengthFileName = getConfig(configDict['OUTPUT_DIR']) + "/OpWiseExcel/SessLength"
    df.to_excel(outputSessLengthFileName + ".xlsx", sheet_name='sheet1', index=False)
    return

def plotQueryProgression(configDict,logFile):
    queryProgDict = {} # key is queryID and value is episode number
    with open(logFile) as f:
        for line in f:
            if line.startswith("#Episodes"):
                tokens = line.strip().split(";")
                episodeID = int(tokens[0].split(":")[1])
                queryID = int(tokens[len(tokens)-1].split(":")[1])
                if queryID not in queryProgDict:
                    queryProgDict[queryID] = episodeID
    queryProgKeys = []
    queryProgVals = []
    for key in sorted(queryProgDict):
        queryProgKeys.append(key)
        queryProgVals.append(queryProgDict[key])
    df = DataFrame(
        {'queryID': queryProgKeys, 'episodeID': queryProgVals})
    outputQueryProgFileName = getConfig(configDict['OUTPUT_DIR']) + "/OpWiseExcel/QueryProg"
    df.to_excel(outputQueryProgFileName + ".xlsx", sheet_name='sheet1', index=False)
    return

def plotTemporality(configDict, logFile):
    plotSessionLengths(configDict)
    plotQueryProgression(configDict,logFile)

def incrementVal(dictName, key):
    try:
        dictName[key] = dictName[key]+1
    except:
        dictName[key]=1
    return dictName

def analyzePR(configDict, logFile):
    numPREqualOne = 0
    numPREqualNotOne = 0
    numPRNotEqual = 0
    notEqualBinKeys = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    notEqualBinFreqDict = {}
    avgDiffPR = 0.0
    count =0
    with open(logFile) as f:
        for line in f:
            if line.startswith("#Episodes"):
                tokens = line.strip().split(";")
                precision = float(tokens[1].split(":")[1])
                recall = float(tokens[2].split(":")[1])
                FMeasure = float(tokens[3].split(":")[1])
                avgDiffPR += abs(precision-recall)
                count+=1
                if precision == recall:
                    if precision == 1.0:
                        numPREqualOne+=1
                    else:
                        numPREqualNotOne+=1
                else:
                    numPRNotEqual+=1
                    for key in notEqualBinKeys:
                        if FMeasure < key:
                            notEqualBinFreqDict = incrementVal(notEqualBinFreqDict,key)
                            break
        notEqualBinVals = []
        for key in notEqualBinKeys:
            try:
                notEqualBinVals.append(notEqualBinFreqDict[key])
                print "key: "+str(key)+", val: "+str(notEqualBinFreqDict[key])
            except:
                notEqualBinVals.append(0.0)
    avgDiffPR = float(avgDiffPR)/float(count)
    print "numPREqualOne: "+str(numPREqualOne)+", numPREqualNotOne: "+str(numPREqualNotOne)+", numPRNotEqual: "+str(numPRNotEqual)+", avgDiffPR: "+str(avgDiffPR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="config file to parse", type=str, required=True)
    parser.add_argument("-log", help="log filename to analyze", type=str, required=True)
    #parser.add_argument("-lineNum", help="line Number to analyze", type=int, required=True)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    #plotTemporality(configDict, args.log)
    analyzePR(configDict, args.log)