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
import random

def readSessQueryIDs(logFile):
    sessQueryIDs = []
    with open(logFile) as f:
        for line in f:
            if line.startswith("#Episodes"):
                tokens = line.strip().split(";")
                sessQueryIDs.append(tokens[len(tokens)-2]+";"+tokens[len(tokens)-1])
    return sessQueryIDs

def checkPresenceAndOutput(logFile, sessQueryID):
    flag = False
    with open(logFile) as f:
        for line in f:
            if line.startswith("----") and flag == True:
                flag = False
                return
            if line.startswith("#Episodes") and sessQueryID in line:
                flag = True
            if flag == True:
                print line
    return

def origDemo(configDict, qlLogFile, rnnNovelLogFile, svdLogFile, cfLogFile):
    sessQueryIDs = readSessQueryIDs(qlLogFile)
    # parse issues: "Session:32442;Query:2", "Session:28383;Query:28"
    termFlag = input('Enter True to continue and False to terminate: ')
    while (termFlag == True):
        sessQueryID = random.choice(sessQueryIDs)
        print "Selected " + sessQueryID
        print "-----------PREDICTION BY Q-LEARNING------------"
        checkPresenceAndOutput(qlLogFile, sessQueryID)
        print "-----------PREDICTION BY RNN(Novel)------------"
        checkPresenceAndOutput(rnnNovelLogFile, sessQueryID)
        print "-----------PREDICTION BY SVD------------"
        checkPresenceAndOutput(svdLogFile, sessQueryID)
        print "-----------PREDICTION BY CFCOSINESIM------------"
        checkPresenceAndOutput(cfLogFile, sessQueryID)
        termFlag = input('Enter True to continue and False to terminate: ')
    return

def ourDemo(qlLogFile, rnnNovelLogFile, svdLogFile, cfLogFile):
    sessQueryIDs = ["Session:5780;Query:1", "Session:26137;Query:0", "Session:12635;Query:5",
     "Session:12495;Query:4", "Session:4542;Query:0", "Session:1140;Query:0", "Session:33536;Query:4", "Session:31559;Query:11",
     "Session:6032;Query:1", "Session:16399;Query:6", "Session:14062;Query:0", "Session:25895;Query:0",
     "Session:19808;Query:0", "Session:11882;Query:0", "Session:26835;Query:3", "Session:34616;Query:4", "Session:32281;Query:1",
     "Session:27832;Query:1", "Session:18512;Query:3"]
    termFlag = input('Enter True to continue and False to terminate: ')
    while(termFlag == True):
        sessQueryID = random.choice(sessQueryIDs)
        sessQueryIDs.remove(sessQueryID)
        print "Selected "+sessQueryID
        print "-----------PREDICTION BY Q-LEARNING------------"
        checkPresenceAndOutput(qlLogFile, sessQueryID)
        print "-----------PREDICTION BY RNN(Novel)------------"
        checkPresenceAndOutput(rnnNovelLogFile, sessQueryID)
        print "-----------PREDICTION BY SVD------------"
        checkPresenceAndOutput(svdLogFile, sessQueryID)
        print "-----------PREDICTION BY CFCOSINESIM------------"
        checkPresenceAndOutput(cfLogFile, sessQueryID)
        termFlag = input('Enter True to continue and False to terminate: ')
    return

def showDemo(configDict):
    inputDir = "/Users/postgres/Documents/DataExploration-Research/MINC/InputOutput/ClusterRuns/NovelTables-113K-Pruned/sustenance_0.8"
    qlLogFile = inputDir+"/QL/outputSQL_QL_Log"
    rnnNovelLogFile = inputDir+"/NovelRNN/outputSQL_Novel_RNN_Log"
    svdLogFile = inputDir+"/SVD/outputSQL_SVD_Log"
    cfLogFile = inputDir+"/CFCosineSim/outputSQL_CFCosineSim_Log"
    #origDemo(configDict, qlLogFile, rnnNovelLogFile, svdLogFile, cfLogFile)
    ourDemo(qlLogFile, rnnNovelLogFile, svdLogFile, cfLogFile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="config file to parse", type=str, required=True)
    #parser.add_argument("-lineNum", help="line Number to analyze", type=int, required=True)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    #plotTemporality(configDict, args.log)
    showDemo(configDict)
