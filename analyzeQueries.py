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
import QueryRecommender as QR
import CreateSQLFromIntentVec
import ReverseEnggQueries

def writeSetToFile(tabColSet, fn):
    try:
        os.remove(fn)
    except OSError:
        pass
    with open(fn, 'a') as f:
        for elem in tabColSet:
            f.write(elem+"\n")
        f.flush()
        f.close()
    print "Wrote to file " + fn

def countConstTabPreds(configDict, schemaDicts):
    intentSessionFile = QR.fetchIntentFileFromConfigDict(configDict)
    totalTables = set()
    selTables = set()
    selCols = set()
    joinTables = set()
    joinCols = set()
    count = 0
    with open(intentSessionFile) as f:
        for line in f:
            (sessID, queryID, curQueryIntent) = QR.retrieveSessIDQueryIDIntent(line, configDict)
            intentObj = CreateSQLFromIntentVec.regenerateSQL(curQueryIntent, schemaDicts)
            totalTables.update(intentObj.tables)
            if intentObj.joinPreds is not None and len(intentObj.joinPreds) > 0:
                for joinPred in intentObj.joinPreds:
                    leftJoinCol = joinPred.split(",")[0]
                    rightJoinCol = joinPred.split(",")[1]
                    leftjoinTab = leftJoinCol.split(".")[1]
                    rightJoinTab = rightJoinCol.split(".")[1]
                    joinCols.add(leftJoinCol)
                    joinCols.add(rightJoinCol)
                    joinTables.add(leftjoinTab)
                    joinTables.add(rightJoinTab)
            if intentObj.selCols is not None and len(intentObj.selCols) > 0:
                selCols.update(intentObj.selCols)
                for selCol in intentObj.selCols:
                    selTable = selCol.split(".")[0]
                    selTables.add(selTable)
            count+=1
            if count % 1000 == 0:
                print "len(totalTables): " + str(len(totalTables)) + ", len(selTables): " + str(
                    len(selTables)) + ", len(selCols): " + str(len(selCols)) + ", len(joinTables): " + str(
                    len(joinTables)) + ", len(joinCols): " + str(len(joinCols))
    print "len(totalTables): " + str(len(totalTables)) + ", len(selTables): " + str(
        len(selTables)) + ", len(selCols): " + str(len(selCols)) + ", len(joinTables): " + str(
        len(joinTables)) + ", len(joinCols): " + str(len(joinCols))
    writeSetToFile(totalTables, getConfig(configDict['OUTPUT_DIR'])+"TotalTables")
    writeSetToFile(selTables, getConfig(configDict['OUTPUT_DIR'])+"SelTables")
    writeSetToFile(selCols, getConfig(configDict['OUTPUT_DIR'])+"SelCols")
    writeSetToFile(joinTables, getConfig(configDict['OUTPUT_DIR']) + "JoinTables")
    writeSetToFile(joinCols, getConfig(configDict['OUTPUT_DIR']) + "JoinCols")
    QR.writeToPickleFile(getConfig(configDict['OUTPUT_DIR'])+"TotalTables.pickle", totalTables)
    QR.writeToPickleFile(getConfig(configDict['OUTPUT_DIR']) + "SelTables.pickle", selTables)
    QR.writeToPickleFile(getConfig(configDict['OUTPUT_DIR']) + "SelCols.pickle", selCols)
    QR.writeToPickleFile(getConfig(configDict['OUTPUT_DIR']) + "JoinTables.pickle", joinTables)
    QR.writeToPickleFile(getConfig(configDict['OUTPUT_DIR']) + "JoinCols.pickle", joinCols)
    return

def evalSelCols(configDict, schemaDicts):
    totalTables = QR.readFromPickleFile(getConfig(configDict['OUTPUT_DIR'])+"TotalTables.pickle")
    selTables = QR.readFromPickleFile(getConfig(configDict['OUTPUT_DIR'])+"SelTables.pickle")
    selCols = QR.readFromPickleFile(getConfig(configDict['OUTPUT_DIR'])+"SelCols.pickle")
    joinTables = QR.readFromPickleFile(getConfig(configDict['OUTPUT_DIR'])+"JoinTables.pickle")
    joinCols = QR.readFromPickleFile(getConfig(configDict['OUTPUT_DIR'])+"JoinCols.pickle")
    print "joinTables - selTables: " + str(joinTables - selTables)
    print "joinCols - selCols: "+str(joinCols - selCols)
    print "totalTables - selTables: "+str(totalTables - selTables)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="config file to parse", type=str, required=True)
    #parser.add_argument("-log", help="log filename to analyze", type=str, required=True)
    #parser.add_argument("-lineNum", help="line Number to analyze", type=int, required=True)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    schemaDicts = ReverseEnggQueries.readSchemaDicts(configDict)
    countConstTabPreds(configDict, schemaDicts)
    evalSelCols(configDict, schemaDicts)