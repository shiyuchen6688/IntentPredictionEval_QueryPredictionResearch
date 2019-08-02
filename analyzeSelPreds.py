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
                    leftjoinTab = leftJoinCol.split(".")[0]
                    rightJoinTab = rightJoinCol.split(".")[0]
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
    writeSetToFile(totalTables, getConfig(configDict['OUTPUT_DIR'])+"../MincTotalTables")
    writeSetToFile(selTables, getConfig(configDict['OUTPUT_DIR'])+"../MincSelTables")
    writeSetToFile(selCols, getConfig(configDict['OUTPUT_DIR'])+"../MincSelCols")
    writeSetToFile(joinTables, getConfig(configDict['OUTPUT_DIR']) + "../MincJoinTables")
    writeSetToFile(joinCols, getConfig(configDict['OUTPUT_DIR']) + "../MincJoinCols")
    QR.writeToPickleFile(getConfig(configDict['OUTPUT_DIR'])+"../MincTotalTables.pickle", totalTables)
    QR.writeToPickleFile(getConfig(configDict['OUTPUT_DIR']) + "../MincSelTables.pickle", selTables)
    QR.writeToPickleFile(getConfig(configDict['OUTPUT_DIR']) + "../MincSelCols.pickle", selCols)
    QR.writeToPickleFile(getConfig(configDict['OUTPUT_DIR']) + "../MincJoinTables.pickle", joinTables)
    QR.writeToPickleFile(getConfig(configDict['OUTPUT_DIR']) + "../MincJoinCols.pickle", joinCols)
    return

def evalSelCols(configDict, schemaDicts):
    colTypeDict = ReverseEnggQueries.readColDict(getConfig(configDict['MINC_COL_TYPES']))
    totalTables = QR.readFromPickleFile(getConfig(configDict['OUTPUT_DIR'])+"../MincTotalTables.pickle")
    selTables = QR.readFromPickleFile(getConfig(configDict['OUTPUT_DIR'])+"../MincSelTables.pickle")
    selCols = QR.readFromPickleFile(getConfig(configDict['OUTPUT_DIR'])+"../MincSelCols.pickle")
    selCols.add('jos_community_photos.permissions')
    selCols.add('jos_community_photos.owner')
    selCols.add('jos_community_courses.approvals')
    QR.writeToPickleFile(getConfig(configDict['OUTPUT_DIR'])+"../MincSelCols.pickle", selCols)
    joinTables = QR.readFromPickleFile(getConfig(configDict['OUTPUT_DIR'])+"../MincJoinTables.pickle")
    joinCols = QR.readFromPickleFile(getConfig(configDict['OUTPUT_DIR'])+"../MincJoinCols.pickle")
    print "joinTables - selTables: " + str(joinTables - selTables)
    print "joinCols - selCols: "+str(joinCols - selCols)
    print "totalTables - selTables: "+str(totalTables - selTables)
    selJoinCols = selCols.union(joinCols)
    selJoinColTypeDict = {}
    for selJoinCol in selJoinCols:
        selJoinTab = selJoinCol.split(".")[0]
        selJoinColName = selJoinCol.split(".")[1]
        offset = schemaDicts.colDict[selJoinTab].index(selJoinColName)
        selJoinColType = colTypeDict[selJoinTab][offset]
        selJoinColTypeDict[selJoinCol] = selJoinColType
        print selJoinCol + ": "+selJoinColType
    print set(selJoinColTypeDict.values())
    print "len(totalTables): " + str(len(totalTables)) + ", len(selTables): " + str(
        len(selTables)) + ", len(selCols): " + str(len(selCols)) + ", len(joinTables): " + str(
        len(joinTables)) + ", len(joinCols): " + str(len(joinCols)) + ", len(selJoinCols): "+str(len(selJoinColTypeDict))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="config file to parse", type=str, required=True)
    #parser.add_argument("-log", help="log filename to analyze", type=str, required=True)
    #parser.add_argument("-lineNum", help="line Number to analyze", type=int, required=True)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    schemaDicts = ReverseEnggQueries.readSchemaDicts(configDict)
    #countConstTabPreds(configDict, schemaDicts)
    evalSelCols(configDict, schemaDicts)