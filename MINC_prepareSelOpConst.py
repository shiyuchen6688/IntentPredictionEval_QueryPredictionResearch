import sys
import os
import time, argparse
from bitmap import BitMap
import ParseConfigFile as parseConfig
import QueryParser as qp
import TupleIntent as ti
import re
import mysql.connector
from mysql.connector import errorcode
from ParseConfigFile import getConfig
import QueryRecommender as QR
import CreateSQLFromIntentVec
import ReverseEnggQueries
import MINC_prepareJoinKeyPairs
import unicodedata

class SelPredObj:
    def __init__(self, configDict):
        self.configDict = configDict
        self.schemaDicts = ReverseEnggQueries.readSchemaDicts(configDict)
        self.colTypeDict = ReverseEnggQueries.readColDict(getConfig(configDict['MINC_COL_TYPES']))
        self.cnx = MINC_prepareJoinKeyPairs.connectToMySQL(configDict)
        self.totalTables = QR.readFromPickleFile(getConfig(configDict['OUTPUT_DIR']) + "../MincTotalTables.pickle")
        self.selTables = QR.readFromPickleFile(getConfig(configDict['OUTPUT_DIR']) + "../MincSelTables.pickle")
        self.selCols = QR.readFromPickleFile(getConfig(configDict['OUTPUT_DIR']) + "../MincSelCols.pickle")
        self.joinTables = QR.readFromPickleFile(getConfig(configDict['OUTPUT_DIR']) + "../MincJoinTables.pickle")
        self.joinCols = QR.readFromPickleFile(getConfig(configDict['OUTPUT_DIR']) + "../MincJoinCols.pickle")
        self.selPredCols = list(self.selCols.union(self.joinCols)) # we union selCols with joinCols to build selPredCols -- to account for misses
        self.selPredOpBitPosDict = {} # key is selPredCol and value is startBitPos,endBitPos reqd for eq,neq,leq,geq,lt,gt in the same order
        self.selPredColRangeBinDict = {} # key is selPredCol and value is the list of range bins [(minBinVal,maxBinval),..,]
        self.selPredColBitPosDict = {} # key is selPredCol and value is startBitPos,endBitPos reqd for that selPredCol

def projectDistinctVals(selPredObj, tableName, colName):
    distinctVals = []
    query = "SELECT DISTINCT " + tableName+"."+colName + " FROM " + tableName
    cursor = selPredObj.cnx.cursor()
    cursor.execute(query)
    for row in cursor:
        assert len(row) == 1  # single column projected
        try:
            rowVal = str(row[0])
        except:
            rowVal = unicodedata.normalize('NFKD', row[0]).encode('ascii', 'ignore') # for unicode and non-ascii
        distinctVals.append(rowVal)
    return distinctVals

def createSortedRangesPerCol(distinctVals):
    # hardcoding for 10 bins -- can add a configParam later
    numBins = 10
    distinctVals.sort()
    rangeBinsCol = []
    binSize = 1
    if len(distinctVals) == 0:
        print "Length 0 !!"
        return None
    elif len(distinctVals) < numBins:
        binSize = 1
    else:
        binSize = len(distinctVals) / 10
    startIndex = 0
    while startIndex < len(distinctVals):
        endIndex = min(startIndex + binSize, len(distinctVals)-1)
        pair = (distinctVals[startIndex], distinctVals[endIndex])
        rangeBinsCol.append(pair)
        startIndex = endIndex + 1
    return rangeBinsCol

def createSelPredColRangeBins(selPredObj):
    for selPredCol in selPredObj.selPredCols:
        tableName = selPredCol.split(".")[0]
        colName = selPredCol.split(".")[1]
        print "Creating Sorted Range Bins for column "+selPredCol
        distinctVals = projectDistinctVals(selPredObj, tableName, colName)
        rangeBinsCol = createSortedRangesPerCol(distinctVals)
        if rangeBinsCol is not None:
            selPredObj.selPredColRangeBinDict[selPredCol] = rangeBinsCol
    return

def createSelPredOpBitPosDict(selPredObj):
    endPos = -1
    comparisonOps = ['eq', 'neq', 'leq', 'geq', 'lt', 'gt']
    for selPredCol in selPredObj.selPredCols:
        startPos = endPos + 1
        endPos = startPos + len(comparisonOps) - 1 # for the 6 comparison ops -- eq,neq,leq,geq,lt,gt
        selPredObj.selPredOpBitPosDict[selPredCol] = str(startPos) + "," + str(endPos)
    return

def createSelPredColBitPosDict(selPredObj):
    endPos = -1
    for selPredCol in selPredObj.selPredCols:
        startPos = endPos+1
        try:
            endPos = startPos+len(selPredObj.selPredColRangeBinDict[selPredCol])
        except:
            continue
        selPredObj.selPredColBitPosDict[selPredCol] = str(startPos)+","+str(endPos)
    return

def writeSchemaInfoToFiles(selPredObj):
    MINC_prepareJoinKeyPairs.writeSchemaInfoToFile(selPredObj.selPredOpBitPosDict, getConfig(configDict['MINC_SEL_PRED_OP_BIT_POS']))
    MINC_prepareJoinKeyPairs.writeSchemaInfoToFile(selPredObj.selPredColRangeBinDict, getConfig(configDict['MINC_SEL_PRED_COL_RANGE_BINS']))
    MINC_prepareJoinKeyPairs.writeSchemaInfoToFile(selPredObj.selPredColBitPosDict, getConfig(configDict['MINC_SEL_PRED_COL_BIT_POS']))
    return

def buildSelPredDicts(configDict):
    selPredObj = SelPredObj(configDict)
    createSelPredOpBitPosDict(selPredObj)
    createSelPredColRangeBins(selPredObj)
    createSelPredColBitPosDict(selPredObj)
    print "Writing Dictionaries To Files"
    writeSchemaInfoToFiles(selPredObj)
    return

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("MINC_configFile.txt")
    buildSelPredDicts(configDict)