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
import math

class SelPredObj:
    def __init__(self, configDict):
        self.configDict = configDict
        self.schemaDicts = ReverseEnggQueries.readSchemaDicts(configDict)
        self.colTypeDict = ReverseEnggQueries.readColDict(getConfig(configDict['MINC_COL_TYPES']))
        self.cnx = MINC_prepareJoinKeyPairs.connectToDB(configDict)
        self.totalTables = QR.readFromPickleFile(getConfig(configDict['OUTPUT_DIR']) + "../MincTotalTables.pickle")
        self.selTables = QR.readFromPickleFile(getConfig(configDict['OUTPUT_DIR']) + "../MincSelTables.pickle")
        self.selCols = QR.readFromPickleFile(getConfig(configDict['OUTPUT_DIR']) + "../MincSelCols.pickle")
        self.joinTables = QR.readFromPickleFile(getConfig(configDict['OUTPUT_DIR']) + "../MincJoinTables.pickle")
        self.joinCols = QR.readFromPickleFile(getConfig(configDict['OUTPUT_DIR']) + "../MincJoinCols.pickle")
        self.selPredCols = list(self.selCols.union(self.joinCols)) # we union selCols with joinCols to build selPredCols -- to account for misses
        self.selPredOpBitPosDict = {} # key is selPredCol and value is startBitPos,endBitPos reqd for eq,neq,leq,geq,lt,gt in the same order
        self.selPredColRangeBinDict = {} # key is selPredCol and value is the list of range bins [(minBinVal,maxBinval),..,]
        self.selPredColRangeBitPosDict = {} # key is selPredCol and value is startBitPos,endBitPos reqd for that selPredCol
        self.commonDataTypes = ['int(11) unsigned', 'varchar(100)', 'int(11)', 'int(10)', 'int(16)', 'tinyint(3) unsigned', 'varchar(61)', 'text', 'int(1)', 'tinyint(1) unsigned', 'varchar(200)', 'varchar(255)', 'tinyint(3)', 'varchar(64)', 'tinyint(4)', 'int(255)', 'varchar(14)', 'int(10) unsigned', 'tinyint(1)']

def projectDistinctVals(selPredObj, tableName, colName, colType):
    distinctVals = []
    query = "SELECT DISTINCT " + tableName+"."+colName + " FROM " + tableName
    cursor = selPredObj.cnx.cursor()
    cursor.execute(query)
    for row in cursor:
        assert len(row) == 1  # single column projected
        try:
            if 'text' in colType or 'varchar' in colType:
                rowVal = str(row[0])
            elif 'int' in colType:
                rowVal = int(row[0])
            else:
                print 'Unknown datatype !' # datetime stored as string
                sys.exit(0)
        except:
            rowVal = unicodedata.normalize('NFKD', row[0]).encode('ascii', 'ignore') # for unicode and non-ascii
        distinctVals.append(rowVal)
    return distinctVals

def createSortedRangesPerCol(distinctVals, numBins):
    distinctVals.sort()
    rangeBinsCol = []
    binSize = 1
    if len(distinctVals) == 0:
        print "Length 0 !!"
        return None
    elif len(distinctVals) < numBins:
        binSize = 1
    else:
        binSize = int(math.ceil(len(distinctVals) / float(10)))
    startIndex = 0
    while startIndex < len(distinctVals):
        #diff = len(distinctVals) - 1 - (startIndex +binSize - 1)
        #if diff < binSize:
        #    endIndex = len(distinctVals) - 1 # merge the last bin into the before last bin
        #else:
        endIndex = min(startIndex + binSize-1, len(distinctVals)-1)
        pair = str(distinctVals[startIndex]).replace("\'","")+"%"+str(distinctVals[endIndex]).replace("\'","")
        rangeBinsCol.append(pair)
        startIndex = endIndex + 1
    return rangeBinsCol

def findColType(tableName, colName, selPredObj):
    offset = selPredObj.schemaDicts.colDict[tableName].index(colName)
    colType = selPredObj.colTypeDict[tableName][offset]
    return colType

def createSelPredColRangeBins(selPredObj):
    numBins = int(selPredObj.configDict['MINC_NUM_BINS_PER_SEL_COL'])
    for selPredCol in selPredObj.selPredCols:
        tableName = selPredCol.split(".")[0]
        colName = selPredCol.split(".")[1]
        colType = findColType(tableName, colName, selPredObj)
        print "Creating Sorted Range Bins for column "+selPredCol
        distinctVals = projectDistinctVals(selPredObj, tableName, colName, colType)
        rangeBinsCol = createSortedRangesPerCol(distinctVals, numBins)
        if rangeBinsCol is None:
            rangeBinsCol = []
        rangeBinsCol.append('NULL%NULL') # extra bin for comparison with null -- IS (NOT) NULL
        selPredObj.selPredColRangeBinDict[selPredCol] = rangeBinsCol
        if len(rangeBinsCol) > numBins+1:
            print "numBins > Limit of "+str(numBins+1)
    return

def createSelPredOpBitPosDict(selPredObj):
    endPos = -1
    comparisonOps = ['eq', 'neq', 'leq', 'geq', 'lt', 'gt', 'LIKE']
    for selPredCol in selPredObj.selPredCols:
        startPos = endPos + 1
        endPos = startPos + len(comparisonOps) - 1 # for the 7 comparison ops -- eq,neq,leq,geq,lt,gt,like
        selPredObj.selPredOpBitPosDict[selPredCol] = str(startPos) + "," + str(endPos)
    return

def createSelPredColRangeBitPosDict(selPredObj):
    endPos = -1
    for selPredCol in selPredObj.selPredCols:
        startPos = endPos+1
        endPos = startPos+len(selPredObj.selPredColRangeBinDict[selPredCol])-1
        selPredObj.selPredColRangeBitPosDict[selPredCol] = str(startPos)+","+str(endPos)
    return

def writeSelPredColDict(selPredCols, fn):
    try:
        os.remove(fn)
    except OSError:
        pass
    # key,value pair from each dictionary is written as key:value in each separate row in the file
    selPredColIndex = 0
    with open(fn, 'a') as f:
        for key in selPredCols:
            f.write(str(key)+":"+str(selPredColIndex)+"\n")
            selPredColIndex += 1
        f.flush()
        f.close()
    print "Wrote to file "+fn

def writeSchemaInfoToFiles(selPredObj):
    writeSelPredColDict(selPredObj.selPredCols, getConfig(selPredObj.configDict['MINC_SEL_PRED_COLS']))
    MINC_prepareJoinKeyPairs.writeSchemaInfoToFile(selPredObj.selPredOpBitPosDict, getConfig(selPredObj.configDict['MINC_SEL_PRED_OP_BIT_POS']))
    MINC_prepareJoinKeyPairs.writeSchemaInfoToFile(selPredObj.selPredColRangeBinDict, getConfig(selPredObj.configDict['MINC_SEL_PRED_COL_RANGE_BINS']))
    MINC_prepareJoinKeyPairs.writeSchemaInfoToFile(selPredObj.selPredColRangeBitPosDict, getConfig(selPredObj.configDict['MINC_SEL_PRED_COL_RANGE_BIT_POS']))
    return

def buildSelPredDicts(configDict):
    selPredObj = SelPredObj(configDict)
    createSelPredOpBitPosDict(selPredObj)
    createSelPredColRangeBins(selPredObj)
    createSelPredColRangeBitPosDict(selPredObj)
    print "Writing Dictionaries To Files"
    writeSchemaInfoToFiles(selPredObj)
    return

if __name__ == "__main__":
    #configDict = parseConfig.parseConfigFile("MINC_configFile.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="Config parameters file", type=str, required=True)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    buildSelPredDicts(configDict)