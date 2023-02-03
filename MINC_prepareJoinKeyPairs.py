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
import operator
import collections
import psycopg2
import psycopg2.extras

def connectToBusTracker(configDict):
    try:
        uname = configDict['PSQL_UNAME']
        dbname = configDict['PSQL_DB']
        conn = psycopg2.connect("dbname='vmeduri' user='vmeduri' password=''")
    except:
        print("I am unable to connect to the database.")
    return conn

def connectToMinc(configDict):
    try:
        uname=configDict['MYSQL_UNAME']
        passwd=configDict['MYSQL_PASSWORD']
        hostname=configDict['MYSQL_HOST']
        dbname=configDict['MYSQL_DB']
        cnx = mysql.connector.connect(user=uname, password=passwd, host=hostname, database=dbname, auth_plugin='mysql_native_password')
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    return cnx

def connectToDB(configDict):
    try:
        if configDict['DATASET'] == 'MINC':
            return connectToMinc(configDict)
        elif configDict['DATASET'] == 'BusTracker':
            return connectToBusTracker(configDict)
    except:
            print("Error in DBName !")
            sys.exit(0)


def createBusTrackerTableDict(cnx, configDict):
    cursor = cnx.cursor()
    tableDict = {}
    psqlSchemaStr = configDict['PSQL_SCHEMA']
    schemaList = psqlSchemaStr.split(",")
    index = 0
    for psqlSchema in schemaList:
        cursor.execute("SET search_path TO " + psqlSchema)
        query = "select table_name from information_schema.tables where table_schema =\'" + psqlSchema + "\'"
        cursor = cnx.cursor()
        cursor.execute(query)
        for cols in cursor:
            tableName = psqlSchema + "_" + str(cols[0])
            assert tableName not in tableDict
            tableDict[tableName] = index
            print("tablename: " + str(tableName) + ", index: " + str(index))
            index += 1
    sorted_x = sorted(tableDict.items(), key=operator.itemgetter(1))
    tableDict = collections.OrderedDict(sorted_x)
    return tableDict


def createMincTableDict(cnx):
    tableDict = {}
    #cursor = execShowTableQuery(cnx, configDict)
    query = "SHOW TABLES"
    cursor = cnx.cursor()
    cursor.execute(query)
    index = 0
    for cols in cursor:
        tableName = str(cols[0])
        assert tableName not in tableDict
        tableDict[tableName] = index
        print("tablename: "+str(tableName)+", index: "+str(index))
        index+=1
    sorted_x = sorted(tableDict.items(), key=operator.itemgetter(1))
    tableDict = collections.OrderedDict(sorted_x)
    return tableDict

def execDescTableQuery(cnx, table, configDict):
    cursor = cnx.cursor()
    if configDict['DATASET'] == 'MINC':
        query = "desc "+table
    elif configDict['DATASET'] == 'BusTracker':
        psqlSchema = table.split("_")[0]
        subList = table.split("_")[1:]
        table = "_".join(subList)
        cursor.execute("SET search_path TO " + psqlSchema)
        query = "select column_name, udt_name from information_schema.columns where table_schema =\'"+psqlSchema +\
                "\' and table_name =\'" +table+"\'"
    else:
        print("Error in execShowTableQuery !")
        sys.exit(0)
    cursor = cnx.cursor()
    cursor.execute(query)
    return cursor

def createTabColDict(cnx, tableDict):
    tabColDict = {}
    tabColTypeDict = {}
    # key is tableID and value is list of name,type
    for table in tableDict:
        cursor = execDescTableQuery(cnx, table, configDict)
        #query = "desc "+table
        #cursor = cnx.cursor()
        #cursor.execute(query)
        #desc query o/p has 5 fields: field, type, null, key, default, extra
        colList = []
        colTypeList = []
        for cols in cursor:
            colName = str(cols[0])
            colType = str(cols[1])
            colList.append(colName)
            colTypeList.append(colType)
        tabColDict[table] = colList
        tabColTypeDict[table] = colTypeList
    return (tabColDict, tabColTypeDict)

def createSelfJoinPairs(curTabID, tabColDict, joinPairDict):
    #curTabID = tabColDict.keys()[tabIndex]
    # key is curTabID,curTabID and value is a list of columns paired to themselves
    joinPairDict[str(curTabID)+","+str(curTabID)] = []
    for colName in tabColDict[curTabID]:
        joinPairDict[str(curTabID) + "," + str(curTabID)].append(colName+","+colName)
    return joinPairDict

def checkIfSameSchema(curTabName, nextTabName):
    curSchema = curTabName.split("_")[0]
    nextSchema = nextTabName.split("_")[0]
    if curSchema == nextSchema:
        return True
    return False

def createCrossJoinPairs(dataset, curTabName, tabColDict, tabColTypeDict, tableDict, joinPairDict):
    #curTabName = tabColDict.keys()[tabIndex]
    tabIndex = tableDict[curTabName]
    curTabCols = tabColDict[curTabName]
    curTabColTypes = tabColTypeDict[curTabName]
    nextTabIndex = tabIndex+1
    while nextTabIndex < len(tableDict):
        nextTabName = tableDict.keys()[nextTabIndex]
        assert tableDict[nextTabName] == nextTabIndex
        if dataset == "BusTracker":
            isValid = checkIfSameSchema(curTabName, nextTabName)
            if isValid is False:
                nextTabIndex += 1
                continue
        nextTabCols = tabColDict[nextTabName]
        nextTabColTypes = tabColTypeDict[nextTabName]
        joinPairDict[str(curTabName)+","+str(nextTabName)] = []
        for curTabColName in curTabCols:
            curTabColIndex = curTabCols.index(curTabColName)
            curTabColType = curTabColTypes[curTabColIndex]
            for nextTabColName in nextTabCols:
                nextTabColIndex = nextTabCols.index(nextTabColName)
                nextTabColType = nextTabColTypes[nextTabColIndex]
                # if same data types, add them as a possible join key pair to the dictionary
                if curTabColType.split(" ")[0].split("(")[0] == nextTabColType.split(" ")[0].split("(")[0]: # ignore unsigned and num bytes from 'int unsigned' while comparison
                    joinPairDict[str(curTabName) + "," + str(nextTabName)].append(curTabColName+","+nextTabColName)
        nextTabIndex+=1
    return joinPairDict

def pruneEmptyJoinPairs(joinPairDict):
    tabPairs = joinPairDict.keys()
    for tabPair in tabPairs:
        if len(joinPairDict[tabPair])==0:
            del joinPairDict[tabPair]
    return joinPairDict

def createJoinPairDict(dataset, tabColDict, tabColTypeDict, tableDict):
    joinPairDict = {}
    # key is the table ID pair and value is the list of possible join key pairs, we store both self join and cross-table join.
    # if in the future query a join predicate has tab2.Col2 = tab1.Col3, it will always be chronologically stored as tab1.Col3 = tab2.Col2
    # we restrict self-joins to be on the same col. That is a reasonable assumption & will confine the dimensionality.
    for tableName in tableDict:
        joinPairDict = createSelfJoinPairs(tableName, tabColDict, joinPairDict)
        joinPairDict = createCrossJoinPairs(dataset, tableName, tabColDict, tabColTypeDict, tableDict, joinPairDict)
    return joinPairDict

def createJoinPredBitPosDict(joinPairDict):
    # key is tabPair and value is startBitPos,endBitPos reqd for that tabPair combination
    bitPosDict = {}
    endPos = -1
    for tabPair in joinPairDict:
        startPos = endPos+1
        endPos = startPos+len(joinPairDict[tabPair])-1
        bitPosDict[tabPair] = str(startPos)+","+str(endPos)
    return bitPosDict

def writeSchemaInfoToFile(dict, fn):
    try:
        os.remove(fn)
    except OSError:
        pass
    # key,value pair from each dictionary is written as key:value in each separate row in the file
    with open(fn, 'a') as f:
        for key in dict:
            f.write(str(key)+":"+str(dict[key])+"\n")
        f.flush()
        f.close()
    print("Wrote to file "+fn)

def writeSchemaInfoToFiles(tableDict, tabColDict, tabColTypeDict, tabColBitPosDict, joinPairDict, joinPredBitPosDict, configDict):
    writeSchemaInfoToFile(tableDict, getConfig(configDict['MINC_TABLES']))
    writeSchemaInfoToFile(tabColDict, getConfig(configDict['MINC_COLS']))
    writeSchemaInfoToFile(tabColTypeDict, getConfig(configDict['MINC_COL_TYPES']))
    writeSchemaInfoToFile(tabColBitPosDict, getConfig(configDict['MINC_COL_BIT_POS']))
    writeSchemaInfoToFile(joinPairDict, getConfig(configDict['MINC_JOIN_PREDS']))
    writeSchemaInfoToFile(joinPredBitPosDict, getConfig(configDict['MINC_JOIN_PRED_BIT_POS']))

def createTabColBitPosDict(tabColDict, tableDict):
    sorted_tableDict = sorted(tableDict.items(), key=operator.itemgetter(1))
    tabIndex=0
    colIndex = 0
    tabColBitPosDict = {}
    for (table, tabIndex) in sorted_tableDict:
        assert tableDict[table] == tabIndex
        colArr = tabColDict[table]
        for col in colArr:
            tabColBitPosDict[table+"."+col] = colIndex
            colIndex+=1
        tabIndex+=1
    return tabColBitPosDict

def fetchSchema(configDict):
    cnx = connectToDB(configDict)
    tableDict = None
    if configDict['DATASET'] == 'MINC':
        tableDict = createMincTableDict(cnx)
    elif configDict['DATASET'] == 'BusTracker':
        tableDict = createBusTrackerTableDict(cnx, configDict)
    (tabColDict, tabColTypeDict) = createTabColDict(cnx, tableDict)
    tabColBitPosDict = createTabColBitPosDict(tabColDict, tableDict)
    joinPairDict = createJoinPairDict(configDict['DATASET'], tabColDict, tabColTypeDict, tableDict)
    joinPairDict = pruneEmptyJoinPairs(joinPairDict)
    joinPredBitPosDict = createJoinPredBitPosDict(joinPairDict)

    print("Writing Dictionaries To Files")
    writeSchemaInfoToFiles(tableDict, tabColDict, tabColTypeDict, tabColBitPosDict, joinPairDict, joinPredBitPosDict, configDict)
    return (tableDict, tabColDict, tabColTypeDict, tabColBitPosDict, joinPairDict, joinPredBitPosDict)

def countTables(fileName):
    count = 0
    with open(fileName) as f:
        for line in f:
            count+=1
    return count

def countCols(fileName):
    count = 0
    with open(fileName) as f:
        for line in f:
            count += len(line.split(":")[1].split(","))
    return count

def countJoinPredsFromBitPos(fileName):
    count = 0
    with open(fileName) as f:
        for line in f:
            startEndPos = line.split(":")[1].split(",")
            count += int(startEndPos[1]) - int(startEndPos[0]) + 1
    return count

def countColsFromBitPos(fileName):
    count = 0
    maxBitPos = -1
    with open(fileName) as f:
        for line in f:
            bitPos = int(line.split(":")[1])
            if bitPos > maxBitPos:
                maxBitPos = bitPos
            count += 1
    assert count == maxBitPos+1
    return count

def printStats(configDict):
    print("# Tables: "+str(countTables(getConfig(configDict['MINC_TABLES']))))
    print("# Columns: "+str(countCols(getConfig(configDict['MINC_COLS']))))
    print("# Columns from bit positions: " + str(countColsFromBitPos(getConfig(configDict['MINC_COL_BIT_POS']))))
    joinPredCountFromBitPos = countJoinPredsFromBitPos(getConfig(configDict['MINC_JOIN_PRED_BIT_POS']))
    print("# JoinPreds from bit positions: "+str(joinPredCountFromBitPos))

if __name__ == "__main__":
    #configDict = parseConfig.parseConfigFile("MINC_configFile.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="Config parameters file", type=str, required=True)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    (tableDict, tabColDict, tabColTypeDict, tabColBitPosDict, joinPairDict, joinPredBitPosDict) = fetchSchema(configDict)
    printStats(configDict)