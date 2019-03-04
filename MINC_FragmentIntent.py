import sys
import os
import time, argparse
from bitmap import BitMap
import ParseConfigFile as parseConfig
import QueryParser as qp
import TupleIntent as ti
import re
import MINC_prepareJoinKeyPairs
from ParseConfigFile import getConfig
import random
import ReverseEnggQueries
#import MINC_QueryParser as MINC_QP

def concatenateSeqIntentVectorFiles(configDict):
    splitDir = getConfig(configDict['BIT_FRAGMENT_SEQ_SPLITS'])
    splitFileName = configDict['BIT_FRAGMENT_SEQ_SPLIT_NAME_FORMAT']
    numFiles = int(configDict['BIT_FRAGMENT_SEQ_SPLIT_THREADS'])
    sessionQueryDict = {} # key is session ID and value is line
    queryCount = 0
    prevSessName = None
    sessID = -1
    for i in range(numFiles):
        fileNamePerThread = splitDir+"/"+splitFileName+str(i)
        with open(fileNamePerThread) as f:
            for line in f:
                line = line.strip()
                if len(line.split(";")) > 3:
                    line = removeExcessDelimiters(line)
                assert len(line.split(";")) == 3
                tokens = line.split(";")
                sessName = tokens[0].split(", ")[0].split(" ")[1]
                if sessName != prevSessName:
                    sessID+=1
                    prevSessName = sessName
                if sessID not in sessionQueryDict:
                    sessionQueryDict[sessID] = []
                sessionQueryDict[sessID].append(line)
                queryCount +=1
                #if queryCount >20000:
                    #break
                if queryCount % 10000 == 0:
                    print ("Query count so far: "+str(queryCount))
    return sessionQueryDict

def createQuerySessions(sessionQueryDict, configDict):
    sessQueryFile = getConfig(configDict['QUERYSESSIONS'])
    try:
        os.remove(sessQueryFile)
    except OSError:
        pass
    for sessID in sessionQueryDict:
        output_str = "Session "+str(sessID)+";"
        for i in range(len(sessionQueryDict[sessID])):
            output_str = output_str + sessionQueryDict[sessID][i].split(";")[1].split(": ")[1]+";"
        ti.appendToFile(sessQueryFile, output_str)

def removeExcessDelimiters(sessQueryIntent):
    tokens = sessQueryIntent.split(";")
    ret_str = tokens[0]+";"
    for i in range(1,len(tokens)-1):
        if i == 1:
            ret_str = ret_str+tokens[i]
        else:
            ret_str = ret_str+","+tokens[i]
    ret_str = ret_str + ";"+tokens[len(tokens)-1]
    return ret_str

def tryDeletionIfExists(fileName):
    try:
        os.remove(fileName)
    except OSError:
        pass
    return

def createConcurrentIntentVectors(sessionQueryDict, configDict):
    intentFile = getConfig(configDict['BIT_FRAGMENT_INTENT_SESSIONS'])
    concurrentFile = getConfig(configDict['CONCURRENT_QUERY_SESSIONS'])
    tableIntentFile = getConfig(configDict['BIT_FRAGMENT_TABLE_INTENT_SESSIONS'])
    schemaDicts = ReverseEnggQueries.readSchemaDicts(configDict)
    tryDeletionIfExists(intentFile)
    tryDeletionIfExists(concurrentFile)
    tryDeletionIfExists(tableIntentFile)
    queryCount = 0
    queryIndex = 0
    absCount = 0
    numSessions = len(sessionQueryDict)
    while len(sessionQueryDict)!=0:
        keyList = sessionQueryDict.keys()
        random.shuffle(keyList)
        queryIndex += 1
        for sessIndex in keyList:
            sessQueryIntent = sessionQueryDict[sessIndex][0]
            sessionQueryDict[sessIndex].remove(sessQueryIntent)
            if len(sessionQueryDict[sessIndex]) == 0:
                del sessionQueryDict[sessIndex]
            queryIndexRec = sessQueryIntent.split(";")[0].split(", ")[1].split(" ")[1]
            if queryIndexRec != str(queryIndex):
                print "queryIndexRec != queryIndex !!"
            assert queryIndexRec == str(queryIndex)
            tokens = sessQueryIntent.split(";")
            assert len(tokens) == 3
            assert queryCount>=0
            if queryCount == 0:
                output_str = "Session " + str(sessIndex) + ", Query " + str(queryIndex) + ";" + tokens[1] + ";" + tokens[2]
                output_table_str = "Session " + str(sessIndex) + ", Query " + str(queryIndex) + ";" + tokens[1] + ";" + \
                             tokens[2][schemaDicts.tableStartBitIndex:schemaDicts.tableStartBitIndex+len(schemaDicts.tableDict)]
                conc_str = "Session " + str(sessIndex) + ", Query " + str(queryIndex) + ";" + tokens[1].split(":")[1]
            else:
                output_str += "\nSession " + str(sessIndex) + ", Query " + str(queryIndex) + ";" + tokens[1] + ";" + tokens[2]
                output_table_str += "\nSession " + str(sessIndex) + ", Query " + str(queryIndex) + ";" + tokens[1] + ";" + \
                                   tokens[2][schemaDicts.tableStartBitIndex:schemaDicts.tableStartBitIndex + len(
                                       schemaDicts.tableDict)]
                conc_str += "\nSession " + str(sessIndex) + ", Query " + str(queryIndex) + ";" + tokens[1].split(":")[1]
            queryCount += 1
            absCount+=1
            if queryCount % 100 == 0:
                ti.appendToFile(intentFile, output_str)
                ti.appendToFile(concurrentFile, conc_str)
                ti.appendToFile(tableIntentFile, output_table_str)
                queryCount = 0
            if absCount % 10000 == 0:
                print ("appended Session " + str(sessIndex) + ", Query " + str(queryIndex) + ", absQueryCount: " + str(absCount))
    if queryCount > 0:
        ti.appendToFile(intentFile, output_str)
        ti.appendToFile(concurrentFile, conc_str)
        ti.appendToFile(tableIntentFile, output_table_str)
    print ("Created intent vectors for # Sessions: "+str(numSessions)+" and # Queries: "+str(absCount))


if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("MINC_configFile.txt")
    assert configDict["BIT_OR_WEIGHTED"] == "BIT"
    fragmentIntentSessionsFile = getConfig(configDict['BIT_FRAGMENT_INTENT_SESSIONS'])
    try:
        os.remove(fragmentIntentSessionsFile)
    except OSError:
        pass
    sessionQueryDict = concatenateSeqIntentVectorFiles(configDict)
    createQuerySessions(sessionQueryDict, configDict)
    createConcurrentIntentVectors(sessionQueryDict, configDict)
