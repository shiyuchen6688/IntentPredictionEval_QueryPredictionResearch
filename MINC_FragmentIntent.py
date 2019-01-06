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
#import MINC_QueryParser as MINC_QP

def concatenateSeqIntentVectorFiles(configDict):
    splitDir = getConfig(configDict['BIT_FRAGMENT_SEQ_SPLITS'])
    splitFileName = configDict['BIT_FRAGMENT_SEQ_SPLIT_NAME_FORMAT']
    numFiles = int(configDict['BIT_FRAGMENT_SEQ_SPLIT_THREADS'])
    sessionQueryDict = {} # key is session ID and value is line
    queryCount = 0
    for i in range(numFiles):
        fileNamePerThread = splitDir+"/"+splitFileName+str(i)
        with open(fileNamePerThread) as f:
            for line in f:
                line = line.strip()
                tokens = line.split(";")
                sessID = tokens[0].split(",")[0].split(" ")[1]
                if sessID not in sessionQueryDict:
                    sessionQueryDict[sessID] = []
                sessionQueryDict[sessID].append(line)
                queryCount +=1
                if queryCount % 10000 == 0:
                    print ("Query count so far: "+str(queryCount))
    return sessionQueryDict

def createConcurrentIntentVectors(sessionQueryDict, configDict):
    intentFile = getConfig(configDict['BIT_FRAGMENT_INTENT_SESSIONS'])
    try:
        os.remove(intentFile)
    except OSError:
        pass
    queryCount = 0
    queryIndex = 0
    while len(sessionQueryDict)!=0:
        random.shuffle(sessionQueryDict.keys())
        queryIndex += 1
        for sessIndex in sessionQueryDict.keys():
            sessQueryIntent = sessionQueryDict[sessIndex][0]
            sessionQueryDict[sessIndex].remove(sessQueryIntent)
            if len(sessionQueryDict[sessIndex]) == 0:
                del sessionQueryDict[sessIndex]
            queryIndexRec = sessQueryIntent.split(";")[0].split(",")[1].split(" ")[1]
            assert queryIndexRec == str(queryIndex)
            assert queryCount>=0
            if queryCount == 0:
                output_str = sessQueryIntent
            else:
                output_str += "\n"+sessQueryIntent
            queryCount += 1
            if queryCount % 100 == 0:
                print ("appended Session " + str(sessIndex) + ", Query " + str(queryIndex) + ", queryCount: " + str(queryCount))
                ti.appendToFile(intentFile, output_str)
                queryCount = 0
    #ti.appendToFile(intentFile, output_str)
    print ("appended Sessions and Queries for a queryCount: "+str(queryCount))


if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("MINC_configFile.txt")
    assert configDict["BIT_OR_WEIGHTED"] == "BIT"
    fragmentIntentSessionsFile = getConfig(configDict['BIT_FRAGMENT_INTENT_SESSIONS'])
    try:
        os.remove(fragmentIntentSessionsFile)
    except OSError:
        pass
    sessionQueryDict = concatenateSeqIntentVectorFiles(configDict)
    createConcurrentIntentVectors(sessionQueryDict, configDict)
