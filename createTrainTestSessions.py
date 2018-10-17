from __future__ import division
import sys
import os
import time
import QueryRecommender as QR
from bitmap import BitMap
import math
import heapq
import TupleIntent as ti
import ParseConfigFile as parseConfig
import ConcurrentSessions
import ParseResultsToExcel
import random

def createIntentVectors(testSessNamesFold, foldID, configDict, sessNames, intentSessionFile, sessionLengthDict):
    fileNameWithoutDir = intentSessionFile.split("/")[len(intentSessionFile.split("/"))-1]
    outputIntentTrainSessions = configDict['KFOLD_INPUT_DIR']+fileNameWithoutDir+"_TRAIN_FOLD_"+str(foldID)
    outputIntentTestSessions = configDict['KFOLD_INPUT_DIR']+fileNameWithoutDir + "_TEST_FOLD_" + str(foldID)
    try:
        os.remove(outputIntentTrainSessions)
        os.remove(outputIntentTestSessions)
    except OSError:
        pass
    sessionLineDict = {}
    with open(intentSessionFile) as f:
        for line in f:
            sessionLineDict = QR.updateSessionLineDict(line, configDict, sessionLineDict)
    f.close()
    for sessName in sessNames:
        sessID = int(sessName.split(" ")[1])
        numSessQueries = sessionLengthDict[sessID]
        for queryID in range(numSessQueries):
            lineToOutput = sessionLineDict[str(sessID)+","+str(queryID)]
            if sessName in testSessNamesFold:
                ti.appendToFile(outputIntentTestSessions, lineToOutput)
            else:
                ti.appendToFile(outputIntentTrainSessions, lineToOutput)
    return

def prepareVariableTrainFixedTest(configDict, intentSessionFile):
    inputSessionFile = configDict['QUERYSESSIONS']
    #sessID and queryID should start from 0
    sessNames = []
    with open(inputSessionFile) as f:
        for line in f:
            sessName = line.split(";")[0]
            sessNames.append(sessName)
    f.close()
    random.shuffle(sessNames)
    kFold = int(configDict['KFOLD'])
    testFrac = 1.0/float(kFold)
    numTest = int(testFrac * len(sessNames))
    print "numTest: "+str(numTest)
    # for K fold CV
    testSessNames = [[] for i in range(kFold)]
    testEndIndex = -1
    sessionLengthDict = ConcurrentSessions.countQueries(configDict['QUERYSESSIONS'])
    for i in range(kFold):
        testStartIndex = testEndIndex+1
        testEndIndex = testStartIndex + numTest
        if i == kFold - 1:
            testEndIndex = len(sessNames)-1
        for index in range(testStartIndex, testEndIndex+1):
            testSessNames[i].append(sessNames[index])
        print "Fold "+str(i)+", StartIndex="+str(testStartIndex)+", EndIndex="+str(testEndIndex)

        createIntentVectors(testSessNames[i], i, configDict, sessNames, intentSessionFile, sessionLengthDict)

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("configFile.txt")
    if configDict['INTENT_REP']=='TUPLE':
        intentSessionFile = configDict['TUPLEINTENTSESSIONS']
    elif configDict['INTENT_REP']=='FRAGMENT' and configDict['BIT_OR_WEIGHTED']=='BIT':
        intentSessionFile = configDict['BIT_FRAGMENT_INTENT_SESSIONS']
    elif configDict['INTENT_REP']=='FRAGMENT' and configDict['BIT_OR_WEIGHTED']=='WEIGHTED':
        intentSessionFile = configDict['WEIGHTED_FRAGMENT_INTENT_SESSIONS']
    elif configDict['INTENT_REP']=='QUERY':
        intentSessionFile = configDict['QUERY_INTENT_SESSIONS']
    else:
        print "ConfigDict['INTENT_REP'] must either be TUPLE or FRAGMENT or QUERY !!"
        sys.exit(0)
    prepareVariableTrainFixedTest(configDict, intentSessionFile)