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
from ParseConfigFile import getConfig
import ConcurrentSessions
import ParseResultsToExcel
import random
import argparse

def compareForSanity(newSessionLengthDict, sessionLengthDict):
    assert len(sessionLengthDict) == len(newSessionLengthDict)
    #sessCount = 0
    for key in sessionLengthDict:
        #assert key in newSessionLengthDict
        #if sessCount % 100 == 0:
            #print "Checked Sanity for "+str(sessCount)+" so far"
        #sessCount+=1
        if sessionLengthDict[key] != newSessionLengthDict[key]:
            print "newSessionLengthDict["+str(key)+"]: "+str(newSessionLengthDict[key])+", sessionLengthDict["+str(key)+"]: "+str(sessionLengthDict[key])

def createConcIntentVectors(testSessNamesFold, foldID, configDict, intentSessionFile, allQueries):
    fileNameWithoutDir = intentSessionFile.split("/")[len(intentSessionFile.split("/"))-1]
    outputIntentConcTrainSessions = getConfig(configDict['KFOLD_INPUT_DIR'])+fileNameWithoutDir+"_CONC_TRAIN_FOLD_"+str(foldID)
    outputIntentConcTestSessions = getConfig(configDict['KFOLD_INPUT_DIR'])+fileNameWithoutDir + "_CONC_TEST_FOLD_" + str(foldID)
    try:
        os.remove(outputIntentConcTrainSessions)
    except OSError:
        pass
    try:
        os.remove(outputIntentConcTestSessions)
    except OSError:
        pass
    for line in allQueries:
        sessName = line.split(";")[0].split(",")[0]
        if sessName in testSessNamesFold:
            ti.appendToFile(outputIntentConcTestSessions, line)
        else:
            ti.appendToFile(outputIntentConcTrainSessions, line)
    return


def createSeqIntentVectors(testSessNamesFold, foldID, configDict, sessNames, intentSessionFile, sessionLengthDict, sessionLineDict):
    fileNameWithoutDir = intentSessionFile.split("/")[len(intentSessionFile.split("/"))-1]
    outputIntentSeqTrainSessions = getConfig(configDict['KFOLD_INPUT_DIR'])+fileNameWithoutDir+"_SEQ_TRAIN_FOLD_"+str(foldID)
    outputIntentSeqTestSessions = getConfig(configDict['KFOLD_INPUT_DIR'])+fileNameWithoutDir + "_SEQ_TEST_FOLD_" + str(foldID)
    try:
        os.remove(outputIntentSeqTrainSessions)
    except:
        pass
    try:
        os.remove(outputIntentSeqTestSessions)
    except OSError:
        pass
    #sessCount = 0
    #sessQueryCount = 0
    for sessName in sessNames:
        sessID = int(sessName.split(" ")[1])
        numSessQueries = sessionLengthDict[sessID]
        #if sessID == 36 or sessID == 30:
            #print "hi in createTrainTest"
        for queryID in range(numSessQueries):
            lineToOutput = sessionLineDict[str(sessID)+","+str(queryID)]
            #sessQueryCount += 1
            if sessName in testSessNamesFold:
                ti.appendToFile(outputIntentSeqTestSessions, lineToOutput)
                #if sessQueryCount%10000 == 0:
                    #print "Sess: "+str(sessCount)+", sessQueryCount: "+str(sessQueryCount)
            else:
                ti.appendToFile(outputIntentSeqTrainSessions, lineToOutput)
                #if sessQueryCount%10000 == 0:
                    #print "Sess: "+str(sessCount)+", sessQueryCount: "+str(sessQueryCount)
        #sessCount+=1
    return


def prepareKFoldTrainTest(configDict, intentSessionFile):
    inputSessionFile = getConfig(configDict['QUERYSESSIONS'])
    #sessID and queryID should start from 0
    sessNames = []
    #sessDict = {}
    with open(inputSessionFile) as f:
        for line in f:
            sessName = line.split(";")[0]
            sessNames.append(sessName)
            #sessDict[sessName] = line.split(";")
    f.close()
    random.shuffle(sessNames)
    kFold = int(configDict['KFOLD'])
    testFrac = 1.0/float(kFold)
    numTest = int(testFrac * len(sessNames))
    print "Total Sessions: "+ str(len(sessNames))+", numTest: "+str(numTest)
    # for K fold CV
    testSessNames = [[] for i in range(kFold)]
    testEndIndex = -1
    assert configDict['SEQ_OR_CONC_TRAIN_TEST'] == 'SEQ' or configDict['SEQ_OR_CONC_TRAIN_TEST'] == 'CONC'
    if configDict['SEQ_OR_CONC_TRAIN_TEST'] == 'SEQ':
        sessionLengthDict = ConcurrentSessions.countQueries(getConfig(configDict['QUERYSESSIONS']))
        sessionLineDict = {}
        newSessionLengthDict = {}
        with open(intentSessionFile) as f:
            for line in f:
                (sessionLineDict, newSessionLengthDict) = QR.updateSessionLineDict(line, configDict, sessionLineDict, newSessionLengthDict)
        f.close()
        compareForSanity(newSessionLengthDict, sessionLengthDict)
        print "Checked Sanity for all Sessions"
    else:
        allQueries =[]
        with open(intentSessionFile) as f:
            for line in f:
                allQueries.append(line.strip())
        f.close()
    for i in range(kFold):
        testStartIndex = testEndIndex+1
        testEndIndex = testStartIndex + numTest
        if i == kFold - 1:
            testEndIndex = len(sessNames)-1
        for index in range(testStartIndex, testEndIndex+1):
            testSessNames[i].append(sessNames[index])
        print "Fold "+str(i)+", StartIndex="+str(testStartIndex)+", EndIndex="+str(testEndIndex)
        if configDict['SEQ_OR_CONC_TRAIN_TEST'] == 'SEQ':
            createSeqIntentVectors(testSessNames[i], i, configDict, sessNames, intentSessionFile, sessionLengthDict, sessionLineDict)
        else:
            createConcIntentVectors(testSessNames[i], i, configDict, intentSessionFile, allQueries)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="Config parameters file", type=str, required=True)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    #configDict = parseConfig.parseConfigFile("configFile.txt")
    if configDict['INTENT_REP']=='TUPLE':
        intentSessionFile = getConfig(configDict['TUPLEINTENTSESSIONS'])
    elif configDict['INTENT_REP']=='FRAGMENT' and configDict['BIT_OR_WEIGHTED']=='BIT':
        if configDict['RNN_PREDICT_QUERY_OR_TABLE'] == 'TABLE':
            intentSessionFile = getConfig(configDict['BIT_FRAGMENT_TABLE_INTENT_SESSIONS'])
        else:
            intentSessionFile = getConfig(configDict['BIT_FRAGMENT_INTENT_SESSIONS'])
    elif configDict['INTENT_REP']=='FRAGMENT' and configDict['BIT_OR_WEIGHTED']=='WEIGHTED':
        intentSessionFile = getConfig(configDict['WEIGHTED_FRAGMENT_INTENT_SESSIONS'])
    elif configDict['INTENT_REP']=='QUERY':
        intentSessionFile = getConfig(configDict['QUERY_INTENT_SESSIONS'])
    else:
        print "ConfigDict['INTENT_REP'] must either be TUPLE or FRAGMENT or QUERY !!"
        sys.exit(0)
    prepareKFoldTrainTest(configDict, intentSessionFile)
