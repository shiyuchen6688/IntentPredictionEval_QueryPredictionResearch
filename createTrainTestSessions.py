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

def prepareVariableTrainFixedTest(configDict):
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
    for i in range(kFold):
        testStartIndex = testEndIndex+1
        testEndIndex = testStartIndex + numTest
        if i == kFold - 1:
            testEndIndex = len(sessNames)-1
        for index in range(testStartIndex, testEndIndex+1):
            testSessNames[i].append(sessNames[index])
        print "Fold "+str(i)+", StartIndex="+str(testStartIndex)+", EndIndex="+str(testEndIndex)
    


if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("configFile.txt")
    prepareVariableTrainFixedTest(configDict)