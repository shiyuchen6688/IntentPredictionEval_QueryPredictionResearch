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
    print sessNames

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("configFile.txt")
    prepareVariableTrainFixedTest(configDict)