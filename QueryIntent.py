import sys
import os
import time, argparse
from bitmap import BitMap
import ParseConfigFile as parseConfig
from ParseConfigFile import getConfig
import QueryParser as qp
import TupleIntent as ti
import re

def createQueryIntentRep(sessQuery, configDict, queryVocabulary):
    if sessQuery not in queryVocabulary:
        newBitPos = len(queryVocabulary)
        queryVocabulary[sessQuery] = newBitPos
    resObj = BitMap(len(queryVocabulary))
    resObj.set(queryVocabulary[sessQuery])
    return (queryVocabulary,resObj)

def createQueryIntentRepFullDimensionality(sessQuery, configDict, queryVocabulary):
    FIXED_SIZE = int(configDict['NUMQUERIES'])
    if sessQuery not in queryVocabulary:
        newBitPos = len(queryVocabulary)
        queryVocabulary[sessQuery] = newBitPos
    resObj = BitMap(FIXED_SIZE)
    resObj.set(queryVocabulary[sessQuery])
    return (queryVocabulary,resObj)


if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("configFile.txt")
    if configDict["INTENT_REP"] == "QUERY":
        queryIntentSessionsFile = getConfig(configDict['QUERY_INTENT_SESSIONS'])
    else:
        print("This supports only query intent gen !!")
        sys.exit(0)
    try:
        os.remove(queryIntentSessionsFile)
    except OSError:
        pass
    with open(getConfig(configDict['CONCURRENT_QUERY_SESSIONS'])) as f:
        queryVocabulary = {}  # dict with query as key and bit position/dimension as value
        queryCount = 0
        for line in f:
            tokens = line.split(";")
            sessQueryName = tokens[0]
            sessQuery = tokens[1].strip()
            (queryVocabulary, resObj) = createQueryIntentRepFullDimensionality(sessQuery, configDict,
                                                             queryVocabulary)  # rowIDs passed should be None, else it won't fill up
            outputIntentLine = sessQueryName + "; OrigQuery: " + sessQuery + ";" + str(resObj)
            ti.appendToFile(queryIntentSessionsFile, outputIntentLine)
            queryCount = queryCount + 1
            print("Generated fragment for " + sessQueryName + ", #distinct queries so far: " + str(
                len(queryVocabulary)) + ", total #queries: " + str(queryCount))

