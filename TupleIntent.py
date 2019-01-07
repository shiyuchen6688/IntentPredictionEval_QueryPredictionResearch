import sys
import os
import time, argparse
from bitmap import BitMap
import ParseConfigFile as parseConfig
from ParseConfigFile import getConfig
import QueryParser as qp

def createTupleIntentRep(rowIDs, sessQuery, configDict):
    resObj = None
    if configDict['EXEC_SAMPLE']:
        totalDataSize = int(configDict['SAMPLETABLESIZE'])
    else:
        totalDataSize = int(configDict['FULLTABLESIZE'])
    resObj = BitMap(totalDataSize)
    if rowIDs is None:
        (newQuery,rowIDs) = qp.fetchRowIDs(sessQuery, configDict)
    if rowIDs is None:
        for rowID in range(totalDataSize):
            resObj.set(rowID)  # here rowID was forced to start from  0 in the for loop as all rows are being set to 1
    else:
        for rowID in rowIDs:
            resObj.set(rowID-1) # because rowIDs start from 1 but bit positions start from 0
    return (newQuery,resObj)

def appendToFile(outputFile, outputLine):
    with open(outputFile, 'a') as outFile:
        outFile.write(outputLine+"\n")

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("configFile.txt")
    tupleIntentSessionsFile = getConfig(configDict['TUPLEINTENTSESSIONS'])
    try:
        os.remove(tupleIntentSessionsFile)
    except OSError:
        pass
    with open(getConfig(configDict['CONCURRENT_QUERY_SESSIONS'])) as f:
        for line in f:
            sessQueries = line.split(";")
            sessQueryName = sessQueries[0]
            sessQuery = sessQueries[1].strip()
            (newQuery, resObj) = createTupleIntentRep(None, sessQuery,
                                                      configDict)  # rowIDs passed should be None, else it won't fill up
            if newQuery is None:
                outputIntentLine = sessQueryName + "; OrigQuery: " + sessQuery + ";" + str(resObj)
            else:
                outputIntentLine = sessQueryName + ";" + newQuery + ";" + str(resObj)
            appendToFile(tupleIntentSessionsFile, outputIntentLine)
            print ("Executed " + sessQueryName)



