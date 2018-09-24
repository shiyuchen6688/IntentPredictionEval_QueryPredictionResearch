import sys
import os
import time, argparse
from bitmap import BitMap
import ParseConfigFile as parseConfig
def createTupleIntentRep(rowIDs, configDict):
    resObj = None
    if configDict['EXEC_SAMPLE']:
        totalDataSize = int(configDict['SAMPLETABLESIZE'])
    else:
        totalDataSize = int(configDict['FULLTABLESIZE'])
    resObj = BitMap(totalDataSize)
    for rowID in rowIDs:
        resObj.set(rowID-1) # because rowIDs start from 1 but bit positions start from 0

    return resObj

if __name__ == "__main__":
    rowIDs=[11, 1, 2, 3, 4, 109673] # rowIDs start from 1
    configDict = parseConfig.parseConfigFile("configFile.txt")
    createTupleIntentRep(rowIDs,configDict)
