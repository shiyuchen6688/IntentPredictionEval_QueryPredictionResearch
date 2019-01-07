import sys
import os
import time, argparse
import QueryParser as qp
import ParseConfigFile as parseConfig
from ParseConfigFile import getConfig
import random
import TupleIntent as ti

def countQueries(inputFile):
    sessionQueryCountDict = {}
    with open(inputFile) as f:
        for line in f:
            sessTokens = line.strip().split(";")
            # sessionIndices need to be noted that they are neither sequential nor complete. For instance session 15 or 16 does not exist.
            sessIndex = int(sessTokens[0].split(" ")[1])
            sessionQueryCountDict[sessIndex] = line.count(";")-1 #line ends with a semicolon but has the first token as session name which is ignored
    return sessionQueryCountDict

def retrieveQueryFromFile(inputFile, coveredSessQueries, sessIndex):
    with open(inputFile) as f:
        for line in f:
            sessTokens = line.strip().split(";")
            curSessIndex = int(sessTokens[0].split(" ")[1])
            if sessIndex == curSessIndex:
                # here we assume queryIndex starts from 1, count of queries covered so far gives the index of the next uncovered query
                # but sessionName is the 0th token, so we need to add a 1 to get the query index
                if sessIndex not in coveredSessQueries:
                    queryIndex = 1
                else:
                    queryIndex = coveredSessQueries[sessIndex] + 1
                sessQuery = sessTokens[queryIndex].split("~")[0]
                sessQuery = ' '.join(sessQuery.split()) # eliminate extra spaces within the SQL query
                return (sessQuery,queryIndex)

def createConcurrentSessions(inputFile, outputFile):
    sessionQueryCountDict = countQueries(inputFile)
    try:
        os.remove(outputFile)
    except OSError:
        pass
    keyList = list(sessionQueryCountDict.keys()) # this actually clones the keys into a new python object keyList, not the same as pointing to the existing list
    coveredSessQueries = {} # key is sessionID and value is the query count covered
    while len(keyList)!=0:
        sessIndex = random.choice(keyList)
        if sessIndex not in coveredSessQueries or coveredSessQueries[sessIndex] < sessionQueryCountDict[sessIndex]:
            (sessQuery,queryIndex) = retrieveQueryFromFile(inputFile, coveredSessQueries, sessIndex)
            if sessQuery == "":
                keyList.remove(sessIndex)
                continue
            if sessIndex not in coveredSessQueries:
                coveredSessQueries[sessIndex] = 1
            else:
                coveredSessQueries[sessIndex] += 1
            output_str="Session "+str(sessIndex)+", Query "+str(queryIndex)+";"+sessQuery
            ti.appendToFile(outputFile, output_str)
            print "appended Session "+str(sessIndex)+", Query "+str(queryIndex)
        else:
            keyList.remove(sessIndex)

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("configFile.txt")
    createConcurrentSessions(getConfig(configDict['QUERYSESSIONS']), getConfig(configDict['CONCURRENT_QUERY_SESSIONS']))
    print "Completed concurrent session order creation"