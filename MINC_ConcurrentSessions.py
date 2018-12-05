import sys
import os
import time, argparse
import QueryParser as qp
import ParseConfigFile as parseConfig
import random
import TupleIntent as ti
from itertools import islice
from ParseConfigFile import getConfig

def cleanQuery(line):
    #remove trailing newline and replace consecutive tabs with a single tab
    line = '\t'.join(line.strip().split("\t"))
    #substitute tabs with spaces
    line = line.replace("\t", " ")
    #replace consecutive spaces with a single space
    line = " ".join(line.split())
    #remove starting spaces
    line = line.strip()
    return line

def countQueries(inputFile): # this is an in-memory version, so holds all the lines in memory in dict
    sessionQueryDict = {}
    queryCount = 0
    violated = 0
    lineNum = 0
    with open(inputFile) as f:
        for line in f:
            if 'Query' in line and line.startswith('\t'):
                line = cleanQuery(line)
                sessTokens = line.split()
                if sessTokens[1] != 'Query':
                    violated += 1
                    print "discovered violations so far: "+str(violated)
                    continue # because the pattern is messed up and such queries can be ignored
                sessIndex = sessTokens[0] # the ID is used as a string
                if sessIndex not in sessionQueryDict:
                    sessionQueryDict[sessIndex] = []
                sessQuery = " ".join(sessTokens[2:])
                sessionQueryDict[sessIndex].append(sessQuery)
                queryCount +=1 # note that queryCount is not the same as lineNum
                if queryCount % 1000000 == 0:
                    print "Query count so far: "+str(queryCount)
            lineNum+=1
    print "Total Query Count: " + str(queryCount)+", session count: "+str(len(sessionQueryDict))
    return sessionQueryDict

def raiseError():
    print "Error as correct line not found !!"
    exit(0)

def retrieveQueryFromFile(inputFile, coveredSessQueries, sessIndex, sessionQueryPosDict):
    if sessIndex not in coveredSessQueries:
        queryPos = 0
    else:
        queryPos = coveredSessQueries[sessIndex]
    queryLineNum = sessionQueryPosDict[sessIndex][queryPos]
    with open(inputFile) as f:
        lines = list(islice(f, int(queryLineNum), int(queryLineNum)+1)) # only one line
    for line in lines:
        print sessIndex+","+line
    line = lines[0]
    if 'Query' in line and line.startswith('\t'):
        line = cleanQuery(line)
        sessTokens = line.split()
        curSessIndex = sessTokens[0]
        if sessTokens[1] != 'Query':
            print "Error following pattern !!"
            exit(0)
        if sessIndex == curSessIndex:
            # here we assume queryIndex starts from 1, count of queries covered so far gives the index of the next uncovered query
            # but sessionName is the 0th token, so we need to add a 1 to get the query index
            if sessIndex not in coveredSessQueries:
                queryIndex = 1
            else:
                queryIndex = coveredSessQueries[sessIndex] + 1
            sessQuery = " ".join(sessTokens[2:])
            return (sessQuery,queryIndex)
        else:
            raiseError()
    raiseError()

def retrieveQueryFromMemory(coveredSessQueries, sessIndex, sessionQueryDict):
    if sessIndex not in coveredSessQueries:
        queryPos = 0
    else:
        queryPos = coveredSessQueries[sessIndex]
    sessQuery = sessionQueryDict[sessIndex][queryPos]
    queryIndex = queryPos+1
    return (sessQuery, queryIndex)

def createConcurrentSessions(inputFile, outputFile):
    sessionQueryDict = countQueries(inputFile)
    try:
        os.remove(outputFile)
    except OSError:
        pass
    keyList = list(sessionQueryDict.keys()) # this actually clones the keys into a new python object keyList, not the same as pointing to the existing list
    coveredSessQueries = {} # key is sessionID and value is the query count covered
    queryCount = 0
    while len(keyList)!=0:
        sessIndex = random.choice(keyList)
        if sessIndex not in coveredSessQueries or coveredSessQueries[sessIndex] < len(sessionQueryDict[sessIndex]):
            (sessQuery,queryIndex) = retrieveQueryFromMemory(coveredSessQueries, sessIndex, sessionQueryDict)
            if sessQuery == "":
                keyList.remove(sessIndex)
                continue
            if sessIndex not in coveredSessQueries:
                coveredSessQueries[sessIndex] = 1
            else:
                coveredSessQueries[sessIndex] += 1
            output_str="Session "+str(sessIndex)+", Query "+str(queryIndex)+";"+sessQuery
            ti.appendToFile(outputFile, output_str)
            queryCount+=1
            if queryCount % 1000000 == 0:
                print "appended Session "+str(sessIndex)+", Query "+str(queryIndex)
        else:
            keyList.remove(sessIndex)
            print "ERROR: invalid SessIndex !!"+str(sessIndex)
            sys.exit(0)

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("MINC_configFile.txt")
    createConcurrentSessions(getConfig(configDict['QUERYSESSIONS']), getConfig(configDict['CONCURRENT_QUERY_SESSIONS']))
    print "Completed concurrent session order creation"
