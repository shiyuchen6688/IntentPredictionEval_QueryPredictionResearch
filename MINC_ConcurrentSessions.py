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

def assertConcurrentSessions(rawFile, concSessFile):
    inputQueries = []
    violated = 0
    with open(rawFile) as f:
        for line in f:
            if 'Query' in line and line.startswith('\t'):
                line = cleanQuery(line)
                sessTokens = line.split()
                if sessTokens[1] != 'Query':
                    violated += 1
                    print ("discovered violations so far: "+str(violated))
                    continue # because the pattern is messed up and such queries can be ignored
                sessQuery = " ".join(sessTokens[2:])
                inputQueries.append(sessQuery)
    print("inputQueries is ", inputQueries)
    queryCount = 0
    with open(concSessFile) as f:
        for line in f:
            sessQuery = line.strip().split(";")[1]
            if sessQuery not in inputQueries:
                print("query not present !! exiting")
                print(sessQuery)
                # print(inputQueries)
                exit(0)
            inputQueries.remove(sessQuery)
            queryCount+=1
            if queryCount % 1000000 == 0:
                print("Completed Assertion for "+str(queryCount)+" queries")
    print("Assertion Met !!")
    return

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
                    print("discovered violations so far: "+str(violated))
                    continue # because the pattern is messed up and such queries can be ignored
                sessIndex = sessTokens[0] # the ID is used as a string
                if sessIndex not in sessionQueryDict:
                    sessionQueryDict[sessIndex] = []
                sessQuery = " ".join(sessTokens[2:])
                sessionQueryDict[sessIndex].append(sessQuery)
                queryCount +=1 # note that queryCount is not the same as lineNum
                if queryCount % 1000000 == 0:
                    print("Query count so far: "+str(queryCount))
            lineNum+=1
    print("Total Query Count: " + str(queryCount)+", session count: "+str(len(sessionQueryDict)))
    return sessionQueryDict

def raiseError():
    print("Error as correct line not found !!")
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
        print(sessIndex+","+line)
    line = lines[0]
    if 'Query' in line and line.startswith('\t'):
        line = cleanQuery(line)
        sessTokens = line.split()
        curSessIndex = sessTokens[0]
        if sessTokens[1] != 'Query':
            print("Error following pattern !!")
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

def createConcurrentSessionsIdeal(inputFile, outputFile):
    sessionQueryDict = countQueries(inputFile)
    try:
        os.remove(outputFile)
    except OSError:
        pass
    coveredSessQueries = {} # key is sessionID and value is the query count covered
    queryCount = 0
    aggQueryCount = 0
    while len(sessionQueryDict)!=0:
        sessIndex = sessionQueryDict.keys()[random.randint(0,len(sessionQueryDict))]
        #sessIndex = random.choice(sessionQueryDict.keys())
        sessQuery = sessionQueryDict[sessIndex][0]
        if sessIndex in coveredSessQueries:
            coveredSessQueries[sessIndex] += 1
        else:
            coveredSessQueries[sessIndex] = 0
        sessionQueryDict[sessIndex].remove(sessQuery)
        if len(sessionQueryDict[sessIndex]) == 0:
            del sessionQueryDict[sessIndex]
        queryIndex = coveredSessQueries[sessIndex]+1
        if queryCount == 0:
            output_str ="Session "+str(sessIndex)+", Query "+str(queryIndex)+";"+sessQuery
        elif queryCount > 1:
            output_str += "\nSession " + str(sessIndex) + ", Query " + str(queryIndex) + ";" + sessQuery
        queryCount+=1
        aggQueryCount += queryCount
        if queryCount % 1000000 == 0:
            print("appended Session " + str(sessIndex) + ", Query " + str(queryIndex) + ", queryCount: " + str(queryCount))
            #ti.appendToFile(outputFile, output_str)
            #queryCount = 0
    if queryCount > 0:
        ti.appendToFile(outputFile, output_str)
        print("appended Sessions and Queries for a queryCount: "+str(queryCount))

def createConcurrentSessions(inputFile, outputFile):
    sessionQueryDict = countQueries(inputFile)
    try:
        os.remove(outputFile)
    except OSError:
        pass
    queryCount = 0
    queryIndex = 0
    while len(sessionQueryDict)!=0:
        keyList = sessionQueryDict.keys()
        random.shuffle(keyList)
        queryIndex += 1
        for sessIndex in keyList:
            sessQuery = sessionQueryDict[sessIndex][0]
            sessionQueryDict[sessIndex].remove(sessQuery)
            if len(sessionQueryDict[sessIndex]) == 0:
                del sessionQueryDict[sessIndex]
            if queryCount == 0:
                output_str = "Session " + str(sessIndex) + ", Query " + str(queryIndex) + ";" + sessQuery
            else:
                output_str += "\nSession " + str(sessIndex) + ", Query " + str(queryIndex) + ";" + sessQuery
            queryCount += 1
            if queryCount % 1000000 == 0:
                print("appended Session " + str(sessIndex) + ", Query " + str(queryIndex) + ", queryCount: " + str(queryCount))
                #ti.appendToFile(outputFile, output_str)
                #queryCount = 0
    ti.appendToFile(outputFile, output_str)
    print("appended Sessions and Queries for a queryCount: "+str(queryCount))

if __name__ == "__main__":
    # configDict = parseConfig.parseConfigFile("MINC_configFile.txt")
    configDict = parseConfig.parseConfigFile("MINC_configFile_updated_file_path.txt")
    #createConcurrentSessions(getConfig(configDict['QUERYSESSIONS']), getConfig(configDict['CONCURRENT_QUERY_SESSIONS']))
    #print("Completed concurrent session order creation")
    print("configDict['QUERYSESSIONS'] is ", configDict['QUERYSESSIONS'])
    assertConcurrentSessions(getConfig(configDict['QUERYSESSIONS']), getConfig(configDict['CONCURRENT_QUERY_SESSIONS']))
    print("Completed assertion of concurrent sessions")
