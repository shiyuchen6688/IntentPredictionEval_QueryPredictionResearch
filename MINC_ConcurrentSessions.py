import sys
import os
import time, argparse
import QueryParser as qp
import ParseConfigFile as parseConfig
import random
import TupleIntent as ti

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

def countQueries(inputFile):
    sessionQueryCountDict = {}
    queryCount = 0
    with open(inputFile) as f:
        for line in f:
            if 'Query' in line and line.startswith('\t'):
                line = cleanQuery(line)
                sessTokens = line.split()
                assert sessTokens[1] == 'Query'
                sessIndex = sessTokens[0] # the ID is used as a string
                if sessIndex in sessionQueryCountDict:
                    sessionQueryCountDict[sessIndex] += 1
                else:
                    sessionQueryCountDict[sessIndex] = 1
                queryCount +=1
                #if queryCount % 10000 == 0:
                #    print "Query count so far: "+str(queryCount)
    print "Total Query Count: " + str(queryCount)
    return sessionQueryCountDict

def retrieveQueryFromFile(inputFile, coveredSessQueries, sessIndex):
    with open(inputFile) as f:
        for line in f:
            line = cleanQuery(line)
            sessTokens = line.split()
            curSessIndex = sessTokens[0]
            assert sessTokens[1] == 'Query'
            if sessIndex == curSessIndex:
                # here we assume queryIndex starts from 1, count of queries covered so far gives the index of the next uncovered query
                # but sessionName is the 0th token, so we need to add a 1 to get the query index
                if sessIndex not in coveredSessQueries:
                    queryIndex = 1
                else:
                    queryIndex = coveredSessQueries[sessIndex] + 1
                sessQuery = " ".join(sessTokens[2:])
                return (sessQuery,queryIndex)

def createConcurrentSessions(inputFile, outputFile):
    sessionQueryCountDict = countQueries(inputFile)
    try:
        os.remove(outputFile)
    except OSError:
        pass
    keyList = list(sessionQueryCountDict.keys()) # this actually clones the keys into a new python object keyList, not the same as pointing to the existing list
    coveredSessQueries = {} # key is sessionID and value is the query count covered
    queryCount = 0
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
            queryCount+=1
            if queryCount % 100000 == 0:
                print "appended Session "+str(sessIndex)+", Query "+str(queryIndex)
        else:
            keyList.remove(sessIndex)
            print "ERROR: invalid SessIndex !!"+str(sessIndex)
            sys.exit(0)

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("MINC_configFile.txt")
    createConcurrentSessions(configDict['QUERYSESSIONS'], configDict['CONCURRENT_QUERY_SESSIONS'])
    print "Completed concurrent session order creation"
