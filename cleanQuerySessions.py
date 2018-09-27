import sys
import os
import time, argparse
import QueryParser as qp

def appendToFile(outputFile, outputLine):
    with open(outputFile, 'a') as outFile:
        outFile.write(outputLine+"\n")

def cleanEachQuery(sessQuery):
    sessQuery = ' '.join(sessQuery.split())
    (sessQuery, selectList, projectList, groupByList, havingList, tableList) = qp.parseNYCQuery(sessQuery)
    for parseLevel in projectList:
        cols = projectList[parseLevel].split(",")
        for col in cols:
            if " AS " in col:
                lenCol = len(col.split("AS "))
                colAlias = col.split("AS ")[lenCol-1]
                newColAlias = colAlias.replace("(","_").replace(")","_").replace(" ","_")
                sessQuery = sessQuery.replace(colAlias, newColAlias)
    return sessQuery

def cleanQuerySessions(log, outputFile):
    try:
        os.remove(outputFile)
    except OSError:
        pass
    sessionIndex = 0
    outputLine = "Session " + str(sessionIndex) + ";"
    with open(log) as f:
        for line in f:
            if "------------" in line:
                appendToFile(outputFile, outputLine)
                sessionIndex = sessionIndex + 1
                outputLine = "Session "+str(sessionIndex)+";"
            elif line.startswith("SELECT"):
                cleanedQuery = cleanEachQuery(line)
                outputLine = outputLine+cleanedQuery.strip()+";"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", help="Input Query Log File", type=str, required=True)
    parser.add_argument("-output", help="Output Query Log File", type=str, required=True)
    args = parser.parse_args()
    cleanQuerySessions(args.input, args.output)



