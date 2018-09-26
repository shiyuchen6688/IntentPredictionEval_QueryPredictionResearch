import sys
import os
import time, argparse
import ParseConfigFile as parseConfig

def setTabFlagTrue(selFlag, projFlag, grpFlag, havFlag, tabFlag, parseLevel):
    projFlag[parseLevel] = False
    selFlag[parseLevel] = False
    grpFlag[parseLevel] = False
    havFlag[parseLevel] = False
    tabFlag[parseLevel] = True

def setHavFlagTrue(selFlag, projFlag, grpFlag, havFlag, tabFlag, parseLevel):
    projFlag[parseLevel] = False
    selFlag[parseLevel] = False
    grpFlag[parseLevel] = False
    havFlag[parseLevel] = True
    tabFlag[parseLevel] = False

def setGrpFlagTrue(selFlag, projFlag, grpFlag, havFlag, tabFlag, parseLevel):
    projFlag[parseLevel] = False
    selFlag[parseLevel] = False
    grpFlag[parseLevel] = True
    havFlag[parseLevel] = False
    tabFlag[parseLevel] = False

def setSelFlagTrue(selFlag, projFlag, grpFlag, havFlag, tabFlag, parseLevel):
    projFlag[parseLevel] = False
    selFlag[parseLevel] = True
    grpFlag[parseLevel] = False
    havFlag[parseLevel] = False
    tabFlag[parseLevel] = False

def setProjFlagTrue(selFlag, projFlag, grpFlag, havFlag, tabFlag, parseLevel):
    projFlag[parseLevel] = True
    selFlag[parseLevel] = False
    grpFlag[parseLevel] = False
    havFlag[parseLevel] = False
    tabFlag[parseLevel] = False

def setAllFlagsFalse(selFlag, projFlag, grpFlag, havFlag, tabFlag, parseLevel):
    projFlag[parseLevel] = False
    selFlag[parseLevel] = False
    grpFlag[parseLevel] = False
    havFlag[parseLevel] = False
    tabFlag[parseLevel] = False

def parseQueryLevelWise(sessQuery, parseLevel, selectList, projectList, groupByList, havingList, tableList, selFlag, projFlag, grpFlag, havFlag, tabFlag, stackParenth, extractFieldNum):
    tokens = sessQuery.split() #accommodates one or more spaces
    stackParenth[parseLevel] = 0  # 0 opening  (
    extractFieldNum[parseLevel] = 0 # 0 EXTRACTS so far
    for token in tokens:
        if "EXTRACT" in token:
            extractFieldNum[parseLevel] = extractFieldNum[parseLevel] + 1
        if "(" in token:
            stackParenth[parseLevel] = stackParenth[parseLevel] + token.count("(")
        if ")" in token:
            stackParenth[parseLevel] = stackParenth[parseLevel] - token.count(")")
        if stackParenth < 0:
            parseLevel = parseLevel - 1 # finished a subquery
        if "SELECT" in token:
            parseLevel =parseLevel + 1
            setProjFlagTrue(selFlag, projFlag, grpFlag, havFlag, tabFlag, parseLevel)
            stackParenth[parseLevel] = 0 # 0 opening  (
            extractFieldNum[parseLevel] = 0 # 0 Extracts so far
        elif "FROM" in token:
            if extractFieldNum[parseLevel] > 0:
                extractFieldNum[parseLevel] = extractFieldNum[parseLevel] - 1  # this FROM is attached to EXTRACT and not to the table
            else:
                setTabFlagTrue(selFlag, projFlag, grpFlag, havFlag, tabFlag, parseLevel) # this is FROM followed by tablename(s)
        elif "WHERE" in token:
            setSelFlagTrue(selFlag, projFlag, grpFlag, havFlag, tabFlag, parseLevel)
        elif "GROUP BY" in token:
            setGrpFlagTrue(selFlag, projFlag, grpFlag, havFlag, tabFlag, parseLevel)
        elif "HAVING" in token:
            setHavFlagTrue(selFlag, projFlag, grpFlag, havFlag, tabFlag, parseLevel)
        elif projFlag[parseLevel] == 1:
            if parseLevel not in projectList:
                projectList[parseLevel] = token
            else:
                projectList[parseLevel] = projectList[parseLevel] + " " + token
        elif selFlag[parseLevel] == 1:
            if parseLevel not in selectList:
                selectList[parseLevel] = token
            else:
                selectList[parseLevel] = selectList[parseLevel] + " " + token
        elif grpFlag[parseLevel] == 1:
            if parseLevel not in groupByList:
                groupByList[parseLevel] = token
            else:
                groupByList[parseLevel] = groupByList[parseLevel] + " " + token
        elif havFlag[parseLevel] == 1:
            if parseLevel not in havingList:
                havingList[parseLevel] = token
            else:
                havingList[parseLevel] = havingList[parseLevel] + " " + token
        elif tabFlag[parseLevel] == 1:
            if parseLevel not in tableList:
                tableList[parseLevel] = token
            else:
                tableList[parseLevel] = tableList[parseLevel] + " " + token
    return selectList, projectList, groupByList, havingList, tableList

def parseNYCQuery(sessQuery):
    selectList = {}
    projectList = {}
    groupByList = {}
    havingList = {}
    tableList = {}
    selFlag = {}
    projFlag = {}
    grpFlag = {}
    havFlag = {}
    tabFlag = {}
    parseLevel = -1
    stackParenth = {}
    extractFieldNum = {}
    (selectList, projectList, groupByList, havingList, tableList) = parseQueryLevelWise(sessQuery, parseLevel, selectList, projectList, groupByList, havingList, tableList, selFlag, projFlag, grpFlag, havFlag, tabFlag, stackParenth, extractFieldNum)
    return (sessQuery, selectList, projectList, groupByList, havingList, tableList)

def rewriteHavingGroupByComplexQuery(sessQuery, selectList, projectList, groupByList, havingList, tableList):
    parseLevel = 0
    colAliasList = []
    cols = projectList[parseLevel].split()
    for col in cols:
        lenCol = col.split("AS")
        colAlias = col.split("AS")[lenCol-1]
        colAliasList.append(colAlias)
    tempQuery = sessQuery.replace(projectList[parseLevel], projectList[parseLevel] + " INTO TEMP")
    tableAlias = tableList[parseLevel].split()[1] # this gives the alias
    tempProjectList = projectList[parseLevel].replace(tableAlias, "TEMP")
    joinSelList = ""
    colCount = 0
    for colAlias in colAliasList:
        tempProjectList = tempProjectList.replace("AS "+colAlias, "AS TEMP."+colAlias)
        if colCount == 0:
            joinSelList = joinSelList + "WHERE "
        else:
            joinSelList = joinSelList+" AND "
        joinSelList = joinSelList+tableAlias+"."+colAlias+"=TEMP."+colAlias
    joinProjectList = projectList[parseLevel]+", "+tempProjectList
    newQuery = tempQuery+";"+"SELECT "+joinProjectList+" FROM "+tableList[parseLevel]+", TEMP "+joinSelList
    return newQuery

def rewriteQuery(sessQuery, selectList, projectList, groupByList, havingList, tableList):
    parseLevel = 0
    if "WHERE" not in sessQuery and ("HAVING" not in sessQuery or "HAVING (COUNT(1) > 0)" in sessQuery):
        return None  # either one of HAVING or WHERE must be in the sessQuery, else every tuple ends up being a witness
    elif "WHERE" in sessQuery and "HAVING" not in sessQuery and "GROUP BY" not in sessQuery:
        newQuery = sessQuery.replace(projectList[parseLevel], projectList[parseLevel] + ", id")  # we are projecting id as well
    elif "WHERE" in sessQuery and "GROUP BY" in sessQuery and "HAVING" not in sessQuery:
        newQuery = sessQuery.replace(projectList[parseLevel], projectList[parseLevel] + ", id")  # we are projecting id as well
        projectList[parseLevel] = projectList[parseLevel] + ", id"
        numAttrsProjected = projectList[parseLevel].count(",")+1
        newQuery = newQuery.replace(groupByList[parseLevel], groupByList[parseLevel] + ", " + str(numAttrsProjected)) # we are grouping by id as well, via its index ID in the projections
        groupByList[parseLevel] = groupByList[parseLevel] + ", " + str(numAttrsProjected)
    elif "GROUP BY" in sessQuery and "HAVING" in sessQuery:
        # if MIN/MAX/AVG/SUM in sessQuery, every tuple is a witness
        aggrKeywords = ["MIN", "MAX", "AVG", "COUNT", "SUM"]
        if any(aggrKeyword in projectList for aggrKeyword in aggrKeywords):
            return None
        newQuery = rewriteHavingGroupByComplexQuery(sessQuery, selectList, projectList, groupByList, havingList, tableList)
    return newQuery


def rewriteQueryForProvenance(sessQuery, configDict):
    # write grammer to parse query and then rewrite
    if configDict['DATASET']=='NYCTaxiTrips':
        (sessQuery, selectList, projectList, groupByList, havingList, tableList) = parseNYCQuery(sessQuery)
        newQuery = rewriteQuery(sessQuery, selectList, projectList, groupByList, havingList, tableList)
        return newQuery

# the whole point of this parsing is to check if in the results we have a column called rowID. Else, we want to get rowID.
# Several possible cases:
#
def fetchRowIDs(sessQuery, rows, configDict):
    rowIDs = []
    if 'id' in rows[0]:
        for row in rows:
            rowIDs.append(row['id'])
    else:
        newQuery = rewriteQueryForProvenance(sessQuery, configDict)
        if newQuery is None:
            return None

    return rowIDs

# rowIDs = []
#    for row in rows:
#        rowIDs.append(row['id'])
#    del rows
#    gc.collect()

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("configFile.txt")
    with open(configDict['QUERYSESSIONS']) as f:
        for line in f:
            sessQueries = line.split(";")
            sessName = sessQueries[0]
            for i in range(1,len(sessQueries)):
                sessQuery = sessQueries[i]
                newQuery = rewriteQueryForProvenance(sessQuery, configDict)
                print "Session "+sessName+", Query "+str(i)+": \n"
                print "OrigQuery: "+sessQuery+"\n"
                print "Provenance Query: "+newQuery+"\n"