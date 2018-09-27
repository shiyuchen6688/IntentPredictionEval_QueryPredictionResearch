import sys
import os
import time, argparse
import ParseConfigFile as parseConfig
import QueryExecution as QExec
import re, gc

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

def checkFlagAndSetList(parseLevel, selectList, projectList, groupByList, havingList, tableList, selFlag, projFlag, grpFlag, havFlag, tabFlag, token):
    if projFlag[parseLevel]:
        if parseLevel not in projectList:
            projectList[parseLevel] = token
        else:
            projectList[parseLevel] = projectList[parseLevel] + " " + token
    elif selFlag[parseLevel]:
        if parseLevel not in selectList:
            selectList[parseLevel] = token
        else:
            selectList[parseLevel] = selectList[parseLevel] + " " + token
    elif grpFlag[parseLevel]:
        if parseLevel not in groupByList:
            groupByList[parseLevel] = token
        else:
            groupByList[parseLevel] = groupByList[parseLevel] + " " + token
    elif havFlag[parseLevel]:
        if parseLevel not in havingList:
            havingList[parseLevel] = token
        else:
            havingList[parseLevel] = havingList[parseLevel] + " " + token
    elif tabFlag[parseLevel]:
        if parseLevel not in tableList:
            tableList[parseLevel] = token
        else:
            tableList[parseLevel] = tableList[parseLevel] + " " + token
    return selectList, projectList, groupByList, havingList, tableList

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
                selectList, projectList, groupByList, havingList, tableList = checkFlagAndSetList(parseLevel,
                                                                                                  selectList,
                                                                                                  projectList,
                                                                                                  groupByList,
                                                                                                  havingList, tableList,
                                                                                                  selFlag, projFlag,
                                                                                                  grpFlag, havFlag,
                                                                                                  tabFlag, token)
            else:
                setTabFlagTrue(selFlag, projFlag, grpFlag, havFlag, tabFlag, parseLevel) # this is FROM followed by tablename(s)
        elif "WHERE" in token:
            setSelFlagTrue(selFlag, projFlag, grpFlag, havFlag, tabFlag, parseLevel)
        elif "GROUP" in token:
            setGrpFlagTrue(selFlag, projFlag, grpFlag, havFlag, tabFlag, parseLevel)
        elif "HAVING" in token:
            setHavFlagTrue(selFlag, projFlag, grpFlag, havFlag, tabFlag, parseLevel)
        else:
            selectList, projectList, groupByList, havingList, tableList = checkFlagAndSetList(parseLevel, selectList, projectList, groupByList, havingList, tableList, selFlag, projFlag, grpFlag, havFlag, tabFlag, token)
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
    cols = projectList[parseLevel].split(",")
    actualColList = []
    for col in cols:
        lenCol = len(col.split(" AS "))
        actualCol = ""
        for i in range(lenCol-1):
            actualCol += col.split(" AS ")[i]
            if i<lenCol-2:
                actualCol += " AS "
        actualColList.append(actualCol)
        colAlias = col.split(" AS ")[lenCol-1]
        colAliasList.append(colAlias)
    if projectList[parseLevel] not in sessQuery:
        print "Incorrect sessQuery with extra spaces !!"
        exit(0)
    tempQuery = sessQuery.replace(projectList[parseLevel], projectList[parseLevel] + " INTO TEMPTABLE")
    #tempQuery = "CREATE TABLE TEMPTABLE AS "+sessQuery

    tableAlias = tableList[parseLevel].split()[1] # this gives the alias
    #tempProjectList = projectList[parseLevel].replace(tableAlias, "TEMPTABLE")
    joinSelList = ""
    assert len(colAliasList) == len(actualColList)
    for colID in range(len(colAliasList)):
        #tempProjectList = tempProjectList.replace("AS "+colAlias, "AS TEMPTABLE_"+colAlias)
        #projectList[parseLevel] = projectList[parseLevel].replace(" "+colAlias, " "+tableAlias+"_"+colAlias)
        colAlias = colAliasList[colID]
        if colID == 0:
            joinSelList = joinSelList + "WHERE "
        else:
            joinSelList = joinSelList+" AND "
        joinSelList = joinSelList+actualColList[colID]+"=TEMPTABLE."+colAlias
    joinProjectList = projectList[parseLevel]+", "+tableAlias+".id" #ignored for now
    #joinProjectList = "DISTINCT "+tableAlias+".id"
    newQuery = tempQuery+";"+"SELECT "+joinProjectList+" FROM "+tableList[parseLevel]+", TEMPTABLE "+joinSelList
    return newQuery

def rewriteQuery(sessQuery, selectList, projectList, groupByList, havingList, tableList):
    parseLevel = 0
    #newQuery = None
    if "WHERE" not in sessQuery and ("HAVING" not in sessQuery or "HAVING (COUNT(1) > 0)" in sessQuery):
        return None  # either one of HAVING or WHERE must be in the sessQuery, else every tuple ends up being a witness
    elif "WHERE" in sessQuery and "GROUP BY" not in sessQuery: #HAVING may or may not be in the query, does not change anything
        if projectList[parseLevel] not in sessQuery:
            print "Incorrect sessQuery with extra spaces !!"
            exit(0)
        newQuery = sessQuery.replace(projectList[parseLevel], projectList[parseLevel] + ", id")  # we are projecting id as well
        #projectList[parseLevel] = projectList[parseLevel] + ", id"
        #numAttrsProjected = projectList[parseLevel].count(",") + 1
        if "HAVING (COUNT(1) > 0)" in newQuery:
            newQuery = newQuery.replace("HAVING (COUNT(1) > 0)", "GROUP BY id HAVING (COUNT(1) > 0)")
        #newQuery = sessQuery.replace(projectList[parseLevel], "DISTINCT id")  # we are projecting id as well
    elif "WHERE" in sessQuery and "GROUP BY" in sessQuery and "HAVING" not in sessQuery:
        if projectList[parseLevel] not in sessQuery:
            print "Incorrect sessQuery with extra spaces !!"
            exit(0)
        newQuery = sessQuery.replace(projectList[parseLevel], projectList[parseLevel] + ", id")  # we are projecting id as well
        projectList[parseLevel] = projectList[parseLevel] + ", id"
        #numAttrsProjected = projectList[parseLevel].count(",")+1
        if groupByList[parseLevel] not in newQuery:
            print "Incorrect newQuery without groupBy list !!"
            exit(0)
        #newQuery = newQuery.replace(groupByList[parseLevel], groupByList[parseLevel] + ", " + str(numAttrsProjected)) # we are grouping by id as well, via its index ID in the projections
        #groupByList[parseLevel] = groupByList[parseLevel] + ", " + str(numAttrsProjected)
        newQuery = newQuery.replace(groupByList[parseLevel], groupByList[parseLevel] + ", id")  # we are grouping by id as well, via its index ID in the projections
        groupByList[parseLevel] = groupByList[parseLevel] + ", id"
    elif "GROUP BY" in sessQuery and "HAVING" in sessQuery:
        # if MIN/MAX/AVG/SUM in sessQuery, every tuple is a witness
        aggrKeywords = ["MIN", "MAX", "AVG", "COUNT", "SUM"]
        if any(aggrKeyword in projectList[parseLevel] for aggrKeyword in aggrKeywords):
            return None
        newQuery = rewriteHavingGroupByComplexQuery(sessQuery, selectList, projectList, groupByList, havingList, tableList)
    return newQuery


def rewriteQueryForProvenance(sessQuery, configDict):
    # write grammar to parse query and then rewrite
    #sessQuery = ' '.join(sessQuery.split('\t'))
    #while '  ' in sessQuery:
        #sessQuery = sessQuery.replace('  ', ' ')
    sessQuery = ' '.join(sessQuery.split())
    if configDict['DATASET']=='NYCTaxiTrips':
        (sessQuery, selectList, projectList, groupByList, havingList, tableList) = parseNYCQuery(sessQuery)
        newQuery = rewriteQuery(sessQuery, selectList, projectList, groupByList, havingList, tableList)
        return newQuery

# the whole point of this parsing is to check if in the results we have a column called rowID. Else, we want to get rowID.
# Several possible cases:
#
def fetchRowIDs(sessQuery, configDict):
    rowIDs = []
    newQuery = rewriteQueryForProvenance(sessQuery, configDict)
    if newQuery is None:
        return (newQuery, None)
    if ";" in newQuery:  # happens for combined provenance queries generated for HAVING, GROUP BY combination
        tempQuery = newQuery.split(";")[0]
        QExec.executeQuery("drop table TEMPTABLE", configDict)
        QExec.executeQuery(tempQuery, configDict)
        print "successfully created temptable"
        #QExec.executeQuery("select * from temptable", configDict)
        newQuery = newQuery.split(";")[1]
    cur = QExec.executeQuery(newQuery, configDict) # without intent
    rowIDs = QExec.getRowIDs(cur)
    return (newQuery,rowIDs)

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
            for i in range(1,len(sessQueries)-1): # we need to ignore the empty query coming from the end of line semicolon ;
                sessQuery = sessQueries[i].split("~")[0]
                #sessQuery = "SELECT nyc_yellow_tripdata_2016_06_sample_1_percent.dropoff_latitude AS dropoff_latitude, nyc_yellow_tripdata_2016_06_sample_1_percent.dropoff_longitude AS dropoff_longitude, nyc_yellow_tripdata_2016_06_sample_1_percent.fare_amount AS fare_amount FROM public.nyc_yellow_tripdata_2016_06_sample_1_percent nyc_yellow_tripdata_2016_06_sample_1_percent GROUP BY 1, 2, 3 HAVING ((CAST(MIN(nyc_yellow_tripdata_2016_06_sample_1_percent.fare_amount) AS DOUBLE PRECISION) >= 11.999999999999879) AND (CAST(MIN(nyc_yellow_tripdata_2016_06_sample_1_percent.fare_amount) AS DOUBLE PRECISION) <= 14.00000000000014))"
                sessQuery = ' '.join(sessQuery.split())
                #(newQuery, rowIDs) = fetchRowIDs(sessQuery, configDict)
                newQuery = rewriteQueryForProvenance(sessQuery, configDict)
                print sessName+", Query "+str(i)+": \n"
                print "OrigQuery: "+sessQuery+"\n"
                if newQuery is not None:
                    print "Provenance Query: "+newQuery+"\n"
                else:
                    print "Provenance Query: None\n"