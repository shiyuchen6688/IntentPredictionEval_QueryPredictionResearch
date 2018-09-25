import sys
import os
import time, argparse
# the whole point of this parsing is to check if in the results we have a column called rowID. Else, we want to get rowID.
# Several possible cases:
#
def fetchRowIDs(sessQuery, rows, configDict):
    rowIDs = []
    if 'ID' in rows[0]:
        for row in rows:
            rowIDs.append(row['ID'])
    else:
        sessQuery = rewriteQueryForProvenance(sessQuery, configDict)
    return rowIDs

# rowIDs = []
#    for row in rows:
#        rowIDs.append(row['id'])
#    del rows
#    gc.collect()

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

def parseQueryLevelWise(sessQuery, parseLevel, selectList, projectList, groupByList, havingList, tableList, selFlag, projFlag, grpFlag, havFlag, tabFlag, stackParenth):
    tokens = sessQuery.split() #accommodates one or more spaces
    stackParenth[parseLevel] = 0  # 0 opening  (
    for token in tokens:
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
        elif "FROM" in token:
            setTabFlagTrue(selFlag, projFlag, grpFlag, havFlag, tabFlag, parseLevel)
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
    return selectList, projectList, groupByList, havingList, tableList, stackParenth

def rewriteQuery(sessQuery, selectList, projectList, groupByList, havingList, tableList, stackParenth):
    

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
    (selectList, projectList, groupByList, havingList, tableList, stackParenth) = parseQueryLevelWise(sessQuery, parseLevel, selectList, projectList, groupByList, havingList, tableList, selFlag, projFlag, grpFlag, havFlag, tabFlag, stackParenth)
    rewriteQuery(sessQuery, selectList, projectList, groupByList, havingList, tableList, stackParenth)

def rewriteQueryForProvenance(sessQuery, configDict):
    # write grammer to parse query and then rewrite
    if configDict['DATASET']=='NYCTaxiTrips':
        parseNYCQuery(sessQuery)