import sys
import os
import time, argparse
from bitmap import BitMap
import ParseConfigFile as parseConfig
from ParseConfigFile import getConfig
import QueryParser as qp
import TupleIntent as ti
import re

def setColIndexInDict(targetDict, bitOrWeighted, countOp, colIndex):
    if countOp > 0:
        if bitOrWeighted == "WEIGHTED":
            if colIndex in targetDict:
                targetDict[colIndex] = targetDict[colIndex]+countOp
            else:
                targetDict[colIndex] = countOp
        elif bitOrWeighted == "BIT":
            targetDict[colIndex] = 1
    return targetDict

def findTableAlias(tableList, levelID, configDict):
    tableToks = tableList[levelID].split(" ")
    tableAlias = tableToks[len(tableToks) - 1]
    if configDict['DATASET'] == "NYCTaxiTrips" and configDict['EXEC_SAMPLE']=="True" and tableAlias != "nyc_yellow_tripdata_2016_06_sample_1_percent":
        tableAlias = "nyc_yellow_tripdata_2016_06_sample_1_percent"
    elif configDict['DATASET'] == "NYCTaxiTrips" and configDict['EXEC_SAMPLE']!="True" and tableAlias != "nyc_yellow_tripdata_2016_06":
        tableAlias = "nyc_yellow_tripdata_2016_06"
    return tableAlias

def parseOpDict(opDict, bitOrWeighted, numAttrs):
    resObjOp =  None
    if bitOrWeighted == "BIT":
        resObjOp = BitMap(numAttrs)
        for colIndex in opDict:
            resObjOp.set(colIndex)
    elif bitOrWeighted == "WEIGHTED":
        resObjOp = ""
        for colIndex in range(numAttrs):
            if colIndex in opDict:
                resObjOp = resObjOp+str(opDict[colIndex])+";"
            else:
                resObjOp = resObjOp+"0;" # note that ; is the separator between dimensions for weighted vector
    return resObjOp

def trackColumnOrder(projectList, schemaDict):
    colOrderDict = {}
    for levelID in projectList:
        attrListStr = projectList[levelID]
        #attrList = attrListStr.split(",")
        attrList = re.split(r',\s*(?![^()]*\))', attrListStr)
        colOrderDict[levelID] = list()
        for attrStr in attrList:
            foundAttrStr = 0
            for col in schemaDict:
                colIndex = schemaDict[col]
                if col in attrStr:
                    foundAttrStr = 1
                    colOrderDict[levelID].append(colIndex)
            if foundAttrStr == 0:
                colOrderDict[levelID].append(-1) # this covers bad cases such as SUM(1), COUNT(*), AVG(1), MIN(1), MAX(1)
    return colOrderDict

def createFragmentsFromAttrList(attrListDict, schemaDict, opString, configDict, tableList, projectList):
    resObjOp = None
    numLimit = 0
    if opString == "LIMIT":
        if attrListDict is not None:
            for levelID in attrListDict:
                numLimit = numLimit+attrListDict[levelID].count(opString)
        if configDict["BIT_OR_WEIGHTED"] == "BIT":
            if numLimit > 0:
                resObjOp = BitMap.fromstring("1")
            else:
                resObjOp = BitMap.fromstring("0")
        elif configDict["BIT_OR_WEIGHTED"] == "WEIGHTED":
            resObjOp = str(numLimit)+";"
    elif opString == "TABLE":
        if configDict["BIT_OR_WEIGHTED"] == "BIT":
            resObjOp = BitMap.fromstring("1")
        elif configDict["BIT_OR_WEIGHTED"] == "WEIGHTED":
            resObjOp = str(len(attrListDict)) # attrListDict comes as a dictionary with key as levelNumber, table is trailing bit dimension so no semicolon at end
    elif attrListDict is None:
        if opString != "PROJECT":
            resObjOp = BitMap(len(schemaDict))
        else:
            resObjOp = BitMap(len(schemaDict)*6) # 1 for projection and 5 for AVG, MIN, MAX, SUM, COUNT
    elif attrListDict is not None:
        colDict = {}
        avgDict = {}
        minDict = {}
        maxDict = {}
        sumDict = {}
        countDict = {}
        aggrKeywords = ["AVG", "MIN", "MAX", "SUM", "COUNT"]
        colOrderDict = trackColumnOrder(projectList, schemaDict)
        for col in schemaDict:
            colIndex = schemaDict[col]
            for levelID in attrListDict:
                if opString == "GROUP BY" or opString == "ORDER BY":
                    colOrder = colOrderDict[levelID]
                    countCol = 0
                    groupByColIndices = attrListDict[levelID].split(",") # u get the group by or order by indices
                    actualGroupByIndexes = []
                    for grpByColIndex in groupByColIndices:
                        grpByColIndex = grpByColIndex.replace("BY ","")
                        if " " in grpByColIndex and ("ASC" in grpByColIndex or "DESC" in grpByColIndex):
                            grpByColIndex = grpByColIndex.split(" ")[0]
                        grpByColIndex = int(grpByColIndex) - 1 # starts from 1
                        #if grpByColIndex in colOrder:
                        actualGroupByIndex = colOrder[grpByColIndex]  #actualGroupByIndex could be -1 for cases such as SUM(1), COUNT(*), MIN(1), MAX(1), AVG(1) but it can't be found be in schemaDict, so no issues
                        actualGroupByIndexes.append(actualGroupByIndex)
                    if colIndex in actualGroupByIndexes:
                        countCol = 1  # within a single group by a col can occur only once
                else:
                    countCol = attrListDict[levelID].count(col)
                colDict = setColIndexInDict(colDict, configDict["BIT_OR_WEIGHTED"], countCol, colIndex)
                if opString == "PROJECT":
                    tableAlias = findTableAlias(tableList, levelID, configDict)
                    countAvg = attrListDict[levelID].count("AVG(" + tableAlias + "." + col)
                    if countAvg == 0 and "AVG("+tableAlias not in attrListDict[levelID]:
                        countAvg = attrListDict[levelID].count("AVG(")  # AVG(1) is covered as all columns
                    avgDict = setColIndexInDict(avgDict, configDict["BIT_OR_WEIGHTED"], countAvg, colIndex)
                    countMin = attrListDict[levelID].count("MIN(" + tableAlias + "." + col)
                    if countMin == 0 and "MIN("+tableAlias not in attrListDict[levelID]:
                        countMin = attrListDict[levelID].count("MIN(")  # MIN(1) is covered as all columns
                    minDict = setColIndexInDict(minDict, configDict["BIT_OR_WEIGHTED"], countMin, colIndex)
                    countMax = attrListDict[levelID].count("MAX(" + tableAlias + "." + col)
                    if countMax == 0 and "MAX("+tableAlias not in attrListDict[levelID]:
                        countMax = attrListDict[levelID].count("MAX(") # MAX(1) is covered as all columns
                    maxDict = setColIndexInDict(maxDict, configDict["BIT_OR_WEIGHTED"], countMax, colIndex)
                    countSum = attrListDict[levelID].count("SUM(" + tableAlias + "." + col)
                    if countSum == 0 and "SUM("+tableAlias not in attrListDict[levelID]:
                        countSum = attrListDict[levelID].count("SUM(")  # SUM(1) is covered as all columns
                    sumDict = setColIndexInDict(sumDict, configDict["BIT_OR_WEIGHTED"], countSum, colIndex)
                    countCount = attrListDict[levelID].count("COUNT(" + tableAlias + "." + col)
                    if countCount == 0 and "COUNT("+tableAlias not in attrListDict[levelID]:
                        countCount = attrListDict[levelID].count("COUNT(")  # COUNT(1) is covered as all columns, also COUNT(*)
                    countDict = setColIndexInDict(countDict, configDict["BIT_OR_WEIGHTED"], countCount, colIndex)
        resObjOp = parseOpDict(colDict, configDict["BIT_OR_WEIGHTED"], len(schemaDict))
        if opString == "PROJECT":
            resObjAvg = parseOpDict(avgDict, configDict["BIT_OR_WEIGHTED"], len(schemaDict))
            resObjMin = parseOpDict(minDict, configDict["BIT_OR_WEIGHTED"], len(schemaDict))
            resObjMax = parseOpDict(maxDict, configDict["BIT_OR_WEIGHTED"], len(schemaDict))
            resObjSum = parseOpDict(sumDict, configDict["BIT_OR_WEIGHTED"], len(schemaDict))
            resObjCount = parseOpDict(countDict, configDict["BIT_OR_WEIGHTED"], len(schemaDict))
            if configDict["BIT_OR_WEIGHTED"] == "BIT":
                resObjOp = BitMap.fromstring(resObjOp.tostring()+resObjAvg.tostring()+resObjMin.tostring()+resObjMax.tostring()+resObjSum.tostring()+resObjCount.tostring())
            elif configDict["BIT_OR_WEIGHTED"] == "WEIGHTED":
                resObjOp = resObjOp+resObjAvg+resObjMin+resObjMax+resObjSum+resObjCount
    return resObjOp

def createFragmentIntentRep(sessQuery, configDict):
    schemaDict = parseConfig.parseSchema(getConfig(configDict['SCHEMA']))
    numCols = len(schemaDict)
    ### expected dimensionality is numCols * 10 + 1 -- 5*numCols dims for AVG, MIN, MAX, SUM, COUNT, 5*numCols dims for SELECT, PROJECT, GROUP BY, ORDER BY, HAVING, 1 Dim for LIMIT
    # 1 for TableDim ###
    numExpectedDims = numCols * 10 + 2
    (sessQuery, selectList, projectList, groupByList, havingList, tableList, orderByList, limitList) = qp.parseNYCQuery(sessQuery)
    selObj = createFragmentsFromAttrList(selectList, schemaDict, "SELECT", configDict, tableList, projectList)
    projAggrObj = createFragmentsFromAttrList(projectList, schemaDict, "PROJECT", configDict, tableList, projectList)
    groupByObj = createFragmentsFromAttrList(groupByList, schemaDict, "GROUP BY", configDict, tableList, projectList)
    havingObj = createFragmentsFromAttrList(havingList, schemaDict, "HAVING", configDict, tableList, projectList)
    tableObj = createFragmentsFromAttrList(tableList, schemaDict, "TABLE", configDict, tableList, projectList)
    orderByObj = createFragmentsFromAttrList(orderByList, schemaDict, "ORDER BY", configDict, tableList, projectList)
    limitObj = createFragmentsFromAttrList(limitList, schemaDict, "LIMIT", configDict, tableList, projectList)
    if configDict["BIT_OR_WEIGHTED"] == "BIT":
        resObj = BitMap.fromstring(projAggrObj.tostring()+selObj.tostring()+groupByObj.tostring()+orderByObj.tostring()+havingObj.tostring()+limitObj.tostring()+tableObj.tostring())
        expectedPaddedBitCount = (6+4)*5+7+7 # reasoning is 19 bits get padded up to 24 and 1 bit gets padded up to 8, coz bitmaps are created in bytes not in bits
        assert resObj.size() == numExpectedDims+expectedPaddedBitCount
    elif configDict["BIT_OR_WEIGHTED"] == "WEIGHTED":
        resObj = projAggrObj+selObj+groupByObj+orderByObj+havingObj+limitObj+tableObj
        assert resObj.count(";")+1 == numExpectedDims
    return resObj

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("configFile.txt")
    if configDict["BIT_OR_WEIGHTED"] == "BIT":
        fragmentIntentSessionsFile = getConfig(configDict['BIT_FRAGMENT_INTENT_SESSIONS'])
    elif configDict["BIT_OR_WEIGHTED"] == "WEIGHTED":
        fragmentIntentSessionsFile = getConfig(configDict['WEIGHTED_FRAGMENT_INTENT_SESSIONS'])
    else:
        print("BIT_OR_WEIGHTED should be set either to BIT or WEIGHTED in the Config File !!")
        sys.exit(0)
    try:
        os.remove(fragmentIntentSessionsFile)
    except OSError:
        pass
    with open(getConfig(configDict['CONCURRENT_QUERY_SESSIONS'])) as f:
        for line in f:
            tokens = line.split(";")
            sessQueryName = tokens[0]
            sessQuery = tokens[1].strip()
            resObj = createFragmentIntentRep(sessQuery,
                                             configDict)  # rowIDs passed should be None, else it won't fill up
            outputIntentLine = sessQueryName + "; OrigQuery: " + sessQuery + ";" + str(resObj)
            ti.appendToFile(fragmentIntentSessionsFile, outputIntentLine)
            print("Generated fragment for " + sessQueryName)

