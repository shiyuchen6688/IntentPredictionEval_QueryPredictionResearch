import sys
import os
import time, argparse
from bitmap import BitMap
import ParseConfigFile as parseConfig
import QueryParser as qp
import TupleIntent as ti

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
    tableAlias = tableList[len(tableToks) - 1]
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

def createFragmentsFromAttrList(attrListDict, schemaDict, opString, configDict, tableList):
    resObjOp = None
    if opString == "TABLE":
        if configDict["BIT_OR_WEIGHTED_FRAGMENT"] == "BIT":
            resObjOp = BitMap("1")
        elif configDict["BIT_OR_WEIGHTED_FRAGMENT"] == "WEIGHTED":
            resObjOp = str(len(attrListDict)) # attrList comes as a dictionary with key as levelNumber
    elif attrListDict is not None:
        colDict = {}
        avgDict = {}
        minDict = {}
        maxDict = {}
        sumDict = {}
        countDict = {}
        aggrKeywords = ["AVG", "MIN", "MAX", "SUM", "COUNT"]
        for col in schemaDict:
            colIndex = schemaDict[col]
            for levelID in attrListDict:
                countCol = attrListDict[levelID].count(col)
                colDict = setColIndexInDict(colDict, configDict["BIT_OR_WEIGHTED_FRAGMENT"], countCol, colIndex)
                if opString == "PROJECT":
                    tableAlias = findTableAlias(tableList, levelID, configDict)
                    countAvg = attrListDict[levelID].count("AVG(" + tableAlias + "." + col)
                    avgDict = setColIndexInDict(avgDict, configDict["BIT_OR_WEIGHTED_FRAGMENT"], countAvg, colIndex)
                    countMin = attrListDict[levelID].count("MIN(" + tableAlias + "." + col)
                    minDict = setColIndexInDict(minDict, configDict["BIT_OR_WEIGHTED_FRAGMENT"], countMin, colIndex)
                    countMax = attrListDict[levelID].count("MAX(" + tableAlias + "." + col)
                    maxDict = setColIndexInDict(maxDict, configDict["BIT_OR_WEIGHTED_FRAGMENT"], countMax, colIndex)
                    countSum = attrListDict[levelID].count("SUM(" + tableAlias + "." + col)
                    sumDict = setColIndexInDict(sumDict, configDict["BIT_OR_WEIGHTED_FRAGMENT"], countSum, colIndex)
                    countCount = attrListDict[levelID].count("COUNT(" + tableAlias + "." + col)
                    countDict = setColIndexInDict(countDict, configDict["BIT_OR_WEIGHTED_FRAGMENT"], countCount, colIndex)
        resObjOp = parseOpDict(colDict, configDict["BIT_OR_WEIGHTED_FRAGMENT"], len(schemaDict))
        if opString == "PROJECT":
            resObjAvg = parseOpDict(avgDict, configDict["BIT_OR_WEIGHTED_FRAGMENT"], len(schemaDict))
            resObjMin = parseOpDict(minDict, configDict["BIT_OR_WEIGHTED_FRAGMENT"], len(schemaDict))
            resObjMax = parseOpDict(maxDict, configDict["BIT_OR_WEIGHTED_FRAGMENT"], len(schemaDict))
            resObjSum = parseOpDict(maxDict, configDict["BIT_OR_WEIGHTED_FRAGMENT"], len(schemaDict))
            resObjCount = parseOpDict(countDict, configDict["BIT_OR_WEIGHTED_FRAGMENT"], len(schemaDict))
            if configDict["BIT_OR_WEIGHTED_FRAGMENT"] == "BIT":
                resObjOp = BitMap.fromstring(resObjOp.tostring()+resObjAvg.tostring()+resObjMin.tostring()+resObjMax.tostring()+resObjSum.tostring()+resObjCount.tostring())
            elif configDict["BIT_OR_WEIGHTED_FRAGMENT"] == "WEIGHTED":
                resObjOp = resObjOp+resObjAvg+resObjMin+resObjMax+resObjSum+resObjCount
        return resObjOp

def createFragmentIntentRep(sessQuery, configDict):
    schemaDict = parseConfig.parseSchema(configDict['SCHEMA'])
    numCols = len(schemaDict)
    ### expected dimensionality is numCols * 11 -- 5 dims for AVG, MIN, MAX, SUM, COUNT, 6 dims for SELECT, PROJECT, GROUP BY, ORDER BY, LIMIT, HAVING ###
    numDims = numCols * 11
    resObj = BitMap(numDims)
    (sessQuery, selectList, projectList, groupByList, havingList, tableList, orderByList, limitList) = qp.parseNYCQuery(sessQuery)
    selObj = createFragmentsFromAttrList(selectList, schemaDict, "SELECT", configDict, tableList)
    projAggrObj = createFragmentsFromAttrList(projectList, schemaDict, "PROJECT", configDict, tableList)
    groupByObj = createFragmentsFromAttrList(groupByList, schemaDict, "GROUP BY", configDict, tableList)
    havingObj = createFragmentsFromAttrList(havingList, schemaDict, "HAVING", configDict, tableList)
    tableObj = createFragmentsFromAttrList(tableList, schemaDict, "TABLE", configDict, tableList)
    orderByObj = createFragmentsFromAttrList(orderByList, schemaDict, "ORDER BY", configDict, tableList)
    limitObj = createFragmentsFromAttrList(limitList, schemaDict, "LIMIT", configDict, tableList)
    if configDict["BIT_OR_WEIGHTED_FRAGMENT"] == "BIT":
        resObj = BitMap.fromstring(projAggrObj.tostring()+selObj.tostring()+groupByObj.tostring()+orderByObj.tostring()+havingObj.tostring()+limitObj.tostring()+tableObj.tostring())
    elif configDict["BIT_OR_WEIGHTED_FRAGMENT"] == "WEIGHTED":
        resObj = projAggrObj+selObj+groupByObj+orderByObj+havingObj+limitObj+tableObj
    return resObj

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("configFile.txt")
    if configDict["BIT_OR_WEIGHTED_FRAGMENT"] == "BIT":
        fragmentIntentSessionsFile = configDict['BIT_FRAGMENT_INTENT_SESSIONS']
    elif configDict["BIT_OR_WEIGHTED_FRAGMENT"] == "WEIGHTED":
        fragmentIntentSessionsFile = configDict['WEIGHTED_FRAGMENT_INTENT_SESSIONS']
    else:
        print "BIT_OR_WEIGHTED_FRAGMENT should be set either to BIT or WEIGHTED in the Config File !!"
        sys.exit(0)
    try:
        os.remove(fragmentIntentSessionsFile)
    except OSError:
        pass
    with open(configDict['QUERYSESSIONS']) as f:
        for line in f:
            sessQueries = line.split(";")
            sessName = sessQueries[0]
            for i in range(1, len(sessQueries) - 1):  # we need to ignore the empty query coming from the end of line semicolon ;
                sessQuery = sessQueries[i].split("~")[0]
                # sessQuery = "SELECT nyc_yellow_tripdata_2016_06_sample_1_percent.dropoff_latitude AS dropoff_latitude, nyc_yellow_tripdata_2016_06_sample_1_percent.dropoff_longitude AS dropoff_longitude, nyc_yellow_tripdata_2016_06_sample_1_percent.fare_amount AS fare_amount FROM public.nyc_yellow_tripdata_2016_06_sample_1_percent nyc_yellow_tripdata_2016_06_sample_1_percent GROUP BY 1, 2, 3 HAVING ((CAST(MIN(nyc_yellow_tripdata_2016_06_sample_1_percent.fare_amount) AS DOUBLE PRECISION) >= 11.999999999999879) AND (CAST(MIN(nyc_yellow_tripdata_2016_06_sample_1_percent.fare_amount) AS DOUBLE PRECISION) <= 14.00000000000014))"
                sessQuery = ' '.join(sessQuery.split())
                resObj = createFragmentIntentRep(sessQuery, configDict) #rowIDs passed should be None, else it won't fill up
                queryName = sessName+", Query "+str(i)
                outputIntentLine = queryName+"; OrigQuery: "+sessQuery+";"+str(resObj)
                ti.appendToFile(fragmentIntentSessionsFile,outputIntentLine)
                print "Generated fragment for "+queryName
