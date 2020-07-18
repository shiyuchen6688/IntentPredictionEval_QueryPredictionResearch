from __future__ import division
import sys
import os
import time
import QueryExecution as QExec
import QueryRecommender as QR
import ReverseEnggQueries_selOpConst
import CreateSQLFromIntentVec_selOpConst
from bitmap import BitMap
import CFCosineSim
import TupleIntent as ti
import ParseConfigFile as parseConfig
from ParseConfigFile import getConfig
import pickle
import argparse
from pandas import DataFrame
from openpyxl import load_workbook
import pandas as pd
import operator
import collections
import mysql.connector
import unicodedata

def updateArrWithDictEntry(arr, evalOpsObjDict, epIndex, numOpQueryCountDict):
    try:
        arr.append(float(evalOpsObjDict[epIndex])/float(numOpQueryCountDict[epIndex]))
    except:
        arr.append("")
    return

def updateArrWithCountEntry(arr, numOpQueryCountDict, key):
    try:
        arr.append(numOpQueryCountDict[key])
    except:
        arr.append(0.0)
    return

def updateAggMetricWithDictEntry(avgMetric, evalOpsObjDict, epIndex):
    try:
        avgMetric += float(evalOpsObjDict[epIndex])
    except:
        avgMetric += 0.0
    return avgMetric

def fetchAlgoName(evalOpsObj):
    algoName = evalOpsObj.configDict['ALGORITHM']
    if evalOpsObj.configDict['ALGORITHM'] == 'RNN' or evalOpsObj.configDict['ALGORITHM'] == 'LSTM' or \
                    evalOpsObj.configDict['ALGORITHM'] == 'GRU':
        if evalOpsObj.configDict['RNN_PREDICT_NOVEL_QUERIES'] == 'True':
            algoName = "Novel" + algoName
        elif evalOpsObj.configDict['RNN_PREDICT_NOVEL_QUERIES'] == 'False':
            algoName = "Historical" + algoName
    elif evalOpsObj.configDict['ALGORITHM'] == 'QLEARNING':
        if evalOpsObj.configDict['QL_BOOLEAN_NUMERIC_REWARD'] == 'NUMERIC':
            algoName = "QLNumeric"
        elif evalOpsObj.configDict['QL_BOOLEAN_NUMERIC_REWARD'] == 'BOOLEAN':
            algoName = "QLBoolean"
    return algoName

def plotQueryTypeDistribution(evalOpsObj):
    episodes = []
    numSelectQueryType = []
    numInsertQueryType = []
    numUpdateQueryType = []
    numDeleteQueryType = []
    totalSelect = 0.0
    totalInsert = 0.0
    totalUpdate = 0.0
    totalDelete = 0.0
    for key in sorted(evalOpsObj.meanReciprocalRank.keys()):
        episodes.append(key)
        updateArrWithCountEntry(numSelectQueryType, evalOpsObj.numSelectQueryType, key)
        updateArrWithCountEntry(numInsertQueryType, evalOpsObj.numInsertQueryType, key)
        updateArrWithCountEntry(numUpdateQueryType, evalOpsObj.numUpdateQueryType, key)
        updateArrWithCountEntry(numDeleteQueryType, evalOpsObj.numDeleteQueryType, key)
        totalSelect = updateAggMetricWithDictEntry(totalSelect, evalOpsObj.numSelectQueryType, key)
        totalInsert = updateAggMetricWithDictEntry(totalInsert, evalOpsObj.numInsertQueryType, key)
        totalUpdate = updateAggMetricWithDictEntry(totalUpdate, evalOpsObj.numUpdateQueryType, key)
        totalDelete = updateAggMetricWithDictEntry(totalDelete, evalOpsObj.numDeleteQueryType, key)
    df = DataFrame(
        {'episodes': episodes, '# SELECT': numSelectQueryType, '# INSERT': numInsertQueryType, '# UPDATE': numUpdateQueryType,
         '# DELETE': numDeleteQueryType})
    algoName = fetchAlgoName(evalOpsObj)
    outputOpWiseQualityFileName = getConfig(evalOpsObj.configDict['OUTPUT_DIR']) + "/OpWiseExcel/QTDist" + \
                                  algoName
    df.to_excel(outputOpWiseQualityFileName + ".xlsx", sheet_name='sheet1', index=False)
    totalSelList = []
    totalSelList.append(totalSelect)
    totalInsList = []
    totalInsList.append(totalInsert)
    totalUpdList = []
    totalUpdList.append(totalUpdate)
    totalDelList = []
    totalDelList.append(totalDelete)
    df = DataFrame({'totalSEL': totalSelList, 'totalINS': totalInsList, 'totalUPD': totalUpdList, 'totalDEL': totalDelList})
    outputOpWiseQualityFileName = getConfig(evalOpsObj.configDict['OUTPUT_DIR']) + "/OpWiseExcel/TotalQT_" + \
                                  algoName
    df.to_excel(outputOpWiseQualityFileName + ".xlsx", sheet_name='sheet2', index=False)
    return


def plotMeanReciprocalRank(evalOpsObj):
    episodes = []
    meanReciprocalRank = []
    numEpQueries = []
    avgMRR = 0.0
    for key in sorted(evalOpsObj.meanReciprocalRank.keys()):
        episodes.append(key)
        updateArrWithCountEntry(numEpQueries, evalOpsObj.numEpQueries, key)
        updateArrWithDictEntry(meanReciprocalRank, evalOpsObj.meanReciprocalRank, key, evalOpsObj.numEpQueries)
        avgMRR = updateAggMetricWithDictEntry(avgMRR, evalOpsObj.meanReciprocalRank, key)
    df = DataFrame(
        {'episodes': episodes, 'meanReciprocalRank': meanReciprocalRank, 'numMRRQueries': numEpQueries})
    algoName = fetchAlgoName(evalOpsObj)
    outputOpWiseQualityFileName = getConfig(evalOpsObj.configDict['OUTPUT_DIR']) + "/OpWiseExcel/Output_MRR_" + algoName
    df.to_excel(outputOpWiseQualityFileName + ".xlsx", sheet_name='sheet1', index=False)
    totalQueryCount = float(sum(numEpQueries))
    if totalQueryCount > 0.0:
        avgMRR = avgMRR / totalQueryCount
    avgMRRList = []
    avgMRRList.append(avgMRR)
    totalQueryCountList = []
    totalQueryCountList.append(totalQueryCount)
    df = DataFrame({'avgMRR': avgMRRList, 'numMRRQueries': totalQueryCountList})
    outputOpWiseQualityFileName = getConfig(evalOpsObj.configDict['OUTPUT_DIR']) + "/OpWiseExcel/AggrOutput_MRR_" + algoName
    df.to_excel(outputOpWiseQualityFileName + ".xlsx", sheet_name='sheet2', index=False)
    return

def plotOp(evalOpsP, evalOpsR, evalOpsF, numOpQueryCountDict, evalOpsObj, opString):
    episodes = []
    resP = []
    resR = []
    resF = []
    numEpQueries = []
    avgResP = 0.0
    avgResR = 0.0
    avgResF = 0.0
    for key in sorted(evalOpsObj.queryTypeP.keys()):
        episodes.append(key)
        updateArrWithCountEntry(numEpQueries, numOpQueryCountDict, key)
        updateArrWithDictEntry(resP, evalOpsP, key, numOpQueryCountDict)
        updateArrWithDictEntry(resR, evalOpsR, key, numOpQueryCountDict)
        updateArrWithDictEntry(resF, evalOpsF, key, numOpQueryCountDict)
        avgResP = updateAggMetricWithDictEntry(avgResP, evalOpsP, key)
        avgResR = updateAggMetricWithDictEntry(avgResR, evalOpsR, key)
        avgResF = updateAggMetricWithDictEntry(avgResF, evalOpsF, key)
    headerP = evalOpsObj.configDict['ALGORITHM']+"(P)"
    headerR = evalOpsObj.configDict['ALGORITHM']+"(R)"
    headerF = evalOpsObj.configDict['ALGORITHM'] + "(F)"
    headerQ = 'num'+opString+'Queries'
    df = DataFrame(
        {'episodes': episodes, headerP: resP, headerR: resR, headerF: resF, headerQ:numEpQueries})
    algoName = fetchAlgoName(evalOpsObj)
    outputOpWiseQualityFileName = getConfig(evalOpsObj.configDict['OUTPUT_DIR']) + "/OpWiseExcel/Output_" + opString + "_" + \
                                  algoName
    df.to_excel(outputOpWiseQualityFileName + ".xlsx", sheet_name='sheet1', index=False)
    totalQueryCount = float(sum(numEpQueries))
    if totalQueryCount > 0.0:
        avgResP = avgResP / totalQueryCount
        avgResR = avgResR / totalQueryCount
        avgResF = avgResF / totalQueryCount
    avgResPList = []
    avgResPList.append(avgResP)
    avgResRList = []
    avgResRList.append(avgResR)
    avgResFList = []
    avgResFList.append(avgResF)
    totalQueryCountList = []
    totalQueryCountList.append(totalQueryCount)
    df = DataFrame({headerP: avgResPList, headerR: avgResRList, headerF: avgResFList, headerQ: totalQueryCountList})
    outputOpWiseQualityFileName = getConfig(
        evalOpsObj.configDict['OUTPUT_DIR']) + "/OpWiseExcel/AggrOutput_" + opString + "_" + algoName
    df.to_excel(outputOpWiseQualityFileName + ".xlsx", sheet_name='sheet1', index=False)
    return

def plotEvalMetricsOpWise(evalOpsObj):
    plotQueryTypeDistribution(evalOpsObj)
    plotMeanReciprocalRank(evalOpsObj)
    plotOp(evalOpsObj.queryTypeP, evalOpsObj.queryTypeR, evalOpsObj.queryTypeF, evalOpsObj.numQueryTypeQueries, evalOpsObj, "QUERYTYPE")
    plotOp(evalOpsObj.tablesP, evalOpsObj.tablesR, evalOpsObj.tablesF, evalOpsObj.numTablesQueries, evalOpsObj, "TABLES")
    plotOp(evalOpsObj.projColsP, evalOpsObj.projColsR, evalOpsObj.projColsF, evalOpsObj.numProjColsQueries, evalOpsObj,
           "PROJ")
    plotOp(evalOpsObj.avgColsP, evalOpsObj.avgColsR, evalOpsObj.avgColsF, evalOpsObj.numAvgColsQueries, evalOpsObj,
           "AVG")
    plotOp(evalOpsObj.minColsP, evalOpsObj.minColsR, evalOpsObj.minColsF, evalOpsObj.numMinColsQueries, evalOpsObj,
           "MIN")
    plotOp(evalOpsObj.maxColsP, evalOpsObj.maxColsR, evalOpsObj.maxColsF, evalOpsObj.numMaxColsQueries, evalOpsObj,
           "MAX")
    plotOp(evalOpsObj.sumColsP, evalOpsObj.sumColsR, evalOpsObj.sumColsF, evalOpsObj.numSumColsQueries, evalOpsObj,
           "SUM")
    plotOp(evalOpsObj.countColsP, evalOpsObj.countColsR, evalOpsObj.countColsF, evalOpsObj.numCountColsQueries, evalOpsObj,
           "COUNT")
    plotOp(evalOpsObj.selPredsP, evalOpsObj.selPredsR, evalOpsObj.selPredsF, evalOpsObj.numSelPredsQueries, evalOpsObj,
           "SEL")
    plotOp(evalOpsObj.condSelPredsP, evalOpsObj.condSelPredsR, evalOpsObj.condSelPredsF, evalOpsObj.numCondSelPredsQueries, evalOpsObj,
           "CONDSEL")
    plotOp(evalOpsObj.groupByColsP, evalOpsObj.groupByColsR, evalOpsObj.groupByColsF, evalOpsObj.numGroupByColsQueries, evalOpsObj,
           "GROUPBY")
    plotOp(evalOpsObj.orderByColsP, evalOpsObj.orderByColsR, evalOpsObj.orderByColsF, evalOpsObj.numOrderByColsQueries, evalOpsObj,
           "ORDERBY")
    plotOp(evalOpsObj.havingColsP, evalOpsObj.havingColsR, evalOpsObj.havingColsF, evalOpsObj.numHavingColsQueries, evalOpsObj,
           "HAVING")
    plotOp(evalOpsObj.limitP, evalOpsObj.limitR, evalOpsObj.limitF, evalOpsObj.numLimitColsQueries, evalOpsObj,
           "LIMIT")
    plotOp(evalOpsObj.joinPredsP, evalOpsObj.joinPredsR, evalOpsObj.joinPredsF, evalOpsObj.numJoinPredsColsQueries, evalOpsObj,
           "JOIN")
    plotOp(evalOpsObj.condJoinPredsP, evalOpsObj.condJoinPredsR, evalOpsObj.condJoinPredsF, evalOpsObj.numCondJoinPredsColsQueries,
           evalOpsObj,
           "CONDJOIN")
    return

class evalOps:
    def __init__(self, configFileName, logFile):
        self.configFileName = configFileName
        self.configDict = parseConfig.parseConfigFile(configFileName)
        self.logFile = logFile
        self.curEpisode = 0
        self.numSelectQueryType = {}
        self.numInsertQueryType = {}
        self.numUpdateQueryType = {}
        self.numDeleteQueryType = {}
        self.numEpQueries = {}
        self.curQueryIndex = -1
        self.meanReciprocalRank = {}
        self.episode = {}
        self.queryTypeP = {}
        self.queryTypeR = {}
        self.queryTypeF = {}
        self.numQueryTypeQueries = {}
        self.tablesP = {}
        self.tablesR = {}
        self.tablesF = {}
        self.numTablesQueries = {}
        self.projColsP = {}
        self.projColsR = {}
        self.projColsF = {}
        self.numProjColsQueries = {}
        self.avgColsP = {}
        self.avgColsR = {}
        self.avgColsF = {}
        self.numAvgColsQueries = {}
        self.minColsP = {}
        self.minColsR = {}
        self.minColsF = {}
        self.numMinColsQueries = {}
        self.maxColsP = {}
        self.maxColsR = {}
        self.maxColsF = {}
        self.numMaxColsQueries = {}
        self.sumColsP = {}
        self.sumColsR = {}
        self.sumColsF = {}
        self.numSumColsQueries = {}
        self.countColsP = {}
        self.countColsR = {}
        self.countColsF = {}
        self.numCountColsQueries = {}
        self.selPredsP = {}
        self.selPredsR = {}
        self.selPredsF = {}
        self.numSelPredsQueries = {}
        self.condSelPredsP = {}
        self.condSelPredsR = {}
        self.condSelPredsF = {}
        self.numCondSelPredsQueries = {}
        self.groupByColsP = {}
        self.groupByColsR = {}
        self.groupByColsF = {}
        self.numGroupByColsQueries = {}
        self.orderByColsP = {}
        self.orderByColsR = {}
        self.orderByColsF = {}
        self.numOrderByColsQueries = {}
        self.havingColsP = {}
        self.havingColsR = {}
        self.havingColsF = {}
        self.numHavingColsQueries = {}
        self.limitP = {}
        self.limitR = {}
        self.limitF = {}
        self.numLimitColsQueries = {}
        self.joinPredsP = {}
        self.joinPredsR = {}
        self.joinPredsF = {}
        self.numJoinPredsColsQueries = {}
        self.condJoinPredsP = {}
        self.condJoinPredsR = {}
        self.condJoinPredsF = {}
        self.numCondJoinPredsColsQueries = {}

def createSQLFragmentDict(intentSessionFile, schemaDicts, configDict):
    SQLFragmentDict = {}
    count = 0
    with open(intentSessionFile) as f:
        for line in f:
            if count > int(configDict['RNN_SUSTENANCE_TRAIN_LIMIT']):
                f.close()
                return SQLFragmentDict
            (sqlQuery, curQueryIntent) = QR.retrieveQueryAndIntent(line, configDict)
            intentObj = CreateSQLFromIntentVec_selOpConst.regenerateSQL(None, curQueryIntent, schemaDicts)
            SQLFragStr = CreateSQLFromIntentVec_selOpConst.createSQLString(intentObj)
            # the following may overwrite key.value pairs but that is fine
            SQLFragmentDict[SQLFragStr] = sqlQuery
            count+=1
    f.close()
    return SQLFragmentDict

def printSQLFragmentDict(SQLFragmentDict):
    count = 0
    for key in SQLFragmentDict.keys():
        print "<key,value> pair for SQL query #"+str(count)
        print key
        print SQLFragmentDict[key]
        print "================================================"
        count+=1
    return

class evalExec:
    def __init__(self, configFileName, logFile):
        self.configFileName = configFileName
        self.configDict = parseConfig.parseConfigFile(configFileName)
        self.logFile = logFile
        self.curEpisode = 0
        self.schemaDicts = ReverseEnggQueries_selOpConst.readSchemaDicts(self.configDict)
        self.colTypeDict = ReverseEnggQueries_selOpConst.readColDict(getConfig(self.configDict['MINC_COL_TYPES']))
        assert self.configDict['BORROW_OR_RECONSTRUCT_QUERY'] == "BORROW" or self.configDict['BORROW_OR_RECONSTRUCT_QUERY'] == "RECONSTRUCT"
        if self.configDict['BORROW_OR_RECONSTRUCT_QUERY'] == "BORROW":
            try:
                self.SQLFragmentDict = QR.readFromPickleFile(getConfig(self.configDict['PICKLE_TEMP_OUTPUT_DIR'])+"SQLFragmentDict.pickle")
                print "Read SQLFragmentDict"
            except:
                print "Create SQLFragmentDict"
                self.intentSessionFile = QR.fetchIntentFileFromConfigDict(self.configDict)
                self.SQLFragmentDict = createSQLFragmentDict(self.intentSessionFile, self.schemaDicts, self.configDict)
                QR.writeToPickleFile(getConfig(self.configDict['PICKLE_TEMP_OUTPUT_DIR'])+"SQLFragmentDict.pickle", self.SQLFragmentDict)
                print "len(SQLFragmentDict): " + str(len(self.SQLFragmentDict))
                printSQLFragmentDict(self.SQLFragmentDict)

class nextActualOps:
    def __init__(self):
        self.queryType = None
        self.tables = None
        self.projCols = None
        self.avgCols = None
        self.minCols = None
        self.maxCols = None
        self.sumCols = None
        self.countCols = None
        self.selCols = None
        self.selPredOps = None
        self.selPredColRangeBins = None
        self.groupByCols = None
        self.orderByCols = None
        self.havingCols = None
        self.limit = None
        self.joinPreds = None

def parseLineAddOp(line, actualOrPredObj):
    if line.startswith("Query Type"):
        actualOrPredObj.queryType = line.strip().split(": ")[1]
    elif line.startswith("Limit"):
        actualOrPredObj.limit = line.strip().split(": ")[1]
    elif line.startswith("Tables"):
        actualOrPredObj.tables = eval(line.strip().split(": ")[1])
    elif line.startswith("Projected"):
        actualOrPredObj.projCols = eval(line.strip().split(": ")[1])
    elif line.startswith("AVG"):
        actualOrPredObj.avgCols = eval(line.strip().split(": ")[1])
    elif line.startswith("MIN"):
        actualOrPredObj.minCols = eval(line.strip().split(": ")[1])
    elif line.startswith("MAX"):
        actualOrPredObj.maxCols = eval(line.strip().split(": ")[1])
    elif line.startswith("SUM"):
        actualOrPredObj.sumCols = eval(line.strip().split(": ")[1])
    elif line.startswith("COUNT"):
        actualOrPredObj.countCols = eval(line.strip().split(": ")[1])
    elif line.startswith("SEL Columns"):
        actualOrPredObj.selCols = eval(line.strip().split(": ")[1])
    elif line.startswith("SEL PRED Ops"):
        actualOrPredObj.selPredOps = eval(line.strip().split(": ")[1])
    elif line.startswith("SEL PRED ColRangeBins"):
        actualOrPredObj.selPredColRangeBins = eval(line.strip().split(": ")[1])
    elif line.startswith("GROUP"):
        actualOrPredObj.groupByCols = eval(line.strip().split(": ")[1])
    elif line.startswith("ORDER"):
        actualOrPredObj.orderByCols = eval(line.strip().split(": ")[1])
    elif line.startswith("HAVING"):
        actualOrPredObj.havingCols = eval(line.strip().split(": ")[1])
    elif line.startswith("JOIN"):
        actualOrPredObj.joinPreds = eval(line.strip().split(": ")[1])
    return

def updateMetricDict(metricDict, key, val):
    if key not in metricDict:
        metricDict[key] = val
    else:
        metricDict[key] = float(metricDict[key]+val)
    return

def computeOpF1(predOpList, actualOpList):
    if predOpList is None and actualOpList is not None:
        return (0.0, 0.0, 0.0)
    elif predOpList is not None and actualOpList is not None:
        TP = len(set(predOpList).intersection(set(actualOpList)))
        FP = len(set(predOpList) - set(actualOpList))
        FN = len(set(actualOpList) - set(predOpList))
        P = float(TP)/float(TP+FP)
        R = float(TP)/float(TP+FN)
        if P == 0.0 or R == 0.0:
            F = 0.0
        else:
            F = 2*P*R / float(P+R)
        return (P, R, F)
    else:
        return (None, None, None)

def updateOpMetrics(P, R, F, evalOpsP, evalOpsR, evalOpsF, evalOpsQueryCountDict, evalOpsObj):
    if P is not None and R is not None and F is not None:
        if evalOpsObj.curEpisode not in evalOpsQueryCountDict:
            evalOpsQueryCountDict[evalOpsObj.curEpisode] = 1
        else:
            evalOpsQueryCountDict[evalOpsObj.curEpisode] += 1
        updateMetricDict(evalOpsP, evalOpsObj.curEpisode, P)
        updateMetricDict(evalOpsR, evalOpsObj.curEpisode, R)
        updateMetricDict(evalOpsF, evalOpsObj.curEpisode, F)
    return

def computeRelevantJoinPreds(accTables, predictedOrActualJoinPreds):
    if predictedOrActualJoinPreds is None:
        return None
    relJoinPreds = []
    for joinPred in predictedOrActualJoinPreds:
        leftTable = joinPred.split(",")[0].split(".")[0]
        rightTable = joinPred.split(",")[1].split(".")[0]
        if leftTable in accTables and rightTable in accTables:
            relJoinPreds.append(joinPred)
    if len(relJoinPreds) == 0:
        return None
    return relJoinPreds

def computeRelevantSelColOpColRangeBins(accTables, predOrActualColOpColRangeBins):
    if predOrActualColOpColRangeBins is None:
        return None
    relColOpColRangeBins = []
    for colOpColRangeBin in predOrActualColOpColRangeBins:
        tableName = colOpColRangeBin.split(".")[0]
        if tableName in accTables:
            relColOpColRangeBins.append(colOpColRangeBin)
    if len(relColOpColRangeBins) == 0:
        return None
    return relColOpColRangeBins

def compUpdateOpMetrics(predOpList, actualOpList, evalOpsP, evalOpsR, evalOpsF, evalOpsQueryCountDict, evalOpsObj):
    (P,R,F) = computeOpF1(predOpList, actualOpList)
    updateOpMetrics(P, R, F, evalOpsP, evalOpsR, evalOpsF, evalOpsQueryCountDict, evalOpsObj)
    return

def compUpdateSelMetrics(predOpsObj, nextActualOpsObj, evalOpsObj):
    try:
        predOpList = predOpsObj.selCols
        predSelOps = predOpsObj.selPredOps
        predSelColRangeBins = predOpsObj.selPredColRangeBins
        if predSelOps is not None and len(predSelOps) > 0:
            predOpList = list(set().union(predOpList, predSelOps))
        if predSelColRangeBins is not None and len(predSelColRangeBins) > 0:
            predOpList = list(set().union(predOpList,predSelColRangeBins))
        actualOpList = nextActualOpsObj.selCols
        actualSelOps = nextActualOpsObj.selPredOps
        actualSelColRangeBins = nextActualOpsObj.selPredColRangeBins
        if actualSelOps is not None and len(actualSelOps) > 0:
            actualOpList = list(set().union(actualOpList, actualSelOps))
        if actualSelColRangeBins is not None and len(actualSelColRangeBins) > 0:
            actualOpList = list(set().union(actualOpList, actualSelColRangeBins))
        compUpdateOpMetrics(predOpList, actualOpList, evalOpsObj.selPredsP, evalOpsObj.selPredsR, evalOpsObj.selPredsF,
                            evalOpsObj.numSelPredsQueries, evalOpsObj)
    except:
        pass
    return

def compUpdateCondSelMetrics(predOpsObj, nextActualOpsObj, evalOpsObj):
    try:
        if evalOpsObj.tablesF[evalOpsObj.curEpisode] == 1.0 and evalOpsObj.curEpisode in evalOpsObj.selPredsP \
                and evalOpsObj.curEpisode in evalOpsObj.selPredsR and evalOpsObj.curEpisode in evalOpsObj.selPredsF:
            updateOpMetrics(evalOpsObj.selPredsP[evalOpsObj.curEpisode], evalOpsObj.selPredsR[evalOpsObj.curEpisode],
                            evalOpsObj.selPredsF[evalOpsObj.curEpisode], evalOpsObj.condSelPredsP, evalOpsObj.condSelPredsR,
                            evalOpsObj.condSelPredsF, evalOpsObj.numCondSelPredsQueries, evalOpsObj)
        elif evalOpsObj.tablesF[evalOpsObj.curEpisode] > 0.0 and evalOpsObj.curEpisode in evalOpsObj.selPredsP \
                and evalOpsObj.curEpisode in evalOpsObj.selPredsR and evalOpsObj.curEpisode in evalOpsObj.selPredsF: # partial overlap of tables
            accTables = list(set(predOpsObj.tables).intersection(set(nextActualOpsObj.tables)))
            if len(accTables) > 0:
                relPredOpList = computeRelevantSelColOpColRangeBins(accTables, predOpsObj.selCols)
                relPredOps = computeRelevantSelColOpColRangeBins(accTables, predOpsObj.selPredOps)
                relPredColRangeBins = computeRelevantSelColOpColRangeBins(accTables, predOpsObj.selPredColRangeBins)
                if relPredOps is not None and len(relPredOps) > 0:
                    relPredOpList = list(set().union(relPredOpList, relPredOps))
                if relPredColRangeBins is not None and len(relPredColRangeBins) > 0:
                    relPredOpList = list(set().union(relPredOpList, relPredColRangeBins))
                relActualOpList = computeRelevantSelColOpColRangeBins(accTables, nextActualOpsObj.selCols)
                relActualOps = computeRelevantSelColOpColRangeBins(accTables, nextActualOpsObj.selPredOps)
                relActualColRangeBins = computeRelevantSelColOpColRangeBins(accTables, nextActualOpsObj.selPredColRangeBins)
                if relActualOps is not None and len(relActualOps) > 0:
                    relActualOpList = list(set().union(relActualOpList, relActualOps))
                if relActualColRangeBins is not None and len(relActualColRangeBins) > 0:
                    relActualOpList = list(set().union(relActualOpList, relActualColRangeBins))
                compUpdateOpMetrics(relPredOpList, relActualOpList, evalOpsObj.condSelPredsP, evalOpsObj.condSelPredsR,
                                    evalOpsObj.condSelPredsF, evalOpsObj.numCondSelPredsQueries, evalOpsObj)
    except:
        pass
    return

def compUpdateCondJoinMetrics(predOpsObj, nextActualOpsObj, evalOpsObj):
    try:
        if evalOpsObj.tablesF[evalOpsObj.curEpisode] == 1.0 and evalOpsObj.curEpisode in evalOpsObj.joinPredsP \
                and evalOpsObj.curEpisode in evalOpsObj.joinPredsR and evalOpsObj.curEpisode in evalOpsObj.joinPredsF:
            updateOpMetrics(evalOpsObj.joinPredsP[evalOpsObj.curEpisode], evalOpsObj.joinPredsR[evalOpsObj.curEpisode],
                            evalOpsObj.joinPredsF[evalOpsObj.curEpisode], evalOpsObj.condJoinPredsP, evalOpsObj.condJoinPredsR,
                            evalOpsObj.condJoinPredsF, evalOpsObj.numCondJoinPredsColsQueries, evalOpsObj)
        elif evalOpsObj.tablesF[evalOpsObj.curEpisode] > 0.0 and evalOpsObj.curEpisode in evalOpsObj.joinPredsP \
                and evalOpsObj.curEpisode in evalOpsObj.joinPredsR and evalOpsObj.curEpisode in evalOpsObj.joinPredsF: # partial overlap of tables
            accTables = list(set(predOpsObj.tables).intersection(set(nextActualOpsObj.tables)))
            if len(accTables) > 0:
                relPredictedJoinPreds = computeRelevantJoinPreds(accTables, predOpsObj.joinPreds)
                relActualJoinPreds = computeRelevantJoinPreds(accTables, nextActualOpsObj.joinPreds)
                compUpdateOpMetrics(relPredictedJoinPreds, relActualJoinPreds, evalOpsObj.condJoinPredsP, evalOpsObj.condJoinPredsR,
                                    evalOpsObj.condJoinPredsF, evalOpsObj.numCondJoinPredsQueries, evalOpsObj)
    except:
        pass
    return

def computeF1(evalOpsObj, predOpsObj, nextActualOpsObj):
    if nextActualOpsObj.queryType == "select":
        updateMetricDict(evalOpsObj.numSelectQueryType, evalOpsObj.curEpisode, 1.0)
    elif nextActualOpsObj.queryType == "insert":
        updateMetricDict(evalOpsObj.numInsertQueryType, evalOpsObj.curEpisode, 1.0)
    elif nextActualOpsObj.queryType == "update":
        updateMetricDict(evalOpsObj.numUpdateQueryType, evalOpsObj.curEpisode, 1.0)
    elif nextActualOpsObj.queryType == "delete":
        updateMetricDict(evalOpsObj.numDeleteQueryType, evalOpsObj.curEpisode, 1.0)
    if predOpsObj.queryType == nextActualOpsObj.queryType:
        updateOpMetrics(1.0, 1.0, 1.0, evalOpsObj.queryTypeP, evalOpsObj.queryTypeR,
                        evalOpsObj.queryTypeF, evalOpsObj.numQueryTypeQueries, evalOpsObj)
    elif predOpsObj.queryType != nextActualOpsObj.queryType:
        updateOpMetrics(0.0, 0.0, 0.0, evalOpsObj.queryTypeP, evalOpsObj.queryTypeR,
                        evalOpsObj.queryTypeF, evalOpsObj.numQueryTypeQueries, evalOpsObj)
    if nextActualOpsObj.limit is not None:
        if predOpsObj.limit == nextActualOpsObj.limit:
            updateOpMetrics(1.0, 1.0, 1.0, evalOpsObj.limitP, evalOpsObj.limitR,
                            evalOpsObj.limitF, evalOpsObj.numLimitColsQueries, evalOpsObj)
        else:
            updateOpMetrics(0.0, 0.0, 0.0, evalOpsObj.limitP, evalOpsObj.limitR,
                            evalOpsObj.limitF, evalOpsObj.numLimitColsQueries, evalOpsObj)
    compUpdateOpMetrics(predOpsObj.tables, nextActualOpsObj.tables, evalOpsObj.tablesP,
                        evalOpsObj.tablesR, evalOpsObj.tablesF, evalOpsObj.numTablesQueries, evalOpsObj)
    compUpdateOpMetrics(predOpsObj.projCols, nextActualOpsObj.projCols, evalOpsObj.projColsP,
                        evalOpsObj.projColsR, evalOpsObj.projColsF, evalOpsObj.numProjColsQueries, evalOpsObj)
    compUpdateOpMetrics(predOpsObj.avgCols, nextActualOpsObj.avgCols, evalOpsObj.avgColsP,
                        evalOpsObj.avgColsR, evalOpsObj.avgColsF, evalOpsObj.numAvgColsQueries, evalOpsObj)
    compUpdateOpMetrics(predOpsObj.minCols, nextActualOpsObj.minCols, evalOpsObj.minColsP,
                        evalOpsObj.minColsR, evalOpsObj.minColsF, evalOpsObj.numMinColsQueries, evalOpsObj)
    compUpdateOpMetrics(predOpsObj.maxCols, nextActualOpsObj.maxCols, evalOpsObj.maxColsP,
                        evalOpsObj.maxColsR, evalOpsObj.maxColsF, evalOpsObj.numMaxColsQueries, evalOpsObj)
    compUpdateOpMetrics(predOpsObj.sumCols, nextActualOpsObj.sumCols, evalOpsObj.sumColsP,
                        evalOpsObj.sumColsR, evalOpsObj.sumColsF, evalOpsObj.numSumColsQueries, evalOpsObj)
    compUpdateOpMetrics(predOpsObj.countCols, nextActualOpsObj.countCols, evalOpsObj.countColsP,
                        evalOpsObj.countColsR, evalOpsObj.countColsF, evalOpsObj.numCountColsQueries, evalOpsObj)
    compUpdateOpMetrics(predOpsObj.groupByCols, nextActualOpsObj.groupByCols, evalOpsObj.groupByColsP,
                        evalOpsObj.groupByColsR, evalOpsObj.groupByColsF, evalOpsObj.numGroupByColsQueries, evalOpsObj)
    compUpdateOpMetrics(predOpsObj.orderByCols, nextActualOpsObj.orderByCols, evalOpsObj.orderByColsP,
                        evalOpsObj.orderByColsR, evalOpsObj.orderByColsF, evalOpsObj.numOrderByColsQueries, evalOpsObj)
    compUpdateOpMetrics(predOpsObj.havingCols, nextActualOpsObj.havingCols, evalOpsObj.havingColsP,
                        evalOpsObj.havingColsR, evalOpsObj.havingColsF, evalOpsObj.numHavingColsQueries, evalOpsObj)
    compUpdateOpMetrics(predOpsObj.joinPreds, nextActualOpsObj.joinPreds, evalOpsObj.joinPredsP,
                        evalOpsObj.joinPredsR, evalOpsObj.joinPredsF, evalOpsObj.numJoinPredsColsQueries, evalOpsObj)
    '''
    compUpdateOpMetrics(predOpsObj.selCols, nextActualOpsObj.selCols, evalOpsObj.selColsP,
                        evalOpsObj.selColsR, evalOpsObj.selColsF, evalOpsObj.numSelColsQueries, evalOpsObj)
    '''
    compUpdateSelMetrics(predOpsObj, nextActualOpsObj, evalOpsObj)
    compUpdateCondSelMetrics(predOpsObj, nextActualOpsObj, evalOpsObj)
    compUpdateCondJoinMetrics(predOpsObj, nextActualOpsObj, evalOpsObj)
    return

            
def createEvalMetricsOpWise(evalOpsObj):
    prevEpisode = -1
    rank = float("-inf")
    nextActualOpsObj = None
    predOpsObj = None
    with open(evalOpsObj.logFile) as f:
        for line in f:
            if line.startswith("#Episodes"):
                evalOpsObj.curEpisode = int(line.strip().split(";")[0].split(":")[1])
                numTokens = len(line.strip().split(";"))
                rank = int(line.strip().split(";")[numTokens-3].split(":")[1])
                if rank == -1: # this can happen when all predicted queries are equally bad
                    rank = 0
                assert rank >= 0 and rank < int(evalOpsObj.configDict['TOP_K'])
                MRR = float(1.0) / float(rank+1)
                if evalOpsObj.curEpisode != prevEpisode:
                    evalOpsObj.numEpQueries[evalOpsObj.curEpisode] = 1
                    assert evalOpsObj.curEpisode not in evalOpsObj.meanReciprocalRank
                    evalOpsObj.meanReciprocalRank[evalOpsObj.curEpisode] = MRR
                else:
                    evalOpsObj.numEpQueries[evalOpsObj.curEpisode] += 1
                    evalOpsObj.meanReciprocalRank[evalOpsObj.curEpisode] = (evalOpsObj.meanReciprocalRank[evalOpsObj.curEpisode] + MRR)
            elif line.startswith("Actual SQL"):
                evalOpsObj.curQueryIndex = -100
                nextActualOpsObj = nextActualOps()
            elif line.startswith("Predicted SQL Ops"):
                substrTokens = line.strip().split(":")[0].split(" ")
                evalOpsObj.curQueryIndex = int(substrTokens[len(substrTokens)-1])
                if evalOpsObj.curQueryIndex == rank:
                    predOpsObj = nextActualOps()
            elif line.startswith("---") and predOpsObj is not None and evalOpsObj is not None:
                computeF1(evalOpsObj, predOpsObj, nextActualOpsObj)
            elif evalOpsObj.curQueryIndex == -100:
                parseLineAddOp(line, nextActualOpsObj)
            elif evalOpsObj.curQueryIndex == rank:
                parseLineAddOp(line, predOpsObj)
            prevEpisode = evalOpsObj.curEpisode
    return evalOpsObj

def appendProjCols(predictedQuery, predOpsObj):
    for i in range(len(predOpsObj.projCols)):
        colName = predOpsObj.projCols[i]
        colToAppend = colName.replace("'","")
        if predOpsObj.groupByCols is not None and colName in predOpsObj.groupByCols:
            colToAppend = colToAppend # do nothing, do not check for further stuff
        elif predOpsObj.avgCols is not None and colName in predOpsObj.avgCols:
            colToAppend = "AVG("+colToAppend+")"
        elif predOpsObj.minCols is not None and colName in predOpsObj.minCols:
            colToAppend = "MIN("+colToAppend+")"
        elif predOpsObj.maxCols is not None and colName in predOpsObj.maxCols:
            colToAppend = "MAX("+colToAppend+")"
        elif predOpsObj.sumCols is not None and colName in predOpsObj.sumCols:
            colToAppend = "SUM("+colToAppend+")"
        elif predOpsObj.countCols is not None and colName in predOpsObj.countCols:
            colToAppend = "COUNT("+colToAppend+")"
        predictedQuery += " "+colToAppend
        if i<len(predOpsObj.projCols)-1:
            predictedQuery += ","
    return predictedQuery

def appendTables(predictedQuery, predOpsObj):
    predictedQuery += " FROM"
    for i in range(len(predOpsObj.tables)):
        tabName = predOpsObj.tables[i].replace("'","")
        predictedQuery += " "+tabName
        if i < len(predOpsObj.tables)-1:
            predictedQuery += ","
    return predictedQuery

def findColType(tableName, colName, evalExecObj):
    offset = evalExecObj.schemaDicts.colDict[tableName].index(colName)
    colType = evalExecObj.colTypeDict[tableName][offset]
    return colType

def findQuoteIncl(selCol, evalExecObj):
    selTabName = selCol.split(".")[0]
    selColName = selCol.split(".")[1]
    selColType = findColType(selTabName, selColName, evalExecObj)
    includeQuote = 1
    if 'int' in selColType or 'double' in selColName or 'float' in selColType:
        includeQuote = 0
    return includeQuote

def quoteModify(selConst, includeQuote):
    if includeQuote == 0:
        return selConst
    else:
        #print "Including quote"
        return "'" +selConst + "'"

def appendSelPreds(predictedQuery, predOpsObj, evalExecObj):
    if predOpsObj.selCols is None:
        return predictedQuery
    strToAppend = ""
    for i in range(len(predOpsObj.selCols)):
        if strToAppend != "":
            if i == 1: # you are in iteration 1 when you append for iteration 0
                predictedQuery += strToAppend # WHERE has already been added
            elif i > 1:
                predictedQuery += " AND " + strToAppend
            strToAppend = ""
        selCol = predOpsObj.selCols[i].replace("'","")
        includeQuote = findQuoteIncl(selCol, evalExecObj)
        for selPredOp in predOpsObj.selPredOps:
            if strToAppend != "":
                break
            if selCol in selPredOp:
                opStr = selPredOp.replace(selCol,"").replace(".","")
                opSym = None
                if opStr == "eq":
                    opSym = "="
                elif opStr == "neq":
                    opSym = "<>"
                elif opStr == "lt":
                    opSym = "<"
                elif opStr == "gt":
                    opSym = ">"
                elif opStr == "leq":
                    opSym = "<="
                elif opStr == "geq":
                    opSym = ">="
                elif opStr == "LIKE":
                    opSym = "LIKE"
                else:
                    continue
                for selColBin in predOpsObj.selPredColRangeBins:
                    if strToAppend != "":
                        break
                    if selCol in selColBin:
                        constBin = selColBin.replace(selCol,"").strip()
                        if constBin.startswith("."):
                            constBinToks = constBin.replace(".","").split("%")
                            assert len(constBinToks) == 2
                            if constBinToks[0] != "NULL" and constBinToks[1] != "NULL":
                                if constBinToks[0] == constBinToks[1]:
                                    strToAppend = selCol + " " +opSym + " " + quoteModify(constBinToks[0], includeQuote)
                                elif constBinToks[0] != constBinToks[1]:
                                    if opSym == "=":
                                        strToAppend = selCol + " >= " + quoteModify(constBinToks[0], includeQuote) + " AND " + selCol +" <= "+ quoteModify(constBinToks[1], includeQuote)
                                    elif opSym == "<=" or opSym == "<":
                                        strToAppend = selCol + " " + opSym + " " + quoteModify(constBinToks[1], includeQuote)
                                    elif opSym == ">=" or opSym == ">":
                                        strToAppend = selCol + " " + opSym + " " + quoteModify(constBinToks[0], includeQuote)
                                    elif opSym == "<>":
                                        strToAppend = selCol + "<>" + quoteModify(constBinToks[0], includeQuote) + "AND" + selCol + "<>" + quoteModify(constBinToks[1], includeQuote)
                                    elif opSym == "LIKE":
                                        strToAppend = "(" + selCol + "LIKE" + quoteModify(constBinToks[0], includeQuote) + "OR" + selCol + "LIKE" + quoteModify(constBinToks[1], includeQuote) + ")"
    if strToAppend != "":
        if len(predOpsObj.selCols) == 1:  # you are in iteration 1 when you append for iteration 0
            predictedQuery += strToAppend  # WHERE has already been added
        elif len(predOpsObj.selCols) > 1:
            predictedQuery += " AND " + strToAppend
    return predictedQuery

def appendJoinPreds(predictedQuery, predOpsObj):
    if predOpsObj.joinPreds is None:
        return predictedQuery
    strToAppend = ""
    for i in range(len(predOpsObj.joinPreds)):
        if strToAppend != "":
            if i == 1: # you are in iteration 1 when you append for iteration 0
                predictedQuery += strToAppend # WHERE has already been added
            elif i > 1:
                predictedQuery += " AND " + strToAppend
            strToAppend = ""
        joinPred = predOpsObj.joinPreds[i].replace("'","")
        leftTabCol = joinPred.split(",")[0]
        rightTabCol = joinPred.split(",")[1]
        strToAppend = leftTabCol + "=" + rightTabCol
    if strToAppend != "":
        if len(predOpsObj.joinPreds) == 1:  # you are in iteration 1 when you append for iteration 0
            predictedQuery += strToAppend  # WHERE has already been added
        elif len(predOpsObj.joinPreds) > 1:
            predictedQuery += " AND " + strToAppend
    return predictedQuery

def appendGrpByPred(predictedQuery, predOpsObj):
    if predOpsObj.groupByCols is None:
        return predictedQuery
    predictedQuery += " GROUP BY "
    for i in range(len(predOpsObj.groupByCols)):
        grpByCol = predOpsObj.groupByCols[i].replace("'", "")
        predictedQuery += grpByCol
        if i<len(predOpsObj.projCols)-1:
            predictedQuery += ", "
    return predictedQuery

def appendOrderByPred(predictedQuery, predOpsObj):
    if predOpsObj.orderByCols is None:
        return predictedQuery
    predictedQuery += " ORDER BY "
    for i in range(len(predOpsObj.orderByCols)):
        sortCol = predOpsObj.orderByCols[i].replace("'", "")
        predictedQuery += sortCol
        if i<len(predOpsObj.orderByCols)-1:
            predictedQuery += ", "
    return predictedQuery

def createPredictedQuery(evalExecObj, predOpsObj):
    predictedQuery = predOpsObj.queryType.replace("'","")
    if predOpsObj.projCols is not None:
        predictedQuery = appendProjCols(predictedQuery, predOpsObj)
    if predOpsObj.tables is not None:
        predictedQuery = appendTables(predictedQuery, predOpsObj)
    if predOpsObj.selCols is not None or predOpsObj.joinPreds is not None:
        predictedQuery += " WHERE "
        predictedQuery = appendSelPreds(predictedQuery, predOpsObj, evalExecObj)
        predictedQuery = appendJoinPreds(predictedQuery, predOpsObj)
        if predictedQuery.endswith(" WHERE "):
            predictedQuery = predictedQuery.replace(" WHERE ", "")
    if predOpsObj.groupByCols is not None:
        predictedQuery = appendGrpByPred(predictedQuery, predOpsObj)
    if predOpsObj.orderByCols is not None:
        predictedQuery = appendOrderByPred(predictedQuery, predOpsObj)
    return predictedQuery
'''
        self.queryType = None
        self.tables = None
        self.projCols = None
        self.avgCols = None
        self.minCols = None
        self.maxCols = None
        self.sumCols = None
        self.countCols = None
        self.selCols = None
        self.selPredOps = None
        self.selPredColRangeBins = None
        self.groupByCols = None
        self.orderByCols = None
        self.havingCols = None
        self.limit = None
        self.joinPreds = None
'''

def borrowQueryIfPossible(evalExecObj, predOpsObj, nextQuery):
    predictedSQLFragStr = CreateSQLFromIntentVec_selOpConst.createSQLString(predOpsObj)
    borrowedQuery = False
    try:
        predictedQuery = evalExecObj.SQLFragmentDict[predictedSQLFragStr]
        borrowedQuery = True
    except:
        predictedQuery = createPredictedQuery(evalExecObj, predOpsObj)
    return (borrowedQuery, predictedQuery)

def extractCols(cursor):
    num_fields = len(cursor.description)
    field_names = [i[0] for i in cursor.description]
    return (num_fields, field_names)

def computeColF1(nextQueryCursor, predictedQueryCursor):
    (num_next_fields, nextQueryCols) = extractCols(nextQueryCursor)
    (num_predicted_fields, predictedQueryCols) = extractCols(predictedQueryCursor)
    TP_cols = list(set(nextQueryCols) & set(predictedQueryCols))
    FP_cols = list(set(predictedQueryCols) - set(TP_cols))
    FN_cols = list(set(nextQueryCols) - set(TP_cols))
    if len(TP_cols) == 0:
        return (0.0, 0.0, 0.0, TP_cols, predictedQueryCols, nextQueryCols)
    col_prec = float(len(TP_cols))/float(len(TP_cols) + len(FP_cols))
    col_rec = float(len(TP_cols))/float(len(TP_cols) + len(FN_cols))
    col_F1 = float(2*col_prec*col_rec) / float(col_prec+col_rec)
    #print col_F1
    #print TP_cols
    #print predictedQueryCols
    #print nextQueryCols
    return (col_F1, col_prec, col_rec, TP_cols, predictedQueryCols, nextQueryCols)

def find_matching_indices(TP_cols, predictedQueryCols, nextQueryCols):
    if TP_cols is None or len(TP_cols) == 0:
        return (None, None)
    pred_indices = []
    next_indices = []
    for tp_col in TP_cols:
        pred_index = predictedQueryCols.index(tp_col)
        pred_indices.append(pred_index)
        next_index = nextQueryCols.index(tp_col)
        next_indices.append(next_index)
    return (pred_indices, next_indices)

def computeTupSet(queryRes, indices):
    resultSet = set()
    for row in queryRes:
        result = ""
        for i in range(len(indices)):
            colIndex = indices[i]
            try:
                result += str(row[colIndex])
            except:
                result += unicodedata.normalize('NFKD', row[colIndex]).encode('ascii', 'ignore')
            if i < len(indices)-1:
                result += ","
            resultSet.add(result)
    return resultSet

def computeTupF1(predictedQueryRes, nextQueryRes, pred_indices, next_indices):
    nextTupSet = computeTupSet(nextQueryRes, next_indices)
    predTupSet = computeTupSet(predictedQueryRes, pred_indices)
    TP_tups = list(nextTupSet & predTupSet)
    FP_tups = list(predTupSet - set(TP_tups))
    FN_tups = list(nextTupSet - set(TP_tups))
    if len(TP_tups) == 0:
        return (0.0, 0.0, 0.0)
    tup_prec = float(len(TP_tups)) / float(len(TP_tups) + len(FP_tups))
    tup_rec = float(len(TP_tups)) / float(len(TP_tups) + len(FN_tups))
    tup_F1 = float(2*tup_prec*tup_rec)/float(tup_prec+tup_rec)
    return (tup_F1, tup_prec, tup_rec)

def execF1(evalExecObj, predOpsObj, predictedQuery, nextQuery):
    (nextQueryCursor, nextQueryRes) = QExec.executeMINCQueryCursor(nextQuery, evalExecObj.configDict)
    (predictedQueryCursor, predictedQueryRes) = QExec.executeMINCQueryCursor(predictedQuery, evalExecObj.configDict)
    if nextQueryRes is None and predictedQueryRes is None:
        return (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    elif nextQueryRes is None or predictedQueryRes is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    elif len(list(nextQueryRes)) == 0 and len(list(predictedQueryRes)) > 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    elif len(list(nextQueryRes)) > 0 and len(list(predictedQueryRes)) == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    (col_F1, col_prec, col_rec, TP_cols, predictedQueryCols, nextQueryCols) = computeColF1(nextQueryCursor, predictedQueryCursor)
    if len(list(nextQueryRes)) == 0 and len(list(predictedQueryRes)) == 0:
        return (col_F1, col_prec, col_rec, 1.0, 1.0, 1.0, 0.2*col_F1+0.8, 0.2*col_prec+0.8, 0.2*col_rec+0.8)
    (pred_indices, next_indices) = find_matching_indices(TP_cols, predictedQueryCols, nextQueryCols)
    if pred_indices is not None and next_indices is not None:
        (tup_F1, tup_prec, tup_rec) = computeTupF1(predictedQueryRes, nextQueryRes, pred_indices, next_indices)
    else:
        tup_F1 = 0.0
        tup_prec = 0.0
        tup_rec = 0.0
    total_F1 = col_F1 * 0.2 + tup_F1 * 0.8
    total_prec = col_prec * 0.2 + tup_prec * 0.8
    total_rec = col_rec * 0.2 + tup_rec * 0.8
    return (col_F1, col_prec, col_rec, tup_F1, tup_prec, tup_rec, total_F1, total_prec, total_rec)

def computeExecF1(evalExecObj, predOpsObj, nextQuery):
    borrow_or_reconstruct = evalExecObj.configDict['BORROW_OR_RECONSTRUCT_QUERY']
    assert borrow_or_reconstruct == "BORROW" or borrow_or_reconstruct == "RECONSTRUCT"
    borrowedQuery = False
    if borrow_or_reconstruct == "BORROW":
        (borrowedQuery, predictedQuery) = borrowQueryIfPossible(evalExecObj, predOpsObj, nextQuery)
    else:
        CreateSQLFromIntentVec_selOpConst.printSQLOps(predOpsObj)
        predictedQuery = createPredictedQuery(evalExecObj, predOpsObj)
    if predictedQuery.lower().strip().startswith("select") and nextQuery.lower().strip().startswith("select"):
        (col_F1, col_prec, col_rec, tup_F1, tup_prec, tup_rec, total_F1, total_prec, total_rec) = execF1(evalExecObj, predOpsObj, predictedQuery, nextQuery)
        print "NextQuery: " + nextQuery
        print "PredictedQuery: " + predictedQuery
        print "col_F1: "+ str(col_F1) + ", col_prec: "+ str(col_prec) + ", col_rec: "+ str(col_rec) + ", tup_F1: "+ str(tup_F1) + ", tup_prec: "+ str(tup_prec) + ", tup_rec: "+ str(tup_rec) + ", total_F1: " + str(total_F1) + ", total_prec: " + str(total_prec) + ", total_rec: " + str(total_rec)
        # print "PredictedSQLFragStr: " + predictedSQLFragStr + "\n"
        #print "BorrowedQuery: " + str(borrowedQuery)
        return (col_F1, col_prec, col_rec, tup_F1, tup_prec, tup_rec, total_F1, total_prec, total_rec, borrowedQuery)
    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, borrowedQuery)

def executeExpectedQueries(evalExecObj):
    newEpFlg = 0
    nextQueryCount = 0
    insQueryCount = 0
    updQueryCount = 0
    delQueryCount = 0
    predictedQueryCount = 0
    nextQuery = None
    predictedQuery = None
    missedNextQueryExec = 0
    zeroResCount = 0
    nonZeroResCount = 0
    missedPredQueryExec = 0
    rank = float("-inf")
    curQueryIndex = float("-inf")
    predOpsObj = None
    borrowedQueryCount = 0
    execF1Count = 0
    execColF1 = 0.0
    execColPrec = 0.0
    execColRec = 0.0
    execTupF1 = 0.0
    execTupPrec = 0.0
    execTupRec = 0.0
    execTotalF1 = 0.0
    execTotalPrec = 0.0
    execTotalRec = 0.0
    with open(evalExecObj.logFile) as f:
        for line in f:
            if line.startswith("#Episodes"):
                newEpFlg = 1
                numTokens = len(line.strip().split(";"))
                rank = int(line.strip().split(";")[numTokens - 3].split(":")[1])
                if rank == -1:  # this can happen when all predicted queries are equally bad
                    rank = 0
                assert rank >= 0 and rank < int(evalExecObj.configDict['TOP_K'])
            if line.startswith("Next Query"):
                nextQuery = line.strip().split(": ")[1].strip()
                # convert nextquery to lower case and compare with start INSERT, UPDATE, DELETE to ignore them
                if nextQuery.lower().startswith("select"):
                    records = QExec.executeMINCQuery(nextQuery, evalExecObj.configDict)
                    if records is None:
                        missedNextQueryExec += 1
                        print "NonExecQuery: "+nextQuery
                    else:
                        #print "#Records: "+str(len(records))
                        if len(records) == 0:
                            zeroResCount += 1
                        else:
                            nonZeroResCount += 1
                    #records = cursor.fetchall()
                    #print "Total rows are: " +str(len(records))
                    nextQueryCount+=1
                    if nextQueryCount % 10000 == 0:
                        print "Total #queries: " + str(nextQueryCount) + ", #misses: " + str(missedNextQueryExec) + ", #zeroRes: " + str(zeroResCount) + ", #nonZeroRes: " + str(nonZeroResCount)
                elif nextQuery.lower().startswith("insert"):
                        insQueryCount += 1
                elif nextQuery.lower().startswith("update"):
                        updQueryCount += 1
                elif nextQuery.lower().startswith("delete"):
                        delQueryCount += 1
            if line.startswith("Predicted SQL Ops"):
                substrTokens = line.strip().split(":")[0].split(" ")
                curQueryIndex = int(substrTokens[len(substrTokens) - 1])
                if curQueryIndex == rank:
                    predOpsObj = nextActualOps()
            elif line.startswith("---") and nextQuery is not None and predOpsObj is not None and nextQuery.lower().strip().startswith("select"):
                (col_F1, col_prec, col_rec, tup_F1, tup_prec, tup_rec, total_F1, total_prec, total_rec, borrowedQuery) = computeExecF1(evalExecObj, predOpsObj, nextQuery)
                if borrowedQuery == True:
                    borrowedQueryCount+=1
                execF1Count += 1
                execColF1 += col_F1
                execColPrec += col_prec
                execColRec += col_rec
                execTupF1 += tup_F1
                execTupPrec += tup_prec
                execTupRec += tup_rec
                execTotalF1 += total_F1
                execTotalPrec += total_prec
                execTotalRec += total_rec
                #computeF1(evalOpsObj, predOpsObj, nextActualOpsObj)
            elif curQueryIndex == rank:
                    parseLineAddOp(line, predOpsObj)
    avgColF1 = float(execColF1) / float(execF1Count)
    avgColPrec = float(execColPrec) / float(execF1Count)
    avgColRec = float(execColRec) / float(execF1Count)
    avgTupF1 = float(execTupF1) / float(execF1Count)
    avgTupPrec = float(execTupPrec) / float(execF1Count)
    avgTupRec = float(execTupPrec) / float(execF1Count)
    avgExecF1 = float(execTotalF1) / float(execF1Count)
    avgExecPrec = float(execTotalPrec) / float(execF1Count)
    avgExecRec = float(execTotalRec) / float(execF1Count)
    print "avgColF1: "+str(avgColF1)+", avgColPrec: "+str(avgColPrec)+", avgColRec: "+str(avgColRec)+", avgTupF1: "+str(avgTupF1)+", avgTupPrec: "+str(avgTupPrec)+", avgTupRec: "+str(avgTupRec)+", avgExecF1: "+str(avgExecF1)+", avgExecPrec: "+str(avgExecPrec)+", avgExecRec: "+str(avgExecRec)
    print "Total Test #SELECT queries: " +str(nextQueryCount)+", #misses: "+str(missedNextQueryExec) +", #zeroRes: "+str(zeroResCount)+", #nonZeroRes: "+str(nonZeroResCount) + ", #borrowedQuery: "+str(borrowedQueryCount)
    print "Total Test #INSERT queries: "+str(insQueryCount)+", #UPDATES: "+str(updQueryCount)+", #DELETES: "+str(delQueryCount)
    return
'''
    def createEvalMetricsOpWise(evalOpsObj):
        prevEpisode = -1
        rank = float("-inf")
        nextActualOpsObj = None
        predOpsObj = None
        with open(evalOpsObj.logFile) as f:
            for line in f:
                if line.startswith("#Episodes"):
                    evalOpsObj.curEpisode = int(line.strip().split(";")[0].split(":")[1])
                    numTokens = len(line.strip().split(";"))
                    rank = int(line.strip().split(";")[numTokens - 3].split(":")[1])
                    if rank == -1:  # this can happen when all predicted queries are equally bad
                        rank = 0
                    assert rank >= 0 and rank < int(evalOpsObj.configDict['TOP_K'])
                    MRR = float(1.0) / float(rank + 1)
                    if evalOpsObj.curEpisode != prevEpisode:
                        evalOpsObj.numEpQueries[evalOpsObj.curEpisode] = 1
                        assert evalOpsObj.curEpisode not in evalOpsObj.meanReciprocalRank
                        evalOpsObj.meanReciprocalRank[evalOpsObj.curEpisode] = MRR
                    else:
                        evalOpsObj.numEpQueries[evalOpsObj.curEpisode] += 1
                        evalOpsObj.meanReciprocalRank[evalOpsObj.curEpisode] = (
                                    evalOpsObj.meanReciprocalRank[evalOpsObj.curEpisode] + MRR)
                elif line.startswith("Actual SQL"):
                    evalOpsObj.curQueryIndex = -100
                    nextActualOpsObj = nextActualOps()
                elif line.startswith("Predicted SQL Ops"):
                    substrTokens = line.strip().split(":")[0].split(" ")
                    evalOpsObj.curQueryIndex = int(substrTokens[len(substrTokens) - 1])
                    if evalOpsObj.curQueryIndex == rank:
                        predOpsObj = nextActualOps()
                elif line.startswith("---") and predOpsObj is not None and evalOpsObj is not None:
                    computeF1(evalOpsObj, predOpsObj, nextActualOpsObj)
                elif evalOpsObj.curQueryIndex == -100:
                    parseLineAddOp(line, nextActualOpsObj)
                elif evalOpsObj.curQueryIndex == rank:
                    parseLineAddOp(line, predOpsObj)
                prevEpisode = evalOpsObj.curEpisode
        return evalOpsObj
'''

def findTableRowStats(evalExecObj):
    tableDict = {}
    # cursor = execShowTableQuery(cnx, configDict)
    query = "SHOW TABLES"
    records = QExec.executeMINCQuery(query, evalExecObj.configDict)
    index = 0
    for row in records:
        tableName = str(row[0])
        assert tableName not in tableDict
        query = "SELECT COUNT(*) FROM "+tableName
        countRec = QExec.executeMINCQuery(query, evalExecObj.configDict)
        tableDict[tableName] = int(countRec[0][0])
        print "tablename: " + str(tableName) + ", count: " + str(tableDict[tableName])
        index += 1
    sorted_x = sorted(tableDict.items(), key=operator.itemgetter(1))
    tableDict = collections.OrderedDict(sorted_x)
    return tableDict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="config file to parse", type=str, required=True)
    parser.add_argument("-log", help="log filename to analyze", type=str, required=True)
    #parser.add_argument("-lineNum", help="line Number to analyze", type=int, required=True)
    args = parser.parse_args()
    evalExecObj = evalExec(args.config, args.log)
    #findTableRowStats(evalExecObj)

    executeExpectedQueries(evalExecObj)

    #evalOpsObj = evalOps(args.config, args.log)
    #evalOpsObj = createEvalMetricsOpWise(evalOpsObj)
    #plotEvalMetricsOpWise(evalOpsObj)