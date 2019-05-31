from __future__ import division
import sys
import os
import time
import QueryExecution as QExec
from bitmap import BitMap
import CFCosineSim
import TupleIntent as ti
import ParseConfigFile as parseConfig
from ParseConfigFile import getConfig
import pickle
import argparse
from pandas import DataFrame

def updateArrWithDictEntry(arr, evalOpsObjDict, epIndex, evalOpsObj):
    try:
        arr.append(float(evalOpsObjDict[epIndex])/float(evalOpsObj.numEpQueries[epIndex]))
    except:
        arr.append("")
    return

def plotEvalMetricsOpWise(evalOpsObj):
    episodes = []
    numEpQueries = []
    meanReciprocalRank = []
    queryTypeP= []
    queryTypeR = []
    queryTypeF = []
    tablesP = []
    tablesR = []
    tablesF = []
    projColsP = []
    projColsR = []
    projColsF = []
    avgColsP = []
    avgColsR = []
    avgColsF = []
    minColsP = []
    minColsR = []
    minColsF = []
    maxColsP = []
    maxColsR = []
    maxColsF = []
    sumColsP = []
    sumColsR = []
    sumColsF = []
    countColsP = []
    countColsR = []
    countColsF = []
    selColsP = []
    selColsR = []
    selColsF = []
    condSelColsP = []
    condSelColsR = []
    condSelColsF = []
    groupByColsP = []
    groupByColsR = []
    groupByColsF = []
    orderByColsP = []
    orderByColsR = []
    orderByColsF = []
    havingColsP = []
    havingColsR = []
    havingColsF = []
    limitP = []
    limitR = []
    limitF = []
    joinPredsP = []
    joinPredsR = []
    joinPredsF = []
    for i in range(evalOpsObj.curEpisode + 1):
        episodes.append(i)
        updateArrWithDictEntry(meanReciprocalRank, evalOpsObj.meanReciprocalRank, i, evalOpsObj)
        updateArrWithDictEntry(queryTypeP, evalOpsObj.queryTypeP, i, evalOpsObj)
        updateArrWithDictEntry(queryTypeR, evalOpsObj.queryTypeR, i, evalOpsObj)
        updateArrWithDictEntry(queryTypeF, evalOpsObj.queryTypeF, i, evalOpsObj)
        updateArrWithDictEntry(tablesP, evalOpsObj.tablesP, i, evalOpsObj)
        updateArrWithDictEntry(tablesR, evalOpsObj.tablesR, i, evalOpsObj)
        updateArrWithDictEntry(tablesF, evalOpsObj.tablesF, i, evalOpsObj)
        updateArrWithDictEntry(projColsP, evalOpsObj.projColsP, i, evalOpsObj)
        updateArrWithDictEntry(projColsR, evalOpsObj.projColsR, i, evalOpsObj)
        updateArrWithDictEntry(projColsF, evalOpsObj.projColsF, i, evalOpsObj)
        updateArrWithDictEntry(avgColsP, evalOpsObj.avgColsP, i, evalOpsObj)
        updateArrWithDictEntry(avgColsR, evalOpsObj.avgColsR, i, evalOpsObj)
        updateArrWithDictEntry(avgColsF, evalOpsObj.avgColsF, i, evalOpsObj)
        updateArrWithDictEntry(minColsP, evalOpsObj.minColsP, i, evalOpsObj)
        updateArrWithDictEntry(minColsR, evalOpsObj.minColsR, i, evalOpsObj)
        updateArrWithDictEntry(minColsF, evalOpsObj.minColsF, i, evalOpsObj)
        updateArrWithDictEntry(maxColsP, evalOpsObj.maxColsP, i, evalOpsObj)
        updateArrWithDictEntry(maxColsR, evalOpsObj.maxColsR, i, evalOpsObj)
        updateArrWithDictEntry(maxColsF, evalOpsObj.maxColsF, i, evalOpsObj)
        updateArrWithDictEntry(sumColsP, evalOpsObj.sumColsP, i, evalOpsObj)
        updateArrWithDictEntry(sumColsR, evalOpsObj.sumColsR, i, evalOpsObj)
        updateArrWithDictEntry(sumColsF, evalOpsObj.sumColsF, i, evalOpsObj)
        updateArrWithDictEntry(countColsP, evalOpsObj.countColsP, i, evalOpsObj)
        updateArrWithDictEntry(countColsR, evalOpsObj.countColsR, i, evalOpsObj)
        updateArrWithDictEntry(countColsF, evalOpsObj.countColsF, i, evalOpsObj)
        updateArrWithDictEntry(selColsP, evalOpsObj.selColsP, i, evalOpsObj)
        updateArrWithDictEntry(selColsR, evalOpsObj.selColsR, i, evalOpsObj)
        updateArrWithDictEntry(selColsF, evalOpsObj.selColsF, i, evalOpsObj)
        updateArrWithDictEntry(condSelColsP, evalOpsObj.condSelColsP, i, evalOpsObj)
        updateArrWithDictEntry(condSelColsR, evalOpsObj.condSelColsR, i, evalOpsObj)
        updateArrWithDictEntry(condSelColsF, evalOpsObj.condSelColsF, i, evalOpsObj)
        updateArrWithDictEntry(groupByColsP, evalOpsObj.groupByColsP, i, evalOpsObj)
        updateArrWithDictEntry(groupByColsR, evalOpsObj.groupByColsR, i, evalOpsObj)
        updateArrWithDictEntry(groupByColsF, evalOpsObj.groupByColsF, i, evalOpsObj)
        updateArrWithDictEntry(orderByColsP, evalOpsObj.orderByColsP, i, evalOpsObj)
        updateArrWithDictEntry(orderByColsR, evalOpsObj.orderByColsR, i, evalOpsObj)
        updateArrWithDictEntry(orderByColsF, evalOpsObj.orderByColsF, i, evalOpsObj)
        updateArrWithDictEntry(havingColsP, evalOpsObj.havingColsP, i, evalOpsObj)
        updateArrWithDictEntry(havingColsR, evalOpsObj.havingColsR, i, evalOpsObj)
        updateArrWithDictEntry(havingColsF, evalOpsObj.havingColsF, i, evalOpsObj)
        updateArrWithDictEntry(limitP, evalOpsObj.limitP, i, evalOpsObj)
        updateArrWithDictEntry(limitR, evalOpsObj.limitR, i, evalOpsObj)
        updateArrWithDictEntry(limitF, evalOpsObj.limitF, i, evalOpsObj)
        updateArrWithDictEntry(joinPredsP, evalOpsObj.joinPredsP, i, evalOpsObj)
        updateArrWithDictEntry(joinPredsR, evalOpsObj.joinPredsR, i, evalOpsObj)
        updateArrWithDictEntry(joinPredsF, evalOpsObj.joinPredsF, i, evalOpsObj)
    df = DataFrame(
        {'episodes': episodes, 'meanReciprocalRank': meanReciprocalRank, 'queryTypeP': queryTypeP, 'queryTypeR': queryTypeR, 'queryTypeF': queryTypeF,
         'tablesP': tablesP, 'tablesR': tablesR, 'tablesF': tablesF,
         'projColsP': projColsP, 'projColsR': projColsR, 'projColsF': projColsF,
         'avgColsP': avgColsP, 'avgColsR': avgColsR, 'avgColsF': avgColsF,
         'minColsP': minColsP, 'minColsR': minColsR, 'minColsF': minColsF,
         'maxColsP': maxColsP, 'maxColsR': maxColsR, 'maxColsF': maxColsF,
         'sumColsP': sumColsP, 'sumColsR': sumColsR, 'sumColsF': sumColsF,
         'countColsP': countColsP, 'countColsR': countColsR, 'countColsF': countColsF,
         'selColsP': selColsP, 'selColsR': selColsR, 'selColsF': selColsF,
         'condSelColsP': condSelColsP, 'condSelColsR': condSelColsR, 'condSelColsF': condSelColsF,
         'groupByColsP': groupByColsP, 'groupByColsR': groupByColsR, 'groupByColsF': groupByColsF,
         'orderByColsP': orderByColsP, 'orderByColsR': orderByColsR, 'orderByColsF': orderByColsF,
         'havingColsP': havingColsP, 'havingColsR': havingColsR, 'havingColsF': havingColsF,
         'limitP': limitP, 'limitR': limitR, 'limitF': limitF,
         'joinPredsP': joinPredsP, 'joinPredsR': joinPredsR, 'joinPredsF': joinPredsF,})
    outputOpWiseQualityFileName = getConfig(evalOpsObj.configDict['OUTPUT_DIR']) + "/OutputOpWiseQuality_" + evalOpsObj.configDict[
        'ALGORITHM']
    df.to_excel(outputOpWiseQualityFileName+".xlsx", sheet_name='sheet1', index=False)

class evalOps:
    def __init__(self, configFileName, logFile):
        self.configDict = parseConfig.parseConfigFile(configFileName)
        self.logFile = logFile
        self.curEpisode = 0
        self.numEpQueries = {}
        self.curQueryIndex = -1
        self.meanReciprocalRank = {}
        self.episode = {}
        self.queryTypeP = {}
        self.queryTypeR = {}
        self.queryTypeF = {}
        self.tablesP = {}
        self.tablesR = {}
        self.tablesF = {}
        self.projColsP = {}
        self.projColsR = {}
        self.projColsF = {}
        self.avgColsP = {}
        self.avgColsR = {}
        self.avgColsF = {}
        self.minColsP = {}
        self.minColsR = {}
        self.minColsF = {}
        self.maxColsP = {}
        self.maxColsR = {}
        self.maxColsF = {}
        self.sumColsP = {}
        self.sumColsR = {}
        self.sumColsF = {}
        self.countColsP = {}
        self.countColsR = {}
        self.countColsF = {}
        self.selColsP = {}
        self.selColsR = {}
        self.selColsF = {}
        self.condSelColsP = {}
        self.condSelColsR = {}
        self.condSelColsF = {}
        self.groupByColsP = {}
        self.groupByColsR = {}
        self.groupByColsF = {}
        self.orderByColsP = {}
        self.orderByColsR = {}
        self.orderByColsF = {}
        self.havingColsP = {}
        self.havingColsR = {}
        self.havingColsF = {}
        self.limitP = {}
        self.limitR = {}
        self.limitF = {}
        self.joinPredsP = {}
        self.joinPredsR = {}
        self.joinPredsF = {}

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
    elif line.startswith("SEL"):
        actualOrPredObj.selCols = eval(line.strip().split(": ")[1])
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
    if (actualOpList is None and predOpList is not None) or\
            (predOpList is None and actualOpList is not None):
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
        return (1.0, 1.0, 1.0)

def updateOpMetrics(P, R, F, evalOpsP, evalOpsR, evalOpsF, evalOpsObj):
    if P is not None and R is not None and F is not None:
        updateMetricDict(evalOpsP, evalOpsObj.curEpisode, P)
        updateMetricDict(evalOpsR, evalOpsObj.curEpisode, R)
        updateMetricDict(evalOpsF, evalOpsObj.curEpisode, F)
    return

def computeRelevantCols(accTables, predOrActualCols):
    relCols = []
    for col in predOrActualCols:
        tableName = col.split(".")[0]
        if tableName in accTables:
            relCols.append(col)
    if len(relCols) == 0:
        return None
    return relCols

def compUpdateOpMetrics(predOpList, actualOpList, evalOpsP, evalOpsR, evalOpsF, evalOpsObj):
    (P,R,F) = computeOpF1(predOpList, actualOpList)
    updateOpMetrics(P, R, F, evalOpsP, evalOpsR, evalOpsF, evalOpsObj)
    return

def compUpdateCondSelMetrics(predOpsObj, nextActualOpsObj, evalOpsObj):
    try:
        if evalOpsObj.tablesF[evalOpsObj.curEpisode] == 1.0 and evalOpsObj.curEpisode in evalOpsObj.selColsP \
                and evalOpsObj.curEpisode in evalOpsObj.selColsR and evalOpsObj.curEpisode in evalOpsObj.selColsF:
            updateOpMetrics(evalOpsObj.selColsP[evalOpsObj.curEpisode], evalOpsObj.selColsR[evalOpsObj.curEpisode], evalOpsObj.selColsF[evalOpsObj.curEpisode], evalOpsObj.condSelColsP, evalOpsObj.condSelColsR, evalOpsObj.condSelColsF, evalOpsObj)
        elif evalOpsObj.tablesF[evalOpsObj.curEpisode] > 0.0: # partial overlap of tables
            accTables = list(set(predOpsObj.tables).intersection(set(nextActualOpsObj.tables)))
            relPredCols = computeRelevantCols(accTables, predOpsObj.selCols)
            relActualCols = computeRelevantCols(accTables, nextActualOpsObj.selCols)
            compUpdateOpMetrics(relPredCols, relActualCols, evalOpsObj.condSelColsP, evalOpsObj.condSelColsR, evalOpsObj.condSelColsF, evalOpsObj)
        else:
            updateOpMetrics(1.0, 1.0, 1.0, evalOpsObj.condSelColsP, evalOpsObj.condSelColsR, evalOpsObj.condSelColsF, evalOpsObj)
    except:
        updateOpMetrics(1.0, 1.0, 1.0, evalOpsObj.condSelColsP, evalOpsObj.condSelColsR, evalOpsObj.condSelColsF,
                        evalOpsObj)
        pass
    return

def computeF1(evalOpsObj, predOpsObj, nextActualOpsObj):
    if predOpsObj.queryType == nextActualOpsObj.queryType:
        updateOpMetrics(1.0, 1.0, 1.0, evalOpsObj.queryTypeP, evalOpsObj.queryTypeR, evalOpsObj.queryTypeF, evalOpsObj)
    elif predOpsObj.queryType != nextActualOpsObj.queryType:
        updateOpMetrics(0.0, 0.0, 0.0, evalOpsObj.queryTypeP, evalOpsObj.queryTypeR, evalOpsObj.queryTypeF, evalOpsObj)
    if predOpsObj.limit == nextActualOpsObj.limit:
        updateOpMetrics(1.0, 1.0, 1.0, evalOpsObj.limitP, evalOpsObj.limitR, evalOpsObj.limitF, evalOpsObj)
    elif predOpsObj.limit != nextActualOpsObj.limit:
        updateOpMetrics(0.0, 0.0, 0.0, evalOpsObj.limitP, evalOpsObj.limitR, evalOpsObj.limitF, evalOpsObj)
    compUpdateOpMetrics(predOpsObj.tables, nextActualOpsObj.tables, evalOpsObj.tablesP,
                        evalOpsObj.tablesR, evalOpsObj.tablesF, evalOpsObj)
    compUpdateOpMetrics(predOpsObj.projCols, nextActualOpsObj.projCols, evalOpsObj.projColsP,
                        evalOpsObj.projColsR, evalOpsObj.projColsF, evalOpsObj)
    compUpdateOpMetrics(predOpsObj.avgCols, nextActualOpsObj.avgCols, evalOpsObj.avgColsP,
                        evalOpsObj.avgColsR, evalOpsObj.avgColsF, evalOpsObj)
    compUpdateOpMetrics(predOpsObj.minCols, nextActualOpsObj.minCols, evalOpsObj.minColsP,
                        evalOpsObj.minColsR, evalOpsObj.minColsF, evalOpsObj)
    compUpdateOpMetrics(predOpsObj.maxCols, nextActualOpsObj.maxCols, evalOpsObj.maxColsP,
                        evalOpsObj.maxColsR, evalOpsObj.maxColsF, evalOpsObj)
    compUpdateOpMetrics(predOpsObj.sumCols, nextActualOpsObj.sumCols, evalOpsObj.sumColsP,
                        evalOpsObj.sumColsR, evalOpsObj.sumColsF, evalOpsObj)
    compUpdateOpMetrics(predOpsObj.countCols, nextActualOpsObj.countCols, evalOpsObj.countColsP,
                        evalOpsObj.countColsR, evalOpsObj.countColsF, evalOpsObj)
    compUpdateOpMetrics(predOpsObj.selCols, nextActualOpsObj.selCols, evalOpsObj.selColsP,
                        evalOpsObj.selColsR, evalOpsObj.selColsF, evalOpsObj)
    compUpdateOpMetrics(predOpsObj.groupByCols, nextActualOpsObj.groupByCols, evalOpsObj.groupByColsP,
                        evalOpsObj.groupByColsR, evalOpsObj.groupByColsF, evalOpsObj)
    compUpdateOpMetrics(predOpsObj.orderByCols, nextActualOpsObj.orderByCols, evalOpsObj.orderByColsP,
                        evalOpsObj.orderByColsR, evalOpsObj.orderByColsF, evalOpsObj)
    compUpdateOpMetrics(predOpsObj.havingCols, nextActualOpsObj.havingCols, evalOpsObj.havingColsP,
                        evalOpsObj.havingColsR, evalOpsObj.havingColsF, evalOpsObj)
    compUpdateOpMetrics(predOpsObj.joinPreds, nextActualOpsObj.joinPreds, evalOpsObj.joinPredsP,
                        evalOpsObj.joinPredsR, evalOpsObj.joinPredsF, evalOpsObj)
    compUpdateCondSelMetrics(predOpsObj, nextActualOpsObj, evalOpsObj)
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
                evalOpsObj.curQueryIndex = -1
                nextActualOpsObj = nextActualOps()
            elif line.startswith("Predicted SQL Ops"):
                substrTokens = line.strip().split(":")[0].split(" ")
                evalOpsObj.curQueryIndex = int(substrTokens[len(substrTokens)-1])
                if evalOpsObj.curQueryIndex == rank:
                    predOpsObj = nextActualOps()
            elif evalOpsObj.curQueryIndex == -1:
                parseLineAddOp(line, nextActualOpsObj)
            elif evalOpsObj.curQueryIndex == rank:
                parseLineAddOp(line, predOpsObj)
            elif line.startswith("---") and predOpsObj is not None and evalOpsObj is not None:
                computeF1(evalOpsObj, predOpsObj, nextActualOpsObj)
            prevEpisode = evalOpsObj.curEpisode
    return evalOpsObj

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="config file to parse", type=str, required=True)
    parser.add_argument("-log", help="log filename to analyze", type=str, required=True)
    #parser.add_argument("-lineNum", help="line Number to analyze", type=int, required=True)
    args = parser.parse_args()
    evalOpsObj = evalOps(args.config, args.log)
    evalOpsObj = createEvalMetricsOpWise(evalOpsObj)
    plotEvalMetricsOpWise(evalOpsObj)