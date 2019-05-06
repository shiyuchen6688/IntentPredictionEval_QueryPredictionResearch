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

def updateArrWithDictEntry(arr, evalOpsObjDict, epIndex):
    try:
        arr.append(evalOpsObjDict[epIndex])
    except:
        arr.append("")
    return

def plotEvalMetricsOpWise(evalOpsObj):
    episodes = []
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
        updateArrWithDictEntry(meanReciprocalRank, evalOpsObj.meanReciprocalRank, i)
        updateArrWithDictEntry(queryTypeP, evalOpsObj.queryTypeP, i)
        updateArrWithDictEntry(queryTypeR, evalOpsObj.queryTypeR, i)
        updateArrWithDictEntry(queryTypeF, evalOpsObj.queryTypeF, i)
        updateArrWithDictEntry(tablesP, evalOpsObj.tablesP, i)
        updateArrWithDictEntry(tablesR, evalOpsObj.tablesR, i)
        updateArrWithDictEntry(tablesF, evalOpsObj.tablesF, i)
        updateArrWithDictEntry(projColsP, evalOpsObj.projColsP, i)
        updateArrWithDictEntry(projColsR, evalOpsObj.projColsR, i)
        updateArrWithDictEntry(projColsF, evalOpsObj.projColsF, i)
        updateArrWithDictEntry(avgColsP, evalOpsObj.avgColsP, i)
        updateArrWithDictEntry(avgColsR, evalOpsObj.avgColsR, i)
        updateArrWithDictEntry(avgColsF, evalOpsObj.avgColsF, i)
        updateArrWithDictEntry(minColsP, evalOpsObj.minColsP, i)
        updateArrWithDictEntry(minColsR, evalOpsObj.minColsR, i)
        updateArrWithDictEntry(minColsF, evalOpsObj.minColsF, i)
        updateArrWithDictEntry(maxColsP, evalOpsObj.maxColsP, i)
        updateArrWithDictEntry(maxColsR, evalOpsObj.maxColsR, i)
        updateArrWithDictEntry(maxColsF, evalOpsObj.maxColsF, i)
        updateArrWithDictEntry(sumColsP, evalOpsObj.sumColsP, i)
        updateArrWithDictEntry(sumColsR, evalOpsObj.sumColsR, i)
        updateArrWithDictEntry(sumColsF, evalOpsObj.sumColsF, i)
        updateArrWithDictEntry(countColsP, evalOpsObj.countColsP, i)
        updateArrWithDictEntry(countColsR, evalOpsObj.countColsR, i)
        updateArrWithDictEntry(countColsF, evalOpsObj.countColsF, i)
        updateArrWithDictEntry(selColsP, evalOpsObj.selColsP, i)
        updateArrWithDictEntry(selColsR, evalOpsObj.selColsR, i)
        updateArrWithDictEntry(selColsF, evalOpsObj.selColsF, i)
        updateArrWithDictEntry(condSelColsP, evalOpsObj.condSelColsP, i)
        updateArrWithDictEntry(condSelColsR, evalOpsObj.condSelColsR, i)
        updateArrWithDictEntry(condSelColsF, evalOpsObj.condSelColsF, i)
        updateArrWithDictEntry(groupByColsP, evalOpsObj.groupByColsP, i)
        updateArrWithDictEntry(groupByColsR, evalOpsObj.groupByColsR, i)
        updateArrWithDictEntry(groupByColsF, evalOpsObj.groupByColsF, i)
        updateArrWithDictEntry(orderByColsP, evalOpsObj.orderByColsP, i)
        updateArrWithDictEntry(orderByColsR, evalOpsObj.orderByColsR, i)
        updateArrWithDictEntry(orderByColsF, evalOpsObj.orderByColsF, i)
        updateArrWithDictEntry(havingColsP, evalOpsObj.havingColsP, i)
        updateArrWithDictEntry(havingColsR, evalOpsObj.havingColsR, i)
        updateArrWithDictEntry(havingColsF, evalOpsObj.havingColsF, i)
        updateArrWithDictEntry(limitP, evalOpsObj.limitP, i)
        updateArrWithDictEntry(limitR, evalOpsObj.limitR, i)
        updateArrWithDictEntry(limitF, evalOpsObj.limitF, i)
        updateArrWithDictEntry(joinPredsP, evalOpsObj.joinPredsP, i)
        updateArrWithDictEntry(joinPredsR, evalOpsObj.joinPredsR, i)
        updateArrWithDictEntry(joinPredsF, evalOpsObj.joinPredsF, i)
    df = DataFrame(
        {'episodes': episodes, 'queryTypeP': queryTypeP, 'queryTypeR': queryTypeR, 'queryTypeF': queryTypeF,
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
        'ALGORITHM'] + ".xlsx"
    df.to_excel(outputOpWiseQualityFileName+".xlsx", sheet_name='sheet1', index=False)

class evalOps:
    def __init__(self, configDict, logFile):
        self.configDict = configDict
        self.logFile = logFile
        self.curEpisode = 0
        self.numEpQueries = 0
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

def updateMetricDict(metricDict, key, val, numEpQueries):
    if key not in metricDict:
        metricDict[key] = val
    else:
        metricDict[key] = float(metricDict[key]+val)/float(numEpQueries)
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
        updateMetricDict(evalOpsP, evalOpsObj.curEpisode, P, evalOpsObj.numEpQueries)
        updateMetricDict(evalOpsR, evalOpsObj.curEpisode, R, evalOpsObj.numEpQueries)
        updateMetricDict(evalOpsF, evalOpsObj.curEpisode, F, evalOpsObj.numEpQueries)
    return

def compUpdateCondSelMetrics(evalOpsObj):
    try:
        if evalOpsObj.tablesF[evalOpsObj.curEpisode] == 1.0 and evalOpsObj.curEpisode in evalOpsObj.selColsP \
                and evalOpsObj.curEpisode in evalOpsObj.selColsR and evalOpsObj.curEpisode in evalOpsObj.selColsF:
            updateOpMetrics(evalOpsObj.selColsP, evalOpsObj.selColsR, evalOpsObj.selColsF, evalOpsObj.condSelColsP, evalOpsObj.condSelColsR, evalOpsObj.condSelColsF, evalOpsObj)
        else:
            updateOpMetrics(0.0, 0.0, 0.0, evalOpsObj.condSelColsP, evalOpsObj.condSelColsR, evalOpsObj.condSelColsF, evalOpsObj)
    except:
        updateOpMetrics(0.0, 0.0, 0.0, evalOpsObj.condSelColsP, evalOpsObj.condSelColsR, evalOpsObj.condSelColsF,
                        evalOpsObj)
        pass

def compUpdateOpMetrics(predOpList, actualOpList, evalOpsP, evalOpsR, evalOpsF, evalOpsObj):
    (P,R,F) = computeOpF1(predOpList, actualOpList)
    updateOpMetrics(P, R, F, evalOpsP, evalOpsR, evalOpsF, evalOpsObj)
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
    compUpdateCondSelMetrics(evalOpsObj)
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
                numTokens = len(line.strip().split(":"))
                rank = int(line.split(";")[numTokens-1].split(":")[1])
                assert rank >= 0 and rank < int(evalOpsObj.configDir['TOP_K'])
                MRR = float(1.0) / float(rank+1)
                if evalOpsObj.curEpisode != prevEpisode:
                    evalOpsObj.numEpQueries = 1
                    assert evalOpsObj.curEpisode not in evalOpsObj.meanReciprocalRank
                    evalOpsObj.meanReciprocalRank[evalOpsObj.curEpisode] = MRR
                else:
                    evalOpsObj.numEpQueries += 1
                    evalOpsObj.meanReciprocalRank[evalOpsObj.curEpisode] = (evalOpsObj.meanReciprocalRank[evalOpsObj.curEpisode] + MRR) \
                                                                           / float(evalOpsObj.numEpQueries)
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
    evalOpsObj = evalOps(args.configDict, args.log)
    evalOpsObj = createEvalMetricsOpWise(evalOpsObj)
    plotEvalMetricsOpWise(evalOpsObj)