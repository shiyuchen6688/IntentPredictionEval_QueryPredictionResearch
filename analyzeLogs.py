from __future__ import division
import sys
import os
import time
import QueryExecution as QExec
from bitmap import BitMap
import CFCosineSim
import TupleIntent as ti
import ParseConfigFile as parseConfig
import pickle
import argparse

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

def parseLineAddOp(line, evalOpsObj, actualOrPredObj):
    print "hi"

def computeF1(evalOpsObj, predOpsObj, nextActualOpsObj):
    print "ji"

            
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
                parseLineAddOp(line, evalOpsObj, nextActualOpsObj)
            elif evalOpsObj.curQueryIndex == rank:
                parseLineAddOp(line, evalOpsObj, predOpsObj)
            elif line.startswith("---") and predOpsObj is not None and evalOpsObj is not None:
                computeF1(evalOpsObj, predOpsObj, nextActualOpsObj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="config file to parse", type=str, required=True)
    parser.add_argument("-log", help="log filename to analyze", type=str, required=True)
    #parser.add_argument("-lineNum", help="line Number to analyze", type=int, required=True)
    args = parser.parse_args()
    evalOpsObj = evalOps(args.configDict, args.log)
    createEvalMetricsOpWise(evalOpsObj)