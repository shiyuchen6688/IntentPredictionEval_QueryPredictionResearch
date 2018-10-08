from __future__ import division
import sys
import os
import time
import QueryExecution as QExec
from bitmap import BitMap
import CFCosineSim
import TupleIntent as ti
import ParseConfigFile as parseConfig

def evaluatePredictions(outputIntentFileName, configDict):
    outputEvalFileName = configDict['OUTPUT_DIR'] + "/OutputEvalFileShortTermIntent_" + configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_K_" + configDict['TOP_K'] + "_EPISODE_SECS_" + configDict['EPISODE_IN_SECONDS']+"_ACCURACY_THRESHOLD_"+str(configDict['ACCURACY_THRESHOLD'])
    try:
        os.remove(outputEvalFileName)
    except OSError:
        pass
    with open(outputIntentFileName) as f:
        for line in f:
            tokens = line.strip().split(";")
            sessID = tokens[0].split(":")[1]
            queryID = tokens[1].split(":")[1]
            numEpisodes = tokens[2].split(":")[1]
            precision = 0.0
            recall = 0.0
            maxCosineSim =0.0
            if configDict['BIT_OR_WEIGHTED'] == 'BIT':
                curQueryIntent = BitMap.fromstring(tokens[3].split(":")[1])
            elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
                curQueryIntent = tokens[3].split(":")[1]
            for i in range(4,len(tokens)):
                if configDict['BIT_OR_WEIGHTED'] == 'BIT':
                    topKQueryIntent = BitMap.fromstring(tokens[i].split(":")[1])
                    cosineSim = CFCosineSim.computeBitCosineSimilarity(curQueryIntent, topKQueryIntent)
                elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
                    topKQueryIntent = tokens[i].split(":")[1]
                    cosineSim = CFCosineSim.computeWeightedCosineSimilarity(curQueryIntent, topKQueryIntent, ",")
                if cosineSim >= configDict['ACCURACY_THRESHOLD']:
                    recall = 1.0
                    precision += 1.0
                if cosineSim > maxCosineSim:
                    maxCosineSim = cosineSim
            precision /= float(len(tokens)-4+1)
            outputEvalStr = "Session:"+str(sessID)+";Query:"+str(queryID)+";#Episodes:"+str(numEpisodes)+";Precision:"+str(precision)+";Recall:"+str(recall)+";Accuracy:"+str(maxCosineSim)
            ti.appendToFile(outputEvalFileName, outputEvalStr)
    print "--Completed Evaluation--"

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("configFile.txt")
    outputIntentFileName = configDict['OUTPUT_DIR']+"/OutputFileShortTermIntent_"+configDict['INTENT_REP']+"_"+configDict['BIT_OR_WEIGHTED']+"_K_"+configDict['TOP_K']+"_EPISODE_SECS_"+configDict['EPISODE_IN_SECONDS']
    evaluatePredictions(outputIntentFileName, configDict)

'''
class TimeStep(object):
    def __init__(self, timeStep, sessQuery, sessQueryIntent, sessLogs):
        self.timeStep = timeStep
        self.sessQuery = sessQuery
        self.sessQueryIntent = sessQueryIntent
        self.sessLogs = sessLogs  # these are tuple/fragment/query vectors

    def updateTimeStep(self, timeStep):
        self.timeStep = timeStep

    def updateSessQueryIntent(self, sessQuery, sessQueryIntent):
        self.sessQuery = sessQuery
        self.sessQueryIntent = sessQueryIntent

    def updateSessLogs(self, resObj, sessIndex, queryIndex):
        if self.sessLogs is None:
            self.sessLogs = dict()
        if sessIndex not in self.sessLogs.keys():
            self.sessLogs[sessIndex] = dict()
        self.sessLogs[sessIndex][queryIndex] = resObj

def recommendQuery(resObj, timeStepObj):
    return None

def simulateHumanQueriesWithCreateIntent(configDict):
    timeStep = 0
    timeStepObj = TimeStep(0,None,None)
    with open(configDict['QUERYSESSIONS']) as f:
        for line in f:
            sessQueries = line.split(";")
            sessName = sessQueries[0]
            for i in range(1,len(sessQueries)):
                sessQuery = sessQueries[i]
                timeStepObj.updateTimeStep(timeStep)
                timeStepObj.updateSessQuery(sessQuery)
                resObj = QExec.executeQueryWithIntent(sessQuery, configDict) # with intent
                predQuery = recommendQuery(resObj, timeStepObj)
                evaluatePredictions(predQuery, timeStepObj)
                timeStepObj.updateSessLogs(resObj,sessName)
'''

