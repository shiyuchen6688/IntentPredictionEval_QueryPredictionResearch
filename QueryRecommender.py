import sys
import os
import time
import QueryExecution as QExec

def evaluatePredictions(predQuery, timeStepObj):
    print "--pending evaluation--"

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