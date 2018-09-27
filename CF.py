import sys
import os
import time
import QueryExecution as QExec
import QueryRecommender as QR

def evaluatePredictions(predQuery, timeStepObj):
    print "--pending evaluation--"

class TimeStep(object):
    def __init__(self, timeStep, sessQuery, sessLogs):
        self.timeStep = timeStep
        self.sessQuery = sessQuery
        self.sessLogs = sessLogs  # these are tuple/fragment/query vectors

    def updateTimeStep(self, timeStep):
        self.timeStep = timeStep

    def updateSessQuery(self, sessQuery):
        self.sessQuery = sessQuery

    def updateSessLogs(self, resObj, sessName):
        if self.sessLogs is None:
            self.sessLogs = dict()
            self.sessLogs[sessName] = resObj
        else:
            if sessName in self.sessLogs.keys():
                self.sessLogs[sessName] = self.sessLogs[sessName]+";"+resObj
            else:
                self.sessLogs[sessName] = resObj

def simulateHumanQueries(configDict):
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
                predQuery = QR.recommendQuery(resObj, timeStepObj)
                evaluatePredictions(predQuery, timeStepObj)
                timeStepObj.updateSessLogs(resObj,sessName)

def runIntentPrediction(configDict):
    simulateHumanQueries(configDict)