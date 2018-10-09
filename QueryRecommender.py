from __future__ import division
import sys
import os
import time
import QueryExecution as QExec
from bitmap import BitMap
import CFCosineSim
import TupleIntent as ti
import ParseConfigFile as parseConfig

def evaluatePredictions(outputIntentFileName, episodeResponseTime, configDict):
    outputEvalQualityFileName = configDict['OUTPUT_DIR'] + "/OutputEvalQualityShortTermIntent_" + configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']+"_ACCURACY_THRESHOLD_"+str(configDict['ACCURACY_THRESHOLD'])
    try:
        os.remove(outputEvalQualityFileName)
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
                    cosineSim = CFCosineSim.computeWeightedCosineSimilarity(curQueryIntent, topKQueryIntent, ",", configDict)
                if cosineSim >= float(configDict['ACCURACY_THRESHOLD']):
                    recall = 1.0
                    precision += 1.0
                if cosineSim > maxCosineSim:
                    maxCosineSim = cosineSim
            precision /= float(len(tokens)-4+1)
            outputEvalQualityStr = "Session:"+str(sessID)+";Query:"+str(queryID)+";#Episodes:"+str(numEpisodes)+";Precision:"+str(precision)+";Recall:"+str(recall)+";Accuracy:"+str(maxCosineSim)
            ti.appendToFile(outputEvalQualityFileName, outputEvalQualityStr)
    outputEvalTimeFileName = configDict['OUTPUT_DIR'] + "/OutputEvalTimeShortTermIntent_" + configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']+"_ACCURACY_THRESHOLD_"+str(configDict['ACCURACY_THRESHOLD'])
    try:
        os.remove(outputEvalTimeFileName)
    except OSError:
        pass
    # Simulate query execution and intent creation to record their times #
    numQueries = 0
    episodeQueryExecutionTime = {}
    episodeIntentCreationTime = {}
    numEpisodes =1
    with open(configDict['QUERYSESSIONS']) as f:
        for line in f:
            numQueries+=1
            if numQueries % int(configDict['EPISODE_IN_QUERIES']) == 0:
                numEpisodes += 1
            sessQueries = line.split(";")
            sessName = sessQueries[0]
            for i in range(1, len(sessQueries) - 1):  # we need to ignore the empty query coming from the end of line semicolon ;
                sessQuery = sessQueries[i].split("~")[0]
                sessQuery = ' '.join(sessQuery.split())
                # sessQuery = "SELECT nyc_yellow_tripdata_2016_06_sample_1_percent.dropoff_latitude AS dropoff_latitude, nyc_yellow_tripdata_2016_06_sample_1_percent.dropoff_longitude AS dropoff_longitude, nyc_yellow_tripdata_2016_06_sample_1_percent.fare_amount AS fare_amount FROM public.nyc_yellow_tripdata_2016_06_sample_1_percent nyc_yellow_tripdata_2016_06_sample_1_percent GROUP BY 1, 2, 3 HAVING ((CAST(MIN(nyc_yellow_tripdata_2016_06_sample_1_percent.fare_amount) AS DOUBLE PRECISION) >= 11.999999999999879) AND (CAST(MIN(nyc_yellow_tripdata_2016_06_sample_1_percent.fare_amount) AS DOUBLE PRECISION) <= 14.00000000000014))"
                queryVocabulary = {}
                (queryVocabulary, resObj, queryExecutionTime, intentCreationTime) = QExec.executeQueryWithIntent(sessQuery, configDict, queryVocabulary)
                if numEpisodes not in episodeIntentCreationTime:
                    episodeIntentCreationTime[numEpisodes] = intentCreationTime
                else:
                    episodeIntentCreationTime[numEpisodes] += intentCreationTime
                if numEpisodes not in episodeQueryExecutionTime:
                    episodeQueryExecutionTime[numEpisodes] = queryExecutionTime
                else:
                    episodeQueryExecutionTime[numEpisodes] += queryExecutionTime
                print "Executed and obtained intent for "+sessName+", Query "+str(i)

    assert len(episodeQueryExecutionTime) == len(episodeResponseTime) and len(episodeIntentCreationTime) == len(episodeResponseTime)
    for episodes in range(1,len(episodeResponseTime)):
        totalResponseTime = float(episodeIntentCreationTime[episodes]) + float(episodeQueryExecutionTime[episodes]) + float(episodeResponseTime[episodes])
        outputEvalTimeStr = "#Episodes:"+str(episodes)+";QueryExecutionTime(secs):"+str(episodeQueryExecutionTime[episodes])+";IntentCreationTime(secs):"+str(episodeIntentCreationTime[episodes])+";IntentPredictionTime(secs):"+str(episodeResponseTime[episodes])+";TotalResponseTime(secs):"+str(totalResponseTime)
        ti.appendToFile(outputEvalTimeFileName, outputEvalTimeStr)
    print "--Completed Evaluation--"
    return

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

