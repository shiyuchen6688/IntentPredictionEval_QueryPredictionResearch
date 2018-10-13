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

def findNextQueryIntent(intentSessionFile, sessID, queryID):
    with open(intentSessionFile) as f:
        for line in f:
            (curSessID, curQueryID, curQueryIntent) = retrieveSessIDQueryIDIntent(line, configDict)
            if curSessID == sessID and curQueryID == queryID:
                f.close()
                return curQueryIntent
    print "Error: Could not find the nextQueryIntent !!"
    sys.exit(0)

def normalizeWeightedVector(curQueryIntent):
    tokens = curQueryIntent.split(";")
    total = 0.0
    for token in tokens:
        total = total+float(token)
    normalizedVector = []
    for token in tokens:
        normalizedVector.append(float(token)/total)
    res = ';'.join(normalizedVector)
    return res

def retrieveSessIDQueryIDIntent(line, configDict):
    tokens = line.strip().split(";")
    sessQueryName = tokens[0]
    sessID = int(sessQueryName.split(", ")[0].split(" ")[1])
    queryID = int(sessQueryName.split(", ")[1].split(" ")[1]) - 1  # coz queryID starts from 1 instead of 0
    curQueryIntent = ';'.join(tokens[2:])
    if ";" not in curQueryIntent and configDict['BIT_OR_WEIGHTED'] == 'BIT':
        curQueryIntent = BitMap.fromstring(curQueryIntent)
    else:
        curQueryIntent = normalizeWeightedVector(curQueryIntent)
    return (sessID, queryID, curQueryIntent)

def appendPredictedRNNIntentToFile(sessID, queryID, cosineSim, numEpisodes, outputEvalQualityFileName):
    startAppendTime = time.time()
    outputEvalQualityStr = "Session:" + str(sessID) + ";Query:" + str(queryID) + ";#Episodes:" + str(numEpisodes) + ";Accuracy:" + str(cosineSim)
    ti.appendToFile(outputEvalQualityFileName, outputEvalQualityStr)
    print "cosine similarity at sessID: " + str(sessID) + ", queryID: " + str(queryID) + " is " + str(cosineSim)
    elapsedAppendTime = float(time.time() - startAppendTime)
    return elapsedAppendTime

def appendPredictedIntentsToFile(topKSessQueryIndices, topKPredictedIntents, sessID, queryID, curQueryIntent, numEpisodes, configDict, outputIntentFileName):
    startAppendTime = time.time()
    output_str = "Session:"+str(sessID)+";Query:"+str(queryID)+";#Episodes:"+str(numEpisodes)+";CurQueryIntent:"
    if configDict['BIT_OR_WEIGHTED'] == 'BIT':
        output_str += curQueryIntent.tostring()
    elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
        if ";" in curQueryIntent:
            curQueryIntent.replace(";",",")
        output_str += curQueryIntent
    assert len(topKSessQueryIndices) == len(topKPredictedIntents)
    for k in range(len(topKPredictedIntents)):
        output_str += ";TOP_" +str(k)+"_PREDICTED_INTENT_"+str(topKSessQueryIndices[k])+":"
        if configDict['BIT_OR_WEIGHTED'] == 'BIT':
            output_str += topKPredictedIntents[k].tostring()
        elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
            output_str += topKPredictedIntents[k].replace(";",",")
    ti.appendToFile(outputIntentFileName, output_str)
    print "Predicted "+str(len(topKPredictedIntents))+" query intent vectors for Session "+str(sessID)+", Query "+str(queryID)
    elapsedAppendTime = float(time.time()-startAppendTime)
    return elapsedAppendTime

def updateResponseTime(episodeResponseTime, numEpisodes, startEpisode, elapsedAppendTime):
    episodeResponseTime[numEpisodes] = float(time.time()-startEpisode) - elapsedAppendTime # we exclude the time consumed by appending predicted intents to the output intent file
    startEpisode = time.time()
    return (episodeResponseTime, startEpisode)

def createQueryExecIntentCreationTimes(configDict):
    numQueries = 0
    episodeQueryExecutionTime = {}
    episodeIntentCreationTime = {}
    numEpisodes = 0
    tempExecTimeEpisode = 0.0
    tempIntentTimeEpisode = 0.0
    with open(configDict['CONCURRENT_QUERY_SESSIONS']) as f:
        for line in f:
            sessQueries = line.split(";")
            sessQueryName = sessQueries[0]
            sessQuery = sessQueries[1].strip()
            queryVocabulary = {}
            (queryVocabulary, resObj, queryExecutionTime, intentCreationTime) = QExec.executeQueryWithIntent(sessQuery,
                                                                                                             configDict,
                                                                                                             queryVocabulary)
            tempExecTimeEpisode += float(queryExecutionTime)
            tempIntentTimeEpisode += float(intentCreationTime)
            print "Executed and obtained intent for " + sessQueryName
            numQueries += 1
            if numQueries % int(configDict['EPISODE_IN_QUERIES']) == 0:
                numEpisodes += 1
                episodeQueryExecutionTime[numEpisodes] = tempExecTimeEpisode
                episodeIntentCreationTime[numEpisodes] = tempIntentTimeEpisode
                tempExecTimeEpisode = 0.0
                tempIntentTimeEpisode = 0.0
    return (episodeQueryExecutionTime, episodeIntentCreationTime)

def readFromPickleFile(fileName):
    with open(fileName, 'rb') as handle:
        readObj = pickle.load(handle)
    return readObj

def writeToPickleFile(fileName, writeObj):
    with open(fileName, 'wb') as handle:
        pickle.dump(writeObj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def evaluateQualityPredictions(outputIntentFileName, configDict, accThres, algoName):
    outputEvalQualityFileName = configDict['OUTPUT_DIR'] + "/OutputEvalQualityShortTermIntent_" + algoName+"_"+configDict[
        'INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                                configDict['EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres)
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
            maxCosineSim = 0.0
            if configDict['BIT_OR_WEIGHTED'] == 'BIT':
                curQueryIntent = BitMap.fromstring(tokens[3].split(":")[1])
            elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
                curQueryIntent = tokens[3].split(":")[1]
            for i in range(4, len(tokens)):
                if configDict['BIT_OR_WEIGHTED'] == 'BIT':
                    topKQueryIntent = BitMap.fromstring(tokens[i].split(":")[1])
                    cosineSim = CFCosineSim.computeBitCosineSimilarity(curQueryIntent, topKQueryIntent)
                elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
                    topKQueryIntent = tokens[i].split(":")[1]
                    cosineSim = CFCosineSim.computeWeightedCosineSimilarity(curQueryIntent, topKQueryIntent, ",",
                                                                            configDict)
                if cosineSim >= float(accThres):
                    recall = 1.0
                    precision += 1.0
                if cosineSim > maxCosineSim:
                    maxCosineSim = cosineSim
            # print "float(len(tokens)-4 ="+str(len(tokens)-4)+", precision = "+str(precision/float(len(tokens)-4))
            precision /= float(len(tokens) - 4)
            outputEvalQualityStr = "Session:" + str(sessID) + ";Query:" + str(queryID) + ";#Episodes:" + str(
                numEpisodes) + ";Precision:" + str(precision) + ";Recall:" + str(recall) + ";Accuracy:" + str(
                maxCosineSim)
            ti.appendToFile(outputEvalQualityFileName, outputEvalQualityStr)

def evaluateTimePredictions(episodeResponseTimeDictName, configDict, algoName):
    outputEvalTimeFileName = configDict['OUTPUT_DIR'] + "/OutputEvalTimeShortTermIntent_" + algoName+"_"+\
                             configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                             configDict['EPISODE_IN_QUERIES']
    try:
        os.remove(outputEvalTimeFileName)
    except OSError:
        pass
    # Simulate or borrow query execution and intent creation to record their times #
    intentCreationTimeDictName = configDict['OUTPUT_DIR'] + "/IntentCreationTimeDict_" + configDict[
        'INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_EPISODE_IN_QUERIES_" + configDict[
                                     'EPISODE_IN_QUERIES'] + ".pickle"
    queryExecutionTimeDictName = configDict['OUTPUT_DIR'] + "/QueryExecutionTimeDict_" + configDict[
        'INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_EPISODE_IN_QUERIES_" + configDict[
                                     'EPISODE_IN_QUERIES'] + ".pickle"
    if os.path.exists(intentCreationTimeDictName) and os.path.exists(queryExecutionTimeDictName):
        episodeQueryExecutionTime = readFromPickleFile(queryExecutionTimeDictName)
        episodeIntentCreationTime = readFromPickleFile(intentCreationTimeDictName)
    else:
        (episodeQueryExecutionTime, episodeIntentCreationTime) = createQueryExecIntentCreationTimes(configDict)
        writeToPickleFile(queryExecutionTimeDictName, episodeQueryExecutionTime)
        writeToPickleFile(intentCreationTimeDictName, episodeIntentCreationTime)

    episodeResponseTime = readFromPickleFile(episodeResponseTimeDictName)

    print "len(episodeQueryExecutionTime) = " + str(
        len(episodeQueryExecutionTime)) + ", len(episodeIntentCreationTime) = " + str(
        len(episodeIntentCreationTime)) + ", len(episodeResponseTime) = " + str(len(episodeResponseTime))

    assert len(episodeQueryExecutionTime) == len(episodeResponseTime) and len(episodeIntentCreationTime) == len(
        episodeResponseTime)
    for episodes in range(1, len(episodeResponseTime)):
        totalResponseTime = float(episodeIntentCreationTime[episodes]) + float(
            episodeQueryExecutionTime[episodes]) + float(episodeResponseTime[episodes])
        outputEvalTimeStr = "#Episodes:" + str(episodes) + ";QueryExecutionTime(secs):" + str(
            episodeQueryExecutionTime[episodes]) + ";IntentCreationTime(secs):" + str(
            episodeIntentCreationTime[episodes]) + ";IntentPredictionTime(secs):" + str(
            episodeResponseTime[episodes]) + ";TotalResponseTime(secs):" + str(totalResponseTime)
        ti.appendToFile(outputEvalTimeFileName, outputEvalTimeStr)

def evaluatePredictions(outputIntentFileName, episodeResponseTimeDictName, configDict):
    evaluateQualityPredictions(outputIntentFileName, configDict, configDict['ACCURACY_THRESHOLD'], configDict['ALGORITHM'])
    evaluateTimePredictions(episodeResponseTimeDictName, configDict, configDict['ALGORITHM'])
    print "--Completed Quality and Time Evaluation--"
    return

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("configFile.txt")
    outputIntentFileName = configDict['OUTPUT_DIR']+"/OutputFileShortTermIntent_"+ configDict['ALGORITHM']+"_"+\
                           configDict['INTENT_REP']+"_"+configDict['BIT_OR_WEIGHTED']+"_TOP_K_"+configDict['TOP_K']+"_EPISODE_IN_QUERIES_"+configDict['EPISODE_IN_QUERIES']
    episodeResponseTimeDictName = configDict['OUTPUT_DIR'] + "/ResponseTimeDict_" +configDict['ALGORITHM']+"_"+configDict['INTENT_REP']+"_"+configDict['BIT_OR_WEIGHTED']+"_TOP_K_"+configDict['TOP_K']+"_EPISODE_IN_QUERIES_"+configDict['EPISODE_IN_QUERIES']+ ".pickle"
    #evaluatePredictions(outputIntentFileName, episodeResponseTimeDictName, configDict)
    accThresList = [0.95]
    for accThres in accThresList:
        evaluateQualityPredictions(outputIntentFileName, configDict, accThres, configDict['ALGORITHM'])
        print "--Completed Quality Evaluation for accThres:"+str(accThres)
    evaluateTimePredictions(episodeResponseTimeDictName, configDict, configDict['ALGORITHM'])

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

