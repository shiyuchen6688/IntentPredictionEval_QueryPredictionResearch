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

def fetchIntentFileFromConfigDict(configDict):
    if configDict['INTENT_REP'] == 'TUPLE':
        intentSessionFile = configDict['TUPLEINTENTSESSIONS']
    elif configDict['INTENT_REP'] == 'FRAGMENT' and configDict['BIT_OR_WEIGHTED'] == 'BIT':
        intentSessionFile = configDict['BIT_FRAGMENT_INTENT_SESSIONS']
    elif configDict['INTENT_REP'] == 'FRAGMENT' and configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
        intentSessionFile = configDict['WEIGHTED_FRAGMENT_INTENT_SESSIONS']
    elif configDict['INTENT_REP'] == 'QUERY':
        intentSessionFile = configDict['QUERY_INTENT_SESSIONS']
    else:
        print "ConfigDict['INTENT_REP'] must either be TUPLE or FRAGMENT or QUERY !!"
        sys.exit(0)
    return intentSessionFile

def updateSessionDict(line, configDict, sessionStreamDict):
    (sessID, queryID, curQueryIntent) = retrieveSessIDQueryIDIntent(line, configDict)
    if str(sessID)+","+str(queryID) in sessionStreamDict:
        print str(sessID)+","+str(queryID)+ " already exists !!"
        sys.exit(0)
    sessionStreamDict[str(sessID)+","+str(queryID)] = curQueryIntent
    return (sessID, queryID, curQueryIntent, sessionStreamDict)

def updateSessionLineDict(line, configDict, sessionLineDict):
    (sessID, queryID, curQueryIntent) = retrieveSessIDQueryIDIntent(line, configDict)
    if str(sessID)+","+str(queryID) in sessionLineDict:
        print str(sessID)+","+str(queryID)+ " already exists !!"
        sys.exit(0)
    sessionLineDict[str(sessID)+","+str(queryID)] = line.strip()
    return sessionLineDict

def findNextQueryIntent(intentSessionFile, sessID, queryID, configDict, lines):
    #with open(intentSessionFile) as f:
    for line in lines:
        (curSessID, curQueryID, curQueryIntent) = retrieveSessIDQueryIDIntent(line, configDict)
        if curSessID == sessID and curQueryID == queryID:
            #f.close()
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
        normalizedVector.append(str(float(token)/total))
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

def appendPredictedIntentsToFile(topKSessQueryIndices, topKPredictedIntents, sessID, queryID, actualQueryIntent, numEpisodes, configDict, outputIntentFileName):
    startAppendTime = time.time()
    output_str = "Session:"+str(sessID)+";Query:"+str(queryID)+";#Episodes:"+str(numEpisodes)+";ActualQueryIntent:"
    if configDict['BIT_OR_WEIGHTED'] == 'BIT':
        output_str += actualQueryIntent.tostring()
    elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
        if ";" in actualQueryIntent:
            actualQueryIntent.replace(";",",")
        output_str += actualQueryIntent
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
    elapsedAppendTime = 0.0
    startEpisode = time.time()
    return (episodeResponseTime, startEpisode, elapsedAppendTime)

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

def computeBitFMeasure(actualQueryIntent, topKQueryIntent):
    assert actualQueryIntent.size() == topKQueryIntent.size()
    TP=0
    FP=0
    TN=0
    FN=0
    for pos in range(actualQueryIntent.size()):
        if actualQueryIntent.test(pos) and topKQueryIntent.test(pos):
            TP+=1
        elif not actualQueryIntent.test(pos) and not topKQueryIntent.test(pos):
            TN+=1
        elif actualQueryIntent.test(pos) and not topKQueryIntent.test(pos):
            FN+=1
        elif not actualQueryIntent.test(pos) and topKQueryIntent.test(pos):
            FP+=1
    if TP == 0 and FP == 0:
        precision = 0.0
    else:
        precision = float(TP)/float(TP+FP)
    if TP == 0 and FN == 0:
        recall = 0.0
    else:
        recall = float(TP)/float(TP+FN)
    if precision == 0.0 and recall == 0.0:
        FMeasure = 0.0
    else:
        FMeasure = 2 * precision * recall / (precision + recall)
    accuracy = float(TP+TN)/float(TP+FP+TN+FN)
    return (precision, recall, FMeasure, accuracy)

def computeWeightedFMeasure(actualQueryIntent, topKQueryIntent, delimiter, configDict):
    groundTruthDims = actualQueryIntent.split(delimiter)
    predictedDims = topKQueryIntent.split(delimiter)
    assert groundTruthDims.size() == predictedDims.size()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for pos in range(groundTruthDims.size()):
        if groundTruthDims[pos] == '1' and predictedDims[pos]  == '1':
            TP += 1
        elif groundTruthDims[pos] == '0' and predictedDims[pos]  == '0':
            TN += 1
        elif groundTruthDims[pos] == '1' and predictedDims[pos]  == '0':
            FN += 1
        elif groundTruthDims[pos] == '0' and predictedDims[pos]  == '1':
            FP += 1
    if TP == 0 and FP == 0:
        precision = 0.0
    else:
        precision = float(TP) / float(TP + FP)
    if TP == 0 and FN == 0:
        recall = 0.0
    else:
        recall = float(TP) / float(TP + FN)
    if precision == 0.0 and recall == 0.0:
        FMeasure = 0.0
    else:
        FMeasure = 2 * precision * recall / (precision + recall)
    accuracy = float(TP + TN) / float(TP + FP + TN + FN)
    return (precision, recall, FMeasure, accuracy)

def computeQueRIEFMeasureForEachEpisode(line, configDict):
    tokens = line.strip().split(";")
    sessID = tokens[0].split(":")[1]
    queryID = tokens[1].split(":")[1]
    numEpisodes = tokens[2].split(":")[1]
    precisionAtMaxFMeasure = 0.0
    recallAtMaxFMeasure = 0.0
    maxFMeasure = 0.0
    accuracyAtMaxFMeasure = 0.0
    if configDict['BIT_OR_WEIGHTED'] == 'BIT':
        actualQueryIntent = BitMap.fromstring(tokens[3].split(":")[1])
    elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
        actualQueryIntent = tokens[3].split(":")[1]
    for i in range(4, len(tokens)):
        if configDict['BIT_OR_WEIGHTED'] == 'BIT':
            topKQueryIntent = BitMap.fromstring(tokens[i].split(":")[1])
            (precision, recall, FMeasure, accuracy) = computeBitFMeasure(actualQueryIntent, topKQueryIntent)
        elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
            topKQueryIntent = tokens[i].split(":")[1]
            (precision, recall, FMeasure, accuracy) = computeWeightedFMeasure(actualQueryIntent, topKQueryIntent, ",",
                                                                    configDict)
        if FMeasure > maxFMeasure:
            maxFMeasure = FMeasure
            precisionAtMaxFMeasure = precision
            recallAtMaxFMeasure = recall
            accuracyAtMaxFMeasure = accuracy
        #if precision > maxPrecision:
        #if recall > maxRecall:
        #if accuracy > maxAccuracy:
    # print "float(len(tokens)-4 ="+str(len(tokens)-4)+", precision = "+str(precision/float(len(tokens)-4))
    return (sessID, queryID, numEpisodes, accuracyAtMaxFMeasure, precisionAtMaxFMeasure, recallAtMaxFMeasure, maxFMeasure)

def computeCosineSimFMeasureForEachEpisode(line, configDict):
    tokens = line.strip().split(";")
    sessID = tokens[0].split(":")[1]
    queryID = tokens[1].split(":")[1]
    numEpisodes = tokens[2].split(":")[1]
    precision = 0.0
    recall = 0.0
    maxCosineSim = 0.0
    accThres = float(configDict['ACCURACY_THRESHOLD'])
    if configDict['BIT_OR_WEIGHTED'] == 'BIT':
        actualQueryIntent = BitMap.fromstring(tokens[3].split(":")[1])
    elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
        actualQueryIntent = tokens[3].split(":")[1]
    for i in range(4, len(tokens)):
        if configDict['BIT_OR_WEIGHTED'] == 'BIT':
            topKQueryIntent = BitMap.fromstring(tokens[i].split(":")[1])
            cosineSim = CFCosineSim.computeBitCosineSimilarity(actualQueryIntent, topKQueryIntent)
        elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
            topKQueryIntent = tokens[i].split(":")[1]
            cosineSim = CFCosineSim.computeWeightedCosineSimilarity(actualQueryIntent, topKQueryIntent, ",",
                                                                    configDict)
        if cosineSim >= float(accThres):
            recall = 1.0
            precision += 1.0
        if cosineSim > maxCosineSim:
            maxCosineSim = cosineSim
    # print "float(len(tokens)-4 ="+str(len(tokens)-4)+", precision = "+str(precision/float(len(tokens)-4))
    precision /= float(len(tokens) - 4)
    if precision == 0 or recall == 0:
        FMeasure = 0
    else:
        FMeasure = 2 * precision * recall / (precision + recall)
    return (sessID, queryID, numEpisodes, maxCosineSim, precision, recall, FMeasure)

def computeAccuracyForEachEpisode(line, configDict):
    assert configDict['COSINESIM_OR_QUERIE_FMEASURE'] == 'COSINESIM' or configDict['COSINESIM_OR_QUERIE_FMEASURE'] == 'QUERIE'
    if configDict['COSINESIM_OR_QUERIE_FMEASURE'] == 'COSINESIM':
        (sessID, queryID, numEpisodes, accuracy, precision, recall, FMeasure) = computeCosineSimFMeasureForEachEpisode(line, configDict)
    elif configDict['COSINESIM_OR_QUERIE_FMEASURE'] == 'QUERIE':
        (sessID, queryID, numEpisodes, accuracy, precision, recall, FMeasure) = computeQueRIEFMeasureForEachEpisode(line, configDict)
    return (sessID, queryID, numEpisodes, accuracy, precision, recall, FMeasure)

def appendToDict(avgDict, key, value):
    if key not in avgDict:
        avgDict[key] = []
    avgDict[key].append(value)
    return avgDict

def computeAvgFoldAccuracy(kFoldOutputIntentFiles, configDict):
    algoName = None
    if configDict['ALGORITHM'] == 'CF':
        algoName = configDict['ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF']
    elif configDict['ALGORITHM'] == 'RNN':
        algoName = configDict['ALGORITHM'] + "_" + configDict['RNN_BACKPROP_LSTM_GRU']
    avgMaxAccuracy = {}
    avgPrecision = {}
    avgRecall = {}
    avgFMeasure = {}
    accThres = configDict['ACCURACY_THRESHOLD']
    for foldOutputIntentFile in kFoldOutputIntentFiles:
        with open(foldOutputIntentFile) as f:
            for line in f:
                (sessID, queryID, numEpisodes, accuracy, precision, recall, FMeasure) = computeAccuracyForEachEpisode(line, configDict)
                avgMaxAccuracy = appendToDict(avgMaxAccuracy, numEpisodes, accuracy)
                avgPrecision = appendToDict(avgPrecision, numEpisodes, precision)
                avgRecall = appendToDict(avgRecall, numEpisodes, recall)
                avgFMeasure = appendToDict(avgFMeasure, numEpisodes, FMeasure)
    outputEvalQualityFileName = configDict['KFOLD_OUTPUT_DIR'] + "/OutputEvalQualityShortTermIntent_" + algoName + "_" + configDict[
        'INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_ACCURACY_THRESHOLD_" + str(accThres)
    try:
        os.remove(outputEvalQualityFileName)
    except OSError:
        pass
    assert len(avgMaxAccuracy) == len(avgPrecision) and len(avgPrecision) == len(avgRecall) and len(avgRecall) == len(avgFMeasure)
    episodeIndex = 0
    for key in avgMaxAccuracy:
        episodeIndex+=1
        outputAccuracy = float(sum(avgMaxAccuracy[key])) / float(len(avgMaxAccuracy[key]))
        outputPrecision = float(sum(avgPrecision[key])) / float(len(avgPrecision[key]))
        outputRecall = float(sum(avgRecall[key])) / float(len(avgRecall[key]))
        outputFMeasure = float(sum(avgFMeasure[key])) / float(len(avgFMeasure[key]))
        outputEvalQualityStr = "#Episodes:" + str(
            episodeIndex) + ";Precision:" + str(outputPrecision) + ";Recall:" + str(outputRecall) + ";FMeasure:" + str(outputFMeasure)+ ";Accuracy:" + str(outputAccuracy)
        ti.appendToFile(outputEvalQualityFileName, outputEvalQualityStr)
    return outputEvalQualityFileName

def evaluateQualityPredictions(outputIntentFileName, configDict, accThres, algoName):
    assert configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY' or configDict['SINGULARITY_OR_KFOLD'] == 'KFOLD'
    outputDir = None
    if configDict['SINGULARITY_OR_KFOLD'] =='SINGULARITY':
        outputDir = configDict['OUTPUT_DIR']
    elif configDict['SINGULARITY_OR_KFOLD'] =='KFOLD':
        outputDir = configDict['KFOLD_OUTPUT_DIR']
    outputEvalQualityFileName = outputDir + "/OutputEvalQualityShortTermIntent_" + algoName+"_"+configDict[
        'INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                                configDict['EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres)
    try:
        os.remove(outputEvalQualityFileName)
    except OSError:
        pass
    with open(outputIntentFileName) as f:
        for line in f:
            (sessID, queryID, numEpisodes, accuracy, precision, recall, FMeasure) = computeAccuracyForEachEpisode(line,
                                                                                                        configDict)
            outputEvalQualityStr = "Session:" + str(sessID) + ";Query:" + str(queryID) + ";#Episodes:" + str(
                numEpisodes) + ";Precision:" + str(precision) + ";Recall:" + str(recall) + ";FMeasure:" + str(FMeasure) +";Accuracy:" + str(
                accuracy)
            ti.appendToFile(outputEvalQualityFileName, outputEvalQualityStr)


def computeAvgFoldTime(kFoldEpisodeResponseTimeDicts, configDict):
    avgKFoldTimeDict = {}
    for kFoldEpisodeTimeDict in kFoldEpisodeResponseTimeDicts:
        episodeResponseTime = readFromPickleFile(kFoldEpisodeTimeDict)
        for episodes in range(1, len(episodeResponseTime)):
            if episodes not in avgKFoldTimeDict:
                avgKFoldTimeDict[episodes] = []
            avgKFoldTimeDict[episodes].append(episodeResponseTime[episodes])
    for episodes in range(1, len(avgKFoldTimeDict)):
        avgKFoldTimeDict[episodes] = float(sum(avgKFoldTimeDict[episodes]))/float(len(avgKFoldTimeDict[episodes]))
    return avgKFoldTimeDict

def evaluateTimePredictions(episodeResponseTimeDictName, configDict, algoName):
    assert configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY' or configDict['SINGULARITY_OR_KFOLD'] == 'KFOLD'
    outputDir = None
    if configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY':
        outputDir = configDict['OUTPUT_DIR']
    elif configDict['SINGULARITY_OR_KFOLD'] == 'KFOLD':
        outputDir = configDict['KFOLD_OUTPUT_DIR']
    outputEvalTimeFileName = outputDir + "/OutputEvalTimeShortTermIntent_" + algoName+"_"+\
                             configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                             configDict['EPISODE_IN_QUERIES']
    try:
        os.remove(outputEvalTimeFileName)
    except OSError:
        pass
    # Simulate or borrow query execution and intent creation to record their times #
    # the following should be configDict['OUTPUT_DIR] and not outputDir because it gets intent creation and queryExec times from the existing pickle files in the outer directory for kfold exp"
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
    return outputEvalTimeFileName

def evaluatePredictions(outputIntentFileName, episodeResponseTimeDictName, configDict):
    evaluateQualityPredictions(outputIntentFileName, configDict, configDict['ACCURACY_THRESHOLD'], configDict['ALGORITHM'])
    evaluateTimePredictions(episodeResponseTimeDictName, configDict, configDict['ALGORITHM'])
    print "--Completed Quality and Time Evaluation--"
    return

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("configFile.txt")
    accThres = float(configDict['ACCURACY_THRESHOLD'])
    algoName = None
    outputDir=None
    outputEvalQualityFileName = None
    if configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY':
        outputDir = configDict['OUTPUT_DIR']
    elif configDict['SINGULARITY_OR_KFOLD'] == 'KFOLD':
        outputDir = configDict['KFOLD_OUTPUT_DIR']
    if configDict['ALGORITHM'] == 'CF':
        algoName = configDict['ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF']
        if configDict['SINGULARITY_OR_KFOLD'] == 'KFOLD':
            outputIntentFileName = configDict[
                                            'KFOLD_OUTPUT_DIR'] + "/OutputEvalQualityShortTermIntent_" + algoName + "_" + \
                                        configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + \
                                        configDict['TOP_K'] + "_ACCURACY_THRESHOLD_" + str(accThres)
        elif configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY':
            outputIntentFileName = outputDir + "/OutputEvalQualityShortTermIntent_" + configDict[
                'ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF'] + "_" + configDict['INTENT_REP'] + "_" + configDict[
                                            'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                                            'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict[
                                            'EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres)
    elif configDict['ALGORITHM'] == 'RNN':
        algoName = configDict['ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"]
        outputIntentFileName = outputDir + "/OutputFileShortTermIntent_" +configDict['ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"] + "_" + \
                                configDict['INTENT_REP'] + "_" + \
                                configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                                configDict['EPISODE_IN_QUERIES']
        #evaluatePredictions(outputIntentFileName, episodeResponseTimeDictName, configDict)
    evaluateQualityPredictions(outputIntentFileName, configDict, accThres, configDict['ALGORITHM'])
    print "--Completed Quality Evaluation for accThres:"+str(accThres)
    if configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY':
        episodeResponseTimeDictName = outputDir + "/ResponseTimeDict_" + algoName + "_" + \
                                      configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + \
                                      configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict[
                                          'EPISODE_IN_QUERIES'] + ".pickle"
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

