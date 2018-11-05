from __future__ import division
import sys, operator
import os
import time
import QueryRecommender as QR
from bitmap import BitMap
import math
import heapq
import TupleIntent as ti
import ParseConfigFile as parseConfig
import ParseResultsToExcel
import ConcurrentSessions
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
import keras
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Activation, SimpleRNN, Dense, TimeDistributed, Flatten, LSTM, Dropout, GRU
import CFCosineSim
import LSTM_RNN
'''
def computePredictedIntentDictRNN(predictedY, sessionDict, configDict, curSessID):
    cosineSimDict = {}
    for sessID in sessionDict:
        if len(sessionDict)>1 and sessID == curSessID: # we are not going to suggest query intents from the same session
            break
        numQueries = len(sessionDict[sessID])
        for queryID in range(numQueries):
            queryIntent = sessionDict[sessID][queryID]
            cosineSim = CFCosineSim.computeListBitCosineSimilarity(predictedY, queryIntent, configDict)
            cosineSimDict[str(sessID)+","+str(queryID)] = cosineSim
    # sorted_d is a list of lists, not a dictionary. Each list entry has key as 0th entry and value as 1st entry, we need the key
    sorted_csd = sorted(cosineSimDict.items(), key=operator.itemgetter(1), reverse=True)
    topKPredictedIntents = {}
    maxTopK = int(configDict['TOP_K'])
    resCount = 0
    for cosSimEntry in sorted_csd:
        sessID = int(cosSimEntry[0].split(",")[0])
        queryID = int(cosSimEntry[0].split(",")[1])
        topKPredictedIntents[sessionDict[sessID][queryID]]=float(cosSimEntry[1])  #picks query intents only from already seen vocabulary
        resCount += 1
        if resCount >= maxTopK:
            break
    del cosineSimDict
    del sorted_csd
    return topKPredictedIntents
'''

def exampleSelection(modelRNN, availTrainDictX, availTrainDictY, holdOutTrainDictX, holdOutTrainDictY, trainSessionDict):
    #delete all the elements in availTrainX and availTrainY thus far because RNNs can be trained incrementally, old train data is redundant
    availTrainDictX.clear()
    availTrainDictY.clear()
    exampleBatchSize = int(configDict['ACTIVE_BATCH_SIZE'])
    minimaxCosineSimDict = {}
    i=0

    for sessIDQueryID in holdOutTrainDictX:
        leftX = np.array(holdOutTrainDictX[sessIDQueryID])
        sessID = int(sessIDQueryID.split(",")[0])
        predictedY = modelRNN.predict(leftX.reshape(1, leftX.shape[0], leftX.shape[1]))
        predictedY = predictedY[0][predictedY.shape[1] - 1]
        # predict topK intent vectors based on the weight vector
        topKPredictedIntents = LSTM_RNN.computePredictedIntentsRNN(predictedY, trainSessionDict, configDict, sessID)
        maxCosineSim = CFCosineSim.computeListBitCosineSimilarity(predictedY, topKPredictedIntents[0], configDict)
        minimaxCosineSimDict[sessIDQueryID] = maxCosineSim
    sorted_minimaxCSD = sorted(minimaxCosineSimDict.items(), key=operator.itemgetter(1)) # we sort in ASC order
    resCount = 0
    for cosSimEntry in sorted_minimaxCSD:
        sessIDQueryID = cosSimEntry[0]
        availTrainDictX[sessIDQueryID] = holdOutTrainDictX[sessIDQueryID]
        availTrainDictY[sessIDQueryID] = holdOutTrainDictY[sessIDQueryID]
        del holdOutTrainDictX[sessIDQueryID]
        del holdOutTrainDictY[sessIDQueryID]
        resCount+=1
        if resCount >= exampleBatchSize:
            break
    return (availTrainDictX, availTrainDictY, holdOutTrainDictX, holdOutTrainDictY)

def createAvailHoldOutDicts(trainX, trainY, trainKeyOrder):
    availTrainDictX = {}
    availTrainDictY = {}  # we can have key from keyOrder paired with dataX and dataY
    holdOutTrainDictX = {}
    holdOutTrainDictY = {}
    assert len(trainX) == len(trainY) and len(trainX) > int(configDict['ACTIVE_SEED_TRAINING_SIZE'])
    totalSize = len(trainX)
    i=0
    # traverse trainKeyOrder
    while i< totalSize:
        curElemX = trainX[0]
        curElemY = trainY[0]
        sessIDQueryID = trainKeyOrder[i]
        if i < int(configDict['ACTIVE_SEED_TRAINING_SIZE']):
            availTrainDictX[sessIDQueryID] = curElemX
            availTrainDictY[sessIDQueryID] = curElemY
        else:
            holdOutTrainDictX[sessIDQueryID] = curElemX
            holdOutTrainDictY[sessIDQueryID] = curElemY
        trainX.pop(0)
        trainY.pop(0)
        i+=1
    assert len(holdOutTrainDictX) == totalSize-int(configDict['ACTIVE_SEED_TRAINING_SIZE']) and len(availTrainDictX) == int(configDict['ACTIVE_SEED_TRAINING_SIZE']) and len(availTrainDictX) == len(availTrainDictY)
    return (availTrainDictX, availTrainDictY, holdOutTrainDictX, holdOutTrainDictY)

def initRNNOneFoldActiveTrainTest(trainIntentSessionFile, testIntentSessionFile, configDict):
    # the purpose of this method is to create two training sets - one available and another hold-out
    (trainSessionDict, sessionLengthDict, trainSessionStreamDict, trainKeyOrder, modelRNN) = LSTM_RNN.initRNNOneFoldTrain(trainIntentSessionFile, configDict)
    # we got sessionLengthDict, sessionStreamDict and keyOrder non null, remaining are just initialized
    (trainX, trainY) = LSTM_RNN.createTemporalPairs(trainKeyOrder, configDict, trainSessionDict, trainSessionStreamDict)
    # keep ACTIVE_SEED_TRAINING_SIZE pairs in available and remaining in hold-out

    (availTrainKeyX, availTrainKeyY, holdOutTrainX, holdOutTrainY) = createAvailHoldOutDicts(trainX, trainY, trainKeyOrder)
    testSessionDict = []
    (testSessionStreamDict, testKeyOrder, testEpisodeResponseTime) = LSTM_RNN.initRNNOneFoldTest(testIntentSessionFile,
                                                                                        configDict)
    return (sessionLengthDict, trainSessionDict, trainSessionStreamDict, trainKeyOrder, modelRNN, testSessionDict, testSessionStreamDict, testKeyOrder, testEpisodeResponseTime, availTrainKeyX, availTrainKeyY, holdOutTrainX, holdOutTrainY)

def testOneFold(foldID, keyOrder, sessionStreamDict, sessionLengthDict, modelRNN, sessionDict, episodeResponseTime, outputIntentFileName, episodeResponseTimeDictName, configDict):
    numEpisodes = 1
    startEpisode = time.time()
    prevSessID = -1
    elapsedAppendTime = 0.0
    for key in keyOrder:
        sessID = int(key.split(",")[0])
        queryID = int(key.split(",")[1])
        curQueryIntent = sessionStreamDict[key]
        if prevSessID != sessID:
            if prevSessID in sessionDict:
                del sessionDict[prevSessID] # bcoz none of the test session queries should be used for test phase prediction for a different session, so delete a test session-info once it is done with
                (episodeResponseTime, startEpisode, elapsedAppendTime) = QR.updateResponseTime(episodeResponseTime,
                                                                                               numEpisodes,
                                                                                               startEpisode,
                                                                                               elapsedAppendTime)
                numEpisodes += 1  # episodes start from 1
            prevSessID = sessID

        #update sessionDict with this new query
        updateSessionDictWithCurrentIntent(sessionDict, sessID, curQueryIntent)

        if modelRNN is not None and queryID < sessionLengthDict[sessID] - 1:
            predictedY = predictTopKIntents(modelRNN, sessionDict, sessID, configDict)
            nextQueryIntent = sessionStreamDict[str(sessID) + "," + str(queryID + 1)]
            nextIntentList = createCharListFromIntent(nextQueryIntent, configDict)
            actual_vector = np.array(nextIntentList).astype(np.int)
            # actual_vector = np.array(actual_vector[actual_vector.shape[0] - 1]).astype(np.int)
            #cosineSim = dot(predictedY, actual_vector) / (norm(predictedY) * norm(actual_vector))
            if configDict['BIT_OR_WEIGHTED'] == 'BIT':
                topKPredictedIntents = computePredictedIntentsRNN(predictedY, sessionDict, configDict, sessID)
            elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
                topKPredictedIntents = QR.computeWeightedVectorFromList(predictedY)
            elapsedAppendTime += QR.appendPredictedRNNIntentToFile(sessID, queryID, topKPredictedIntents, nextQueryIntent, numEpisodes,
                                                                   outputIntentFileName, configDict, foldID)
    (episodeResponseTime, startEpisode, elapsedAppendTime) = QR.updateResponseTime(episodeResponseTime,
                                                                                   numEpisodes,
                                                                                   startEpisode,
                                                                                   elapsedAppendTime) # last session
    QR.writeToPickleFile(episodeResponseTimeDictName, episodeResponseTime)
    return (outputIntentFileName, episodeResponseTimeDictName)

def testActiveRNN(foldID, sessionLengthDict, testSessionDict, testKeyOrder, testSessionStreamDict, modelRNN):
    prevSessID = -1
    numEpisodes = 0
    avgAccuracyPerSession = []
    avgFMeasurePerSession = []
    avgPrecisionPerSession = []
    avgRecallPerSession = []
    avgFMeasure =0.0
    avgAccuracy = 0.0
    avgPrecision = 0.0
    avgRecall =0.0
    for sessIDQueryID in testKeyOrder:
        sessID = int(sessIDQueryID.split(",")[0])
        queryID = int(sessIDQueryID.split(",")[1])
        curQueryIntent = testSessionStreamDict[sessIDQueryID]
        if prevSessID != sessID:
            if prevSessID in testSessionDict:
                assert len(testSessionDict[prevSessID])>1
                avgAccuracyPerSession.append(float(avgAccuracy)/float(len(testSessionDict[prevSessID])-1))
                avgFMeasurePerSession.append(float(avgFMeasure)/float(len(testSessionDict[prevSessID])-1))
                avgPrecisionPerSession.append(float(avgPrecision)/float(len(testSessionDict[prevSessID])-1))
                avgRecallPerSession.append(float(avgRecall)/float(len(testSessionDict[prevSessID])-1))
                avgFMeasure = 0.0
                avgAccuracy = 0.0
                avgPrecision = 0.0
                avgRecall = 0.0
                del testSessionDict[prevSessID] # bcoz none of the test session queries should be used for test phase prediction for a different session, so delete a test session-info once it is done with
                numEpisodes += 1  # episodes start from 1
            prevSessID = sessID
        # update sessionDict with this new query
        LSTM_RNN.updateSessionDictWithCurrentIntent(testSessionDict, sessID, curQueryIntent)
        if modelRNN is not None and queryID < sessionLengthDict[sessID] - 1:
            predictedY = LSTM_RNN.predictTopKIntents(modelRNN, testSessionDict, sessID, configDict)
            if configDict['BIT_OR_WEIGHTED'] == 'BIT':
                topKPredictedIntents = LSTM_RNN.computePredictedIntentsRNN(predictedY, testSessionDict, configDict, sessID)
            elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
                topKPredictedIntents = QR.computeWeightedVectorFromList(predictedY)
            #compare topK with testY
            nextQueryIntent = testSessionStreamDict[str(sessID) + "," + str(queryID + 1)]
            output_str = QR.computePredictedOutputStrRNN(sessID, queryID, topKPredictedIntents, nextQueryIntent, numEpisodes, configDict)
            (sessID, queryID, numEpisodes, accuracyAtMaxFMeasure, precisionAtMaxFMeasure, recallAtMaxFMeasure,
             maxFMeasure) = QR.computeQueRIEFMeasureForEachEpisode(output_str, configDict)
            avgFMeasure+=maxFMeasure
            avgAccuracy+=accuracyAtMaxFMeasure
            avgPrecision+=precisionAtMaxFMeasure
            avgRecall+=recallAtMaxFMeasure
    avgFMeasure=float(sum(avgFMeasurePerSession))/float(numEpisodes)
    avgAccuracy = float(sum(avgAccuracyPerSession))/float(numEpisodes)
    avgPrecision = float(sum(avgPrecisionPerSession))/float(numEpisodes)
    avgRecall =  float(sum(avgRecallPerSession))/float(numEpisodes)
    return (avgFMeasure, avgAccuracy, avgPrecision, avgRecall)


def runActiveRNNKFoldExp(configDict):
    intentSessionFile = QR.fetchIntentFileFromConfigDict(configDict)
    kFoldOutputIntentFiles = []
    kFoldEpisodeResponseTimeDicts = []
    avgTrainTime = []
    avgTestTime = []
    algoName = configDict['ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"]
    for foldID in range(int(configDict['KFOLD'])):
        outputIntentFileName = configDict['KFOLD_OUTPUT_DIR'] + "/OutputFileShortTermIntent_" + algoName + "_" + \
                               configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + \
                               configDict['TOP_K'] + "_FOLD_" + str(foldID)
        episodeResponseTimeDictName = configDict['KFOLD_OUTPUT_DIR'] + "/ResponseTimeDict_" + algoName + "_" + \
                                      configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + \
                                      configDict['TOP_K'] + "_FOLD_" + str(foldID) + ".pickle"
        trainIntentSessionFile = configDict['KFOLD_INPUT_DIR'] + intentSessionFile.split("/")[len(intentSessionFile.split("/")) - 1] + "_TRAIN_FOLD_" + str(foldID)
        testIntentSessionFile = configDict['KFOLD_INPUT_DIR'] + intentSessionFile.split("/")[len(intentSessionFile.split("/")) - 1] + "_TEST_FOLD_" + str(foldID)
        (sessionLengthDict, trainSessionDict, trainSessionStreamDict, trainKeyOrder, modelRNN, testSessionDict,
         testSessionStreamDict, testKeyOrder, testEpisodeResponseTime, availTrainDictX, availTrainDictY, holdOutTrainDictX,
         holdOutTrainDictY) = initRNNOneFoldActiveTrainTest(trainIntentSessionFile, testIntentSessionFile, configDict)
        activeIter = 0
        while len(holdOutTrainDictX) > 0:
            startTrain = time.time()
            modelRNN = LSTM_RNN.trainRNN(availTrainDictX.values(), availTrainDictY.values(), modelRNN)
            trainTime = float(time.time() - startTrain)
            avgTrainTime.append(trainTime)
            # example selection phase
            (availTrainDictX, availTrainDictY, holdOutTrainDictX, holdOutTrainDictY) = exampleSelection(modelRNN, availTrainDictX, availTrainDictY, holdOutTrainDictX, holdOutTrainDictY, trainSessionDict)
            (avgFMeasure, avgAccuracy, avgPrecision, avgRecall) = testActiveRNN(foldID, sessionLengthDict, testSessionDict, testKeyOrder, testSessionStreamDict, modelRNN)
            activeIter+=1

        return

def executeAL(configDict):
    # ActiveLearning runs only on kFold
    assert configDict['SINGULARITY_OR_KFOLD']=='KFOLD'
    runActiveRNNKFoldExp(configDict)
    return

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("configFile.txt")
    executeAL(configDict)