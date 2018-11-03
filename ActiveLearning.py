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

def exampleSelection(modelRNN, availTrainDictX, availTrainDictY, holdOutTrainDictX, holdOutTrainDictY, trainSessionDict, trainKeyOrder):
    exampleBatchSize = int(configDict['ACTIVE_BATCH_SIZE'])
    # python supports for min-heap not max-heap so negate items and insert into min-heap
    minCosineSimHeap = []
    i=0

    for sessIDQueryID in holdOutTrainDictX:
        leftX = np.array(holdOutTrainDictX[sessIDQueryID])
        sessID = int(sessIDQueryID.split(",")[0])
        predictedY = modelRNN.predict(leftX.reshape(1, leftX.shape[0], leftX.shape[1]))
        predictedY = predictedY[0][predictedY.shape[1] - 1]
        # predict topK intent vectors based on the weight vector
        assert configDict['BIT_OR_WEIGHTED'] == 'BIT'
        topKPredictedIntents = computePredictedIntentDictRNN(predictedY, trainSessionDict, configDict, sessID)

    return

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
    (testX, testY) = LSTM_RNN.createTemporalPairs(testKeyOrder, configDict, testSessionDict, testSessionStreamDict)
    return (sessionLengthDict, trainSessionDict, trainSessionStreamDict, trainKeyOrder, modelRNN, testSessionDict, testSessionStreamDict, testKeyOrder, testEpisodeResponseTime, availTrainKeyX, availTrainKeyY, holdOutTrainX, holdOutTrainY, testX, testY)


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
         holdOutTrainDictY, testX, testY) = initRNNOneFoldActiveTrainTest(trainIntentSessionFile, testIntentSessionFile, configDict)
        activeIter = 0
        while len(holdOutTrainDictX) > 0:
            startTrain = time.time()
            modelRNN = LSTM_RNN.trainRNN(availTrainDictX.values(), availTrainDictY.values(), modelRNN)
            trainTime = float(time.time() - startTrain)
            avgTrainTime.append(trainTime)
            # example selection phase
            (availTrainDictX, availTrainDictY) = exampleSelection(modelRNN, availTrainDictX, availTrainDictY, holdOutTrainDictX, holdOutTrainDictY, trainSessionDict, trainKeyOrder)
            activeIter+=1

        (testSessionStreamDict, testKeyOrder, testEpisodeResponseTime) = initRNNOneFoldTest(testIntentSessionFile, configDict)
        startTest = time.time()
        (outputIntentFileName, episodeResponseTimeDictName) = testOneFold(foldID, testKeyOrder, testSessionStreamDict, sessionLengthDict, modelRNN, sessionDict, testEpisodeResponseTime, outputIntentFileName, episodeResponseTimeDictName, configDict)
        testTime = float(time.time() - startTest)
        avgTestTime.append(testTime)
        kFoldOutputIntentFiles.append(outputIntentFileName)
        kFoldEpisodeResponseTimeDicts.append(episodeResponseTimeDictName)
    (avgTrainTimeFN, avgTestTimeFN) = QR.writeKFoldTrainTestTimesToPickleFiles(avgTrainTime, avgTestTime, algoName, configDict)
    QR.avgKFoldTimeAndQualityPlots(kFoldOutputIntentFiles,kFoldEpisodeResponseTimeDicts, avgTrainTimeFN, avgTestTimeFN, algoName, configDict)
    return

def executeAL(configDict):
    # ActiveLearning runs only on kFold
    assert configDict['SINGULARITY_OR_KFOLD']=='KFOLD'
    runActiveRNNKFoldExp(configDict)
    return

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("configFile.txt")
    executeAL(configDict)