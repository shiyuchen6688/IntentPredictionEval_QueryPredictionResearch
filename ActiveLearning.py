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
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Activation, SimpleRNN, Dense, TimeDistributed, Flatten, LSTM, Dropout, GRU
import CFCosineSim
import LSTM_RNN
import random
import argparse

def exampleSelectionRandom(foldID, activeIter, availTrainDictX, availTrainDictY, holdOutTrainDictX, holdOutTrainDictY):
    assert configDict['ACTIVE_EXSEL_STRATEGY_MINIMAX_RANDOM'] == 'RANDOM'
    assert configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL' or configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'FULL'
    # get rid of the data that you trained on so far for incremental train
    if configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL':
        del availTrainDictX
        del availTrainDictY
        availTrainDictX = {}
        availTrainDictY = {}
    exampleBatchSize = int(configDict['ACTIVE_BATCH_SIZE'])
    print "foldID: "+str(foldID)+", activeIter: "+str(activeIter)+", #Hold-out-Pairs: "+str(len(holdOutTrainDictX))
    # random selection
    if len(holdOutTrainDictX) < exampleBatchSize:
        exampleBatchSize = len(holdOutTrainDictX)
    chosenKeys = random.sample(holdOutTrainDictX.keys(), exampleBatchSize)
    for sessIDQueryID in chosenKeys:
        availTrainDictX[sessIDQueryID] = holdOutTrainDictX[sessIDQueryID]
        availTrainDictY[sessIDQueryID] = holdOutTrainDictY[sessIDQueryID]
        del holdOutTrainDictX[sessIDQueryID]
        del holdOutTrainDictY[sessIDQueryID]
        print "foldID: " + str(foldID) + ", activeIter: " + str(activeIter) +", Added "+str(len(chosenKeys))+"th example, sessIDQueryID: "+str(sessIDQueryID)+" to the data"
    return (availTrainDictX, availTrainDictY, holdOutTrainDictX, holdOutTrainDictY)

def exampleSelectionMinimax(foldID, activeIter, modelRNN, max_lookback, availTrainDictX, availTrainDictY, holdOutTrainDictX, holdOutTrainDictY, trainSessionDict):
    assert configDict['ACTIVE_EXSEL_STRATEGY_MINIMAX_RANDOM'] == 'MINIMAX'
    assert configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL' or configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'FULL'
    # get rid of the data that you trained on so far for incremental train
    if configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL':
        del availTrainDictX
        del availTrainDictY
        availTrainDictX = {}
        availTrainDictY = {}
    exampleBatchSize = int(configDict['ACTIVE_BATCH_SIZE'])
    minimaxCosineSimDict = {}
    i=0
    print "foldID: "+str(foldID)+", activeIter: "+str(activeIter)+", #Hold-out-Pairs: "+str(len(holdOutTrainDictX))
    for sessIDQueryID in holdOutTrainDictX:
        leftX = np.array(holdOutTrainDictX[sessIDQueryID])
        leftX = leftX.reshape(1, leftX.shape[0], leftX.shape[1])
        if len(leftX) < max_lookback:
            leftX = pad_sequences(leftX, maxlen=max_lookback, padding='pre')
        sessID = int(sessIDQueryID.split(",")[0])

        predictedY = modelRNN.predict(leftX)
        predictedY = predictedY[0][predictedY.shape[1] - 1]
        # predict topK intent vectors based on the weight vector
        topKPredictedIntents = LSTM_RNN.computePredictedIntentsRNN(predictedY, trainSessionDict, configDict, sessID)
        maxCosineSim = CFCosineSim.computeListBitCosineSimilarity(predictedY, topKPredictedIntents[0], configDict)
        minimaxCosineSimDict[sessIDQueryID] = maxCosineSim
        if i% 50 ==0:
            print "foldID: " + str(foldID) + ", activeIter: " + str(activeIter) + ", #Hold-out-Pairs: " + str(len(holdOutTrainDictX))+" #elemSoFar: "+ str(i+1)
        i+=1
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
        print "foldID: " + str(foldID) + ", activeIter: " + str(activeIter) +", Added "+str(resCount)+"th example, sessIDQueryID: "+str(sessIDQueryID)+" with cosineSim: "+str(cosSimEntry[1])+" to the data"
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
    (testSessionStreamDict, testKeyOrder, testEpisodeResponseTime) = LSTM_RNN.initRNNOneFoldTest(testIntentSessionFile,
                                                                                        configDict)
    return (sessionLengthDict, trainSessionDict, trainSessionStreamDict, trainKeyOrder, modelRNN, testSessionStreamDict, testKeyOrder, testEpisodeResponseTime, availTrainKeyX, availTrainKeyY, holdOutTrainX, holdOutTrainY)


def testActiveRNN(sessionLengthDict, trainSessionDict, testKeyOrder, testSessionStreamDict, modelRNN, max_lookback):
    prevSessID = -1
    numSessions = 0
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
            if prevSessID in trainSessionDict:
                assert len(trainSessionDict[prevSessID])>1
                avgAccuracyPerSession.append(float(avgAccuracy)/float(len(trainSessionDict[prevSessID])-1))
                avgFMeasurePerSession.append(float(avgFMeasure)/float(len(trainSessionDict[prevSessID])-1))
                avgPrecisionPerSession.append(float(avgPrecision)/float(len(trainSessionDict[prevSessID])-1))
                avgRecallPerSession.append(float(avgRecall)/float(len(trainSessionDict[prevSessID])-1))
                avgFMeasure = 0.0
                avgAccuracy = 0.0
                avgPrecision = 0.0
                avgRecall = 0.0
                del trainSessionDict[prevSessID] # bcoz none of the test session queries should be used for test phase prediction for a different session, so delete a test session-info once it is done with
                numSessions = int(numSessions) + 1  # episodes start from 1
            prevSessID = sessID
        # update sessionDict with this new query
        LSTM_RNN.updateSessionDictWithCurrentIntent(trainSessionDict, sessID, curQueryIntent)
        if modelRNN is not None and queryID < sessionLengthDict[sessID] - 1:
            predictedY = LSTM_RNN.predictTopKIntents(modelRNN, trainSessionDict, sessID, max_lookback, configDict)
            if configDict['BIT_OR_WEIGHTED'] == 'BIT':
                topKPredictedIntents = LSTM_RNN.computePredictedIntentsRNN(predictedY, trainSessionDict, configDict, sessID)
            elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
                topKPredictedIntents = QR.computeWeightedVectorFromList(predictedY)
            #compare topK with testY
            nextQueryIntent = testSessionStreamDict[str(sessID) + "," + str(queryID + 1)]
            output_str = QR.computePredictedOutputStrRNN(sessID, queryID, topKPredictedIntents, nextQueryIntent, numSessions, configDict)
            (sessID, queryID, numSessions, accuracyAtMaxFMeasure, precisionAtMaxFMeasure, recallAtMaxFMeasure,
             maxFMeasure, maxFIndex) = QR.computeQueRIEFMeasureForEachEpisode(output_str, configDict)
            avgFMeasure+=maxFMeasure
            avgAccuracy+=accuracyAtMaxFMeasure
            avgPrecision+=precisionAtMaxFMeasure
            avgRecall+=recallAtMaxFMeasure
    if prevSessID in trainSessionDict:
        del trainSessionDict[prevSessID] # you also delete the last dangling test session from the overall session dict so far incl train n test
    avgFMeasure=float(sum(avgFMeasurePerSession))/float(numSessions)
    avgAccuracy = float(sum(avgAccuracyPerSession))/float(numSessions)
    avgPrecision = float(sum(avgPrecisionPerSession))/float(numSessions)
    avgRecall =  float(sum(avgRecallPerSession))/float(numSessions)
    return (avgFMeasure, avgAccuracy, avgPrecision, avgRecall)


def runActiveRNNKFoldExp(configDict):
    intentSessionFile = QR.fetchIntentFileFromConfigDict(configDict)
    avgTrainTime = {} #key should be # AL iteration and value should be a list of results across all the folds
    avgExSelTime = {}
    avgTestTime = {}
    avgKFoldFMeasure = {}
    avgKFoldAccuracy = {}
    avgKFoldPrecision = {}
    avgKFoldRecall = {}
    algoName = configDict['ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"]
    outputDir = configDict['KFOLD_OUTPUT_DIR']
    for foldID in range(int(configDict['KFOLD'])):
        trainIntentSessionFile = configDict['KFOLD_INPUT_DIR'] + intentSessionFile.split("/")[len(intentSessionFile.split("/")) - 1] + "_TRAIN_FOLD_" + str(foldID)
        testIntentSessionFile = configDict['KFOLD_INPUT_DIR'] + intentSessionFile.split("/")[len(intentSessionFile.split("/")) - 1] + "_TEST_FOLD_" + str(foldID)
        (sessionLengthDict, trainSessionDict, trainSessionStreamDict, trainKeyOrder, modelRNN, testSessionStreamDict, testKeyOrder, testEpisodeResponseTime, availTrainDictX, availTrainDictY, holdOutTrainDictX,
         holdOutTrainDictY) = initRNNOneFoldActiveTrainTest(trainIntentSessionFile, testIntentSessionFile, configDict)
        activeIter = 0
        while len(holdOutTrainDictX) > 0:
            startTime = time.time()
            assert configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL' or configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'FULL'
            #reinitialize model to None before training it if it is a full train
            if configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'FULL':
                modelRNN=None
            (modelRNN, max_lookback) = LSTM_RNN.trainRNN(availTrainDictX.values(), availTrainDictY.values(), modelRNN, configDict)
            trainTime = float(time.time() - startTime)
            startTime = time.time()
            # example selection phase
            assert configDict['ACTIVE_EXSEL_STRATEGY_MINIMAX_RANDOM'] == 'MINIMAX' or configDict['ACTIVE_EXSEL_STRATEGY_MINIMAX_RANDOM'] == 'RANDOM'
            if configDict['ACTIVE_EXSEL_STRATEGY_MINIMAX_RANDOM'] == 'MINIMAX':
                (availTrainDictX, availTrainDictY, holdOutTrainDictX, holdOutTrainDictY) = exampleSelectionMinimax(foldID, activeIter, modelRNN, max_lookback, availTrainDictX, availTrainDictY, holdOutTrainDictX, holdOutTrainDictY, trainSessionDict)
            else:
                (availTrainDictX, availTrainDictY, holdOutTrainDictX, holdOutTrainDictY) = exampleSelectionRandom(foldID, activeIter, availTrainDictX, availTrainDictY, holdOutTrainDictX, holdOutTrainDictY)
            exSelTime = float(time.time() - startTime)
            print "FoldID: "+str(foldID)+", activeIter: "+str(activeIter)+", Added " + str(len(availTrainDictX)) + " examples to the training data"
            startTime = time.time()
            (avgFMeasure, avgAccuracy, avgPrecision, avgRecall) = testActiveRNN(sessionLengthDict, trainSessionDict, testKeyOrder, testSessionStreamDict, modelRNN, max_lookback)
            testTime = float(time.time() - startTime)
            # update quality measures
            assert len(avgTrainTime) == len(avgTestTime) and len(avgExSelTime) == len(avgTrainTime) and len(avgTrainTime) == len(avgKFoldAccuracy) and len(avgTrainTime) == len(avgKFoldFMeasure) and len(avgTrainTime) == len(avgKFoldPrecision) and len(avgTrainTime) == len(avgKFoldRecall)
            if activeIter not in avgTrainTime:
                avgTrainTime[activeIter] = {}
                avgExSelTime[activeIter] = {}
                avgTestTime[activeIter] = {}
                avgKFoldFMeasure[activeIter] = {}
                avgKFoldAccuracy[activeIter] = {}
                avgKFoldPrecision[activeIter] = {}
                avgKFoldRecall[activeIter] = {}
            avgTrainTime[activeIter][foldID] = trainTime
            avgExSelTime[activeIter][foldID] = exSelTime
            avgTestTime[activeIter][foldID] = testTime
            avgKFoldAccuracy[activeIter][foldID] = avgAccuracy
            avgKFoldFMeasure[activeIter][foldID] = avgFMeasure
            avgKFoldPrecision[activeIter][foldID] = avgPrecision
            avgKFoldRecall[activeIter][foldID] = avgRecall
            activeIter+=1
    saveDictsBeforeAverage(avgTrainTime, avgExSelTime, avgTestTime, avgKFoldFMeasure, avgKFoldAccuracy, avgKFoldPrecision, avgKFoldRecall, configDict)
    # Now take the average
    expectedIterLength = int(configDict['KFOLD'])
    avgTrainTime = computeAvgPerDict(avgTrainTime, expectedIterLength)
    avgExSelTime = computeAvgPerDict(avgExSelTime, expectedIterLength)
    avgTestTime = computeAvgPerDict(avgTestTime, expectedIterLength)
    avgIterTime = computeAvgIterTime(avgTrainTime, avgExSelTime, avgTestTime)
    avgKFoldFMeasure = computeAvgPerDict(avgKFoldFMeasure, expectedIterLength)
    avgKFoldAccuracy = computeAvgPerDict(avgKFoldAccuracy, expectedIterLength)
    avgKFoldPrecision = computeAvgPerDict(avgKFoldPrecision, expectedIterLength)
    avgKFoldRecall = computeAvgPerDict(avgKFoldRecall, expectedIterLength)
    #Now plot the avg Dicts using new methods in ParseResultsToExcel
    ParseResultsToExcel.parseQualityTimeActiveRNN(avgTrainTime, avgExSelTime, avgTestTime, avgIterTime, avgKFoldAccuracy, avgKFoldFMeasure, avgKFoldPrecision, avgKFoldRecall, algoName, outputDir, configDict)
    return

def saveDictsBeforeAverage(avgTrainTime, avgExSelTime, avgTestTime, avgKFoldFMeasure, avgKFoldAccuracy, avgKFoldPrecision, avgKFoldRecall, configDict):
    suffix = "_" + configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_"+configDict['ACTIVE_EXSEL_STRATEGY_MINIMAX_RANDOM']
    QR.writeToPickleFile(configDict['KFOLD_OUTPUT_DIR']+"avgTrainTimeAL"+suffix+".pickle", avgTrainTime)
    QR.writeToPickleFile(configDict['KFOLD_OUTPUT_DIR']+"avgTestTimeAL"+suffix+".pickle", avgTestTime)
    QR.writeToPickleFile(configDict['KFOLD_OUTPUT_DIR']+"avgExSelTime"+suffix+".pickle", avgExSelTime)
    QR.writeToPickleFile(configDict['KFOLD_OUTPUT_DIR']+"avgKFoldFMeasure"+suffix+".pickle", avgKFoldFMeasure)
    QR.writeToPickleFile(configDict['KFOLD_OUTPUT_DIR']+"avgKFoldAccuracy"+suffix+".pickle", avgKFoldAccuracy)
    QR.writeToPickleFile(configDict['KFOLD_OUTPUT_DIR']+"avgKFoldPrecision"+suffix+".pickle", avgKFoldPrecision)
    QR.writeToPickleFile(configDict['KFOLD_OUTPUT_DIR']+"avgKFoldRecall"+suffix+".pickle", avgKFoldRecall)
    return

def computeAvgIterTime(avgTrainTime, avgExSelTime, avgTestTime):
    avgIterTime = {}
    for key in avgTrainTime:
        avgIterTime[key] = avgTrainTime[key] + avgExSelTime[key] + avgTestTime[key]
    return avgIterTime

def computeAvgPerDict(avgDict, expectedIterLength):
    # Each key represents an active learning iteration. A few folds may have more iterations than others coz they may get slightly more training data than others
    # In such a case include the before last active learning iteration's avg performance also into the last iteration, because both represent convergence
    maxValidKey = -1
    for key in avgDict:
        if int(key) > maxValidKey and len(avgDict[key]) < expectedIterLength:
            maxValidKey = key
            if maxValidKey < len(avgDict)-1: # only the last iteration is allowed to have fewer than kfold iteration length - coz remainder occurs only at the end
                print "Invalid Max Key !!"
                sys.exit(0)
    avgOutputDict = {}
    for key in avgDict:
        if key == maxValidKey and key>=1:
            for prevFoldID in avgDict[key-1]:
                if prevFoldID not in avgDict[maxValidKey]:
                    avgDict[maxValidKey][prevFoldID] = avgDict[key-1][prevFoldID]
        avgOutputDict[key] = float(sum(avgDict[key].values())) / float(len(avgDict[key]))
    del avgDict
    return avgOutputDict

'''
def computeAvgPerDict(avgDict, expectedIterLength):
    # Each key represents an active learning iteration. A few folds may have more iterations than others coz they may get slightly more training data than others
    # In such a case include the before last active learning iteration's avg performance also into the last iteration, because both represent convergence
    maxValidKey = -1
    for key in avgDict:
        if int(key) > maxValidKey and len(avgDict[key]) < expectedIterLength:
            maxValidKey = key
            if maxValidKey < len(avgDict)-1: # only the last iteration is allowed to have fewer than kfold iteration length - coz remainder occurs only at the end
                print "Invalid Max Key !!"
                sys.exit(0)
    prevLen = 1
    for key in avgDict:
        if int(key) == maxValidKey and key>=1:
            prevKey = key-1
            numerator = sum(avgDict[key])+avgDict[prevKey]*prevLen
            denominator = len(avgDict[key])+prevLen
            avgDict[key] = float(numerator) / float(denominator)
        else:
            prevLen = len(avgDict[key])
            avgDict[key] = float(sum(avgDict[key])) / float(len(avgDict[key]))
    return avgDict
'''


def executeAL(configDict):
    # ActiveLearning runs only on kFold
    assert configDict['SINGULARITY_OR_KFOLD']=='KFOLD'
    assert configDict['ALGORITHM'] == 'RNN'
    runActiveRNNKFoldExp(configDict)
    return

if __name__ == "__main__":
    #configDict = parseConfig.parseConfigFile("configFile.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="Config parameters file", type=str, required=True)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    executeAL(configDict)