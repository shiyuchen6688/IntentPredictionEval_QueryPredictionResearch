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
#import numpy as np
#import pandas as pd
#from numpy import dot
#from numpy.linalg import norm
#import matplotlib.pyplot as plt
#import keras
#from keras.datasets import imdb
#from keras.preprocessing import sequence
#from keras.preprocessing.sequence import pad_sequences
#from keras import regularizers
#from keras.callbacks import ModelCheckpoint
#from keras.models import Sequential
#from keras.layers import Activation, SimpleRNN, Dense, TimeDistributed, Flatten, LSTM, Dropout, GRU
import CFCosineSim
import LSTM_RNN_Parallel
import random
import argparse
from ParseConfigFile import getConfig
import threading
import copy
import multiprocessing
import ReverseEnggQueries


def exampleSelectionRandom(foldID, activeIter, availTrainKeyOrder, holdOutTrainKeyOrder, availTrainSampledQueryHistory, sessionStreamDict):
    assert configDict['ACTIVE_EXSEL_STRATEGY_MINIMAX_RANDOM'] == 'RANDOM'
    assert configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL' or configDict[
                                                                               'RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'FULL'
    # get rid of the availTrainKeys you trained on so far for incremental train but not the dictionary or sample coz they are used for query recommendation
    if configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL':
        del availTrainKeyOrder
        availTrainKeyOrder = []
    exampleBatchSize = int(configDict['ACTIVE_BATCH_SIZE'])
    print "foldID: " + str(foldID) + ", activeIter: " + str(activeIter) + ", #Hold-out-Pairs: " + str(len(holdOutTrainKeyOrder))
    # random selection
    if len(holdOutTrainKeyOrder) < exampleBatchSize:
        exampleBatchSize = len(holdOutTrainKeyOrder)
    chosenKeys = random.sample(holdOutTrainKeyOrder, exampleBatchSize)
    for sessIDQueryID in chosenKeys:
        availTrainKeyOrder.append(sessIDQueryID)
        holdOutTrainKeyOrder.remove(sessIDQueryID)
        print "foldID: " + str(foldID) + ", activeIter: " + str(activeIter) + ", Added " + str(
            len(chosenKeys)) + "th example, sessIDQueryID: " + str(sessIDQueryID) + " to the data"
    assert configDict['RNN_PREDICT_NOVEL_QUERIES'] == 'True' or configDict['RNN_PREDICT_NOVEL_QUERIES'] == 'False'
    if configDict['RNN_PREDICT_NOVEL_QUERIES'] == 'False':
        LSTM_RNN_Parallel.updateSampledQueryHistory(configDict, availTrainSampledQueryHistory, availTrainKeyOrder,
                                                    sessionStreamDict)
    return (availTrainSampledQueryHistory, availTrainKeyOrder, holdOutTrainKeyOrder, exampleBatchSize)


def createSortedMiniMaxCSD(minimaxCosineSimDict, modelRNN, max_lookback, holdOutTrainKeyOrder, availTrainDictGlobal, availTrainSampledQueryHistory, sessionLengthDict, sessionStreamDict, configDict, schemaDicts):
    lo = 0
    hi = len(holdOutTrainKeyOrder) - 1
    holdOutResultDict = {}
    holdOutResultDict = LSTM_RNN_Parallel.predictIntentsWithoutCurrentBatch(lo, hi, holdOutTrainKeyOrder, schemaDicts, holdOutResultDict,
                                                                     availTrainDictGlobal,
                                                                     availTrainSampledQueryHistory, sessionStreamDict,
                                                                     sessionLengthDict,
                                                                     modelRNN, max_lookback, configDict)
    for threadID in holdOutResultDict:
        for i in range(len(holdOutResultDict[threadID])):
            (sessID, queryID, predictedY, topKPredictedIntents, nextQueryIntent) = holdOutResultDict[threadID][i]
            maxCosineSim = CFCosineSim.computeListBitCosineSimilarity(predictedY, topKPredictedIntents[0], configDict)
            minimaxCosineSimDict[str(sessID)+","+str(queryID)] = maxCosineSim
    sorted_minimaxCSD = sorted(minimaxCosineSimDict.items(), key=operator.itemgetter(1))  # we sort in ASC order
    return sorted_minimaxCSD

def findAverageTopProbs(predictedY, schemaDicts):
    # no need of weight thresholds, just pick the top three dimensions
    # simplest query is always select col from table -- which has 3 bits set for query type, single column and single table
    # these three dimensions are always mandated to have the highest weights or probabilities. Our aim is to check which predictions
    # that the RNN is least confident about. So we look for those weight vectors which have the least average weight of the Top-3 dimensions (miniMax)
    startBit = len(predictedY) - schemaDicts.allOpSize
    predictedY = predictedY[startBit:len(predictedY)]
    predictedY.sort(reverse=True)
    avgMaxProb = 0.0
    numTopKDims = min(3, len(predictedY))
    for i in range(0, numTopKDims):
        avgMaxProb += float(predictedY[i])
    if avgMaxProb > 0:
        avgMaxProb = float(avgMaxProb) / float(numTopKDims)
    return avgMaxProb


def createSortedMiniMaxProbDict(minimaxProbDict, modelRNN, max_lookback, holdOutTrainKeyOrder,
                                                   sessionStreamDict, configDict, schemaDicts):
    for sessQueryID in holdOutTrainKeyOrder:
        sessID = int(sessQueryID.split(",")[0])
        queryID = int(sessQueryID.split(",")[1])
        predictedY = LSTM_RNN_Parallel.predictWeightVector(modelRNN, sessionStreamDict, sessID, queryID, max_lookback, configDict)
        avgMaxProb = findAverageTopProbs(predictedY, schemaDicts)
        minimaxProbDict[sessQueryID] = avgMaxProb
    sorted_minimaxProbDict = sorted(minimaxProbDict.items(), key=operator.itemgetter(1))  # we sort in ASC order
    return sorted_minimaxProbDict

def exampleSelectionMinimax(foldID, activeIter, modelRNN, max_lookback, availTrainKeyOrder, holdOutTrainKeyOrder, availTrainDictGlobal, availTrainSampledQueryHistory, sessionLengthDict, sessionStreamDict, configDict, schemaDicts):
    assert configDict['ACTIVE_EXSEL_STRATEGY_MINIMAX_RANDOM'] == 'MINIMAX'
    assert configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL' or configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'FULL'
    # get rid of the availTrainKeys you trained on so far for incremental train but not the dictionary or sample coz they are used for query recommendation
    if configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL':
        del availTrainKeyOrder
        availTrainKeyOrder = []
    exampleBatchSize = int(configDict['ACTIVE_BATCH_SIZE'])
    minimaxDict = {}
    i = 0
    print "foldID: " + str(foldID) + ", activeIter: " + str(activeIter) + ", #Avail-Dict-Pairs: " + str(len(availTrainKeyOrder))  + ", #Hold-Out-Dict-Pairs: " + str(len(holdOutTrainKeyOrder))
    assert configDict['RNN_PREDICT_NOVEL_QUERIES'] == 'True' or configDict['RNN_PREDICT_NOVEL_QUERIES'] == 'False'
    if configDict['RNN_PREDICT_NOVEL_QUERIES'] == 'False':
        sorted_minimaxDict = createSortedMiniMaxCSD(minimaxDict, modelRNN, max_lookback, holdOutTrainKeyOrder,
                                                   availTrainDictGlobal, availTrainSampledQueryHistory, sessionLengthDict,
                                                   sessionStreamDict, configDict, schemaDicts)
    elif configDict['RNN_PREDICT_NOVEL_QUERIES'] == 'True':
        sorted_minimaxDict = createSortedMiniMaxProbDict(minimaxDict, modelRNN, max_lookback, holdOutTrainKeyOrder,
                                                   sessionStreamDict, configDict, schemaDicts)
    resCount = 0
    for minimaxEntry in sorted_minimaxDict:
        sessIDQueryID = minimaxEntry[0]
        availTrainKeyOrder.append(sessIDQueryID)
        holdOutTrainKeyOrder.remove(sessIDQueryID)
        resCount += 1
        if resCount >= exampleBatchSize:
            break
        print "foldID: " + str(foldID) + ", activeIter: " + str(activeIter) + ", Added " + str(
            resCount) + "th example, sessIDQueryID: " + str(sessIDQueryID) + " with cosineSim/prob: " + str(
            minimaxEntry[1]) + " to the data"
    if configDict['RNN_PREDICT_NOVEL_QUERIES'] == 'False':
        LSTM_RNN_Parallel.updateSampledQueryHistory(configDict, availTrainSampledQueryHistory, availTrainKeyOrder,
                                                    sessionStreamDict)
    return (availTrainSampledQueryHistory, availTrainKeyOrder, holdOutTrainKeyOrder, resCount)


def createAvailHoldOutTrainKeyOrder(availTrainKeyOrder, holdOutTrainKeyOrder, lo_index, batchSize):
    hi_index = lo_index+batchSize-1
    if lo_index+batchSize > len(holdOutTrainKeyOrder):
        hi_index = len(holdOutTrainKeyOrder)-1
    for elemIndex in range(lo_index, hi_index+1):
        sessIDQueryID = holdOutTrainKeyOrder[elemIndex]
        availTrainKeyOrder.append(sessIDQueryID)
    for sessIDQueryID in availTrainKeyOrder:
        holdOutTrainKeyOrder.remove(sessIDQueryID)
    return (availTrainKeyOrder, holdOutTrainKeyOrder)

def initRNNOneFoldActiveTrain(trainIntentSessionFile, configDict):
    availTrainDictGlobal = {}  # key is session ID and value is a list of query intent vectors; no need to store the query itself
    sessionLengthDict = ConcurrentSessions.countQueries(getConfig(configDict['QUERYSESSIONS']))
    multiProcessingManager = multiprocessing.Manager()
    sessionStreamDict = multiProcessingManager.dict()

    holdOutTrainKeyOrder = []
    with open(trainIntentSessionFile) as f:
        for line in f:
            (sessID, queryID, curQueryIntent, sessionStreamDict) = QR.updateSessionDict(line, configDict, sessionStreamDict)
            holdOutTrainKeyOrder.append(str(sessID) + "," + str(queryID))

    f.close()
    modelRNN = None
    max_lookback = 0
    return (availTrainDictGlobal, sessionLengthDict, sessionStreamDict, holdOutTrainKeyOrder, modelRNN, max_lookback)

def initRNNOneFoldActiveTrainTest(trainIntentSessionFile, testIntentSessionFile, configDict):
    # the purpose of this method is to create two training sets - one available and another hold-out
    (availTrainDictGlobal, sessionLengthDict, trainSessionStreamDict, holdOutTrainKeyOrder,
     modelRNN, max_lookback) = initRNNOneFoldActiveTrain(trainIntentSessionFile, configDict)

    availTrainKeyOrder = []
    availTrainSampledQueryHistory = set()  # it is a set
    (availTrainKeyOrder, holdOutTrainKeyOrder) = createAvailHoldOutTrainKeyOrder(availTrainKeyOrder, holdOutTrainKeyOrder, 0, int(configDict['ACTIVE_SEED_TRAINING_SIZE']))
    if configDict['RNN_PREDICT_NOVEL_QUERIES'] == 'False':
        availTrainSampledQueryHistory = LSTM_RNN_Parallel.updateSampledQueryHistory(configDict, availTrainSampledQueryHistory, availTrainKeyOrder,
                                                                          trainSessionStreamDict)

    (sessionStreamDict, testKeyOrder, resultDict, testEpisodeResponseTime) = LSTM_RNN_Parallel.initRNNOneFoldTest(trainSessionStreamDict, testIntentSessionFile,
                                                                                        configDict)
    return (resultDict, availTrainSampledQueryHistory, sessionLengthDict, availTrainDictGlobal, availTrainKeyOrder, holdOutTrainKeyOrder, modelRNN, max_lookback, sessionStreamDict, testKeyOrder, testEpisodeResponseTime)


def computeQualityMeasuresPerSession(resultDict, numSessions):
    sessFMeasure = 0.0
    sessAccuracy = 0.0
    sessPrecision = 0.0
    sessRecall = 0.0
    numValidQueries = 0.0
    for threadID in resultDict:
        for i in range(len(resultDict[threadID])):
            (sessID, queryID, predictedY, topKPredictedIntents, nextQueryIntent) = resultDict[threadID][i]
            output_str = QR.computePredictedOutputStrRNN(sessID, queryID, topKPredictedIntents, nextQueryIntent, numSessions, configDict)
            (sessID, queryID, numSessions, accuracyAtMaxFMeasure, precisionAtMaxFMeasure, recallAtMaxFMeasure, maxFMeasure, maxFIndex) = QR.computeQueRIEFMeasureForEachEpisode(output_str, configDict)
            sessFMeasure += maxFMeasure
            sessAccuracy += accuracyAtMaxFMeasure
            sessPrecision += precisionAtMaxFMeasure
            sessRecall += recallAtMaxFMeasure
            numValidQueries += 1
    if numValidQueries > 0:
        sessFMeasure = float(sessFMeasure)/float(numValidQueries)
        sessAccuracy = float(sessAccuracy)/float(numValidQueries)
        sessPrecision = float(sessPrecision)/float(numValidQueries)
        sessRecall = float(sessRecall)/float(numValidQueries)
    return (sessAccuracy, sessFMeasure, sessPrecision, sessRecall)


def testActiveRNN(schemaDicts, resultDict, availTrainSampledQueryHistory, sessionLengthDict, availTrainDictGlobal, testKeyOrder, sessionStreamDict, modelRNN, max_lookback):
    prevSessID = -1
    numSessions = 0
    avgAccuracyPerSession = []
    avgFMeasurePerSession = []
    avgPrecisionPerSession = []
    avgRecallPerSession = []
    episodeWiseKeys = []

    for key in testKeyOrder:
        sessID = int(key.split(",")[0])
        if prevSessID != sessID:
            assert prevSessID not in availTrainDictGlobal  # because the test sessions are different from availtrainSessions in the AL Experiments
            if len(episodeWiseKeys) > 0:
                lo = 0
                hi = len(episodeWiseKeys) - 1
                resultDict = LSTM_RNN_Parallel.predictIntentsWithoutCurrentBatch(lo, hi, episodeWiseKeys, schemaDicts, resultDict, availTrainDictGlobal,
                                                               availTrainSampledQueryHistory, sessionStreamDict,
                                                               sessionLengthDict,
                                                               modelRNN, max_lookback, configDict)
            numSessions += 1  # episodes start from 1, numEpisodes = numTestSessions
            episodeWiseKeys = []
            prevSessID = sessID
            isEmpty = LSTM_RNN_Parallel.checkResultDictNotEmpty(resultDict)
            assert isEmpty == 'True' or isEmpty == 'False'
            if isEmpty == 'False':
                (sessAccuracy, sessFMeasure, sessPrecision, sessRecall) = computeQualityMeasuresPerSession(resultDict, numSessions)
                avgAccuracyPerSession.append(sessAccuracy)
                avgFMeasurePerSession.append(sessFMeasure)
                avgPrecisionPerSession.append(sessPrecision)
                avgRecallPerSession.append(sessRecall)
                resultDict = LSTM_RNN_Parallel.clear(resultDict)
        episodeWiseKeys.append(key)
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
    schemaDicts = ReverseEnggQueries.readSchemaDicts(configDict)
    for foldID in range(int(configDict['NUM_FOLDS_TO_RUN'])):
        trainIntentSessionFile = getConfig(configDict['KFOLD_INPUT_DIR']) + intentSessionFile.split("/")[len(intentSessionFile.split("/")) - 1] + "_TRAIN_FOLD_" + str(foldID)
        testIntentSessionFile = getConfig(configDict['KFOLD_INPUT_DIR']) + intentSessionFile.split("/")[len(intentSessionFile.split("/")) - 1] + "_TEST_FOLD_" + str(foldID)
        (resultDict, availTrainSampledQueryHistory, sessionLengthDict, availTrainDictGlobal, availTrainKeyOrder, holdOutTrainKeyOrder, modelRNN, max_lookback, sessionStreamDict,
         testKeyOrder, testEpisodeResponseTime) = initRNNOneFoldActiveTrainTest(trainIntentSessionFile, testIntentSessionFile, configDict)
        activeIter = 0
        resCount = 1 # randomly intiialized
        while len(holdOutTrainKeyOrder) > 0 and resCount > 0:
            startTime = time.time()
            assert configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL' or configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'FULL'
            #reinitialize model to None before training it if it is a full train
            if configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'FULL':
                modelRNN=None
            (modelRNN, availTrainDictGlobal, max_lookback) = LSTM_RNN_Parallel.refineTemporalPredictor(availTrainKeyOrder, configDict, availTrainDictGlobal, modelRNN, max_lookback,
                                    sessionStreamDict)
            trainTime = float(time.time() - startTime)

            # test phase first without updating activeKeyOrder or dictionary
            startTime = time.time()
            (avgFMeasure, avgAccuracy, avgPrecision, avgRecall) = testActiveRNN(schemaDicts, resultDict, availTrainSampledQueryHistory, sessionLengthDict, availTrainDictGlobal,
                                                                                testKeyOrder, sessionStreamDict,
                                                                                modelRNN, max_lookback)
            testTime = float(time.time() - startTime)

            # example selection phase
            startTime = time.time()
            assert configDict['ACTIVE_EXSEL_STRATEGY_MINIMAX_RANDOM'] == 'MINIMAX' or configDict['ACTIVE_EXSEL_STRATEGY_MINIMAX_RANDOM'] == 'RANDOM'
            if configDict['ACTIVE_EXSEL_STRATEGY_MINIMAX_RANDOM'] == 'MINIMAX':
                (availTrainSampledQueryHistory, availTrainKeyOrder, holdOutTrainKeyOrder, resCount) = exampleSelectionMinimax(foldID, activeIter, modelRNN, max_lookback, availTrainKeyOrder, holdOutTrainKeyOrder, availTrainDictGlobal, availTrainSampledQueryHistory, sessionLengthDict, sessionStreamDict, configDict, schemaDicts)
            else:
                (availTrainSampledQueryHistory, availTrainKeyOrder, holdOutTrainKeyOrder, resCount) = exampleSelectionRandom(foldID, activeIter, availTrainKeyOrder, holdOutTrainKeyOrder, availTrainSampledQueryHistory, sessionStreamDict)
            exSelTime = float(time.time() - startTime)
            print "FoldID: "+str(foldID)+", activeIter: "+str(activeIter)+", Added " + str(resCount) + " examples to the training data, #Hold-Out Key Pairs: "+str(len(holdOutTrainKeyOrder))

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
    processSavedDicts(configDict)
    return

def processSavedDicts(configDict):
    algoName = configDict['ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"]
    outputDir = getConfig(configDict['KFOLD_OUTPUT_DIR'])
    suffix = "_" + configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_" + configDict[
        'ACTIVE_EXSEL_STRATEGY_MINIMAX_RANDOM']
    avgTrainTime = QR.readFromPickleFile(getConfig(configDict['KFOLD_OUTPUT_DIR'])+"avgTrainTimeAL"+suffix+".pickle")
    avgExSelTime = QR.readFromPickleFile(getConfig(configDict['KFOLD_OUTPUT_DIR'])+"avgExSelTime"+suffix+".pickle")
    avgTestTime = QR.readFromPickleFile(getConfig(configDict['KFOLD_OUTPUT_DIR'])+"avgTestTimeAL"+suffix+".pickle")
    avgKFoldFMeasure = QR.readFromPickleFile(getConfig(configDict['KFOLD_OUTPUT_DIR'])+"avgKFoldFMeasure"+suffix+".pickle")
    avgKFoldPrecision = QR.readFromPickleFile(getConfig(configDict['KFOLD_OUTPUT_DIR'])+"avgKFoldPrecision"+suffix+".pickle")
    avgKFoldRecall = QR.readFromPickleFile(getConfig(configDict['KFOLD_OUTPUT_DIR'])+"avgKFoldRecall"+suffix+".pickle")
    avgKFoldAccuracy = QR.readFromPickleFile(getConfig(configDict['KFOLD_OUTPUT_DIR'])+"avgKFoldAccuracy"+suffix+".pickle")
    # Now take the average
    expectedIterLength = int(configDict['NUM_FOLDS_TO_RUN'])
    avgTrainTime = computeAvgPerDict(avgTrainTime, expectedIterLength)
    avgExSelTime = computeAvgPerDict(avgExSelTime, expectedIterLength)
    avgTestTime = computeAvgPerDict(avgTestTime, expectedIterLength)
    avgIterTime = computeAvgIterTime(avgTrainTime, avgExSelTime, avgTestTime)
    avgKFoldFMeasure = computeAvgPerDict(avgKFoldFMeasure, expectedIterLength)
    avgKFoldAccuracy = computeAvgPerDict(avgKFoldAccuracy, expectedIterLength)
    avgKFoldPrecision = computeAvgPerDict(avgKFoldPrecision, expectedIterLength)
    avgKFoldRecall = computeAvgPerDict(avgKFoldRecall, expectedIterLength)
    # Now plot the avg Dicts using new methods in ParseResultsToExcel
    ParseResultsToExcel.parseMincQualityTimeActiveRNN(avgTrainTime, avgExSelTime, avgTestTime, avgIterTime,
                                                  avgKFoldAccuracy, avgKFoldFMeasure, avgKFoldPrecision, avgKFoldRecall,
                                                  algoName, outputDir, configDict)
    return


def saveDictsBeforeAverage(avgTrainTime, avgExSelTime, avgTestTime, avgKFoldFMeasure, avgKFoldAccuracy, avgKFoldPrecision, avgKFoldRecall, configDict):
    suffix = "_" + configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_"+configDict['ACTIVE_EXSEL_STRATEGY_MINIMAX_RANDOM']
    QR.writeToPickleFile(getConfig(configDict['KFOLD_OUTPUT_DIR'])+"avgTrainTimeAL"+suffix+".pickle", avgTrainTime)
    QR.writeToPickleFile(getConfig(configDict['KFOLD_OUTPUT_DIR'])+"avgTestTimeAL"+suffix+".pickle", avgTestTime)
    QR.writeToPickleFile(getConfig(configDict['KFOLD_OUTPUT_DIR'])+"avgExSelTime"+suffix+".pickle", avgExSelTime)
    QR.writeToPickleFile(getConfig(configDict['KFOLD_OUTPUT_DIR'])+"avgKFoldFMeasure"+suffix+".pickle", avgKFoldFMeasure)
    QR.writeToPickleFile(getConfig(configDict['KFOLD_OUTPUT_DIR'])+"avgKFoldAccuracy"+suffix+".pickle", avgKFoldAccuracy)
    QR.writeToPickleFile(getConfig(configDict['KFOLD_OUTPUT_DIR'])+"avgKFoldPrecision"+suffix+".pickle", avgKFoldPrecision)
    QR.writeToPickleFile(getConfig(configDict['KFOLD_OUTPUT_DIR'])+"avgKFoldRecall"+suffix+".pickle", avgKFoldRecall)
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
    assert configDict['RUN_FROM_EXISTING_OUTPUT'] == 'True' or configDict['RUN_FROM_EXISTING_OUTPUT'] == 'False'
    if configDict['RUN_FROM_EXISTING_OUTPUT'] == 'False':
        executeAL(configDict)
    elif configDict['RUN_FROM_EXISTING_OUTPUT'] == 'True':
        processSavedDicts(configDict)

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
    
def DeprecatedCreateAvailHoldOutDicts(trainX, trainY, trainKeyOrder):
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

def DeprecatedCreateAvailDict(availTrainDictGlobal, trainKeyOrder):
    prevSessID = None
    trainKeyIndex = 0
    activeSeedSize = int(configDict['ACTIVE_SEED_TRAINING_SIZE'])
    assert len(trainKeyOrder) > activeSeedSize
    while trainKeyIndex+1 < activeSeedSize:
        sessQueryID = trainKeyOrder[trainKeyIndex]
        sessID = int(sessQueryID.split(",")[0])
        queryID = int(sessQueryID.split(",")[1])
        availTrainDictGlobal[sessID] = queryID
        if sessID != prevSessID:
            prevSessID = sessID
        trainKeyIndex +=1
    return availTrainDictGlobal
    
def DeprecatedExampleSelectionMinimax(foldID, activeIter, modelRNN, max_lookback, trainKeyOrder, availTrainDictGlobal, availTrainDictX, availTrainDictY, holdOutTrainDictX, holdOutTrainDictY, sessionStreamDict):
    assert configDict['ACTIVE_EXSEL_STRATEGY_MINIMAX_RANDOM'] == 'MINIMAX'
    assert configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL' or configDict[
                                                                               'RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'FULL'
    # get rid of the data that you trained on so far for incremental train
    if configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL':
        del availTrainDictGlobal
        del availTrainDictX
        del availTrainDictY
        availTrainDictGlobal = {}
        availTrainDictX = {}
        availTrainDictY = {}
    exampleBatchSize = int(configDict['ACTIVE_BATCH_SIZE'])
    minimaxCosineSimDict = {}
    i = 0
    print "foldID: " + str(foldID) + ", activeIter: " + str(activeIter) + ", #Avail-Dict-Pairs: " + str(
        len(availTrainDictX))
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
        if i % 50 == 0:
            print "foldID: " + str(foldID) + ", activeIter: " + str(activeIter) + ", #Hold-out-Pairs: " + str(
                len(holdOutTrainDictX)) + " #elemSoFar: " + str(i + 1)
        i += 1
    sorted_minimaxCSD = sorted(minimaxCosineSimDict.items(), key=operator.itemgetter(1))  # we sort in ASC order
    resCount = 0
    for cosSimEntry in sorted_minimaxCSD:
        sessIDQueryID = cosSimEntry[0]
        availTrainDictX[sessIDQueryID] = holdOutTrainDictX[sessIDQueryID]
        availTrainDictY[sessIDQueryID] = holdOutTrainDictY[sessIDQueryID]
        del holdOutTrainDictX[sessIDQueryID]
        del holdOutTrainDictY[sessIDQueryID]
        resCount += 1
        if resCount >= exampleBatchSize:
            break
        print "foldID: " + str(foldID) + ", activeIter: " + str(activeIter) + ", Added " + str(
            resCount) + "th example, sessIDQueryID: " + str(sessIDQueryID) + " with cosineSim: " + str(
            cosSimEntry[1]) + " to the data"
    return (availTrainDictX, availTrainDictY, holdOutTrainDictX, holdOutTrainDictY)

def DeprecatedExampleSelectionRandom(foldID, activeIter, availTrainDictX, availTrainDictY, holdOutTrainDictX, holdOutTrainDictY):
    assert configDict['ACTIVE_EXSEL_STRATEGY_MINIMAX_RANDOM'] == 'RANDOM'
    assert configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL' or configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'FULL'
    # get rid of the availtrainKy that you trained on so far for incremental train
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
'''