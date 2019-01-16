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
import tensorflow as tf
from keras import backend as K
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Activation, SimpleRNN, Dense, TimeDistributed, Flatten, LSTM, Dropout, GRU
import CFCosineSim
import argparse
from ParseConfigFile import getConfig
import threading
import copy

#graph = tf.get_default_graph()

class ThreadSafeDict(dict) :
    def __init__(self, * p_arg, ** n_arg) :
        dict.__init__(self, * p_arg, ** n_arg)
        self._lock = threading.Lock()

    def __enter__(self) :
        self._lock.acquire()
        return self

    def __exit__(self, type, value, traceback) :
        self._lock.release()

def createCharListFromIntent(intent, configDict):
    intentStrList = []
    if configDict['BIT_OR_WEIGHTED'] == 'BIT':
        intentStr = intent.tostring()
        for i in range(len(intentStr)):
            intentStrList.append(intentStr[i])
    elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
        intentStrList = intent.split(';')
    return intentStrList


def perform_input_padding(x_train):
    if(len(x_train) > 0):
        max_lookback = len(x_train[0])
    else:
        max_lookback = 0

    for i in range(1, len(x_train)):
        if len(x_train[i]) > max_lookback:
            max_lookback = len(x_train[i])

    x_train = pad_sequences(x_train, maxlen = max_lookback, padding='pre')
    return (x_train, max_lookback)

def updateRNNIncrementalTrainBackUp(modelRNN, x_train, y_train, configDict):
    for i in range(len(x_train)):
        sample_input = np.array(x_train[i])
        sample_output = np.array(y_train[i])
        modelRNN.fit(sample_input.reshape(1, sample_input.shape[0], sample_input.shape[1]),
                     sample_output.reshape(1, sample_output.shape[0], sample_output.shape[1]), epochs=int(configDict['RNN_FULL_TRAIN_EPOCHS'])) # incremental needs only one epoch
    return (modelRNN,0)

def updateRNNIncrementalTrain(modelRNN, max_lookback, x_train, y_train, configDict):
    (x_train, max_lookback_this) = perform_input_padding(x_train)
    y_train = np.array(y_train)
    modelRNN.fit(x_train, y_train, epochs=int(configDict['RNN_FULL_TRAIN_EPOCHS']), batch_size=len(x_train))
    if max_lookback_this > max_lookback:
        max_lookback = max_lookback_this
    return (modelRNN, max_lookback)

def updateRNNFullTrain(modelRNN, x_train, y_train, configDict):
    (x_train, max_lookback) = perform_input_padding(x_train)
    y_train = np.array(y_train)
    modelRNN.fit(x_train, y_train, epochs=int(configDict['RNN_FULL_TRAIN_EPOCHS']), batch_size=len(x_train))
    return (modelRNN, max_lookback)
    '''
       for i in range(len(x_train)):
        sample_input = np.array(x_train[i])
        sample_output = np.array(y_train[i])
        modelRNN.fit(sample_input.reshape(1, sample_input.shape[0], sample_input.shape[1]), sample_output.reshape(1, sample_output.shape[0], sample_output.shape[1]), epochs = 1)
        return modelRNN
    '''


def initializeRNN(n_features, n_memUnits, configDict):
    modelRNN = Sequential()
    assert configDict['RNN_BACKPROP_LSTM_GRU'] == 'LSTM' or configDict['RNN_BACKPROP_LSTM_GRU'] == 'BACKPROP' or configDict['RNN_BACKPROP_LSTM_GRU'] == 'GRU'
    if configDict['RNN_BACKPROP_LSTM_GRU'] == 'LSTM':
        modelRNN.add(LSTM(n_memUnits, input_shape=(None, n_features), return_sequences=True))
    elif configDict['RNN_BACKPROP_LSTM_GRU'] == 'BACKPROP':
        modelRNN.add(SimpleRNN(n_memUnits, input_shape=(None, n_features), return_sequences=True))
    elif configDict['RNN_BACKPROP_LSTM_GRU'] == 'GRU':
        modelRNN.add(GRU(n_memUnits, input_shape=(None, n_features), return_sequences=True))
    # model.add(Dropout(0.1))
    modelRNN.add(Dense(n_features, activation="sigmoid"))
    modelRNN.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
    return modelRNN

def appendTrainingXY(sessionStreamDict, sessID, queryID, configDict, dataX, dataY):
    sessIntentList = []
    for qid in range(queryID+1):
        sessIntentList.append(sessionStreamDict[str(sessID)+","+str(qid)])
    numQueries = len(sessIntentList)
    xList = []
    for i in range(numQueries-1):
        prevIntent = sessIntentList[i]
        intentStrList = createCharListFromIntent(prevIntent, configDict)
        xList.append(intentStrList)
    yList = createCharListFromIntent(sessIntentList[numQueries-1], configDict)
    dataX.append(xList)
    dataY.append([yList])
    return (dataX, dataY)


def updateSessionDictWithCurrentIntent(sessionDictGlobal, sessID, queryID):
    # update sessionDict with this new query ID in the session ID
    if sessID not in sessionDictGlobal:
        sessionDictGlobal[sessID] = []
    sessionDictGlobal[sessID] = queryID
    return sessionDictGlobal

def trainRNN(dataX, dataY, modelRNN, max_lookback, configDict):
    n_features = len(dataX[0][0])
    # assert configDict['INTENT_REP'] == 'FRAGMENT' or configDict['INTENT_REP'] == 'QUERY' or configDict['INTENT_REP'] == 'TUPLE'
    # if configDict['INTENT_REP'] == 'FRAGMENT' or configDict['INTENT_REP'] == 'QUERY':
    #   n_memUnits = len(dataX[0][0])
    # elif configDict['INTENT_REP'] == 'TUPLE':
    n_memUnits = int(configDict['RNN_NUM_MEM_UNITS'])
    if modelRNN is None:
        modelRNN = initializeRNN(n_features, n_memUnits, configDict)
    assert configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL' or configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'FULL'
    if configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL':
        (modelRNN, max_lookback) = updateRNNIncrementalTrain(modelRNN, max_lookback, dataX, dataY, configDict)
    elif configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'FULL':
        (modelRNN, max_lookback) = updateRNNFullTrain(modelRNN, dataX, dataY, configDict)
    return (modelRNN, max_lookback)

def createTemporalPairs(queryKeysSetAside, configDict, sessionDictGlobal, sessionStreamDict):
    dataX = []
    dataY = []
    for key in queryKeysSetAside:
        sessID = int(key.split(",")[0])
        queryID = int(key.split(",")[1])
        #because for Kfold this is training phase but for singularity it would already have been added
        if configDict['SINGULARITY_OR_KFOLD']=='KFOLD':
            updateSessionDictWithCurrentIntent(sessionDictGlobal, sessID, queryID)
        if int(queryID) > 0:
            (dataX, dataY) = appendTrainingXY(sessionStreamDict, sessID, queryID, configDict, dataX, dataY)
    return (dataX, dataY)


def refineTemporalPredictor(queryKeysSetAside, configDict, sessionDictGlobal, modelRNN, sessionStreamDict):
    (dataX, dataY) = createTemporalPairs(queryKeysSetAside, configDict, sessionDictGlobal, sessionStreamDict)
    max_lookback = -1
    if len(dataX) > 0:
        (modelRNN, max_lookback) = trainRNN(dataX, dataY, modelRNN, max_lookback, configDict)
    return (modelRNN, sessionDictGlobal, max_lookback)

def predictTopKIntents(modelRNNThread, sessionStreamDict, sessID, queryID, max_lookback, configDict):
    # predicts the next query to the query indexed by queryID in the sessID session
    numQueries = queryID + 1
    testX = []
    for i in range(numQueries):
        curSessQueryID = str(sessID) + "," + str(queryID)
        curSessIntent = sessionStreamDict[curSessQueryID]
        intentStrList = createCharListFromIntent(curSessIntent, configDict)
        testX.append(intentStrList)
    # modify testX to be compatible with the RNN prediction
    testX = np.array(testX)
    testX = testX.reshape(1, testX.shape[0], testX.shape[1])
    if len(testX) < max_lookback:
        testX = pad_sequences(testX, maxlen=max_lookback, padding='pre')
    predictedY = modelRNNThread.predict(testX)
    predictedY = predictedY[0][predictedY.shape[1] - 1]
    return predictedY

def testOneFold(foldID, keyOrder, sessionStreamDict, sessionLengthDict, modelRNN, max_lookback, sessionDict, episodeResponseTime, outputIntentFileName, episodeResponseTimeDictName, configDict):
    try:
        os.remove(outputIntentFileName)
    except OSError:
        pass
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
            predictedY = predictTopKIntents(modelRNN, sessionDict, sessID, max_lookback, configDict)
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

def computePredictedIntentsRNN(predictedY, configDict, curSessID, curQueryID, sessionDictCurThread, sessionStreamDict):
    cosineSimDict = {}
    for sessID in sessionDictCurThread:
        if len(sessionDictCurThread) == 1 or sessID != curSessID: # we are not going to suggest query intents from the same session unless it is the only session in the dictionary
            numQueries = sessionDictCurThread[sessID]+1
            for queryID in range(numQueries):
                queryIntent = sessionStreamDict[str(sessID)+","+str(queryID)]
                cosineSim = CFCosineSim.computeListBitCosineSimilarity(predictedY, queryIntent, configDict)
                cosineSimDict[str(sessID) + "," + str(queryID)] = cosineSim
    # sorted_d is a list of lists, not a dictionary. Each list entry has key as 0th entry and value as 1st entry, we need the key
    sorted_csd = sorted(cosineSimDict.items(), key=operator.itemgetter(1), reverse=True)
    topKPredictedIntents = []
    maxTopK = int(configDict['TOP_K'])
    resCount = 0
    for cosSimEntry in sorted_csd:
        sessID = int(cosSimEntry[0].split(",")[0])
        queryID = int(cosSimEntry[0].split(",")[1])
        topKPredictedIntents.append(sessionStreamDict[str(sessID)+","+str(queryID)])  #picks query intents only from already seen vocabulary
        resCount += 1
        if resCount >= maxTopK:
            break
    del cosineSimDict
    del sorted_csd
    return topKPredictedIntents


def predictTopKIntentsPerThread(t_lo, t_hi, keyOrder, modelRNNThread, resList, sessionDictCurThread, sessionStreamDict, sessionLengthDict, max_lookback, configDict):
    #resList  = list()
    #with graph.as_default():
        #modelRNNThread = keras.models.load_model(modelRNNFileName)
        for i in range(t_lo, t_hi+1):
            sessQueryID = keyOrder[i]
            sessID = int(sessQueryID.split(",")[0])
            queryID = int(sessQueryID.split(",")[1])
            if queryID < sessionLengthDict[sessID]-1:
                predictedY = predictTopKIntents(modelRNNThread, sessionStreamDict, sessID, queryID, max_lookback, configDict)
                nextQueryIntent = sessionStreamDict[str(sessID) + "," + str(queryID + 1)]
                nextIntentList = createCharListFromIntent(nextQueryIntent, configDict)
                actual_vector = np.array(nextIntentList).astype(np.int)
                if configDict['BIT_OR_WEIGHTED'] == 'BIT':
                    topKPredictedIntents = computePredictedIntentsRNN(predictedY, configDict, sessID, queryID, sessionDictCurThread, sessionStreamDict)
                elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
                    topKPredictedIntents = QR.computeWeightedVectorFromList(predictedY)
                resList.append((sessID, queryID, topKPredictedIntents, nextQueryIntent))
        #QR.deleteIfExists(modelRNNFileName)
        return resList

def updateSessionDictsThreads(threadID, sessionDictsThreads, t_lo, t_hi, keyOrder):
    sessionDictCurThread = sessionDictsThreads[threadID]
    cur = t_lo
    while(cur < t_hi+1):
        sessQueryID = keyOrder[cur]
        sessID = int(sessQueryID.split(",")[0])
        queryID = int(sessQueryID.split(",")[1])
        sessionDictCurThread[sessID]= queryID # key is sessID and value is the latest queryID
        cur+=1
    for i in range(threadID+1, len(sessionDictsThreads)):
        sessionDictsThreads[i].update(sessionDictCurThread)
    return sessionDictsThreads

def predictIntents(lo, hi, keyOrder, resultDict, sessionDictsThreads, sessionStreamDict, sessionLengthDict, modelRNN, max_lookback, configDict):
    numThreads = int(configDict['RNN_THREADS'])
    numKeysPerThread = int(float(hi-lo+1)/float(numThreads))
    threads = {}
    t_loHiDict = {}
    t_hi = lo-1
    for i in range(numThreads):
        t_lo = t_hi + 1
        if i == numThreads -1:
            t_hi = hi
        else:
            t_hi = t_lo + numKeysPerThread - 1
        t_loHiDict[i] = (t_lo, t_hi)
        sessionDictsThreads = updateSessionDictsThreads(i, sessionDictsThreads, t_lo, t_hi, keyOrder)
        resultDict[i] = list()
    print "Updated Session Dictionaries for Threads"
    for i in range(numThreads):
        (t_lo, t_hi) = t_loHiDict[i]
        assert i in sessionDictsThreads.keys()
        sessionDictCurThread = sessionDictsThreads[i]
        resList = resultDict[i]
        #modelRNNFileName = getConfig(configDict['OUTPUT_DIR'])+'/Thread_Model_'+str(i)+'.h5'
        #modelRNN.save(modelRNNFileName, overwrite=True)
        modelRNN._make_predict_function()
        #predictTopKIntentsPerThread(t_lo, t_hi, keyOrder, modelRNNFileName, resList, sessionDictCurThread, sessionStreamDict, sessionLengthDict, max_lookback, configDict)
        threads[i] = threading.Thread(target=predictTopKIntentsPerThread, args=(t_lo, t_hi, keyOrder, modelRNN, resList, sessionDictCurThread, sessionStreamDict, sessionLengthDict, max_lookback, configDict))
        threads[i].start()
    for i in range(numThreads):
        threads[i].join()
    return resultDict

def updateGlobalSessionDict(lo, hi, keyOrder, queryKeysSetAside, sessionDictGlobal):
    cur = lo
    while(cur<hi+1):
        sessQueryID = keyOrder[cur]
        queryKeysSetAside.append(sessQueryID)
        sessID = int(sessQueryID.split(",")[0])
        queryID = int(sessQueryID.split(",")[1])
        sessionDictGlobal[sessID]= queryID # key is sessID and value is the latest queryID
        cur+=1
    print "updated Global Session Dict"
    return (sessionDictGlobal, queryKeysSetAside)

def copySessionDictsThreads(sessionDictGlobal, sessionDictsThreads, configDict):
    numThreads = int(configDict['RNN_THREADS'])
    for i in range(numThreads):
        if i not in sessionDictsThreads:
            sessionDictsThreads[i] = {}
        sessionDictsThreads[i].update(sessionDictGlobal)
    print "Copied Thread Session Dicts from Global Session Dict"
    return sessionDictsThreads

def appendResultsToFile(resultDict, elapsedAppendTime, numEpisodes, outputIntentFileName, configDict):
    for threadID in resultDict:
        for i in range(len(resultDict[threadID])):
            (sessID, queryID, topKPredictedIntents, nextQueryIntent) = resultDict[threadID][i]
            elapsedAppendTime += QR.appendPredictedRNNIntentToFile(sessID, queryID, topKPredictedIntents,
                                                                   nextQueryIntent, numEpisodes,
                                                                   outputIntentFileName, configDict, -1)
    return elapsedAppendTime

def updateResultsToExcel(configDict, episodeResponseTime, outputIntentFileName):
    episodeResponseTimeDictName = getConfig(configDict['OUTPUT_DIR']) + "/ResponseTimeDict_" + configDict[
        'ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"] + "_" + configDict['INTENT_REP'] + "_" + \
                                  configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                                      'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES'] + ".pickle"
    QR.writeToPickleFile(episodeResponseTimeDictName, episodeResponseTime)
    accThres = float(configDict['ACCURACY_THRESHOLD'])
    QR.evaluateQualityPredictions(outputIntentFileName, configDict, accThres,
                                  configDict['ALGORITHM'] + "_" + configDict['RNN_BACKPROP_LSTM_GRU'])
    print "--Completed Quality Evaluation for accThres:" + str(accThres)
    QR.evaluateTimePredictions(episodeResponseTimeDictName, configDict,
                               configDict['ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"])
    outputEvalQualityFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputEvalQualityShortTermIntent_" + configDict[
        'ALGORITHM'] + "_" + configDict['RNN_BACKPROP_LSTM_GRU'] + "_" + configDict['INTENT_REP'] + "_" + configDict[
                                    'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                                configDict['EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres)
    outputExcelQuality = getConfig(configDict['OUTPUT_DIR']) + "/OutputExcelQuality_" + configDict['ALGORITHM'] + "_" + \
                         configDict["RNN_BACKPROP_LSTM_GRU"] + "_" + configDict['INTENT_REP'] + "_" + configDict[
                             'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict[
                             'EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres) + "_" + configDict[
                             'RNN_INCREMENTAL_OR_FULL_TRAIN'] + ".xlsx"
    ParseResultsToExcel.parseQualityFileWithEpisodeRep(outputEvalQualityFileName, outputExcelQuality, configDict)

    outputEvalTimeFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputEvalTimeShortTermIntent_" + configDict[
        'ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"] + "_" + configDict['INTENT_REP'] + "_" + configDict[
                                 'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                             configDict['EPISODE_IN_QUERIES']
    outputExcelTimeEval = getConfig(configDict['OUTPUT_DIR']) + "/OutputExcelTime_" + configDict['ALGORITHM'] + "_" + \
                          configDict["RNN_BACKPROP_LSTM_GRU"] + "_" + configDict['INTENT_REP'] + "_" + configDict[
                              'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                          configDict['EPISODE_IN_QUERIES'] + "_" + configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] + ".xlsx"
    ParseResultsToExcel.parseTimeFile(outputEvalTimeFileName, outputExcelTimeEval)

    print "--Completed Quality and Time Evaluation--"
    return

def clear(resultDict):
    keys = resultDict.keys()
    for resKey in keys:
        del resultDict[resKey]
    return resultDict

def trainTestBatchWise(keyOrder, queryKeysSetAside, startEpisode, numEpisodes, episodeResponseTime, outputIntentFileName, resultDict, sessionDictGlobal, sessionDictsThreads, sessionStreamDict, sessionLengthDict, modelRNN, max_lookback, configDict):
    batchSize = int(configDict['EPISODE_IN_QUERIES'])
    lo = 0
    hi = -1
    resultDict = {}
    while hi<len(keyOrder):
        lo = hi+1
        if len(keyOrder) - lo < batchSize:
            batchSize = len(keyOrder) - lo
        hi = lo + batchSize - 1
        elapsedAppendTime = 0.0

        # test first for each query in the batch if the classifier is not None
        print "Starting prediction in Episode "+str(numEpisodes)
        if modelRNN is not None:
            sessionDictsThreads = copySessionDictsThreads(sessionDictGlobal, sessionDictsThreads, configDict)
            resultDict = predictIntents(lo, hi, keyOrder, resultDict, sessionDictsThreads, sessionStreamDict, sessionLengthDict, modelRNN, max_lookback, configDict)

        print "Starting training in Episode " + str(numEpisodes)
        # update SessionDictGlobal and train with the new batch
        (sessionDictGlobal, queryKeysSetAside) = updateGlobalSessionDict(lo, hi, keyOrder, queryKeysSetAside, sessionDictGlobal)
        (modelRNN, sessionDictGlobal, max_lookback) = refineTemporalPredictor(queryKeysSetAside, configDict, sessionDictGlobal,
                                                                        modelRNN, sessionStreamDict)
        assert configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL' or configDict[
                                                                                   'RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'FULL'
        # we have empty queryKeysSetAside because we want to incrementally train the RNN at the end of each episode
        if configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL':
            del queryKeysSetAside
            queryKeysSetAside = []

        # we record the times including train and test
        numEpisodes += 1
        if len(resultDict)> 0:
            elapsedAppendTime = appendResultsToFile(resultDict, elapsedAppendTime, numEpisodes, outputIntentFileName, configDict)
            (episodeResponseTime, startEpisode, elapsedAppendTime) = QR.updateResponseTime(episodeResponseTime, numEpisodes, startEpisode, elapsedAppendTime)
            resultDict = clear(resultDict)
    # update results to excel sheet
    print "Starting Excel Write after all Episodes: " + str(numEpisodes)
    updateResultsToExcel(configDict, episodeResponseTime, outputIntentFileName)


def initRNNSingularity(configDict):
    intentSessionFile = QR.fetchIntentFileFromConfigDict(configDict)
    numEpisodes = 0
    queryKeysSetAside = []
    episodeResponseTime = {}
    resultDict = None
    sessionDictGlobal = {} # one global session dictionary updated after all the threads have finished execution
    sessionDictsThreads = {} # one session dictionary per thread
    outputIntentFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputFileShortTermIntent_" + \
                           configDict['ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"] + "_" + \
                           configDict['INTENT_REP'] + "_" + \
                           configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                           configDict['EPISODE_IN_QUERIES']
    sessionLengthDict = ConcurrentSessions.countQueries(getConfig(configDict['QUERYSESSIONS']))
    try:
        os.remove(outputIntentFileName)
    except OSError:
        pass
    numQueries = 0
    sessionStreamDict = {}
    keyOrder = []
    with open(intentSessionFile) as f:
        for line in f:
            (sessID, queryID, curQueryIntent, sessionStreamDict) = QR.updateSessionDict(line, configDict,
                                                                                        sessionStreamDict)
            keyOrder.append(str(sessID) + "," + str(queryID))
    f.close()
    startEpisode = time.time()
    predictedY = None
    modelRNN = None
    return (queryKeysSetAside, numEpisodes, episodeResponseTime, numQueries, resultDict, sessionDictGlobal, sessionDictsThreads, sessionLengthDict, sessionStreamDict, keyOrder, startEpisode, outputIntentFileName, modelRNN, predictedY)


def runRNNSingularityExp(configDict):
    (queryKeysSetAside, numEpisodes, episodeResponseTime, numQueries, resultDict, sessionDictGlobal, sessionDictsThreads, sessionLengthDict,
     sessionStreamDict, keyOrder, startEpisode, outputIntentFileName, modelRNN, predictedY) = initRNNSingularity(configDict)
    max_lookback = 0
    trainTestBatchWise(keyOrder, queryKeysSetAside, startEpisode, numEpisodes, episodeResponseTime, outputIntentFileName, resultDict, sessionDictGlobal, sessionDictsThreads, sessionStreamDict, sessionLengthDict, modelRNN, max_lookback, configDict)
    return


def runRNNKFoldExp(configDict):
    intentSessionFile = QR.fetchIntentFileFromConfigDict(configDict)
    kFoldOutputIntentFiles = []
    kFoldEpisodeResponseTimeDicts = []
    avgTrainTime = []
    avgTestTime = []
    algoName = configDict['ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"]
    for foldID in range(int(configDict['KFOLD'])):
        outputIntentFileName = getConfig(configDict['KFOLD_OUTPUT_DIR']) + "/OutputFileShortTermIntent_" + algoName + "_" + \
                               configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + \
                               configDict['TOP_K'] + "_FOLD_" + str(foldID)
        episodeResponseTimeDictName = getConfig(configDict['KFOLD_OUTPUT_DIR']) + "/ResponseTimeDict_" + algoName + "_" + \
                                      configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + \
                                      configDict['TOP_K'] + "_FOLD_" + str(foldID) + ".pickle"
        trainIntentSessionFile = getConfig(configDict['KFOLD_INPUT_DIR']) + intentSessionFile.split("/")[len(intentSessionFile.split("/")) - 1] + "_TRAIN_FOLD_" + str(foldID)
        testIntentSessionFile = getConfig(configDict['KFOLD_INPUT_DIR']) + intentSessionFile.split("/")[len(intentSessionFile.split("/")) - 1] + "_TEST_FOLD_" + str(foldID)
        (sessionDict, sessionLengthDict, sessionStreamDict, keyOrder, modelRNN) = LSTM_RNN.initRNNOneFoldTrain(trainIntentSessionFile, configDict)
        startTrain = time.time()
        (modelRNN, sessionDict, max_lookback) = LSTM_RNN.refineTemporalPredictor(keyOrder, configDict, sessionDict, modelRNN, sessionStreamDict)
        trainTime = float(time.time() - startTrain)
        avgTrainTime.append(trainTime)
        (testSessionStreamDict, testKeyOrder, testEpisodeResponseTime) = LSTM_RNN.initRNNOneFoldTest(testIntentSessionFile, configDict)
        startTest = time.time()
        (outputIntentFileName, episodeResponseTimeDictName) = testOneFold(foldID, testKeyOrder, testSessionStreamDict, sessionLengthDict, modelRNN, max_lookback, sessionDict, testEpisodeResponseTime, outputIntentFileName, episodeResponseTimeDictName, configDict)
        testTime = float(time.time() - startTest)
        avgTestTime.append(testTime)
        kFoldOutputIntentFiles.append(outputIntentFileName)
        kFoldEpisodeResponseTimeDicts.append(episodeResponseTimeDictName)
    (avgTrainTimeFN, avgTestTimeFN) = QR.writeKFoldTrainTestTimesToPickleFiles(avgTrainTime, avgTestTime, algoName, configDict)
    QR.avgKFoldTimeAndQualityPlots(kFoldOutputIntentFiles,kFoldEpisodeResponseTimeDicts, avgTrainTimeFN, avgTestTimeFN, algoName, configDict)
    return

def executeRNN(configDict):
    assert int(configDict['EPISODE_IN_QUERIES'])>=int(configDict['RNN_THREADS'])
    if configDict['SINGULARITY_OR_KFOLD']=='SINGULARITY':
        runRNNSingularityExp(configDict)
    elif configDict['SINGULARITY_OR_KFOLD']=='KFOLD':
        runRNNKFoldExp(configDict)
    return

if __name__ == "__main__":
    #configDict = parseConfig.parseConfigFile("configFile.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="Config parameters file", type=str, required=True)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    executeRNN(configDict)

'''
    for key in keyOrder:
        sessID = int(key.split(",")[0])
        queryID = int(key.split(",")[1])
        curQueryIntent = sessionStreamDict[key]
        # Here we are putting together the predictedIntent from previous step and the actualIntent from the current query, so that it will be easier for evaluation
        elapsedAppendTime = 0.0
        numQueries += 1
        queryKeysSetAside.append(key)
        # update sessionDict with this new query
        updateSessionDictWithCurrentIntent(sessionDict, sessID, curQueryIntent)
        # -- Refinement is done only at the end of episode, prediction could be done outside but no use for CF and response time update also happens at one shot --
        if numQueries % int(configDict['EPISODE_IN_QUERIES']) == 0:
            numEpisodes += 1
            (modelRNN, sessionDict, max_lookback) = refineTemporalPredictor(queryKeysSetAside, configDict, sessionDict, modelRNN, sessionStreamDict)
            assert configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL' or configDict[
                                                                                       'RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'FULL'
            # we have empty queryKeysSetAside because we want to incrementally train the RNN at the end of each episode
            if configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL':
                del queryKeysSetAside
                queryKeysSetAside = []
        if modelRNN is not None and queryID < sessionLengthDict[sessID]-1:
            predictedY = predictTopKIntents(modelRNN, sessionDict, sessID, max_lookback, configDict)
            nextQueryIntent = sessionStreamDict[str(sessID)+","+str(queryID+1)]
            nextIntentList = createCharListFromIntent(nextQueryIntent, configDict)
            actual_vector = np.array(nextIntentList).astype(np.int)
            # actual_vector = np.array(actual_vector[actual_vector.shape[0] - 1]).astype(np.int)
            #cosineSim = dot(predictedY, actual_vector) / (norm(predictedY) * norm(actual_vector))
            if configDict['BIT_OR_WEIGHTED'] == 'BIT':
                topKPredictedIntents = computePredictedIntentsRNN(predictedY, sessionDict, configDict, sessID)
            elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
                topKPredictedIntents = QR.computeWeightedVectorFromList(predictedY)
            elapsedAppendTime += QR.appendPredictedRNNIntentToFile(sessID, queryID, topKPredictedIntents, nextQueryIntent, numEpisodes,
                                                                   outputIntentFileName, configDict, -1)
        if numQueries % int(configDict['EPISODE_IN_QUERIES']) == 0:
            (episodeResponseTime, startEpisode, elapsedAppendTime) = QR.updateResponseTime(episodeResponseTime, numEpisodes,startEpisode, elapsedAppendTime)
    episodeResponseTimeDictName = getConfig(configDict['OUTPUT_DIR']) + "/ResponseTimeDict_" + configDict['ALGORITHM']+"_"+ configDict["RNN_BACKPROP_LSTM_GRU"]+"_"+configDict['INTENT_REP'] + "_" + \
                                  configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                                      'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES'] + ".pickle"
    QR.writeToPickleFile(episodeResponseTimeDictName, episodeResponseTime)
    accThres = float(configDict['ACCURACY_THRESHOLD'])
    QR.evaluateQualityPredictions(outputIntentFileName, configDict, accThres,
                                  configDict['ALGORITHM'] + "_" + configDict['RNN_BACKPROP_LSTM_GRU'])
    print "--Completed Quality Evaluation for accThres:" + str(accThres)
    QR.evaluateTimePredictions(episodeResponseTimeDictName, configDict,configDict['ALGORITHM']+"_"+ configDict["RNN_BACKPROP_LSTM_GRU"])
    outputEvalQualityFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputEvalQualityShortTermIntent_" + configDict['ALGORITHM'] + "_" + configDict['RNN_BACKPROP_LSTM_GRU']+ "_" + configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres)
    outputExcelQuality = getConfig(configDict['OUTPUT_DIR']) + "/OutputExcelQuality_" + configDict['ALGORITHM']+"_"+ configDict["RNN_BACKPROP_LSTM_GRU"]+"_"+ configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']+"_ACCURACY_THRESHOLD_"+str(accThres)+"_"+configDict['RNN_INCREMENTAL_OR_FULL_TRAIN']+".xlsx"
    ParseResultsToExcel.parseQualityFileWithEpisodeRep(outputEvalQualityFileName, outputExcelQuality, configDict)

    outputEvalTimeFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputEvalTimeShortTermIntent_" + configDict['ALGORITHM']+"_"+ configDict["RNN_BACKPROP_LSTM_GRU"]+"_"+ configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']
    outputExcelTimeEval = getConfig(configDict['OUTPUT_DIR']) + "/OutputExcelTime_" + configDict['ALGORITHM']+"_"+ configDict["RNN_BACKPROP_LSTM_GRU"]+"_"+ configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']+"_"+configDict['RNN_INCREMENTAL_OR_FULL_TRAIN']+".xlsx"
    ParseResultsToExcel.parseTimeFile(outputEvalTimeFileName, outputExcelTimeEval)

    print "--Completed Quality and Time Evaluation--"
'''