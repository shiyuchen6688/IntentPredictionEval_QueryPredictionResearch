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
from keras.layers import Activation, SimpleRNN, Dense, TimeDistributed, Flatten, LSTM, Dropout, GRU, BatchNormalization
import CFCosineSim
import argparse
from ParseConfigFile import getConfig
import threading
import copy
import multiprocessing
from multiprocessing.pool import ThreadPool
from multiprocessing import Array
import ReverseEnggQueries


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

def updateRNNIncrementalTrain_Backup(modelRNN, max_lookback, x_train, y_train, configDict):
    (x_train, max_lookback_this) = perform_input_padding(x_train)
    for i in range(len(x_train)):
        sample_input = np.array(x_train[i])
        sample_output = np.array(y_train[i])
        modelRNN.fit(sample_input.reshape(1, sample_input.shape[0], sample_input.shape[1]),
                     sample_output.reshape(1, sample_output.shape[0], sample_output.shape[1]), epochs=int(configDict['RNN_FULL_TRAIN_EPOCHS'])) # incremental needs only one epoch
    if max_lookback_this > max_lookback:
        max_lookback = max_lookback_this
    return (modelRNN,max_lookback)

def updateRNNIncrementalTrain(modelRNN, max_lookback, x_train, y_train, configDict):
    (x_train, max_lookback_this) = perform_input_padding(x_train)
    y_train = np.array(y_train)
    batchSize = min(len(x_train), int(configDict['RNN_BATCH_SIZE']))
    modelRNN.fit(x_train, y_train, epochs=int(configDict['RNN_FULL_TRAIN_EPOCHS']), batch_size=batchSize)
    if max_lookback_this > max_lookback:
        max_lookback = max_lookback_this
    return (modelRNN, max_lookback)

def updateRNNFullTrain(modelRNN, x_train, y_train, configDict):
    (x_train, max_lookback) = perform_input_padding(x_train)
    y_train = np.array(y_train)
    batchSize = min(len(x_train), int(configDict['RNN_BATCH_SIZE']))
    modelRNN.fit(x_train, y_train, epochs=int(configDict['RNN_FULL_TRAIN_EPOCHS']), batch_size=batchSize)
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
    assert int(configDict['RNN_HIDDEN_LAYERS'])==1 or int(configDict['RNN_HIDDEN_LAYERS'])==2
    if configDict['RNN_BACKPROP_LSTM_GRU'] == 'LSTM':
        modelRNN.add(LSTM(n_memUnits, input_shape=(None, n_features), return_sequences=True))
    elif configDict['RNN_BACKPROP_LSTM_GRU'] == 'BACKPROP':
        modelRNN.add(SimpleRNN(n_memUnits, input_shape=(None, n_features), return_sequences=True))
    elif configDict['RNN_BACKPROP_LSTM_GRU'] == 'GRU':
        modelRNN.add(GRU(n_memUnits, input_shape=(None, n_features), return_sequences=True))
    if int(configDict['RNN_HIDDEN_LAYERS'])==1:
        modelRNN.add(Dropout(0.5))
        modelRNN.add(BatchNormalization())
        modelRNN.add(Dense(n_features, activation="sigmoid"))
        modelRNN.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
    elif int(configDict['RNN_HIDDEN_LAYERS'])==2:
        modelRNN.add(Dropout(0.5))
        modelRNN.add(BatchNormalization())
        modelRNN.add(Dense(256, activation='relu'))  # However this size of weight matrix 256 * 100,000 could potentially blow up
        #print "Inside MultiLayer RNN"
        modelRNN.add(Dropout(0.25))
        modelRNN.add(BatchNormalization())
        modelRNN.add(Dense(n_features, activation="sigmoid"))
        modelRNN.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
    return modelRNN

def appendTrainingXY(sessionStreamDict, sessID, queryID, configDict, dataX, dataY):
    sessIntentList = []
    for qid in range(queryID+1):
        sessIntentList.append(sessionStreamDict[str(sessID)+","+str(qid)])
    numQueries = len(sessIntentList)
    xList = []

    sessionMaxLastK = int(configDict['RNN_SESS_VEC_MAX_LAST_K'])
    assert sessionMaxLastK > 0
    if numQueries < sessionMaxLastK:
        sessionMaxLastK = numQueries
    queryStartIndex = numQueries - 1 - sessionMaxLastK
    for i in range(queryStartIndex, numQueries-1):
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
        #because for supervised Kfold this is training phase but for singularity it would already have been added
        if configDict['SINGULARITY_OR_KFOLD']=='KFOLD':
            updateSessionDictWithCurrentIntent(sessionDictGlobal, sessID, queryID)
        if int(queryID) > 0:
            (dataX, dataY) = appendTrainingXY(sessionStreamDict, sessID, queryID, configDict, dataX, dataY)
    return (dataX, dataY)


def refineTemporalPredictor(queryKeysSetAside, configDict, sessionDictGlobal, modelRNN, max_lookback, sessionStreamDict):
    (dataX, dataY) = createTemporalPairs(queryKeysSetAside, configDict, sessionDictGlobal, sessionStreamDict)
    if len(dataX) > 0:
        (modelRNN, max_lookback) = trainRNN(dataX, dataY, modelRNN, max_lookback, configDict)
    return (modelRNN, sessionDictGlobal, max_lookback)

def predictWeightVector(modelRNNThread, sessionStreamDict, sessID, queryID, max_lookback, configDict):
    # predicts the next query to the query indexed by queryID in the sessID session
    numQueries = queryID + 1
    testX = []
    sessionMaxLastK = int(configDict['RNN_SESS_VEC_MAX_LAST_K_PREDICT'])
    assert sessionMaxLastK > 0
    startQueryIndex = numQueries - sessionMaxLastK
    for i in range(startQueryIndex, numQueries):
        curSessQueryID = str(sessID) + "," + str(queryID)
        curSessIntent = sessionStreamDict[curSessQueryID]
        intentStrList = createCharListFromIntent(curSessIntent, configDict)
        testX.append(intentStrList)
    #print "Appended charList sessID: "+str(sessID)+", queryID: "+str(queryID)
    # modify testX to be compatible with the RNN prediction
    testX = np.array(testX)
    testX = testX.reshape(1, testX.shape[0], testX.shape[1])
    if len(testX) < max_lookback:
        testX = pad_sequences(testX, maxlen=max_lookback, padding='pre')
    #print "Padded sequences sessID: " + str(sessID) + ", queryID: " + str(queryID)
    batchSize = min(int(configDict['ACTIVE_BATCH_SIZE']), len(testX))
    predictedY = modelRNNThread.predict(testX, batch_size = batchSize)
    predictedY = predictedY[0][predictedY.shape[1] - 1]
    #print "Completed prediction: " + str(sessID) + ", queryID: " + str(queryID)
    return predictedY


def partitionPrevQueriesAmongThreads(sessionDictCurThread, sampledQueryHistory, numQueries, numSubThreads, configDict):
    assert configDict['RNN_QUERY_HISTORY_SAMPLE_OR_FULL'] == 'SAMPLE' or configDict[
                                                                             'RNN_QUERY_HISTORY_SAMPLE_OR_FULL'] == 'FULL'
    if configDict['RNN_QUERY_HISTORY_SAMPLE_OR_FULL'] == 'FULL':
        return partitionPrevQueriesAmongThreadsFull(sessionDictCurThread, numQueries, numSubThreads)
    else:
        return partitionPrevQueriesAmongThreadsSample(sampledQueryHistory, numSubThreads)

def partitionPrevQueriesAmongThreadsSample(sampledQueryHistory, numSubThreads):
    queryPartitions = {}
    for i in range(numSubThreads):
        queryPartitions[i] = []
    queryCount = 0
    for sessQueryID in sampledQueryHistory:
        queryCount += 1
        threadID = queryCount % numSubThreads
        queryPartitions[threadID].append(sessQueryID)
    return queryPartitions


def partitionPrevQueriesAmongThreadsFull(sessionDictCurThread, numQueries, numSubThreads):
    numQueriesPerThread = int(numQueries/numSubThreads)
    #round robin assignment of queries to threads
    queryPartitions = {}
    for i in range(numSubThreads):
        queryPartitions[i] = []
    queryCount = 0
    for sessID in sessionDictCurThread:
        for queryID in range(sessionDictCurThread[sessID]+1):
            queryCount += 1
            threadID = queryCount % numSubThreads
            queryPartitions[threadID].append(str(sessID)+","+str(queryID))
    return queryPartitions


def singleThreadedTopKDetection(predictedY, cosineSimDict, curSessID, curQueryID, sessionDictCurThread, sampledQueryHistory, sessionStreamDict, configDict):
    assert configDict['RNN_QUERY_HISTORY_SAMPLE_OR_FULL'] == 'SAMPLE' or configDict['RNN_QUERY_HISTORY_SAMPLE_OR_FULL'] == 'FULL'
    if configDict['RNN_QUERY_HISTORY_SAMPLE_OR_FULL'] =='FULL':
        return singleThreadedTopKDetectionFull(predictedY, cosineSimDict, curSessID, curQueryID, sessionDictCurThread,
                                            sessionStreamDict, configDict)
    else:
        for sessQueryID in sampledQueryHistory:
            queryIntent = sessionStreamDict[sessQueryID]
            cosineSim = CFCosineSim.computeListBitCosineSimilarityPredictOnlyOptimized(predictedY, queryIntent,
                                                                                       configDict)
            cosineSimDict[sessQueryID] = cosineSim
        return cosineSimDict


def singleThreadedTopKDetectionFull(predictedY, cosineSimDict, curSessID, curQueryID, sessionDictCurThread, sessionStreamDict, configDict):
    for sessID in sessionDictCurThread:
        #if len(sessionDictCurThread) == 1 or sessID != curSessID: # we are not going to suggest query intents from the same session unless it is the only session in the dictionary
        numQueries = sessionDictCurThread[sessID]+1
        for queryID in range(numQueries):
            #assert configDict['INCLUDE_CUR_SESS'] == 'True' or configDict['INCLUDE_CUR_SESS'] == 'False'
            #if configDict['INCLUDE_CUR_SESS'] == 'False':
                #expToCheck = (len(sessionDictCurThread) == 1 or sessID != curSessID)
            #elif configDict['INCLUDE_CUR_SESS'] == 'True':
                #expToCheck = (sessID != curSessID and queryID != curQueryID)
            if (sessID != curSessID and queryID != curQueryID):
                queryIntent = sessionStreamDict[str(sessID)+","+str(queryID)]
                cosineSim = CFCosineSim.computeListBitCosineSimilarityPredictOnlyOptimized(predictedY, queryIntent, configDict)
                cosineSimDict[str(sessID) + "," + str(queryID)] = cosineSim
    return cosineSimDict


def multiThreadedTopKDetection((threadID, subThreadID, queryPartition, predictedY, curSessID, curQueryID, sessionDictCurThread, sessionStreamDict, configDict)):
    assert configDict['RNN_QUERY_HISTORY_SAMPLE_OR_FULL'] == 'SAMPLE' or configDict['RNN_QUERY_HISTORY_SAMPLE_OR_FULL'] == 'FULL'
    if configDict['RNN_QUERY_HISTORY_SAMPLE_OR_FULL'] == 'FULL':
        return multiThreadedTopKDetectionFull((threadID, subThreadID, queryPartition, predictedY, curSessID, curQueryID, sessionDictCurThread, sessionStreamDict, configDict))
    else:
        return multiThreadedTopKDetectionSample((threadID, subThreadID, queryPartition, predictedY, sessionStreamDict, configDict))


def multiThreadedTopKDetectionSample((threadID, subThreadID, queryPartition, predictedY, sessionStreamDict, configDict)):
    localCosineSimDict = {}
    for sessQueryID in queryPartition:
        queryIntent = sessionStreamDict[sessQueryID]
        cosineSim = CFCosineSim.computeListBitCosineSimilarityPredictOnlyOptimized(predictedY, queryIntent, configDict)
        localCosineSimDict[sessQueryID] = cosineSim
    #print localCosineSimDict
    QR.writeToPickleFile(getConfig(configDict['PICKLE_TEMP_OUTPUT_DIR'])+"localCosineSimDict_"+str(threadID)+"_"+str(subThreadID)+".pickle",localCosineSimDict)
    return localCosineSimDict


def multiThreadedTopKDetectionFull((threadID, subThreadID, queryPartition, predictedY, curSessID, curQueryID, sessionDictCurThread, sessionStreamDict, configDict)):
    localCosineSimDict = {}
    for sessQueryID in queryPartition:
        sessID = sessQueryID.split(",")[0]
        queryID = sessQueryID.split(",")[1]
        #assert configDict['INCLUDE_CUR_SESS'] == 'True' or configDict['INCLUDE_CUR_SESS'] == 'False'
        #if configDict['INCLUDE_CUR_SESS'] == 'False':
            #expToCheck = (len(sessionDictCurThread) == 1 or sessID != curSessID)
        #elif configDict['INCLUDE_CUR_SESS'] == 'True':
            #expToCheck = (sessID != curSessID and queryID != curQueryID)
        #if len(sessionDictCurThread) == 1 or sessID != curSessID:
        if (sessID != curSessID and queryID != curQueryID):
            queryIntent = sessionStreamDict[sessQueryID]
            cosineSim = CFCosineSim.computeListBitCosineSimilarityPredictOnlyOptimized(predictedY, queryIntent, configDict)
            localCosineSimDict[sessQueryID] = cosineSim
    #print localCosineSimDict
    QR.writeToPickleFile(getConfig(configDict['PICKLE_TEMP_OUTPUT_DIR'])+"localCosineSimDict_"+str(threadID)+"_"+str(subThreadID)+".pickle",localCosineSimDict)
    return localCosineSimDict

def concatenateLocalDicts(localCosineSimDicts, cosineSimDict):
    for subThreadID in localCosineSimDicts:
        for sessQueryID in localCosineSimDicts[subThreadID]:
            cosineSimDict[sessQueryID] = localCosineSimDicts[subThreadID][sessQueryID]
    return cosineSimDict

def computePredictedIntentsRNN(threadID, predictedY, schemaDicts, configDict, curSessID, curQueryID, sessionDictCurThread, sampledQueryHistory, sessionStreamDict):
    assert configDict['RNN_PREDICT_NOVEL_QUERIES'] == 'True' or configDict['RNN_PREDICT_NOVEL_QUERIES'] == 'False'
    if configDict['RNN_PREDICT_NOVEL_QUERIES'] == 'False':
        return computePredictedIntentsRNNFromHistory(threadID, predictedY, configDict, curSessID, curQueryID,
                                              sessionDictCurThread, sampledQueryHistory, sessionStreamDict)
    elif configDict['RNN_PREDICT_NOVEL_QUERIES'] == 'True':
        return ReverseEnggQueries.predictTopKNovelIntents(threadID, predictedY, schemaDicts, configDict, sessionStreamDict[str(curSessID)+","+str(curQueryID)])


def computePredictedIntentsRNNFromHistory(threadID, predictedY, configDict, curSessID, curQueryID, sessionDictCurThread, sampledQueryHistory, sessionStreamDict):
    predictedY = ReverseEnggQueries.pruneUnImportantDimensions(predictedY, float(configDict['RNN_WEIGHT_VECTOR_THRESHOLD']))
    cosineSimDict = {}
    numSubThreads = int(configDict['RNN_SUB_THREADS'])
    if numSubThreads == 1:
        cosineSimDict = singleThreadedTopKDetection(predictedY, cosineSimDict, curSessID, curQueryID, sessionDictCurThread, sampledQueryHistory, sessionStreamDict, configDict)
    else:
        numQueries = sum(sessionDictCurThread.values())+len(sessionDictCurThread) # sum of all latest query Ids + 1 per query session to turn it into count
        numSubThreads = min(numSubThreads, numQueries)
        #sharedArr = Array()
        pool = multiprocessing.Pool()
        argList = []
        if numQueries >= numSubThreads:
            queryPartitions = partitionPrevQueriesAmongThreads(sessionDictCurThread, sampledQueryHistory, numQueries, numSubThreads, configDict)
            assert len(queryPartitions) == int(numSubThreads)
            subThreads = {}
            localCosineSimDicts = {}
            for subThreadID in range(len(queryPartitions)):
                #multiThreadedTopKDetection((localCosineSimDicts[i], queryPartitions[i], predictedY, curSessID, curQueryID, sessionDictCurThread,sessionStreamDict, configDict))
                argList.append((threadID, subThreadID, queryPartitions[subThreadID], predictedY, curSessID, curQueryID, sessionDictCurThread, sessionStreamDict, configDict))
                #subThreads[i] = multiprocessing.Process(target=multiThreadedTopKDetection, args=(localCosineSimDicts, i, queryPartitions[i], predictedY, curSessID, curQueryID, sessionDictCurThread, sessionStreamDict, configDict))
                #subThreads[i].start()
            #for i in range(numSubThreads):
                #subThreads[i].join()
            pool.map(multiThreadedTopKDetection, argList)
            pool.close()
            pool.join()
            for subThreadID in range(len(queryPartitions)):
                localCosineSimDicts[subThreadID] = QR.readFromPickleFile(getConfig(configDict['PICKLE_TEMP_OUTPUT_DIR'])+"localCosineSimDict_"+str(threadID)+"_"+str(subThreadID)+".pickle")
            cosineSimDict = concatenateLocalDicts(localCosineSimDicts, cosineSimDict)
        else:
            cosineSimDict = singleThreadedTopKDetection(predictedY, cosineSimDict, curSessID, curQueryID, sessionDictCurThread, sampledQueryHistory, sessionStreamDict, configDict)
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


def predictTopKIntentsPerThread(threadID, t_lo, t_hi, keyOrder, schemaDicts, modelRNNThread, resList, sessionDictCurThread, sampledQueryHistory, sessionStreamDict, sessionLengthDict, max_lookback, configDict):
    #resList  = list()
    #with graph.as_default():
        #modelRNNThread = keras.models.load_model(modelRNNFileName)
    for i in range(t_lo, t_hi+1):
        sessQueryID = keyOrder[i]
        sessID = int(sessQueryID.split(",")[0])
        queryID = int(sessQueryID.split(",")[1])
        #if queryID < sessionLengthDict[sessID]-1:
        if str(sessID) + "," + str(queryID + 1) in sessionStreamDict:
            predictedY = predictWeightVector(modelRNNThread, sessionStreamDict, sessID, queryID, max_lookback, configDict)
            nextQueryIntent = sessionStreamDict[str(sessID) + "," + str(queryID + 1)]
            #nextIntentList = createCharListFromIntent(nextQueryIntent, configDict)
            #print "Created nextIntentList sessID: " + str(sessID) + ", queryID: " + str(queryID)
            #actual_vector = np.array(nextIntentList).astype(np.int)
            if configDict['BIT_OR_WEIGHTED'] == 'BIT':
                topKPredictedIntents = computePredictedIntentsRNN(threadID, predictedY, schemaDicts, configDict, sessID, queryID, sessionDictCurThread, sampledQueryHistory, sessionStreamDict)
            elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
                topKPredictedIntents = QR.computeWeightedVectorFromList(predictedY)
            resList.append((sessID, queryID, predictedY, topKPredictedIntents, nextQueryIntent))
            #print "computed Top-K Candidates sessID: " + str(sessID) + ", queryID: " + str(queryID)
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

def predictIntentsWithoutCurrentBatch(lo, hi, keyOrder, schemaDicts, resultDict, sessionDictGlobal, sampledQueryHistory, sessionStreamDict, sessionLengthDict, modelRNN, max_lookback, configDict):
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
        resultDict[i] = list()
    #print "Set tuple boundaries for Threads"
    if numThreads == 1:
        predictTopKIntentsPerThread(0, lo, hi, keyOrder, schemaDicts, modelRNN, resultDict[0], sessionDictGlobal, sampledQueryHistory, sessionStreamDict,
                                    sessionLengthDict, max_lookback, configDict) # 0 is the threadID
    else:
        #pool = ThreadPool()
        #argsList = []
        for i in range(numThreads):
            (t_lo, t_hi) = t_loHiDict[i]
            resList = resultDict[i]
            #argsList.append((t_lo, t_hi, keyOrder, modelRNN, resList, sessionDictCurThread, sessionStreamDict, sessionLengthDict, max_lookback, configDict))
            modelRNN._make_predict_function()
            threads[i] = threading.Thread(target=predictTopKIntentsPerThread, args=(i, t_lo, t_hi, keyOrder, schemaDicts, modelRNN, resList, sessionDictGlobal, sampledQueryHistory, sessionStreamDict, sessionLengthDict, max_lookback, configDict))
            threads[i].start()
        for i in range(numThreads):
            threads[i].join()
        #pool.map(predictTopKIntentsPerThread, argsList)
        #pool.close()
        #pool.join()
    return resultDict

def predictIntentsIncludeCurrentBatch(lo, hi, keyOrder, schemaDicts, resultDict, sessionDictGlobal, sessionDictsThreads, sampledQueryHistory, sessionStreamDict, sessionLengthDict, modelRNN, max_lookback, configDict):
    sessionDictsThreads = copySessionDictsThreads(sessionDictGlobal, sessionDictsThreads, configDict)
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
    #print "Updated Session Dictionaries for Threads"
    if numThreads == 1:
        predictTopKIntentsPerThread(0, lo, hi, keyOrder, schemaDicts, modelRNN, resultDict[0], sessionDictsThreads[0], sampledQueryHistory, sessionStreamDict,
                                    sessionLengthDict, max_lookback, configDict) # 0 is the threadID
    else:
        #pool = ThreadPool()
        #argsList = []
        for i in range(numThreads):
            (t_lo, t_hi) = t_loHiDict[i]
            assert i in sessionDictsThreads.keys()
            sessionDictCurThread = sessionDictsThreads[i]
            resList = resultDict[i]
            #argsList.append((t_lo, t_hi, keyOrder, modelRNN, resList, sessionDictCurThread, sessionStreamDict, sessionLengthDict, max_lookback, configDict))
            modelRNN._make_predict_function()
            threads[i] = threading.Thread(target=predictTopKIntentsPerThread, args=(i, t_lo, t_hi, keyOrder, schemaDicts, modelRNN, resList, sessionDictCurThread, sampledQueryHistory, sessionStreamDict, sessionLengthDict, max_lookback, configDict))
            threads[i].start()
        for i in range(numThreads):
            threads[i].join()
        #pool.map(predictTopKIntentsPerThread, argsList)
        #pool.close()
        #pool.join()
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
    #print "updated Global Session Dict"
    return (sessionDictGlobal, queryKeysSetAside)

def copySessionDictsThreads(sessionDictGlobal, sessionDictsThreads, configDict):
    numThreads = int(configDict['RNN_THREADS'])
    for i in range(numThreads):
        if i not in sessionDictsThreads:
            sessionDictsThreads[i] = {}
        sessionDictsThreads[i].update(sessionDictGlobal)
    #print "Copied Thread Session Dicts from Global Session Dict"
    return sessionDictsThreads

def appendResultsToFile(resultDict, elapsedAppendTime, numEpisodes, outputIntentFileName, configDict, foldID):
    for threadID in resultDict:
        for i in range(len(resultDict[threadID])):
            (sessID, queryID, predictedY, topKPredictedIntents, nextQueryIntent) = resultDict[threadID][i]
            elapsedAppendTime += QR.appendPredictedRNNIntentToFile(sessID, queryID, topKPredictedIntents,
                                                                   nextQueryIntent, numEpisodes,
                                                                   outputIntentFileName, configDict, foldID)
    return elapsedAppendTime

def updateTimeResultsToExcel(configDict, episodeResponseTimeDictName, outputIntentFileName):
    accThres = float(configDict['ACCURACY_THRESHOLD'])

    '''
    QR.evaluateQualityPredictions(outputIntentFileName, configDict, accThres,
                                  configDict['ALGORITHM'] + "_" + configDict['RNN_BACKPROP_LSTM_GRU'])
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


    print "--Completed Quality Evaluation for accThres:" + str(accThres)
    '''
    QR.evaluateTimePredictions(episodeResponseTimeDictName, configDict,
                               configDict['ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"])

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

def updateQualityResultsToExcel(configDict, episodeResponseTimeDictName, outputIntentFileName):
    accThres = float(configDict['ACCURACY_THRESHOLD'])


    QR.evaluateQualityPredictions(outputIntentFileName, configDict, accThres,
                                  configDict['ALGORITHM'] + "_" + configDict['RNN_BACKPROP_LSTM_GRU'])
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


    print "--Completed Quality Evaluation for accThres:" + str(accThres)
    '''
    QR.evaluateTimePredictions(episodeResponseTimeDictName, configDict,
                               configDict['ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"])

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
    '''

    return


def updateResultsToExcel(configDict, episodeResponseTimeDictName, outputIntentFileName):
    accThres = float(configDict['ACCURACY_THRESHOLD'])


    QR.evaluateQualityPredictions(outputIntentFileName, configDict, accThres,
                                  configDict['ALGORITHM'] + "_" + configDict['RNN_BACKPROP_LSTM_GRU'])
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
    print "--Completed Quality Evaluation for accThres:" + str(accThres)

    QR.evaluateTimePredictions(episodeResponseTimeDictName, configDict,
                               configDict['ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"])

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

def findIfQueryInside(sessQueryID, sessionStreamDict, sampledQueryHistory, distinctQueries):
    for oldSessQueryID in distinctQueries:
        if sessionStreamDict[oldSessQueryID].tostring() == sessionStreamDict[sessQueryID].tostring():
            return "True"
    for oldSessQueryID in sampledQueryHistory:
        if sessionStreamDict[oldSessQueryID].tostring() == sessionStreamDict[sessQueryID].tostring():
            return "True"
    return "False"

def updateSampledQueryHistory(configDict, sampledQueryHistory, queryKeysSetAside, sessionStreamDict):
    distinctQueries = []
    for sessQueryID in queryKeysSetAside:
        if findIfQueryInside(sessQueryID, sessionStreamDict, sampledQueryHistory, distinctQueries) == "False":
            distinctQueries.append(sessQueryID)
    # employ uniform sampling for repeatability on the same dataset
    sampleFrac = float(configDict['RNN_SAMPLING_FRACTION'])
    count = int(float(configDict['EPISODE_IN_QUERIES']) * sampleFrac)
    if len(distinctQueries) < count:
        count = len(distinctQueries)
    if count == 0:
        count = 1
    if count > 0:
        batchSize = int(len(distinctQueries) / count)
        if batchSize == 0:
            batchSize = 1
        curIndex = 0
        covered = 0
        while covered < count and curIndex < len(distinctQueries):
            sampledQueryHistory.add(distinctQueries[curIndex])
            curIndex += batchSize
            covered += 1
    print "len(distinctQueries): "+str(len(distinctQueries))+", len(sampledQueryHistory): "+str(len(sampledQueryHistory))
    return sampledQueryHistory


def trainTestBatchWise(keyOrder, schemaDicts, sampledQueryHistory, queryKeysSetAside, startEpisode, numEpisodes, episodeResponseTimeDictName, episodeResponseTime, outputIntentFileName, resultDict, sessionDictGlobal, sessionDictsThreads, sessionStreamDict, sessionLengthDict, modelRNN, max_lookback, configDict):
    batchSize = int(configDict['EPISODE_IN_QUERIES'])
    lo = 0
    hi = -1
    while hi<len(keyOrder)-1:
        lo = hi+1
        if len(keyOrder) - lo < batchSize:
            batchSize = len(keyOrder) - lo
        hi = lo + batchSize - 1
        elapsedAppendTime = 0.0

        # test first for each query in the batch if the classifier is not None
        print "Starting prediction in Episode "+str(numEpisodes)+", lo: "+str(lo)+", hi: "+str(hi)+", len(keyOrder): "+str(len(keyOrder))
        if modelRNN is not None:
            assert configDict['INCLUDE_CUR_SESS'] == 'True' or configDict['INCLUDE_CUR_SESS'] == 'False'
            if configDict['INCLUDE_CUR_SESS'] == 'True':
                resultDict = predictIntentsIncludeCurrentBatch(lo, hi, keyOrder, schemaDicts, resultDict, sessionDictGlobal, sessionDictsThreads, sampledQueryHistory, sessionStreamDict, sessionLengthDict, modelRNN, max_lookback, configDict)
            else:
                resultDict = predictIntentsWithoutCurrentBatch(lo, hi, keyOrder, schemaDicts, resultDict, sessionDictGlobal, sampledQueryHistory, sessionStreamDict, sessionLengthDict, modelRNN, max_lookback, configDict)

        print "Starting training in Episode " + str(numEpisodes)
        # update SessionDictGlobal and train with the new batch
        (sessionDictGlobal, queryKeysSetAside) = updateGlobalSessionDict(lo, hi, keyOrder, queryKeysSetAside, sessionDictGlobal)
        if configDict['RNN_PREDICT_NOVEL_QUERIES'] == 'False':
            sampledQueryHistory = updateSampledQueryHistory(configDict, sampledQueryHistory, queryKeysSetAside, sessionStreamDict)
        (modelRNN, sessionDictGlobal, max_lookback) = refineTemporalPredictor(queryKeysSetAside, configDict, sessionDictGlobal,
                                                                        modelRNN, max_lookback, sessionStreamDict)
        assert configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL' or configDict[
                                                                                   'RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'FULL'
        # we have empty queryKeysSetAside because we want to incrementally train the RNN at the end of each episode
        if configDict['RNN_INCREMENTAL_OR_FULL_TRAIN'] == 'INCREMENTAL':
            del queryKeysSetAside
            queryKeysSetAside = []

        # we record the times including train and test
        numEpisodes += 1
        if len(resultDict)> 0:
            elapsedAppendTime = appendResultsToFile(resultDict, elapsedAppendTime, numEpisodes, outputIntentFileName, configDict, -1)
            (episodeResponseTimeDictName, episodeResponseTime, startEpisode, elapsedAppendTime) = QR.updateResponseTime(episodeResponseTimeDictName, episodeResponseTime, numEpisodes, startEpisode, elapsedAppendTime)
            resultDict = clear(resultDict)
    updateResultsToExcel(configDict, episodeResponseTimeDictName, outputIntentFileName)

def checkResultDictNotEmpty(resultDict):
    for threadID in resultDict:
        if len(resultDict[threadID]) > 0:
            return 'False'
    return 'True'

def testOneFold(schemaDicts, foldID, keyOrder, sampledQueryHistory, sessionStreamDict, sessionLengthDict, modelRNN, max_lookback, sessionDictGlobal, resultDict, episodeResponseTime, outputIntentFileName, episodeResponseTimeDictName, configDict):
    try:
        os.remove(outputIntentFileName)
    except OSError:
        pass
    numEpisodes = 1 # starts from Episode 1
    startEpisode = time.time()
    prevSessID = -1
    elapsedAppendTime = 0.0

    episodeWiseKeys = []

    for key in keyOrder:
        sessID = int(key.split(",")[0])
        if prevSessID != sessID:
            assert prevSessID not in sessionDictGlobal # because we never add any batch of test queries to the sessionDictGlobal in the kFold Experiments
            if len(episodeWiseKeys) > 0:
                lo = 0
                hi = len(episodeWiseKeys) -1
                resultDict = predictIntentsWithoutCurrentBatch(lo, hi, episodeWiseKeys, schemaDicts, resultDict, sessionDictGlobal,
                                                               sampledQueryHistory, sessionStreamDict,
                                                               sessionLengthDict,
                                                               modelRNN, max_lookback, configDict)
            episodeWiseKeys = []
            prevSessID = sessID
            isEmpty = checkResultDictNotEmpty(resultDict)
            assert isEmpty == 'True' or isEmpty == 'False'
            if isEmpty == 'False':
                elapsedAppendTime = appendResultsToFile(resultDict, elapsedAppendTime, numEpisodes,
                                                        outputIntentFileName, configDict, foldID)
                (episodeResponseTimeDictName, episodeResponseTime, startEpisode,
                 elapsedAppendTime) = QR.updateResponseTime(episodeResponseTimeDictName, episodeResponseTime,
                                                            numEpisodes, startEpisode, elapsedAppendTime)
                #print episodeResponseTime.keys()
                resultDict = clear(resultDict)
                numEpisodes += 1  # episodes start from 1, numEpisodes = numTestSessions
        episodeWiseKeys.append(key)
    return (outputIntentFileName, episodeResponseTimeDictName)



def initRNNSingularity(configDict):
    intentSessionFile = QR.fetchIntentFileFromConfigDict(configDict)
    sampledQueryHistory = set() # it is a set -- required for predicting from historical query pool
    schemaDicts = ReverseEnggQueries.readSchemaDicts(configDict) # -- required for predicting completely new queries
    numEpisodes = 0
    max_lookback = 0
    queryKeysSetAside = []
    episodeResponseTime = {}
    resultDict = {}
    sessionDictGlobal = {} # one global session dictionary updated after all the threads have finished execution
    sessionDictsThreads = {} # one session dictionary per thread
    episodeResponseTimeDictName = getConfig(configDict['OUTPUT_DIR']) + "/ResponseTimeDict_" + configDict[
        'ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"] + "_" + configDict['INTENT_REP'] + "_" + \
                                  configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                                      'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES'] + ".pickle"
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
    manager = multiprocessing.Manager()
    sessionStreamDict = manager.dict()
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
    return (schemaDicts, sampledQueryHistory, queryKeysSetAside, numEpisodes, episodeResponseTimeDictName, episodeResponseTime, numQueries, resultDict, sessionDictGlobal, sessionDictsThreads, sessionLengthDict, sessionStreamDict, keyOrder, startEpisode, outputIntentFileName, modelRNN, max_lookback, predictedY)


def runRNNSingularityExp(configDict):
    (schemaDicts, sampledQueryHistory, queryKeysSetAside, numEpisodes, episodeResponseTimeDictName, episodeResponseTime, numQueries, resultDict, sessionDictGlobal, sessionDictsThreads, sessionLengthDict,
     sessionStreamDict, keyOrder, startEpisode, outputIntentFileName, modelRNN, max_lookback, predictedY) = initRNNSingularity(configDict)
    trainTestBatchWise(keyOrder, schemaDicts, sampledQueryHistory, queryKeysSetAside, startEpisode, numEpisodes, episodeResponseTimeDictName, episodeResponseTime, outputIntentFileName, resultDict, sessionDictGlobal, sessionDictsThreads, sessionStreamDict, sessionLengthDict, modelRNN, max_lookback, configDict)
    return

def initRNNOneFoldTest(sessionStreamDict, testIntentSessionFile, configDict):
    episodeResponseTime = {}
    resultDict = {}
    keyOrder = []
    with open(testIntentSessionFile) as f:
        for line in f:
            (sessID, queryID, curQueryIntent, sessionStreamDict) = QR.updateSessionDict(line, configDict, sessionStreamDict)
            keyOrder.append(str(sessID) + "," + str(queryID))
    f.close()
    return (sessionStreamDict, keyOrder, resultDict, episodeResponseTime)

def initRNNOneFoldTrain(trainIntentSessionFile, configDict):
    sessionDictGlobal = {}  # key is session ID and value is a list of query intent vectors; no need to store the query itself
    sessionLengthDict = ConcurrentSessions.countQueries(getConfig(configDict['QUERYSESSIONS']))
    multiProcessingManager = multiprocessing.Manager()
    sessionStreamDict = multiProcessingManager.dict()
    sampledQueryHistory = set()  # it is a set
    keyOrder = []
    with open(trainIntentSessionFile) as f:
        for line in f:
            (sessID, queryID, curQueryIntent, sessionStreamDict) = QR.updateSessionDict(line, configDict, sessionStreamDict)
            keyOrder.append(str(sessID) + "," + str(queryID))
    if configDict['RNN_PREDICT_NOVEL_QUERIES'] == 'False':
        sampledQueryHistory = updateSampledQueryHistory(configDict, sampledQueryHistory, keyOrder,
                                                    sessionStreamDict)
    f.close()
    modelRNN = None
    max_lookback = 0
    return (sampledQueryHistory, sessionDictGlobal, sessionLengthDict, sessionStreamDict, keyOrder, modelRNN, max_lookback)


def runRNNKFoldExp(configDict):
    intentSessionFile = QR.fetchIntentFileFromConfigDict(configDict)
    kFoldOutputIntentFiles = []
    kFoldEpisodeResponseTimeDicts = []
    avgTrainTime = []
    avgTestTime = []
    algoName = configDict['ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"]
    schemaDicts = ReverseEnggQueries.readSchemaDicts(configDict)
    print "Num Folds to Run: "+str(int(configDict['NUM_FOLDS_TO_RUN']))
    for foldID in range(int(configDict['NUM_FOLDS_TO_RUN'])):
        outputIntentFileName = getConfig(configDict['KFOLD_OUTPUT_DIR']) + "/OutputFileShortTermIntent_" + algoName + "_" + \
                               configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + \
                               configDict['TOP_K'] + "_FOLD_" + str(foldID)
        episodeResponseTimeDictName = getConfig(configDict['KFOLD_OUTPUT_DIR']) + "/ResponseTimeDict_" + algoName + "_" + \
                                      configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + \
                                      configDict['TOP_K'] + "_FOLD_" + str(foldID) + ".pickle"
        trainIntentSessionFile = getConfig(configDict['KFOLD_INPUT_DIR']) + intentSessionFile.split("/")[len(intentSessionFile.split("/")) - 1] + "_TRAIN_FOLD_" + str(foldID)
        testIntentSessionFile = getConfig(configDict['KFOLD_INPUT_DIR']) + intentSessionFile.split("/")[len(intentSessionFile.split("/")) - 1] + "_TEST_FOLD_" + str(foldID)
        (sampledQueryHistory, sessionDictGlobal, sessionLengthDict, sessionStreamDict, keyOrder, modelRNN, max_lookback) = initRNNOneFoldTrain(trainIntentSessionFile, configDict)
        startTrain = time.time()
        (modelRNN, sessionDictGlobal, max_lookback) = refineTemporalPredictor(keyOrder, configDict, sessionDictGlobal, modelRNN, max_lookback, sessionStreamDict)
        trainTime = float(time.time() - startTrain)
        avgTrainTime.append(trainTime)
        (testSessionStreamDict, testKeyOrder, resultDict, testEpisodeResponseTime) = initRNNOneFoldTest(sessionStreamDict, testIntentSessionFile, configDict)
        if modelRNN is not None:
            startTest = time.time()
            (outputIntentFileName, episodeResponseTimeDictName) = testOneFold(schemaDicts, foldID, testKeyOrder, sampledQueryHistory, testSessionStreamDict, sessionLengthDict, modelRNN, max_lookback, sessionDictGlobal, resultDict, testEpisodeResponseTime, outputIntentFileName, episodeResponseTimeDictName, configDict)
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

def runFromExistingOutput(configDict):
    if configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY':
        outputIntentFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputFileShortTermIntent_" + \
                               configDict['ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"] + "_" + \
                               configDict['INTENT_REP'] + "_" + \
                               configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                                   'TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                               configDict['EPISODE_IN_QUERIES']
        episodeResponseTimeDictName = getConfig(configDict['OUTPUT_DIR']) + "/ResponseTimeDict_" + configDict[
            'ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"] + "_" + configDict['INTENT_REP'] + "_" + \
                                      configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                                          'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict[
                                          'EPISODE_IN_QUERIES'] + ".pickle"
        #updateResultsToExcel(configDict, episodeResponseTimeDictName, outputIntentFileName)
        updateTimeResultsToExcel(configDict, episodeResponseTimeDictName, outputIntentFileName)

def runFromExistingOutputInBetween(configDict):
    if configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY':
        outputIntentFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputFileShortTermIntent_" + \
                               configDict['ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"] + "_" + \
                               configDict['INTENT_REP'] + "_" + \
                               configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                                   'TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                               configDict['EPISODE_IN_QUERIES']+"_copy"
        episodeResponseTimeDictName = getConfig(configDict['OUTPUT_DIR']) + "/ResponseTimeDict_" + configDict[
            'ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"] + "_" + configDict['INTENT_REP'] + "_" + \
                                      configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                                          'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict[
                                          'EPISODE_IN_QUERIES'] + ".pickle"
        updateQualityResultsToExcel(configDict, episodeResponseTimeDictName, outputIntentFileName)



if __name__ == "__main__":
    #configDict = parseConfig.parseConfigFile("configFile.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="Config parameters file", type=str, required=True)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    assert configDict['RUN_FROM_EXISTING_OUTPUT'] == 'True' or configDict['RUN_FROM_EXISTING_OUTPUT'] =='False'
    if configDict['RUN_FROM_EXISTING_OUTPUT'] == 'False':
        executeRNN(configDict)
    elif configDict['RUN_FROM_EXISTING_OUTPUT'] == 'True':
        #runFromExistingOutputInBetween(configDict)
        runFromExistingOutput(configDict)

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
    
            #modelRNNFileName = getConfig(configDict['OUTPUT_DIR'])+'/Thread_Model_'+str(i)+'.h5'
        #modelRNN.save(modelRNNFileName, overwrite=True)
        #graph = tf.get_default_graph()
        
def partitionPrevQueriesAmongThreads_Deprecated(sessionDictCurThread, numQueries, numSubThreads):
    numQueriesPerThread = int(numQueries/numSubThreads)
    queryPartitions = {}
    queryCount = 0
    relCount = 0
    threadID = 0
    for sessID in sessionDictCurThread:
        for queryID in range(sessionDictCurThread[sessID]+1):
            queryCount += 1
            relCount+=1
            remCount = numQueries - queryCount
            if queryCount == 1 or relCount == 1:
                p_lo = str(sessID)+","+str(queryID)
            #if (queryCount % numQueriesPerThread == 0 and remCount < numQueriesPerThread):
                #print "I am here "
            if (queryCount % numQueriesPerThread == 0 and remCount >= numQueriesPerThread) or queryCount == numQueries:
                p_hi = str(sessID)+","+str(queryID)
                relCount = 0
                queryPartitions[threadID] = (p_lo, p_hi)
                threadID += 1
    return queryPartitions

def multiThreadedTopKDetection_Deprecated((localCosineSimDict, threadID, queryPartition, predictedY, curSessID, sessionDictCurThread, sessionStreamDict)):
    (loKey, hiKey) = queryPartition
    sessID_lo = int(loKey.split(",")[0])
    sessID_lo_index = sessionDictCurThread.keys().index(sessID_lo)
    queryID_lo = int(loKey.split(",")[1])
    sessID_hi = int(hiKey.split(",")[0])
    sessID_hi_index = sessionDictCurThread.keys().index(sessID_hi)
    queryID_hi = int(hiKey.split(",")[1])
    finishFlag = False
    for index in range(sessID_lo_index, sessID_hi_index+1):
        sessID = sessionDictCurThread.keys()[index]
        if len(sessionDictCurThread) == 1 or sessID != curSessID:
            numQueries = sessionDictCurThread[sessID] + 1
            queryID_lo_index = 0
            queryID_hi_index = numQueries-1
            if sessID == sessID_lo:
                queryID_lo_index = queryID_lo
            elif sessID == sessID_hi:
                queryID_hi_index = queryID_hi
            for queryID in range(queryID_lo_index, queryID_hi_index+1):
                queryIntent = sessionStreamDict[str(sessID) + "," + str(queryID)]
                cosineSim = CFCosineSim.computeListBitCosineSimilarity(predictedY, queryIntent, configDict)
                localCosineSimDict[str(sessID) + "," + str(queryID)] = cosineSim
                if sessID == sessID_hi and queryID == queryID_hi:
                    finishFlag = True
                    break
            if finishFlag:
                break
    QR.writeToPickleFile(getConfig(configDict['PICKLE_TEMP_OUTPUT_DIR'])+"localCosineSimDict_"+str(threadID)+".pickle",localCosineSimDict)
    return localCosineSimDict

def testOneFold(foldID, keyOrder, sessionStreamDict, sessionLengthDict, modelRNN, max_lookback, sessionDictGlobal, episodeResponseTime, outputIntentFileName, episodeResponseTimeDictName, configDict):
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
            if prevSessID in sessionDictGlobal:
                del sessionDictGlobal[prevSessID] # bcoz none of the test session queries should be used for test phase prediction for a different session, so delete a test session-info once it is done with
                (episodeResponseTimeDictName, episodeResponseTime, startEpisode, elapsedAppendTime) = QR.updateResponseTime(episodeResponseTimeDictName, episodeResponseTime, numEpisodes, startEpisode, elapsedAppendTime)
                numEpisodes += 1  # episodes start from 1, numEpisodes = numTestSessions
            prevSessID = sessID

        #update sessionDictGlobal with this new query
        sessionDictGlobal = updateSessionDictWithCurrentIntent(sessionDictGlobal, sessID, queryID)

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

'''