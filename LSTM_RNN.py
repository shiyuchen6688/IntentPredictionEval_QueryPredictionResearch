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

'''
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[(i + 1): (i + 1 + look_back)])
    return np.array(dataX), np.array(dataY)

def create_input_output_pairs(dataset, timesteps):
    dataX, dataY = [], []

    for i in range(len(dataset) - timesteps):
        if dataset[i][0] != dataset[i + timesteps][0]:
            continue

        instance_x = []
        for j in range(i, i + timesteps):
            row_x = []
            for c in dataset[j][2]:
                row_x.extend(c)
            instance_x.append(row_x)
        dataX.append(instance_x)


        row_y = []
        for c in dataset[i + timesteps][2]:
            row_y.extend(c)
        dataY.append(row_y)

    return np.array(dataX), np.array(dataY)

def create_training_dataset2(dataset):
    dataX, dataY = [], []

    current_session = dataset[0][0]
    start_index = 0

    for i in range(1, len(dataset)):
        if dataset[i][0] != current_session:
            current_session = dataset[i][0]
            start_index = i
        else:

            instance_x = []
            instance_y = []
            for j in reversed(range(start_index, i)):
                row_x = []
                for c in dataset[j][2]:
                    row_x.extend(c)
                instance_x.insert(0, row_x)

                row_y = []
                for c in dataset[j + 1][2]:
                    row_y.extend(c)
                instance_y.insert(0, row_y)

                dataX.append(instance_x)
                dataY.append(instance_y)

    return np.array(dataX), np.array(dataY)

def create_training_dataset(dataset):
    session_dict = {}
    dataX, dataY = [], []

    for i in range(len(dataset)):
        if dataset[i][0] not in session_dict:
            session_dict[dataset[i][0]] = []
            session_dict[dataset[i][0]].append(dataset[i][2])
        else:
            session_dict[dataset[i][0]].append(dataset[i][2])

            session_data = session_dict[dataset[i][0]]

            row_y = []
            for c in session_data[len(session_data) - 1]:
                row_y.extend(c)

            instance_x = []
            #instance_y = []
            for j in range(len(session_data) - 1):
                row_x = []
                for c in session_data[j]:
                    row_x.extend(c)
                instance_x.append(row_x)

            dataX.append(instance_x)
            #dataY.append(instance_y)
            dataY.append([row_y])

    return np.array(dataX), np.array(dataY)

#start of main block where execution will begin
if __name__ == '__main__':

    # fix random seed for reproducibility
    np.random.seed(2000)

    #open the file having intent vectors
    file_name = 'InterleavedSessions/NYCBitFragmentIntentSessions'
    obj = pd.read_csv(file_name, sep=";", header=None)
    data = obj.values

    #processing data to remove unnecessary parts
    m, n = data.shape
    records = []
    sequences = []
    for index in range(m):
        parts = data[index][0].split(", ")
        session_parts = parts[0].split(" ")
        query_parts = parts[1].split(" ")
        records.insert(index, [session_parts[1], query_parts[1], data[index][2]])
        sequences.insert(index, data[index][2])

    records = np.array(records)
    sequences = np.array(sequences)

    # split into train and test sets
    train_size = int(len(records) * 0.806)
    test_size = len(records) - train_size
    train, test = records[0:train_size], records[train_size:len(records)]

    x_train, y_train = create_training_dataset(train)
    x_test, y_test = create_training_dataset(test)

    n_features = len(x_train[0][0])
    print(n_features)
    #n_features = 256
    #n_timesteps = 5



    model=Sequential()
    model.add(LSTM(n_features, input_shape = (None, n_features), return_sequences = True))
    model.add(SimpleRNN(n_features, input_shape=(None, n_features), return_sequences=True))
    model.add(GRU(n_features, input_shape=(None, n_features), return_sequences=True))
    #model.add(Dropout(0.1))
    model.add(Dense(n_features, activation  =  "sigmoid"))
    model.compile(loss = "binary_crossentropy", optimizer = "rmsprop", metrics=['accuracy'])


    #model.fit(x_train.reshape, y_train, epochs=5, batch_size = 1)

    for i in range(len(x_train)):
        sample_input = np.array(x_train[i])
        sample_output = np.array(y_train[i])
        model.fit(sample_input.reshape(1, sample_input.shape[0], sample_input.shape[1]), sample_output.reshape(1, sample_output.shape[0], sample_output.shape[1]), epochs = 1)

    #k = 3
    k = len(x_test) - 1
    test_sample = np.array(x_test[k])
    prediction = model.predict(test_sample.reshape(1, test_sample.shape[0], test_sample.shape[1]))
    #print(np.array(y_test[k]).astype(np.int), "\n")
    test_output = np.array(y_test[k])
    print(np.array(test_output[test_output.shape[0] - 1]).astype(np.int), "\n")
    #labels = (prediction > 0.05).astype(np.int)
    print(prediction[0][prediction.shape[1] - 1])

    similarity = []
    index_values = []
    for i in range(len(x_test)):
        test_sample = np.array(x_test[i])
        prediction = model.predict(test_sample.reshape(1, test_sample.shape[0], test_sample.shape[1]))
        prediction = prediction[0][prediction.shape[1] - 1]
        actual_vector = np.array(y_test[i])
        actual_vector = np.array(actual_vector[actual_vector.shape[0] - 1]).astype(np.int)
        cos_sim = dot(prediction, actual_vector)/(norm(prediction) * norm(actual_vector))
        similarity.insert(i, cos_sim)
        index_values.insert(i, i)

    #print(similarity)
    plt.plot(index_values, similarity)
    plt.xlabel("Serial of test samples")
    plt.ylabel("Similarity")
    plt.title("Cosine Similarity")
    plt.show()

'''

def createCharListFromIntent(intent, configDict):
    intentStrList = []
    if configDict['BIT_OR_WEIGHTED'] == 'BIT':
        intentStr = intent.tostring()
        for i in range(len(intentStr)):
            intentStrList.append(intentStr[i])
    elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
        intentStrList = intent.split(';')
    return intentStrList

def appendTrainingXY(sessIntentList, configDict, dataX, dataY):
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

def updateRNNIncrementalTrain(modelRNN, x_train, y_train):
    for i in range(len(x_train)):
        sample_input = np.array(x_train[i])
        sample_output = np.array(y_train[i])
        modelRNN.fit(sample_input.reshape(1, sample_input.shape[0], sample_input.shape[1]), sample_output.reshape(1, sample_output.shape[0], sample_output.shape[1]), epochs = 1)
        return modelRNN

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

def refineTemporalPredictor(queryKeysSetAside, configDict, sessionDict, modelRNN, sessionStreamDict):
    dataX = []
    dataY = []
    for key in queryKeysSetAside:
        sessID = int(key.split(",")[0])
        queryID = int(key.split(",")[1])
        curQueryIntent = sessionStreamDict[key]
        #because for Kfold this is training phase but for singularity it would already have been added
        if configDict['SINGULARITY_OR_KFOLD']=='KFOLD':
            updateSessionDictWithCurrentIntent(sessionDict, sessID, curQueryIntent)
        if int(queryID) == 0:
            continue
        (dataX, dataY) = appendTrainingXY(sessionDict[sessID], configDict, dataX, dataY)
        n_features = len(dataX[0][0])
        #assert configDict['INTENT_REP'] == 'FRAGMENT' or configDict['INTENT_REP'] == 'QUERY' or configDict['INTENT_REP'] == 'TUPLE'
        #if configDict['INTENT_REP'] == 'FRAGMENT' or configDict['INTENT_REP'] == 'QUERY':
         #   n_memUnits = len(dataX[0][0])
        #elif configDict['INTENT_REP'] == 'TUPLE':
        n_memUnits = int(configDict['RNN_NUM_MEM_UNITS'])
        if modelRNN is None:
            modelRNN = initializeRNN(n_features, n_memUnits, configDict)
        modelRNN = updateRNNIncrementalTrain(modelRNN, dataX, dataY)
    return (modelRNN, sessionDict)

def predictTopKIntents(modelRNN, sessionDict, sessID, configDict):
    #predicts the next query to the last query in the sessID session
    sessIntentList = sessionDict[sessID]
    # top-K is 1
    numQueries = len(sessIntentList)
    testX = []
    for i in range(numQueries):
        curSessIntent = sessIntentList[i]
        intentStrList = createCharListFromIntent(curSessIntent, configDict)
        testX.append(intentStrList)
    testX = np.array(testX)
    predictedY = modelRNN.predict(testX.reshape(1, testX.shape[0], testX.shape[1]))
    predictedY = predictedY[0][predictedY.shape[1] - 1]
    return predictedY

def runRNNKFoldExp(configDict):
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
        (sessionDict, sessionLengthDict, sessionStreamDict, keyOrder, modelRNN) = initRNNOneFoldTrain(trainIntentSessionFile, configDict)
        startTrain = time.time()
        (modelRNN, sessionDict) = refineTemporalPredictor(keyOrder, configDict, sessionDict, modelRNN, sessionStreamDict)
        trainTime = float(time.time() - startTrain)
        avgTrainTime.append(trainTime)
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

def initRNNOneFoldTest(testIntentSessionFile, configDict):
    episodeResponseTime = {}
    sessionStreamDict = {}
    keyOrder = []
    with open(testIntentSessionFile) as f:
        for line in f:
            (sessID, queryID, curQueryIntent, sessionStreamDict) = QR.updateSessionDict(line, configDict,
                                                                                        sessionStreamDict)
            keyOrder.append(str(sessID) + "," + str(queryID))
    f.close()
    return (sessionStreamDict, keyOrder, episodeResponseTime)

def initRNNOneFoldTrain(trainIntentSessionFile, configDict):
    sessionDict = {}  # key is session ID and value is a list of query intent vectors; no need to store the query itself
    sessionLengthDict = ConcurrentSessions.countQueries(configDict['QUERYSESSIONS'])
    sessionStreamDict = {}
    keyOrder = []
    with open(trainIntentSessionFile) as f:
        for line in f:
            (sessID, queryID, curQueryIntent, sessionStreamDict) = QR.updateSessionDict(line, configDict,
                                                                                        sessionStreamDict)
            keyOrder.append(str(sessID) + "," + str(queryID))
    f.close()
    modelRNN = None
    return (sessionDict, sessionLengthDict, sessionStreamDict, keyOrder, modelRNN)

def initRNNSingularity(configDict):
    intentSessionFile = QR.fetchIntentFileFromConfigDict(configDict)
    sessionDict = {}  # key is session ID and value is a list of query intent vectors; no need to store the query itself
    numEpisodes = 0
    queryKeysSetAside = []
    episodeResponseTime = {}
    outputIntentFileName = configDict['OUTPUT_DIR'] + "/OutputFileShortTermIntent_" + \
                           configDict['ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"] + "_" + \
                           configDict['INTENT_REP'] + "_" + \
                           configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                           configDict['EPISODE_IN_QUERIES']
    sessionLengthDict = ConcurrentSessions.countQueries(configDict['QUERYSESSIONS'])
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
    return (sessionDict, numEpisodes, queryKeysSetAside, episodeResponseTime, sessionDict, numQueries, sessionLengthDict, sessionStreamDict, keyOrder, startEpisode, outputIntentFileName, modelRNN, predictedY)

def updateSessionDictWithCurrentIntent(sessionDict, sessID, curQueryIntent):
    # update sessionDict with this new query
    if sessID not in sessionDict:
        sessionDict[sessID] = []
    sessionDict[sessID].append(curQueryIntent)
    return sessionDict

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

def computePredictedIntentsRNN(predictedY, sessionDict, configDict, curSessID):
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
    topKPredictedIntents = []
    maxTopK = int(configDict['TOP_K'])
    resCount = 0
    for cosSimEntry in sorted_csd:
        sessID = int(cosSimEntry[0].split(",")[0])
        queryID = int(cosSimEntry[0].split(",")[1])
        topKPredictedIntents.append(sessionDict[sessID][queryID])  #picks query intents only from already seen vocabulary
        resCount += 1
        if resCount >= maxTopK:
            break
    del cosineSimDict
    del sorted_csd
    return topKPredictedIntents

def runRNNSingularityExp(configDict):
    (sessionDict, numEpisodes, queryKeysSetAside, episodeResponseTime, sessionDict, numQueries, sessionLengthDict,
     sessionStreamDict, keyOrder, startEpisode, outputIntentFileName, modelRNN, predictedY) = initRNNSingularity(configDict)
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
            (modelRNN, sessionDict) = refineTemporalPredictor(queryKeysSetAside, configDict, sessionDict, modelRNN, sessionStreamDict)
            del queryKeysSetAside
            queryKeysSetAside = []
        if modelRNN is not None and queryID < sessionLengthDict[sessID]-1:
            predictedY = predictTopKIntents(modelRNN, sessionDict, sessID, configDict)
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
    episodeResponseTimeDictName = configDict['OUTPUT_DIR'] + "/ResponseTimeDict_" + configDict['ALGORITHM']+"_"+ configDict["RNN_BACKPROP_LSTM_GRU"]+"_"+configDict['INTENT_REP'] + "_" + \
                                  configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                                      'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES'] + ".pickle"
    QR.writeToPickleFile(episodeResponseTimeDictName, episodeResponseTime)
    accThres = float(configDict['ACCURACY_THRESHOLD'])
    QR.evaluateQualityPredictions(outputIntentFileName, configDict, accThres,
                                  configDict['ALGORITHM'] + "_" + configDict['RNN_BACKPROP_LSTM_GRU'])
    print "--Completed Quality Evaluation for accThres:" + str(accThres)
    QR.evaluateTimePredictions(episodeResponseTimeDictName, configDict,configDict['ALGORITHM']+"_"+ configDict["RNN_BACKPROP_LSTM_GRU"])
    outputEvalQualityFileName = configDict['OUTPUT_DIR'] + "/OutputEvalQualityShortTermIntent_" + configDict['ALGORITHM'] + "_" + configDict['RNN_BACKPROP_LSTM_GRU']+ "_" + configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres)
    outputExcelQuality = configDict['OUTPUT_DIR'] + "/OutputExcelQuality_" + configDict['ALGORITHM']+"_"+ configDict["RNN_BACKPROP_LSTM_GRU"]+"_"+ configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']+"_ACCURACY_THRESHOLD_"+str(accThres)+".xlsx"
    ParseResultsToExcel.parseQualityFileWithEpisodeRep(outputEvalQualityFileName, outputExcelQuality, configDict)

    outputEvalTimeFileName = configDict['OUTPUT_DIR'] + "/OutputEvalTimeShortTermIntent_" + configDict['ALGORITHM']+"_"+ configDict["RNN_BACKPROP_LSTM_GRU"]+"_"+ configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']
    outputExcelTimeEval = configDict['OUTPUT_DIR'] + "/OutputExcelTime_" + configDict['ALGORITHM']+"_"+ configDict["RNN_BACKPROP_LSTM_GRU"]+"_"+ configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']+".xlsx"
    ParseResultsToExcel.parseTimeFile(outputEvalTimeFileName, outputExcelTimeEval)

    print "--Completed Quality and Time Evaluation--"
    return

def executeRNN(configDict):
    if configDict['SINGULARITY_OR_KFOLD']=='SINGULARITY':
        runRNNSingularityExp(configDict)
    elif configDict['SINGULARITY_OR_KFOLD']=='KFOLD':
        runRNNKFoldExp(configDict)
    return

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("configFile.txt")
    executeRNN(configDict)


'''
def computePredictedIntentsRNNRankedWeights(predictedY, configDict):
    #sort dimension weights in descending order
    weightDict = {}
    for i in range(len(predictedY)):
        weightDict[i] = predictedY[i]
    #sorted_d is a list of lists, not a dictionary. Each list entry has key as 0th entry and value as 1st entry, we need the key
    sorted_d = sorted(weightDict.items(), key = operator.itemgetter(1), reverse=True)
    dimsSoFar = []
    predictedBitMaps = []
    cosineSimDict = {}
    dictIndex = 0
    if configDict['INTENT_REP'] == 'QUERY':
        topDimLimit = 1
    elif configDict['INTENT_REP'] == 'FRAGMENT':
        topDimLimit=int(float(configDict['RNN_TOP_DIM_PERCENT'])*len(sorted_d)/float(100.0))
    elif configDict['INTENT_REP'] == 'TUPLE':
        topDimLimit = 25
    for dimEntry in sorted_d:
        if len(dimsSoFar)>=topDimLimit:
            break
        dimsSoFar.append(dimEntry[0])
        predictedBitMap = BitMap(len(predictedY))
        for dimSoFar in dimsSoFar:
            predictedBitMap.set(dimSoFar)
        predictedBitMaps.append(predictedBitMap)
        cosineSim = CFCosineSim.computeListBitCosineSimilarity(predictedY, predictedBitMap, configDict)
        cosineSimDict[dictIndex] = cosineSim
        dictIndex+=1
    del sorted_d
    sorted_csd = sorted(cosineSimDict.items(), key=operator.itemgetter(1), reverse=True)
    topKPredictedIntents = []
    maxTopK=int(configDict['TOP_K'])
    resCount =0
    for cosSimEntry in sorted_csd:
        topKPredictedIntents.append(predictedBitMaps[cosSimEntry[0]])
        resCount+=1
        if resCount>=maxTopK:
            break
    del cosineSimDict
    del sorted_csd
    return topKPredictedIntents
'''


