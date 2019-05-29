from __future__ import division
import argparse
import sys, os
from pandas import DataFrame
import ParseConfigFile as parseConfig
from ParseConfigFile import getConfig

def parseMincQualityTimeActiveRNN(avgTrainTime, avgExSelTime, avgTestTime, avgIterTime, avgKFoldAccuracy, avgKFoldFMeasure, avgKFoldPrecision, avgKFoldRecall, algoName, outputDir, configDict):
    assert configDict['SINGULARITY_OR_KFOLD'] == 'KFOLD'
    outputExcelQualityTime = outputDir + "/OutputExcelQualityTimeActiveLearn_" + algoName + "_" + configDict['INTENT_REP'] + "_" + \
                         configDict['BIT_OR_WEIGHTED'] + "_"+configDict['ACTIVE_EXSEL_STRATEGY_MINIMAX_RANDOM']+\
                         "_WEIGHT_" + configDict['RNN_WEIGHT_VECTOR_THRESHOLD']+ "_SAMPLE_" + configDict['RNN_SAMPLING_FRACTION'] + \
                         "_TOP_K_" + configDict['TOP_K'] + "_LAST_K_" + configDict['RNN_SESS_VEC_MAX_LAST_K'] + \
                         "_"+configDict['RNN_INCREMENTAL_OR_FULL_TRAIN']+ ".xlsx"
    assert len(avgTrainTime) == len(avgTestTime) and len(avgExSelTime) == len(avgTrainTime) and len(
        avgTrainTime) == len(avgKFoldAccuracy) and len(avgTrainTime) == len(avgKFoldFMeasure) and len(
        avgTrainTime) == len(avgKFoldPrecision) and len(avgTrainTime) == len(avgKFoldRecall) and len(avgTrainTime) == len(avgIterTime)
    print "Lengths of iterations: " + str(len(avgIterTime))
    df = DataFrame(
        {'iterations': avgIterTime.keys(), 'precision': avgKFoldPrecision.values(),
         'recall': avgKFoldRecall.values(), 'FMeasure': avgKFoldFMeasure.values(), 'accuracy': avgKFoldAccuracy.values(),
         'trainTime': avgTrainTime.values(), 'exSelTime': avgExSelTime.values(), 'testTime': avgTestTime.values(),
         'iterTime': avgIterTime.values()})
    df.to_excel(outputExcelQualityTime, sheet_name='sheet1', index=False)

def parseQualityTimeActiveRNN(avgTrainTime, avgExSelTime, avgTestTime, avgIterTime, avgKFoldAccuracy, avgKFoldFMeasure, avgKFoldPrecision, avgKFoldRecall, algoName, outputDir, configDict):
    assert configDict['SINGULARITY_OR_KFOLD'] == 'KFOLD'
    outputExcelQualityTime = outputDir + "/OutputExcelQualityTimeActiveLearn_" + algoName + "_" + configDict['INTENT_REP'] + "_" + \
                         configDict['BIT_OR_WEIGHTED'] + "_"+configDict['ACTIVE_EXSEL_STRATEGY_MINIMAX_RANDOM']+"_TOP_K_" + configDict['TOP_K'] +"_"+configDict['RNN_INCREMENTAL_OR_FULL_TRAIN']+ ".xlsx"
    assert len(avgTrainTime) == len(avgTestTime) and len(avgExSelTime) == len(avgTrainTime) and len(
        avgTrainTime) == len(avgKFoldAccuracy) and len(avgTrainTime) == len(avgKFoldFMeasure) and len(
        avgTrainTime) == len(avgKFoldPrecision) and len(avgTrainTime) == len(avgKFoldRecall) and len(avgTrainTime) == len(avgIterTime)
    print "Lengths of iterations: " + str(len(avgIterTime))
    df = DataFrame(
        {'iterations': avgIterTime.keys(), 'precision': avgKFoldPrecision.values(),
         'recall': avgKFoldRecall.values(), 'FMeasure': avgKFoldFMeasure.values(), 'accuracy': avgKFoldAccuracy.values(),
         'trainTime': avgTrainTime.values(), 'exSelTime': avgExSelTime.values(), 'testTime': avgTestTime.values(),
         'iterTime': avgIterTime.values()})
    df.to_excel(outputExcelQualityTime, sheet_name='sheet1', index=False)

def parseQualityFileWithoutEpisodeRep(fileName, outputExcel, configDict):
    episodes = []
    precision = []
    recall = []
    FMeasure = []
    accuracy = []
    assert configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY' or configDict['SINGULARITY_OR_KFOLD'] == 'KFOLD'
    if configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY':
        episodeIndex = 2
    elif configDict['SINGULARITY_OR_KFOLD'] == 'KFOLD':
        episodeIndex = 0
    with open(fileName) as f:
        for line in f:
            tokens = line.split(";")
            numEpisodes = float(tokens[episodeIndex].split(":")[1])
            precisionPerEpisode = float(tokens[episodeIndex+1].split(":")[1])
            recallPerEpisode = float(tokens[episodeIndex+2].split(":")[1])
            FMeasurePerEpisode = float(tokens[episodeIndex+3].split(":")[1])
            accuracyPerEpisode = float(tokens[episodeIndex + 4].split(":")[1])
            episodes.append(numEpisodes)
            precision.append(precisionPerEpisode)
            recall.append(recallPerEpisode)
            FMeasure.append(FMeasurePerEpisode)
            accuracy.append(accuracyPerEpisode)
    print "Lengths of episodes: "+str(len(episodes))+", len(precision): "+str(len(precision))+", len(recall): "+str(len(recall))+", len(FMeasure): "+str(len(FMeasure))+", len(accuracy): "+str(len(accuracy))
    df = DataFrame(
        {'episodes':episodes, 'precision': precision,
         'recall': recall, 'FMeasure': FMeasure, 'accuracy': accuracy})
    df.to_excel(outputExcel, sheet_name='sheet1', index=False)

def computeStats(prevEpisode, numQueries, episodes, precision, recall, FMeasure, accuracy, numQueriesPerEpisode, precisionPerEpisode, recallPerEpisode, FMeasurePerEpisode, accuracyPerEpisode):
    precisionPerEpisode /= int(numQueriesPerEpisode)
    recallPerEpisode /= int(numQueriesPerEpisode)
    FMeasurePerEpisode /= int(numQueriesPerEpisode)
    accuracyPerEpisode /= int(numQueriesPerEpisode)
    numQueries.append(numQueriesPerEpisode)
    episodes.append(prevEpisode)
    precision.append(precisionPerEpisode)
    recall.append(recallPerEpisode)
    FMeasure.append(FMeasurePerEpisode)
    accuracy.append(accuracyPerEpisode)
    numQueriesPerEpisode = 0
    precisionPerEpisode = 0.0
    recallPerEpisode = 0.0
    FMeasurePerEpisode = 0.0
    accuracyPerEpisode = 0.0
    return (numQueries, episodes, precision, recall, FMeasure, accuracy, numQueriesPerEpisode, precisionPerEpisode, recallPerEpisode, FMeasurePerEpisode, accuracyPerEpisode)

def parseQualityFileWithEpisodeRep(fileName, outputExcel, configDict):
    episodes = []
    precision = []
    recall = []
    FMeasure = []
    accuracy = []
    numQueries = []
    precisionPerEpisode = 0.0
    recallPerEpisode = 0.0
    FMeasurePerEpisode = 0.0
    accuracyPerEpisode = 0.0
    numQueriesPerEpisode = 0
    prevEpisode = None
    curEpisode = None
    assert configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY' or configDict['SINGULARITY_OR_KFOLD'] == 'KFOLD'
    if configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY':
        episodeIndex = 2
    elif configDict['SINGULARITY_OR_KFOLD'] == 'KFOLD':
        episodeIndex = 0
    with open(fileName) as f:
        for line in f:
            tokens = line.split(";")
            episodeID = float(tokens[episodeIndex].split(":")[1])
            precisionPerEpisode += float(tokens[episodeIndex + 1].split(":")[1])
            recallPerEpisode += float(tokens[episodeIndex + 2].split(":")[1])
            FMeasurePerEpisode += float(tokens[episodeIndex + 3].split(":")[1])
            accuracyPerEpisode += float(tokens[episodeIndex + 4].split(":")[1])
            if episodeID != curEpisode:
                prevEpisode = curEpisode
                curEpisode = episodeID
                if numQueriesPerEpisode > 0 and prevEpisode is not None:
                    (numQueries, episodes, precision, recall, FMeasure, accuracy, numQueriesPerEpisode,
                     precisionPerEpisode, recallPerEpisode, FMeasurePerEpisode, accuracyPerEpisode) = computeStats(
                        prevEpisode, numQueries, episodes, precision, recall, FMeasure, accuracy,
                        numQueriesPerEpisode, precisionPerEpisode, recallPerEpisode, FMeasurePerEpisode,
                        accuracyPerEpisode)
            numQueriesPerEpisode += 1
    # following is for the last episode which was not included in the final result -- note that we pass curEpisode as prevEpisode
    prevEpisode = curEpisode
    (numQueries, episodes, precision, recall, FMeasure, accuracy, numQueriesPerEpisode,
     precisionPerEpisode, recallPerEpisode, FMeasurePerEpisode, accuracyPerEpisode) = computeStats(prevEpisode,
                                                                                                   numQueries, episodes,
                                                                                                   precision, recall,
                                                                                                   FMeasure, accuracy,
                                                                                                   numQueriesPerEpisode,
                                                                                                   precisionPerEpisode,
                                                                                                   recallPerEpisode,
                                                                                                   FMeasurePerEpisode,
                                                                                                   accuracyPerEpisode)
    print "Lengths of episodes: "+str(len(episodes))+", len(precision): "+str(len(precision))+", len(recall): "+str(len(recall))+", len(FMeasure): "+str(len(FMeasure))+", len(accuracy): "+str(len(accuracy))
    df = DataFrame(
        {'episodes':episodes, 'precision': precision,
         'recall': recall, 'FMeasure': FMeasure, 'accuracy': accuracy})
    df.to_excel(outputExcel, sheet_name='sheet1', index=False)


def parseQualityFileWithEpisodeRepOld(fileName, outputExcel, configDict):
    episodes = []
    precision = []
    recall = []
    FMeasure = []
    accuracy = []
    precisionPerEpisode = 0.0
    recallPerEpisode = 0.0
    FMeasurePerEpisode = 0.0
    accuracyPerEpisode = 0.0
    numEpisodes = 0
    numQueries = 0
    assert configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY' or configDict['SINGULARITY_OR_KFOLD'] == 'KFOLD'
    if configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY':
        episodeIndex = 2
    elif configDict['SINGULARITY_OR_KFOLD'] == 'KFOLD':
        episodeIndex = 0
    with open(fileName) as f:
        for line in f:
            numQueries += 1
            tokens = line.split(";")
            numEpisodes = float(tokens[episodeIndex].split(":")[1])
            precisionPerEpisode += float(tokens[episodeIndex + 1].split(":")[1])
            recallPerEpisode += float(tokens[episodeIndex + 2].split(":")[1])
            FMeasurePerEpisode += float(tokens[episodeIndex + 3].split(":")[1])
            accuracyPerEpisode += float(tokens[episodeIndex + 4].split(":")[1])
            if numQueries % int(configDict['EPISODE_IN_QUERIES']) == 0:
                precisionPerEpisode /= int(configDict['EPISODE_IN_QUERIES'])
                recallPerEpisode /= int(configDict['EPISODE_IN_QUERIES'])
                FMeasurePerEpisode /= int(configDict['EPISODE_IN_QUERIES'])
                accuracyPerEpisode /= int(configDict['EPISODE_IN_QUERIES'])
                episodes.append(numEpisodes)
                precision.append(precisionPerEpisode)
                recall.append(recallPerEpisode)
                FMeasure.append(FMeasurePerEpisode)
                accuracy.append(accuracyPerEpisode)
                precisionPerEpisode = 0.0
                recallPerEpisode = 0.0
                FMeasurePerEpisode = 0.0
                accuracyPerEpisode = 0.0
    print "Lengths of episodes: "+str(len(episodes))+", len(precision): "+str(len(precision))+", len(recall): "+str(len(recall))+", len(FMeasure): "+str(len(FMeasure))+", len(accuracy): "+str(len(accuracy))
    df = DataFrame(
        {'episodes':episodes, 'precision': precision,
         'recall': recall, 'FMeasure': FMeasure, 'accuracy': accuracy})
    df.to_excel(outputExcel, sheet_name='sheet1', index=False)

def parseTimeFile(fileName, outputExcel):
    episodes = []
    queryExec = []
    intentCreate = []
    intentPredict = []
    responseTime = []
    with open(fileName) as f:
        for line in f:
            tokens = line.split(";")
            episodes.append(int(tokens[0].split(":")[1]))
            queryExec.append(float(tokens[1].split(":")[1]))
            intentCreate.append(float(tokens[2].split(":")[1]))
            intentPredict.append(float(tokens[3].split(":")[1]))
            responseTime.append(float(tokens[4].split(":")[1]))
    print "Lengths of episodes: " + str(len(episodes)) + ", len(queryExec): " + str(
        len(queryExec)) + ", len(intentCreate): " + str(len(intentCreate)) + ", len(intentPredict): " + str(
        len(intentPredict)) + ", len(responseTime): " + str(len(responseTime))
    df = DataFrame(
        {'episodes': episodes, 'queryExec': queryExec,
         'intentCreate': intentCreate, 'intentPredict': intentPredict, 'responseTime': responseTime})
    df.to_excel(outputExcel, sheet_name='sheet1', index=False)

def parseKFoldTimeDict(avgKFoldTimeDict, avgTrainTime, avgTestTime, outputExcelTimeEval, outputExcelKFoldTimeEval):
    episodes = []
    avgIntentPredict = []
    for episodeIndex in avgKFoldTimeDict:
        episodes.append(episodeIndex)
        avgIntentPredict.append(avgKFoldTimeDict[episodeIndex])
    print "Lengths of episodes: " + str(len(episodes)) + ", avgIntentPredict: "+str(len(avgIntentPredict))
    df = DataFrame({'episodes': episodes, 'avgIntentPredict': avgIntentPredict})
    df.to_excel(outputExcelTimeEval, sheet_name='sheet1', index=False)
    foldID = [ i for i in range(len(avgTrainTime))]
    assert len(avgTrainTime) == len(avgTestTime)
    # there are 11 folds for 10 coz last entry is the average time across all the 10 folds
    print "Lengths of folds: " + str(len(foldID)) + ", avgTrainTime: " + str(len(avgTrainTime)) + ", avgTestTime: " + str(len(avgTestTime))
    df = DataFrame({'foldID': foldID, 'avgTrainTime': avgTrainTime, 'avgTestTime': avgTestTime})
    df.to_excel(outputExcelKFoldTimeEval, sheet_name='sheet1', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="Config parameters file", type=str, required=True)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    accThres = float(configDict['ACCURACY_THRESHOLD'])
    algoName = None
    outputEvalQualityFileName = None
    if configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY':
        outputDir = getConfig(configDict['OUTPUT_DIR'])
    elif configDict['SINGULARITY_OR_KFOLD'] == 'KFOLD':
        outputDir = getConfig(configDict['KFOLD_OUTPUT_DIR'])
    if configDict['ALGORITHM'] == 'CF':
        algoName = configDict['ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF']
        if configDict['SINGULARITY_OR_KFOLD']=='KFOLD':
            outputEvalQualityFileName = getConfig(configDict['KFOLD_OUTPUT_DIR']) + "/OutputEvalQualityShortTermIntent_" + algoName + "_" + configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_ACCURACY_THRESHOLD_" + str(accThres)
        elif configDict['SINGULARITY_OR_KFOLD']=='SINGULARITY':
            outputEvalQualityFileName = outputDir + "/OutputEvalQualityShortTermIntent_" + configDict['ALGORITHM']+"_"+configDict['CF_COSINESIM_MF']+"_"+configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']+"_ACCURACY_THRESHOLD_"+str(accThres)
    elif configDict['ALGORITHM'] == 'RNN':
        algoName = configDict['ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"]
        outputEvalQualityFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputEvalQualityShortTermIntent_" + configDict[
        'ALGORITHM'] + "_" + configDict['RNN_BACKPROP_LSTM_GRU'] + "_" + configDict['INTENT_REP'] + "_" + configDict[
                                    'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                                configDict['EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres)
    outputExcelQuality = outputDir + "/OutputExcelQuality_" + algoName+"_"+configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']+"_ACCURACY_THRESHOLD_"+str(accThres)+".xlsx"
    if configDict['SINGULARITY_OR_KFOLD'] == 'KFOLD':
        parseQualityFileWithoutEpisodeRep(outputEvalQualityFileName, outputExcelQuality, configDict)
    else:
        parseQualityFileWithEpisodeRep(outputEvalQualityFileName, outputExcelQuality, configDict)
    outputEvalTimeFileName = outputDir + "/OutputEvalTimeShortTermIntent_" + algoName+"_"+configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']
    outputExcelTimeEval = outputDir + "/OutputExcelTime_" + algoName+"_"+configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']+".xlsx"
    if configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY':
        parseTimeFile(outputEvalTimeFileName, outputExcelTimeEval)

    '''
    trainSize, testSize, posTrain, posTest, precision, recall, accuracy, FMeasure = read(readFile)
    print "Lengths of trainSize: "+str(len(trainSize))+", len(testSize): "+str(len(testSize))+", len(posTrain): "+str(len(posTrain))+", len(posTest): "+str(len(posTest))+", len(precision): "+str(len(precision))+", len(recall): "+str(len(recall))+", len(accuracy): "+str(len(accuracy))+", len(FMeasure): "+str(len(FMeasure))+"\n"
    df = DataFrame({'trainSize': trainSize, 'posTrain': posTrain, 'testSize': testSize, 'posTest': posTest, 'precision': precision, 'recall': recall, 'accuracy': accuracy, 'FMeasure':FMeasure})
    df.to_excel('PerfMetrics.xlsx', sheet_name = 'sheet1', index=False)
    
def parseQualityFileRNNDeprecated(fileName, outputExcel, configDict):
    episodes = []
    accuracy = []
    accuracyPerEpisode = 0.0
    numEpisodes = 0
    numQueries = 0
    with open(fileName) as f:
        for line in f:
            numQueries += 1
            tokens = line.split(";")
            if tokens[3].split(":")[1] == 'nan' or tokens[3].split(":")[1] == 'inf':
                accuracyPerEpisode+= 0.0
            else:
                accuracyPerEpisode += float(tokens[3].split(":")[1])
            if numQueries % int(configDict['EPISODE_IN_QUERIES']) == 0:
                numEpisodes = float(tokens[2].split(":")[1])
                accuracyPerEpisode /= int(configDict['EPISODE_IN_QUERIES'])
                episodes.append(numEpisodes)
                accuracy.append(accuracyPerEpisode)
                accuracyPerEpisode = 0.0
    print "Lengths of episodes: "+str(len(episodes))+", len(accuracy): "+str(len(accuracy))
    df = DataFrame(
        {'episodes':episodes, 'accuracy': accuracy})
    df.to_excel(outputExcel, sheet_name='sheet1', index=False)
    

    '''
