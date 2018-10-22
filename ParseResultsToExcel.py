from __future__ import division
import argparse
import sys, os
from pandas import DataFrame
import ParseConfigFile as parseConfig

def parseQualityFileRNN(fileName, outputExcel, configDict):
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

def parseQualityFileCFCosineSim(fileName, outputExcel, configDict):
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

def parseQualityFile(fileName, outputExcel, configDict):
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
    with open(fileName) as f:
        for line in f:
            numQueries += 1
            tokens = line.split(";")
            precisionPerEpisode += float(tokens[3].split(":")[1])
            recallPerEpisode += float(tokens[4].split(":")[1])
            if precisionPerEpisode == 0 or recallPerEpisode == 0:
                FMeasurePerEpisode += 0
            else:
                FMeasurePerEpisode += 2 * precisionPerEpisode * recallPerEpisode / (precisionPerEpisode+recallPerEpisode)
            accuracyPerEpisode += float(tokens[5].split(":")[1])
            if numQueries % int(configDict['EPISODE_IN_QUERIES']) == 0:
                numEpisodes += 1
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

def parseTimeDict(avgKFoldTimeDict, outputExcelTimeEval):
    episodes = []
    avgIntentPredict = []
    for episodeIndex in avgKFoldTimeDict:
        episodes.append(episodeIndex)
        avgIntentPredict.append(avgKFoldTimeDict[episodeIndex])
    print "Lengths of episodes: " + str(len(episodes)) + ", avgIntentPredict: "+str(len(avgIntentPredict))
    df = DataFrame({'episodes': episodes, 'avgIntentPredict': avgIntentPredict})
    df.to_excel(outputExcelTimeEval, sheet_name='sheet1', index=False)

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("configFile.txt")
    accThres = float(configDict['ACCURACY_THRESHOLD'])
    algoName = None
    outputEvalQualityFileName = None
    if configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY':
        outputDir = configDict['OUTPUT_DIR']
    elif configDict['SINGULARITY_OR_KFOLD'] == 'KFOLD':
        outputDir = configDict['KFOLD_OUTPUT_DIR']
    if configDict['ALGORITHM'] == 'CF':
        algoName = configDict['ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF']
        if configDict['SINGULARITY_OR_KFOLD']=='KFOLD':
            outputEvalQualityFileName = configDict['KFOLD_OUTPUT_DIR'] + "/OutputEvalQualityShortTermIntent_" + algoName + "_" + configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_ACCURACY_THRESHOLD_" + str(accThres)
        elif configDict['SINGULARITY_OR_KFOLD']=='SINGULARITY':
            outputEvalQualityFileName = outputDir + "/OutputEvalQualityShortTermIntent_" + configDict['ALGORITHM']+"_"+configDict['CF_COSINESIM_MF']+"_"+configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']+"_ACCURACY_THRESHOLD_"+str(accThres)
    elif configDict['ALGORITHM'] == 'RNN':
        algoName = configDict['ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"]
        outputEvalQualityFileName = outputDir + "/OutputFileShortTermIntent_" + \
                                configDict['ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"] + "_" + \
                                configDict['INTENT_REP'] + "_" + \
                                configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                                configDict['EPISODE_IN_QUERIES']
    outputExcelQuality = outputDir + "/OutputExcelQuality_" + algoName+"_"+configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']+"_ACCURACY_THRESHOLD_"+str(accThres)+".xlsx"
    if configDict['ALGORITHM'] == 'CF':
        parseQualityFileCFCosineSim(outputEvalQualityFileName, outputExcelQuality, configDict)
    elif configDict['ALGORITHM'] == 'RNN':
        parseQualityFileRNN(outputEvalQualityFileName, outputExcelQuality, configDict)
    outputEvalTimeFileName = outputDir + "/OutputEvalTimeShortTermIntent_" + algoName+"_"+configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']
    outputExcelTimeEval = outputDir + "/OutputExcelTime_" + algoName+"_"+configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']+".xlsx"
    if configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY':
        parseTimeFile(outputEvalTimeFileName, outputExcelTimeEval)

    '''
    trainSize, testSize, posTrain, posTest, precision, recall, accuracy, FMeasure = read(readFile)
    print "Lengths of trainSize: "+str(len(trainSize))+", len(testSize): "+str(len(testSize))+", len(posTrain): "+str(len(posTrain))+", len(posTest): "+str(len(posTest))+", len(precision): "+str(len(precision))+", len(recall): "+str(len(recall))+", len(accuracy): "+str(len(accuracy))+", len(FMeasure): "+str(len(FMeasure))+"\n"
    df = DataFrame({'trainSize': trainSize, 'posTrain': posTrain, 'testSize': testSize, 'posTest': posTest, 'precision': precision, 'recall': recall, 'accuracy': accuracy, 'FMeasure':FMeasure})
    df.to_excel('PerfMetrics.xlsx', sheet_name = 'sheet1', index=False)
    '''
