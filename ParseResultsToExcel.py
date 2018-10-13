from __future__ import division
import argparse
import sys, os
from pandas import DataFrame
import ParseConfigFile as parseConfig

def parseQualityFileRNN(fileName, outputExcel):
    episodes = []
    accuracy = []
    accuracyPerEpisode = 0.0
    numEpisodes = 0
    numQueries = 0
    with open(fileName) as f:
        for line in f:
            numQueries += 1
            tokens = line.split(";")
            accuracyPerEpisode += float(tokens[3].split(":")[1])
            if numQueries % int(configDict['EPISODE_IN_QUERIES']) == 0:
                numEpisodes += 1
                accuracyPerEpisode /= int(configDict['EPISODE_IN_QUERIES'])
                episodes.append(numEpisodes)
                accuracy.append(accuracyPerEpisode)
                accuracyPerEpisode = 0.0
    print "Lengths of episodes: "+str(len(episodes))+", len(accuracy): "+str(len(accuracy))
    df = DataFrame(
        {'episodes':episodes, 'accuracy': accuracy})
    df.to_excel(outputExcel, sheet_name='sheet1', index=False)


def parseQualityFile(fileName, outputExcel):
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
                FMeasurePerEpisode = 0
            else:
                FMeasurePerEpisode = 2 * precisionPerEpisode * recallPerEpisode / (precisionPerEpisode+recallPerEpisode)
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

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("configFile.txt")
    accThresList = [0.75, 0.8, 0.85, 0.9, 0.95]
    for accThres in accThresList:
        outputEvalQualityFileName = configDict['OUTPUT_DIR'] + "/OutputEvalQualityShortTermIntent_" + configDict['ALGORITHM']+"_"+configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']+"_ACCURACY_THRESHOLD_"+str(accThres)
        outputExcelQuality = configDict['OUTPUT_DIR'] + "/OutputExcelQuality_" + configDict['ALGORITHM']+"_"+configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']+"_ACCURACY_THRESHOLD_"+str(accThres)+".xlsx"
        parseQualityFile(outputEvalQualityFileName, outputExcelQuality)

    outputEvalTimeFileName = configDict['OUTPUT_DIR'] + "/OutputEvalTimeShortTermIntent_" + configDict['ALGORITHM']+"_"+configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']
    outputExcelTimeEval = configDict['OUTPUT_DIR'] + "/OutputExcelTime_" + configDict['ALGORITHM']+"_"+configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']+".xlsx"
    parseTimeFile(outputEvalTimeFileName, outputExcelTimeEval)

    '''
    trainSize, testSize, posTrain, posTest, precision, recall, accuracy, FMeasure = read(readFile)
    print "Lengths of trainSize: "+str(len(trainSize))+", len(testSize): "+str(len(testSize))+", len(posTrain): "+str(len(posTrain))+", len(posTest): "+str(len(posTest))+", len(precision): "+str(len(precision))+", len(recall): "+str(len(recall))+", len(accuracy): "+str(len(accuracy))+", len(FMeasure): "+str(len(FMeasure))+"\n"
    df = DataFrame({'trainSize': trainSize, 'posTrain': posTrain, 'testSize': testSize, 'posTest': posTest, 'precision': precision, 'recall': recall, 'accuracy': accuracy, 'FMeasure':FMeasure})
    df.to_excel('PerfMetrics.xlsx', sheet_name = 'sheet1', index=False)
    '''
