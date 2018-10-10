from __future__ import division
import argparse
import sys, os
from pandas import DataFrame
import ParseConfigFile as parseConfig

def parseQualityFile(fileName, outputExcel):
    episodes = []
    precision = []
    recall = []
    FMeasure = []
    accuracy = []
    with open(fileName) as f:
        for line in f:
            tokens = line.split(";")
            episodes.append(int(tokens[2].split(":")[2]))
            p = float(tokens[3].split(":")[2])
            r = float(tokens[4].split(":")[2])
            if p == 0 or r == 0:
                f = 0
            else:
                f = 2 * p * r / (p+r)
            precision.append(p)
            recall.append(r)
            FMeasure.append(f)
            accuracy.append(float(tokens[5].split(":")[2]))
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
            episodes.append(int(tokens[0].split(":")[2]))
            queryExec.append(float(tokens[1].split(":")[2]))
            intentCreate.append(float(tokens[2].split(":")[2]))
            intentPredict.append(float(tokens[3].split(":")[2]))
            responseTime.append(float(tokens[4].split(":")[2]))
    print "Lengths of episodes: " + str(len(episodes)) + ", len(queryExec): " + str(
        len(queryExec)) + ", len(intentCreate): " + str(len(intentCreate)) + ", len(intentPredict): " + str(
        len(intentPredict)) + ", len(responseTime): " + str(len(responseTime))
    df = DataFrame(
        {'episodes': episodes, 'queryExec': queryExec,
         'intentCreate': intentCreate, 'intentPredict': intentPredict, 'responseTime': responseTime})
    df.to_excel(outputExcel, sheet_name='sheet2', index=False)

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("configFile.txt")
    outputEvalQualityFileName = configDict['OUTPUT_DIR'] + "/OutputEvalQualityShortTermIntent_" + configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']+"_ACCURACY_THRESHOLD_"+str(configDict['ACCURACY_THRESHOLD'])
    outputEvalTimeFileName = configDict['OUTPUT_DIR'] + "/OutputEvalTimeShortTermIntent_" + configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']+"_ACCURACY_THRESHOLD_"+str(configDict['ACCURACY_THRESHOLD'])
    outputExcel = configDict['OUTPUT_DIR'] + "/OutputExcel_" + configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']+"_ACCURACY_THRESHOLD_"+str(configDict['ACCURACY_THRESHOLD']+".xlsx")
    parseQualityFile(outputEvalQualityFileName, outputExcel)
    parseTimeFile(outputEvalTimeFileName, outputExcel)

    '''
    trainSize, testSize, posTrain, posTest, precision, recall, accuracy, FMeasure = read(readFile)
    print "Lengths of trainSize: "+str(len(trainSize))+", len(testSize): "+str(len(testSize))+", len(posTrain): "+str(len(posTrain))+", len(posTest): "+str(len(posTest))+", len(precision): "+str(len(precision))+", len(recall): "+str(len(recall))+", len(accuracy): "+str(len(accuracy))+", len(FMeasure): "+str(len(FMeasure))+"\n"
    df = DataFrame({'trainSize': trainSize, 'posTrain': posTrain, 'testSize': testSize, 'posTest': posTest, 'precision': precision, 'recall': recall, 'accuracy': accuracy, 'FMeasure':FMeasure})
    df.to_excel('PerfMetrics.xlsx', sheet_name = 'sheet1', index=False)
    '''
