import sys
import os
import time, argparse
from bitmap import BitMap
import ParseConfigFile as parseConfig
from ParseConfigFile import getConfig
import QueryParser as qp
import TupleIntent as ti
import re

def computeEmbeddingVectors(inputFile, outputFile, bitOrWeighted):
    try:
        os.remove(outputFile)
    except OSError:
        pass
    embedVocabulary = {}  # dict with intent vector as key and bit position/dimension as value
    len = 0
    with open(inputFile) as f:
        for line in f:
            tokens = line.split(";")
            intentVec = ';'.join(tokens[2:])
            if intentVec not in embedVocabulary:
                embedVocabulary[intentVec] = len
                len+=1
    f.close()
    with open(inputFile) as f:
        for line in f:
            tokens = line.split(";")
            intentVec = ';'.join(tokens[2:])
            embedVec = BitMap(len)
            pos = embedVocabulary[intentVec]
            embedVec.set(pos)

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("configFile.txt")
    tupleIntentInputFile = getConfig(configDict['TUPLEINTENTSESSIONS'])
    tupleIntentOutputFile = getConfig(configDict['TUPLEINTENTSESSIONS_RNN_EMBEDDING'])

    bitFragmentIntentInputFile = getConfig(configDict['BIT_FRAGMENT_INTENT_SESSIONS'])
    bitFragmentIntentOutputFile = getConfig(configDict['BIT_FRAGMENT_INTENT_SESSIONS_RNN_EMBEDDING'])

    weightedFragmentIntentInputFile = getConfig(configDict['WEIGHTED_FRAGMENT_INTENT_SESSIONS'])
    weightedFragmentIntentOutputFile = getConfig(configDict['WEIGHTED_FRAGMENT_INTENT_SESSIONS_RNN_EMBEDDING'])

    computeEmbeddingVectors(tupleIntentInputFile, tupleIntentOutputFile, 'BIT')
    computeEmbeddingVectors(bitFragmentIntentInputFile, bitFragmentIntentOutputFile, 'BIT')
    computeEmbeddingVectors(weightedFragmentIntentInputFile, weightedFragmentIntentOutputFile, 'WEIGHTED')