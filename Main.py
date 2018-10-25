# This is the entry point. Has a main function and allows for multiple algorithms modeling multiple formulations of human intent
# Also allows for mutiple datasets, all placed in a configuration file
import sys
import argparse
import os
import time
import ParseConfigFile as parseConfig
import CF
import LSTM_RNN
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="Config parameters file", type=str, required=True)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    if configDict["ALGORITHM"]=="CF":
        CF.runIntentPrediction(configDict)
    elif configDict["ALGORITHM"]=="RNN":
        LSTM_RNN.executeRNN(configDict)