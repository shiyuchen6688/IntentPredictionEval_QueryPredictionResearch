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
import CFCosineSim
import argparse
from ParseConfigFile import getConfig
import threading
import copy
import multiprocessing
from multiprocessing.pool import ThreadPool
from multiprocessing import Array

def pruneUnImportantDimensions(predictedY, configDict):
    newPredictedY = []
    minY = min(predictedY)
    maxY = max(predictedY)
    for y in predictedY:
        newY = float(y-minY)/float(maxY-minY)
        if newY < float(configDict['RNN_WEIGHT_VECTOR_THRESHOLD']):
            newY = 0.0
        newPredictedY.append(newY)
    return newPredictedY

def regenerateQuery(threadID, predictedY, configDict, curSessID, curQueryID, sessionDictCurThread, sampledQueryHistory, sessionStreamDict):
    topKPredictedIntents = []
    return topKPredictedIntents