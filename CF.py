import sys
import os
import time
import QueryRecommender as QR
import CFCosineSim
import CFMF

def runCosineSimOrMF(configDict):
    if configDict['CF_COSINESIM_MF']=='COSINESIM':
        CFCosineSim.runCFCosineSim(configDict)
    elif configDict['CF_COSINESIM_MF']=='MF':
        CFMF.runCF(configDict)
    else:
        print "CF can either be COSINESIM or MF !!"
        sys.exit(0)
    return

def runIntentPrediction(configDict):
    runCosineSimOrMF(configDict)