import sys
import os
import time
import QueryRecommender as QR
import CFCosineSim
import CFMF

def runCosineSimOrMF(configDict):
    if configDict['INTENT_REP']=='TUPLE':
        intentSessionFile = configDict['TUPLEINTENTSESSIONS']
    elif configDict['INTENT_REP']=='FRAGMENT' and configDict['BIT_OR_WEIGHTED']=='BIT':
        intentSessionFile = configDict['BIT_FRAGMENT_INTENT_SESSIONS']
    elif configDict['INTENT_REP']=='FRAGMENT' and configDict['BIT_OR_WEIGHTED']=='WEIGHTED':
        intentSessionFile = configDict['WEIGHTED_FRAGMENT_INTENT_SESSIONS']
    elif configDict['INTENT_REP']=='QUERY':
        intentSessionFile = configDict['QUERY_INTENT_SESSIONS']
    else:
        print "ConfigDict['INTENT_REP'] must either be TUPLE or FRAGMENT or QUERY !!"
        sys.exit(0)
    if configDict['CF_COSINESIM_MF']=='COSINESIM':
        CFCosineSim.runCF(intentSessionFile, configDict)
    elif configDict['CF_COSINESIM_MF']=='MF':
        CFMF.runCF(intentSessionFile, configDict)
    else:
        print "CF can either be COSINESIM or MF !!"
        sys.exit(0)



def runIntentPrediction(configDict):
    runCosineSimOrMF(configDict)