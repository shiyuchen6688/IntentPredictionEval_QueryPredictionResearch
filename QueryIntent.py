import sys
import os
import time, argparse
from bitmap import BitMap
import ParseConfigFile as parseConfig
import QueryParser as qp
import TupleIntent as ti
import re
def createQueryIntentRep(sessQuery, configDict, queryVocabulary):
    if sessQuery not in queryVocabulary:
        newBitPos = len(queryVocabulary)
        queryVocabulary[sessQuery] = newBitPos
    resObj = BitMap(len(queryVocabulary))
    resObj.set(queryVocabulary[sessQuery])
    return (queryVocabulary,resObj)

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("configFile.txt")
    if configDict["INTENT_REP"] == "QUERY":
        queryIntentSessionsFile = configDict['QUERY_INTENT_SESSIONS']
    else:
        print "This supports only query intent gen !!"
        sys.exit(0)
    try:
        os.remove(queryIntentSessionsFile)
    except OSError:
        pass
    with open(configDict['QUERYSESSIONS']) as f:
        for line in f:
            sessQueries = line.split(";")
            sessName = sessQueries[0]
            queryVocabulary = {} #dict with query as key and bit position/dimension as value
            for i in range(1, len(sessQueries) - 1):  # we need to ignore the empty query coming from the end of line semicolon ;
                sessQuery = sessQueries[i].split("~")[0]
                #sessQuery = "SELECT nyc_yellow_tripdata_2016_06_sample_1_percent.store_and_fwd_flag AS store_and_fwd_flag FROM public.nyc_yellow_tripdata_2016_06_sample_1_percent nyc_yellow_tripdata_2016_06_sample_1_percent GROUP BY 1 ORDER BY 1 ASC NULLS FIRST"
                sessQuery = ' '.join(sessQuery.split())
                (queryVocabulary,resObj) = createQueryIntentRep(sessQuery, configDict, queryVocabulary) #rowIDs passed should be None, else it won't fill up
                queryName = sessName+", Query "+str(i)
                outputIntentLine = queryName+"; OrigQuery: "+sessQuery+";"+str(resObj)
                ti.appendToFile(queryIntentSessionsFile,outputIntentLine)
                print "Generated fragment for "+queryName