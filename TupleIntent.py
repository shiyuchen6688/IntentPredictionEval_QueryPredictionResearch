import sys
import os
import time, argparse
from bitmap import BitMap
import ParseConfigFile as parseConfig
import QueryParser as qp
def createTupleIntentRep(rowIDs, sessQuery, configDict):
    resObj = None
    if configDict['EXEC_SAMPLE']:
        totalDataSize = int(configDict['SAMPLETABLESIZE'])
    else:
        totalDataSize = int(configDict['FULLTABLESIZE'])
    resObj = BitMap(totalDataSize)
    if rowIDs is None:
        (newQuery,rowIDs) = qp.fetchRowIDs(sessQuery, configDict)
    if rowIDs is None:
        for rowID in range(totalDataSize):
            resObj.set(rowID)  # here rowID was forced to start from  0 in the for loop as all rows are being set to 1
    else:
        for rowID in rowIDs:
            resObj.set(rowID-1) # because rowIDs start from 1 but bit positions start from 0
    return (newQuery,resObj)

def appendToFile(outputFile, outputLine):
    with open(outputFile, 'a') as outFile:
        outFile.write(outputLine+"\n")

if __name__ == "__main__":
    #rowIDs=[11, 1, 2, 3, 4, 109673] # rowIDs start from 1
    if __name__ == "__main__":
        configDict = parseConfig.parseConfigFile("configFile.txt")
        tupleIntentSessionsFile = configDict['TUPLEINTENTSESSIONS']
        try:
            os.remove(tupleIntentSessionsFile)
        except OSError:
            pass
        with open(configDict['QUERYSESSIONS']) as f:
            for line in f:
                sessQueries = line.split(";")
                sessName = sessQueries[0]
                for i in range(1, len(sessQueries) - 1):  # we need to ignore the empty query coming from the end of line semicolon ;
                    sessQuery = sessQueries[i].split("~")[0]
                    # sessQuery = "SELECT nyc_yellow_tripdata_2016_06_sample_1_percent.dropoff_latitude AS dropoff_latitude, nyc_yellow_tripdata_2016_06_sample_1_percent.dropoff_longitude AS dropoff_longitude, nyc_yellow_tripdata_2016_06_sample_1_percent.fare_amount AS fare_amount FROM public.nyc_yellow_tripdata_2016_06_sample_1_percent nyc_yellow_tripdata_2016_06_sample_1_percent GROUP BY 1, 2, 3 HAVING ((CAST(MIN(nyc_yellow_tripdata_2016_06_sample_1_percent.fare_amount) AS DOUBLE PRECISION) >= 11.999999999999879) AND (CAST(MIN(nyc_yellow_tripdata_2016_06_sample_1_percent.fare_amount) AS DOUBLE PRECISION) <= 14.00000000000014))"
                    sessQuery = ' '.join(sessQuery.split())
                    (newQuery,resObj) = createTupleIntentRep(None, sessQuery, configDict) #rowIDs passed should be None, else it won't fill up
                    queryName = sessName+", Query "+str(i)
                    if newQuery is None:
                        outputIntentLine = queryName+"; OrigQuery: "+sessQuery+";"+str(resObj)
                    else:
                        outputIntentLine = queryName+";"+newQuery+";"+str(resObj)
                    appendToFile(tupleIntentSessionsFile,outputIntentLine)
                    print "Executed "+queryName


