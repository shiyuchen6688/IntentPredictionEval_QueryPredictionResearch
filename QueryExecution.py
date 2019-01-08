import sys
import os
import time, argparse
import psycopg2
import psycopg2.extras
import ParseConfigFile as parseConfig
from ParseConfigFile import getConfig
import TupleIntent as tupleIntent
import FragmentIntent as fragmentIntent
import QueryIntent as queryIntent
import gc
import mysql.connector
from mysql.connector import errorcode
import MINC_prepareJoinKeyPairs as minc_mysql

def replaceTableName(sessQuery, configDict):
    if configDict['DATASET'] == 'NYCTaxiTrips':
        sessQuery.replace(configDict['SAMPLETABLE'], configDict['FULLTABLE'])
    return sessQuery

def getRowIDs(cur):
    rows = cur.fetchall()
    rowIDs = None
    if rows is None or len(rows) == 0:
        return None
    firstRow = rows[0]
    if 'id' in firstRow:
        rowIDs = []
        for row in rows:
            rowIDs.append(int(row['id']))
        del rows
        gc.collect()
    return rowIDs


def executeQuery(sessQuery, configDict):
    execUponSampleData = configDict['EXEC_SAMPLE']
    if execUponSampleData!="True":
        sessQuery = replaceTableName(sessQuery, configDict)
    try:
        conn = psycopg2.connect("dbname='madlibtest' user='postgres' password=''")
    except:
        print ("I am unable to connect to the database.")

    # If we are accessing the rows via column name instead of position we
    # need to add the arguments to conn.cursor instead of conn.cursor()

    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        #sessQuery = "SELECT * from nyc_yellow_tripdata_2016_06 order by id desc limit 10"
        #cur.execute("""SELECT * from nyc_yellow_tripdata_2016_06 order by id desc limit 10""")
        if configDict['DATASET'] == 'NYCTaxiTrips':
            sessQuery = sessQuery.split("~")[0]
        cur.execute(sessQuery)
        conn.commit()
        return cur
    except:
        print ("cannot execute the query on Postgres")
        exit(0)

def executeNYCQueryWithIntent(sessQuery, configDict, queryVocabulary):
    startTime = time.time()
    cur = executeQuery(sessQuery, configDict)
    queryExecutionTime = float(time.time() - startTime)
    startTime = time.time()
    if configDict['INTENT_REP'] == 'TUPLE':
        # rowIDs = getRowIDs(cur)
        (newQuery, resObj) = tupleIntent.createTupleIntentRep(None, sessQuery, configDict)
    elif configDict['INTENT_REP'] == 'FRAGMENT':
        resObj = fragmentIntent.createFragmentIntentRep(sessQuery, configDict)
    elif configDict['INTENT_REP'] == 'QUERY':
        (queryVocabulary, resObj) = queryIntent.createQueryIntentRepFullDimensionality(sessQuery, configDict,
                                                                                       queryVocabulary)
    intentCreationTime = float(time.time() - startTime)
    return (queryVocabulary, resObj, queryExecutionTime, intentCreationTime)


def executeMINCQueryWithIntent(sessQuery, configDict, queryVocabulary):
    resObj = None
    startTime = time.time()
    cnx = minc_mysql.connectToMySQL(configDict)
    cursor = cnx.cursor()
    cursor.execute(sessQuery)
    queryExecutionTime = float(time.time() - startTime)
    # no need to worry about query vocabulary or resObj here
    # intent creation happens in Java and the avg time per query is a second. To be fairer we can run using Python Java connector later
    intentCreationTime = 1.0
    return (queryVocabulary, resObj, queryExecutionTime, intentCreationTime)


def executeQueryWithIntent(sessQuery, configDict, queryVocabulary):
    assert configDict['DATASET'] == 'NYCTaxitrips' or configDict['DATASET'] == 'MINC'
    if configDict['DATASET'] == 'NYCTaxiTrips':
        return executeNYCQueryWithIntent(sessQuery, configDict, queryVocabulary)
    elif configDict['DATASET'] == 'MINC':
        return executeMINCQueryWithIntent(sessQuery, configDict, queryVocabulary)

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("configFile.txt")
    with open(getConfig(configDict['QUERYSESSIONS'])) as f:
        for line in f:
            sessQueries = line.split(";")
            sessName = sessQueries[0]
            for i in range(1, len(sessQueries) - 1):  # we need to ignore the empty query coming from the end of line semicolon ;
                sessQuery = sessQueries[i].split("~")[0]
                # sessQuery = "SELECT nyc_yellow_tripdata_2016_06_sample_1_percent.dropoff_latitude AS dropoff_latitude, nyc_yellow_tripdata_2016_06_sample_1_percent.dropoff_longitude AS dropoff_longitude, nyc_yellow_tripdata_2016_06_sample_1_percent.fare_amount AS fare_amount FROM public.nyc_yellow_tripdata_2016_06_sample_1_percent nyc_yellow_tripdata_2016_06_sample_1_percent GROUP BY 1, 2, 3 HAVING ((CAST(MIN(nyc_yellow_tripdata_2016_06_sample_1_percent.fare_amount) AS DOUBLE PRECISION) >= 11.999999999999879) AND (CAST(MIN(nyc_yellow_tripdata_2016_06_sample_1_percent.fare_amount) AS DOUBLE PRECISION) <= 14.00000000000014))"
                executeQuery(sessQuery, configDict)
                print ("Executed "+sessName+", Query "+str(i))