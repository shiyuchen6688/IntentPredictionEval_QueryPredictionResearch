import sys
import os
import time, argparse
import psycopg2
import TupleIntent as tupleIntent
import FragmentIntent as fragmentIntent
import QueryIntent as queryIntent
import gc
def replaceTableName(sessQuery, configDict):
    if configDict['DATASET'] == 'NYCTaxiTrips':
        sessQuery.replace(configDict['SAMPLETABLE'], configDict['FULLTABLE'])
    return sessQuery

def executeQuery(sessQuery, configDict, withIntent):
    execUponSampleData = bool(configDict['EXEC_SAMPLE'])
    if not execUponSampleData:
        sessQuery = replaceTableName(sessQuery, configDict)
    try:
        conn = psycopg2.connect("dbname='madlibtest' user='postgres' password=''")
    except:
        print "I am unable to connect to the database."

    # If we are accessing the rows via column name instead of position we
    # need to add the arguments to conn.cursor instead of conn.cursor()

    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        #sessQuery = "SELECT * from nyc_yellow_tripdata_2016_06 order by id desc limit 10"
        #cur.execute("""SELECT * from nyc_yellow_tripdata_2016_06 order by id desc limit 10""")
        if configDict['DATASET'] == 'NYCTaxiTrips':
            sessQuery = sessQuery.split("~")[0]
        cur.execute(sessQuery)
        if not withIntent:
            rows = cur.fetchall()
            return rows
    except:
        print "cannot execute the query on Postgres"

    if configDict['INTENT_REP']=='tuple':
        rows = cur.fetchall()
        rowIDs = None
        firstRow = rows[0]
        if 'id' in firstRow:
            rowIDs = []
            for row in rows:
                rowIDs.append(row['id'])
            del rows
            gc.collect()
        resObj = tupleIntent.createTupleIntentRep(rowIDs, sessQuery, configDict)
    elif configDict['INTENT_REP']=='fragment':
        resObj = fragmentIntent.createFragmentIntentRep(sessQuery, configDict)
    elif configDict['INTENT_REP'] == 'query':
        resObj = queryIntent.createQueryIntentRep(sessQuery, configDict)
    return resObj

if __name__ == "__main__":
    executeQuery(None,None,True)
