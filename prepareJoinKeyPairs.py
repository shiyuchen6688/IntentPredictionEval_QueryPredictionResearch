import sys
import os
import time, argparse
from bitmap import BitMap
import ParseConfigFile as parseConfig
import QueryParser as qp
import TupleIntent as ti
import re
import mysql.connector
from mysql.connector import errorcode

def connectToMySQL(configDict):
    try:
        uname=configDict['MYSQL_UNAME']
        passwd=configDict['MYSQL_PASSWORD']
        hostname=configDict['MYSQL_HOST']
        dbname=configDict['MYSQL_DB']
        cnx = mysql.connector.connect(user=uname, password=passwd, host=hostname, database=dbname, auth_plugin='mysql_native_password')
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    return cnx

def createTableDict(cnx):
    tableDict = {}
    query = "SHOW TABLES"
    cursor = cnx.cursor()
    cursor.execute(query)
    index = 0
    for cols in cursor:
        tableName = str(cols[0])
        assert tableName not in tableDict
        tableDict[tableName] = index
        print "tablename: "+str(tableName)+", index: "+str(index)
        index+=1
    return tableDict

def fetchSchema(configDict):
    cnx = connectToMySQL(configDict)
    tableDict = createTableDict(cnx)
    print "connection successful"

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("configFileMinc.txt")
    fetchSchema(configDict)