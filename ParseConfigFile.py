from os.path import expanduser
import socket
def parseSchema(schemaFileName):
    schemaDict = {}
    colIndex = 0  #colIndex starts from 0
    with open(schemaFileName) as f:
        for line in f:
            colName = line.split("=")[0]
            schemaDict[colName]=colIndex
            colIndex=colIndex+1 # no need to keep data type recorded in the schema Dict. It is column as key and colIndex as value
    return schemaDict

def getConfig(relativePath):
    homeDir = expanduser("~")
    if socket.gethostname() == "en4119510l":
        homeDir = "/hdd2/vamsiCodeData"
    absPath = homeDir+"/"+relativePath
    return absPath

def parseConfigFile(fileName):
    configDict = {}
    with open(fileName) as f:
        for line in f:
            (key, val) = line.strip().split("=")
            configDict[key] = val
    return configDict