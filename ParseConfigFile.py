def parseConfigFile(fileName):
    configDict = {}
    with open(fileName) as f:
        for line in f:
            (key, val) = line.strip().split("=")
            configDict[key] = val
    return configDict