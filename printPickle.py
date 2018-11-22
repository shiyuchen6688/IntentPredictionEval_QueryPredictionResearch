import QueryRecommender as QR
import sys
'''
def computeAvgPerDict(avgDict, expectedIterLength):
    # Each key represents an active learning iteration. A few folds may have more iterations than others coz they may get slightly more training data than others
    # In such a case include the before last active learning iteration's avg performance also into the last iteration, because both represent convergence
    maxValidKey = -1
    for key in avgDict:
        if int(key) > maxValidKey and len(avgDict[key]) < expectedIterLength:
            maxValidKey = key
            if maxValidKey < len(avgDict)-1: # only the last iteration is allowed to have fewer than kfold iteration length - coz remainder occurs only at the end
                print "Invalid Max Key !!"
                sys.exit(0)
    prevLen = 1
    for key in avgDict:
        if int(key) == maxValidKey and key>=1:
            prevKey = key-1
            numerator = sum(avgDict[key])+avgDict[prevKey]*prevLen
            denominator = len(avgDict[key])+prevLen
            avgDict[key] = float(numerator) / float(denominator)
        else:
            prevLen = len(avgDict[key])
            avgDict[key] = float(sum(avgDict[key])) / float(len(avgDict[key]))
    return avgDict
'''

def computeAvgPerDict(avgDict, expectedIterLength):
    # Each key represents an active learning iteration. A few folds may have more iterations than others coz they may get slightly more training data than others
    # In such a case include the before last active learning iteration's avg performance also into the last iteration, because both represent convergence
    maxValidKey = -1
    for key in avgDict:
        if int(key) > maxValidKey and len(avgDict[key]) < expectedIterLength:
            maxValidKey = key
            if maxValidKey < len(avgDict)-1: # only the last iteration is allowed to have fewer than kfold iteration length - coz remainder occurs only at the end
                print "Invalid Max Key !!"
                sys.exit(0)
    avgOutputDict = {}
    for key in avgDict:
        if key == maxValidKey and key>=1:
            for prevFoldID in avgDict[key-1]:
                if prevFoldID not in avgDict[maxValidKey]:
                    avgDict[maxValidKey][prevFoldID] = avgDict[key-1][prevFoldID]
        avgOutputDict[key] = float(sum(avgDict[key].values())) / float(len(avgDict[key]))
    del avgDict
    return avgOutputDict

if __name__ == "__main__":
    '''
    readObj = QR.readFromPickleFile('/Users/postgres/Documents/DataExploration-Research/CreditCardDataset/kFold/outputDir/avgKFoldFMeasure.pickle')
    '''
    readObj = {}
    for i in range(5):
        readObj[i] = {}
        numFolds = 10
        if i==4:
            numFolds = 5
        for j in range(numFolds):
            if i==4:
                readObj[i][j] = 0.0
            else:
                readObj[i][j] = 0.85
    for x in readObj:
        print 'key=' + str(x) + ', value=' + str(readObj[x])
    readObj = computeAvgPerDict(readObj, 10)
    for x in readObj:
        print 'key=' + str(x) + ', value=' + str(readObj[x])

