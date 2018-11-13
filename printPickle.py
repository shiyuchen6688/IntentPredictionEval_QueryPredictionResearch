import QueryRecommender as QR
if __name__ == "__main__":
    readObj = QR.readFromPickleFile('/Users/postgres/Documents/DataExploration-Research/CreditCardDataset/kFold/outputDir/avgKFoldFMeasure.pickle')
    for x in readObj:
        print 'key='+str(x)+', value='+str(readObj[x])
