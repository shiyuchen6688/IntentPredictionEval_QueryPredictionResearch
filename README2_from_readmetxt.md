### Creating Schema Dicts -- TODO: Instructions ###

### Creating the feature vectors for NYCTaxiTrips ###

# 0) Step 0: Creating the raw logs from Tableau -- appending new queries to existing queries
python NYCSessionLogCreation.py # (this python file is in ~/Documents/DataExploration-Research/CreditCardDataset)

# 1) Step 1: Creating cleaned sessions from the raw logs
python cleanQuerySessions.py -input /Users/postgres/Documents/DataExploration-Research/CreditCardDataset/NYCOutputSessionLog -output /Users/postgres/Documents/DataExploration-Research/CreditCardDataset/NYCCleanedSessions

# 2) Step 2: Creating concurrent sessions from cleaned sessions
python ConcurrentSessions.py -config configFile.txt # configFile.txt contains the NYCTaxiTrips config parameters

# 3) Step 3: Creating fragment vectors from concurrent query sessions
python FragmentIntent.py # this automatically looks up configFile.txt and creates the FVs

### Creating the feature vectors for MINC dataset ###

# 0) Step 0: Creating the fragment vectors as splits from the query log -- Use the Java code for this 

# In MincJavaConfig.txt, we have a flag MINC_KEEP_PRUNE_MODIFY_REPEATED_QUERIES, if this is set to KEEP OR MODIFY, the fragment vectors will either be created
# by allowing the raw file (mysqld_original.log) query sessions as it is or by modifying the sessions by specifically ignoring the repeated successions of the same query.
# However if we choose PRUNE, it will read sessions from mysqld.log and will write clean query sessions which exclude sessions with repeated queries into a file called
# TempOutput/mysqld.log_SPLIT_OUT_0 -- make sure to set threads to 1 in this case. Then copy that content to mysqld.log

# Once this is done, set the flag MINC_KEEP_PRUNE_MODIFY_REPEATED_QUERIES to PREPROCESS to indicate that mysqld.log already contains preprocessed sessions. Now set 
# threads back to 100 (or 48 depending on how many cores you have) and run the feature vector creation in the following way. This will create feature vectors in 100 
# output files from TempOutput/mysqld.log_SPLIT_OUT_0 to _99. You can set MINC_SEL_OP_CONST=true or false to include constant dimensions or not

nohup sh runAndShutDown.sh > /hdd2/vamsiCodeData/createCleanSess.out 2> /hdd2/vamsiCodeData/createCleanSess.err &
# (make sure you uncomment this line readFromRawSessionsFile(tempLogDir, rawSessFile, intentVectorFile, line, schParse, numThreads, startLineNum, pruneKeepModifyRepeatedQuer# ies, includeSelOpConst);)

# The above command will internally run the following commands:
mvn -DskipTests clean package
mvn exec:java -Dexec.mainClass="MINCFragmentIntent"

# 1) Step 1: Now stitch all these files together which contain sequential sessions into a concurrent query session and concurrent fragment vectors
# This requires you use the python code from IntentPredictionEval

# First copy all the sequential session files from TempOutput to BakOutput folder
cp ~/Documents/DataExploration-Research/MINC/InputOutput/TempOutput/* ~/Documents/DataExploration-Research/MINC/InputOutput/BakOutput/.

# Next run the following command from IntentPredictionEval directory for source code
python MINC_FragmentIntent.py -config configDir/MINC_FragmentQueries_Keep_configFile.txt
# In the config file, set QUERY_LIMIT=0 if you want to create fragments for all the tuples, else to a limit of QUERY_LIMIT=114607 if u want those many tuples

# The output is typically like the following ##
Using TensorFlow backend.
Query count so far: 10000, len(sessionQueryDict): 3417
Query count so far: 20000, len(sessionQueryDict): 7133
Query count so far: 30000, len(sessionQueryDict): 11074
Query count so far: 40000, len(sessionQueryDict): 14890
Query count so far: 50000, len(sessionQueryDict): 18869
Query count so far: 60000, len(sessionQueryDict): 22816
Query count so far: 70000, len(sessionQueryDict): 26799
Query count so far: 80000, len(sessionQueryDict): 30873
Query count so far: 90000, len(sessionQueryDict): 34837
Query count so far: 100000, len(sessionQueryDict): 38582
Query count so far: 110000, len(sessionQueryDict): 42187
appended Session 26289, Query 1, absQueryCount: 10000
appended Session 13714, Query 1, absQueryCount: 20000
appended Session 36649, Query 1, absQueryCount: 30000
appended Session 17874, Query 1, absQueryCount: 40000
appended Session 24404, Query 2, absQueryCount: 50000
appended Session 37465, Query 2, absQueryCount: 60000
appended Session 29851, Query 2, absQueryCount: 70000
appended Session 29611, Query 3, absQueryCount: 80000
appended Session 24310, Query 3, absQueryCount: 90000
appended Session 19728, Query 5, absQueryCount: 100000
appended Session 33893, Query 13, absQueryCount: 110000
Created intent vectors for # Sessions: 43892 and # Queries: 114607

## 2) Step 2: You can run two kinds of experiments with this -- singularity or sustenance (schemaDicts are assumed to have been created)
# Create a folder for this new dataset in InputOutput/ClusterRuns/
mkdir ~/Documents/DataExploration-Research/MINC/InputOutput/ClusterRuns/NovelTables-114607-Constants

# 3) Step 3: Copy feature vectors (full and for tabels alone) MincBitFragmentIntentSessions and MincBitFragmentTableIntentSessions just created to the new folder
# The feature vectors created here will be used for Singularity Experiments
cp ~/Documents/DataExploration-Research/MINC/InputOutput/MincBitFragment* ClusterRuns/NovelTables-114607-Constants/.
mv ~/Documents/DataExploration-Research/MINC/InputOutput/ClusterRuns/NovelTables-114607-Constants/MincBitFragmentIntentSessions{,Singularity}
mv ~/Documents/DataExploration-Research/MINC/InputOutput/ClusterRuns/NovelTables-114607-Constants/MincBitFragmentTableIntentSessions{,Singularity}
cp ~/Documents/DataExploration-Research/MINC/InputOutput/MincQuerySessions ~/Documents/DataExploration-Research/MINC/InputOutput/ClusterRuns/NovelTables-114607-Constants/.
cp ~/Documents/DataExploration-Research/MINC/InputOutput/MincConcurrentSessions ~/Documents/DataExploration-Research/MINC/InputOutput/ClusterRuns/NovelTables-114607-Constants/.

# modified version of above
cp ../../../research/connections/people/shiyuc/BusTracker/InputOutput/MincBitFragment* ClusterRuns/NovelTables-114607-Constants/.
mv ../../../research/connections/people/shiyuc/BusTracker/InputOutput/ClusterRuns/NovelTables-114607-Constants/MincBitFragmentIntentSessions{,Singularity}
mv ../../../research/connections/people/shiyuc/BusTracker/InputOutput/ClusterRuns/NovelTables-114607-Constants/MincBitFragmentTableIntentSessions{,Singularity}
cp ../../../research/connections/people/shiyuc/BusTracker/InputOutput/MincQuerySessions ../../../research/connections/people/shiyuc/BusTracker/InputOutput/ClusterRuns/NovelTables-114607-Constants/.
cp ../../../research/connections/people/shiyuc/BusTracker/InputOutput/MincConcurrentSessions ../../../research/connections/people/shiyuc/BusTracker/InputOutput/ClusterRuns/NovelTables-114607-Constants/.

# 4) Step 4: Create the K-Fold train and test sets with 80% train and 20% test (Required for Sustenance Experiemnts) 
python createTrainTestSessions.py -config configDir/MINC_FragmentQueries_Keep_configFile.txt
# SEQ_OR_CONC_TRAIN_TEST=CONC in the configFile because we want to have concurrent sessions both in train and test individually, KEEP because we have pruned during 
# FV creation phase in the Java code. Now we do not need to prune any recurrent repetition of queries as they no longer exist

# 5) Step 5: Concatenate any single fold of train and test folds created under InputOutput/kFold directory into a single file and copy it to the ClusterRuns/NovelTables-1143# 43-Constants/ 
cat ~/Documents/DataExploration-Research/MINC/InputOutput/kFold/MincBitFragmentIntentSessions_CONC_TRAIN_FOLD_2 ~/Documents/DataExploration-Research/MINC/InputOutput/kFold/MincBitFragmentIntentSessions_CONC_TEST_FOLD_2 > ~/Documents/DataExploration-Research/MINC/InputOutput/ClusterRuns/NovelTables-114607-Constants/MincBitFragmentIntentSessionsConcTrainTestSustenance_0.8
# Above chooses FOLD 2 to concatenate train and test sets from.

# 6) Step 6: Above chooses FOLD 2 to concatenate train and test sets from. Now do the following to run either SINGULARITY or SUSTENANCE experiments
-----------
SINGULARITY
----------- 
cp ~/Documents/DataExploration-Research/MINC/InputOutput/ClusterRuns/NovelTables-114607-Constants/MincBitFragmentIntentSessionsSingularity ~/Documents/DataExploration-Research/MINC/InputOutput/MincBitFragmentIntentSessions
----------
SUSTENANCE
----------
cp ~/Documents/DataExploration-Research/MINC/InputOutput/ClusterRuns/NovelTables-114607-Constants/MincBitFragmentIntentSessionsConcTrainTestSustenance_0.8 ~/Documents/DataExploration-Research/MINC/InputOutput/MincBitFragmentIntentSessions
