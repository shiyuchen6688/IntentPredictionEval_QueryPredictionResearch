#!/bin/sh 
python analyzeLogs_runQueries.py -config configDir/MINC_Novel_LSTM_trainTest_sustenance_0.8_configFile.txt -log ../Documents/DataExploration-Research/MINC/InputOutput/ClusterRuns/NovelTables-114607-Constants/sustenance/NovelLSTM/Sustenance_NovelLSTM_outputSQLLog
# nohup sh scripts/runNovelLSTMSustenance_0.8_runQueries.sh > ../runNovelLSTMSustenance_0.8_runQueries.out 2> ../runNovelLSTMSustenance_0.8_runQueries.err &
