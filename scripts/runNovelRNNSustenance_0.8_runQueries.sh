#!/bin/sh 
python analyzeLogs_runQueries.py -config configDir/MINC_Novel_RNN_sustenance_configFile.txt -log ../Documents/DataExploration-Research/MINC/InputOutput/ClusterRuns/NovelTables-114607-Constants/sustenance/NovelRNN/Sustenance_NovelRNN_outputSQLLog
# nohup sh scripts/runNovelRNNSustenance_0.8_runQueries.sh > ../runNovelRNNSustenance_0.8_runQueries.out 2> ../runNovelRNNSustenance_0.8_runQueries.err &
