#!/bin/sh 
python analyzeLogs_runQueries.py -config configDir/MINC_Novel_GRU_trainTest_sustenance_0.8_configFile.txt -log ../Documents/DataExploration-Research/MINC/InputOutput/ClusterRuns/NovelTables-114607-Constants/sustenance/NovelGRU/Sustenance_NovelGRU_outputSQLLog
# nohup sh scripts/runNovelGRUSustenance_0.8_runQueries.sh > ../runNovelGRUSustenance_0.8_runQueries.out 2> ../runNovelGRUSustenance_0.8_runQueries.err &
