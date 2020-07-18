#!/bin/sh
python analyzeLogs_runQueries.py -config configDir/MINC_CF_COSINESIM_trainTest_sustenance_0.8_configFile.txt -log ../Documents/DataExploration-Research/MINC/InputOutput/ClusterRuns/NovelTables-114607-Constants/sustenance/CFCosineSim/Sustenance_CFCosineSim_outputSQLLog
# nohup sh scripts/runCFCosineSimSustenance_0.8_runQueries.sh > ../runCFCosineSimSustenance_0.8_runQueries.out 2> ../runCFCosineSimSustenance_0.8_runQueries.err &
