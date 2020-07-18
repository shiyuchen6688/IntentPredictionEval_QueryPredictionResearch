#!/bin/sh
python analyzeLogs_runQueries.py -config configDir/MINC_SVD_trainTest_sustenance_0.8_configFile.txt -log ../Documents/DataExploration-Research/MINC/InputOutput/ClusterRuns/NovelTables-114607-Constants/sustenance/SVD/Sustenance_SVD_outputSQLLog
# nohup sh scripts/runSVDSustenance_0.8_runQueries.sh > ../runSVDSustenance_0.8_runQueries.out 2> ../runSVDSustenance_0.8_runQueries.err &
