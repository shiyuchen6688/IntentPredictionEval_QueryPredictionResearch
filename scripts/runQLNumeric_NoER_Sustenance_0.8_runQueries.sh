#!/bin/sh
python analyzeLogs_runQueries.py -config configDir/MINC_QL_trainTest_sustenance_0.8_configFile.txt -log ../Documents/DataExploration-Research/MINC/InputOutput/ClusterRuns/NovelTables-114607-Constants/sustenance/QL-NUMERIC-NoER/Sustenance_QL_outputSQLLog
# nohup sh scripts/runQLNumeric_NoER_Sustenance_0.8_runQueries.sh > ../runQLNumeric_NoER_Sustenance_0.8_runQueries.out 2> ../runQLNumeric_NoER_Sustenance_0.8_runQueries.err &
