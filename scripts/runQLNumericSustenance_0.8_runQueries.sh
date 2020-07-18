#!/bin/sh
python analyzeLogs_runQueries.py -config configDir/MINC_QL_trainTest_sustenance_0.8_configFile.txt -log ../Documents/DataExploration-Research/MINC/InputOutput/ClusterRuns/NovelTables-114607-Constants/sustenance/QL-NUMERIC/Sustenance_QL_outputSQLLog
# nohup sh scripts/runQLNumericSustenance_0.8_runQueries.sh > ../runQLNumericSustenance_0.8_runQueries.out 2> ../runQLNumericSustenance_0.8_runQueries.err &
