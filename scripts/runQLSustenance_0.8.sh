#!/bin/sh
python QLearning_selOpConst.py -config configDir/MINC_QL_trainTest_sustenance_0.8_configFile.txt
python CreateSQLLogs.py -config configDir/MINC_QL_trainTest_sustenance_0.8_configFile.txt
#python QLearning_selOpConst.py -config configDir/MINC_QL_configFile.txt
#python CF_SVD_selOpConst.py -config MINC_configFile.txt
#python CFCosineSim_Parallel.py -config configDir/MINC_CF_COSINESIM_sustenance_configFile.txt
#python ActiveLearning_Parallel.py -config MINC_configFile.txt
#python LSTM_RNN_Parallel_selOpConst.py -config configDir/MINC_RNN_sustenance_configFile.txt
#python ActiveLearning_Parallel.py -config configDir/MINC_configFile_1Fold_Random_Sample_1.0_Incremental_Weight_0.2_Top_3_Last_3.txt
#python ActiveLearning_Parallel.py -config configDir/MINC_configFile_1Fold_Random_Sample_1.0_Incremental_Weight_0.4_Top_3_Last_3.txt
#python ActiveLearning_Parallel.py -config configDir/MINC_configFile_1Fold_Random_Sample_1.0_Incremental_Weight_0.6_Top_3_Last_3.txt
#python ActiveLearning_Parallel.py -config configDir/MINC_configFile_1Fold_Random_Sample_1.0_Incremental_Weight_0.8_Top_3_Last_3.txt
#sudo shutdown
