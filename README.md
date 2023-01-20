# About this repo:

This codebase was written in Python 2.7 and requires the SQL fragment bit-vectors to be pre-created and fed as input to each of the algorithms here.

* QLearning.py: This code uses tabular version of Q-Learning to predict the SQL fragments
* QLearning_selOpConst.py: This code uses Q-Learning to predict SQL fragment vectors with constant bins
* LSTM_RNN_Parallel.py and LSTM_RNN_Parallel_selOpConst.py contain LSTM and RNN-based implementation
* CFCosineSim_Parallel.py and CF_SVD_selOpConst.py contain the cosine similarity-based and SVD-based implementation
* configDir contains the config files and scripts folder contains the shell scripts to run the query predictors

CourseWebsite (MINC) dataset is proprietary and cannot be released. The BusTracker dataset is from an earlier work http://www.cs.cmu.edu/~malin199/data/tiramisu-sample/ 

The version of the BusTracker dataset we used is available at
* https://www.dropbox.com/s/twvsb5f8zj7vqol/BusTracker.zip?dl=0
