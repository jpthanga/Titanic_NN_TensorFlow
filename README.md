###Titanic using Neural Nets and Tensorflow

This program deals with the titanic dataset available on kaggle. The goal of the program is to predict which passengers survived based on a number of criteria provided.

The program is divided into 2 files:
* preProcess.py - Pre-Processes the data converting categorical variables into desired forms and also dealing with missing data. Finally writes into .npy files to be used for training and testing.
* classify.py - trains a 3 layer neural net built using TensorFlow. and writes the predictions into csv file.

The classifier achieved a accuracy of 72%. 

Skills Used: Python with numPy, pandas and TensorFlow.
#####References  
* TensorFlow documentation
* https://beckernick.github.io/neural-network-scratch/
