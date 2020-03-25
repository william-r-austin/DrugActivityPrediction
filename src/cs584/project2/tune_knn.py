'''
Created on Mar 14, 2020

@author: William
'''
import cs584.project2.utilities as utilities
import cs584.project2.constants as constants
import cs584.project2.data_balancing as data_balancing
import cs584.project2.feature_reduction as feature_reduction
from cs584.project2.naive_bayes_classifier import NaiveBayesClassifier
from sklearn.utils.estimator_checks import check_estimator
from imblearn.under_sampling import RandomUnderSampler
import sys

from statistics import mean

from scipy.sparse import vstack as sparse_vstack

from cs584.project2.information_gain_reducer import InformationGainReducer

import numpy as np

from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import cross_val_score
from sklearn.utils import estimator_checks
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix
from audioop import avg
from nltk.tbl import feature
import cs584.project2.common as common
from collections import Counter

from sklearn.model_selection import KFold
from sklearn.metrics.classification import confusion_matrix, f1_score
from sklearn.feature_selection.mutual_info_ import mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def tuneMultimodelKnnIgr(featureSizes, kValues):
    X_raw, y_raw = common.loadTrainingDataSet()
    
    scoreMap = dict()
    for featureSize in featureSizes:
        for kValue in kValues:
            scoreMap[(featureSize, kValue)] = []
        
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    foldNumber = 0

    for train_index, test_index in kf.split(X_raw):
        X_train, X_test = X_raw[train_index], X_raw[test_index]
        y_train, y_test = y_raw[train_index], y_raw[test_index]
        
        reducer = InformationGainReducer()
        reducer.fit(X_train, y_train)
    
        for featureSize in featureSizes:
            reducer.resize(featureSize)
            X_train_reduced = reducer.transform(X_train).toarray()
            X_test_reduced = reducer.transform(X_test).toarray()
            
            for kValue in kValues:
                modelList = []
            
                for modelNum in range(11):
                    rus_rs = 555 + (modelNum * featureSize)
                    rus = RandomUnderSampler(random_state=rus_rs)
                    X_model, y_model = rus.fit_resample(X_train_reduced, y_train)
                    
                    clf = KNeighborsClassifier(n_neighbors=kValue, metric='manhattan')
                    clf.fit(X_model, y_model)
                    
                    modelList.append(clf)
                    print(".", end="")
    
                output = common.predictCombinedSimple(X_test_reduced, modelList)
                combinedModelScore = f1_score(y_test, output)
                scoreMap[(featureSize, kValue)].append(combinedModelScore)
                
                print()
                print("Done with kValue = " + str(kValue) + " for fold #" + str(foldNumber) + " for feature size = " + str(featureSize) + ". F1 = " + str(combinedModelScore))

            print("Done with fold #" + str(foldNumber) + " for feature size = " + str(featureSize))
        
        foldNumber += 1
        
    for featureSize in featureSizes:
        for kValue in kValues:
            meanF1Score = mean(scoreMap[(featureSize, kValue)])
            print("F1 Score for KNN with IGR, K = " + str(kValue) + " and FR size = " + str(featureSize) + " is: " + str(meanF1Score))    
    
if __name__ == '__main__':
    sizes = [11, 15, 31, 47, 63, 95, 127, 191, 255, 383, 511, 767, 1023, 1535, 2047, 3071, 4095]
    tuneMultimodelKnnIgr([11, 31, 63, 127, 255, 511, 1023], [3, 7, 15, 31, 63])
        