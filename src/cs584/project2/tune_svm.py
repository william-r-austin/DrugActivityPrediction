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

from sklearn.svm import SVC

def tuneMultimodelSvm(featureSizes):
    X_raw, y_raw = common.loadTrainingDataSet()
    
    scoreMap = dict()
    for featureSize in featureSizes:
        scoreMap[featureSize] = []
        
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    foldNumber = 0

    for train_index, test_index in kf.split(X_raw):
        X_train, X_test = X_raw[train_index], X_raw[test_index]
        y_train, y_test = y_raw[train_index], y_raw[test_index]
        
        for featureSize in featureSizes:
            reducer = TruncatedSVD(n_components=featureSize)
            X_train_reduced = reducer.fit_transform(X_train)
            
            modelList = []
        
            for modelNum in range(11):
                rus_rs = 555 + (modelNum * featureSize)
                rus = RandomUnderSampler(random_state=rus_rs)
                X_model, y_model = rus.fit_resample(X_train_reduced, y_train)
                
                clf = SVC(gamma='scale')
                clf.fit(X_model, y_model)
                
                modelList.append(clf)
                print(".", end="")

            X_test_reduced = reducer.transform(X_test)
            output = common.predictCombinedSimple(X_test_reduced, modelList)
            combinedModelScore = f1_score(y_test, output)
            scoreMap[featureSize].append(combinedModelScore)
            
            print()
            print("Done with fold #" + str(foldNumber) + " for feature size = " + str(featureSize) + ". F1 = " + str(combinedModelScore))
        
        foldNumber += 1
        
    for featureSize in featureSizes:
        meanF1Score = mean(scoreMap[featureSize])
        print("F1 Score for SVM with Truncated SVD and FR size = " + str(featureSize) + " is: " + str(meanF1Score))    
    
if __name__ == '__main__':
    sizes = [11, 15, 31, 47, 63, 95, 127, 191]
    tuneMultimodelSvm(sizes)
         
        
        
        
        