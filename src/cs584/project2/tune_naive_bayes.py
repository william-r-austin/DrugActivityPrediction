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


def getAvgF1Score(X, y):
    splitCount = 5
    
    kf = KFold(n_splits=splitCount, random_state=42, shuffle=True)
    avgSum = 0.0
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        nbClassifier = NaiveBayesClassifier()
        nbClassifier.fit(X_train, y_train)
        
        output = nbClassifier.predict(X_test)
        avgSum += f1_score(y_test, output)
        
    return avgSum / splitCount

def tuneFeatureCountWithChiSquared(X, y):
    for j in common.getFeatureCountArray():
        reducer = feature_reduction.getChiSquared(X, y, j)
        #featureReducer = SelectKBest(chi2, k=j)
        #featureReducer.fit(X, y)
    
        X_new = feature_reduction.transform(reducer, X)
        
        f1 = getAvgF1Score(X_new, y)
        print("J = " + str(j) + ",     F1 =     " + str(f1))

def tuneFeatureCountWithTruncatedSVD(X, y):
    for j in common.getFeatureCountArray():
        reducer = feature_reduction.getTruncatedSVD(X, y, j)
        #featureReducer = SelectKBest(chi2, k=j)
        #featureReducer.fit(X, y)
    
        X_new = feature_reduction.transform(reducer, X)
        
        f1 = getAvgF1Score(X_new, y)
        print("J = " + str(j) + ",     F1 =     " + str(f1))        

def getRangeList(n, k):
    ranges = []
    for j in range(0, k):
        lower = (j * n) // k
        upper = ((j + 1) * n) // k
        
        ranges.append(list(range(lower, upper)))
    return ranges

def testMultiModel(X, y, numModels):
    activeIndexTuple = y.nonzero()
    activeIndexValues = activeIndexTuple[0]
    activeTotalCount = activeIndexValues.shape[0]
    
    X_active = X[activeIndexValues, :]
    
    fs = frozenset(activeIndexValues)
            
    allIndices = [k for k in range(len(y))]
    nonActiveIndices =  list(filter(lambda q: q not in fs, allIndices))
    nonActiveIndexValues = np.array(nonActiveIndices, dtype=np.int64)
    X_nonActive = X[nonActiveIndexValues, :]
        
    modelRangeList = getRangeList(len(nonActiveIndices), numModels)
    
    returnList = []
    
    for modelIndex in range(numModels):
        currentZerosList = modelRangeList[modelIndex]
        currentZerosArray = np.array(currentZerosList, dtype=np.int64)
        X_nonActiveCurrent = X_nonActive[currentZerosArray, :]
        
        #X_model = np.append(X_active, X_nonActiveCurrent)
        
        X_model = sparse_vstack([X_active, X_nonActiveCurrent]).tolil()
        
        y_model = [1] * X_active.shape[0]
        y_model.extend([0] * X_nonActiveCurrent.shape[0])
        
        print("Sub model X = " + str(X_model.shape))
        print("Sub model y = " + str(len(y_model)))
        
        print("Constructing model #" + str(modelIndex))
        
        returnList.append((X_model, y_model))
    
    return returnList
    
def testCustomUnderfitting():
    #####################
    # Part 1. Balance the dataset
    #####################
    X, y = common.loadTrainingDataSet()
    #xBalanced, yBalanced = data_balancing.balanceDatasetWithRandomOversampling(xRawData, yRawData)
    
    #tuneFeatureCountWithChiSquared(xBalanced, yBalanced)
    #tuneFeatureCountWithTruncatedSVD(X, y)
    
    datasets = testMultiModel(X, y, 9)
    
    for modelNum in range(9):
        dataset = datasets[modelNum]
        X_sub = dataset[0]
        y_sub = dataset[1]
        
        reducer = feature_reduction.getChiSquared(X_sub, y_sub, 300)
        
    
        X_sub_new = feature_reduction.transform(reducer, X_sub)
        y_sub_new = np.array(y_sub, dtype=np.int64)
        
        modelScore = getAvgF1Score(X_sub_new, y_sub_new)
        print("Model score for model " + str(modelNum) + " = " + str(modelScore))

def tuneNaiveBayesMultiModelWithSize():
    sizes = [55, 105, 155, 205, 255, 305, 375, 505, 655, 805, 1005, 1305]
    
    for size in sizes:
        tuneNaiveBayesMultiModel(size, 9)

def tuneNaiveBayesMultiModel(featureSize, modelCount):
    X, y = common.loadTrainingDataSet()
    
    #print("Counter(y) = " + str(Counter(y)))
    
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    splitIndex = 0
    f1ScoreList = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        modelTransformerList = []
        
        for modelNum in range(modelCount):
            rs = 42 + modelNum
            rus = RandomUnderSampler(random_state=rs)
            X_model_full, y_model = rus.fit_resample(X_train, y_train) 
              
            reducer = SelectKBest(chi2, k=featureSize)
            X_model = reducer.fit_transform(X_model_full, y_model).toarray()
            
            nbClassifier = NaiveBayesClassifier()
            nbClassifier.fit(X_model, y_model)
    
            #X_test_2 = reducer.transform(X_test).toarray()
            #output = nbClassifier.predict(X_test_2)
            #modelScore = f1_score(y_test, output)
            
            #print("Split Index = " + str(splitIndex) + ", Model Num = " + str(modelNum) + ", F1 = " + str(modelScore))
            
            modelTransformerList.append((nbClassifier, reducer)) 
        
        combinedModelOutput = common.predictCombined(X_test, modelTransformerList)
        combinedModelScore = f1_score(y_test, combinedModelOutput)
        f1ScoreList.append(combinedModelScore)
        #print("Combined Model Score = " + str(combinedModelScore))
       
        splitIndex += 1
    
    print("F1 Score for FR size = " + str(featureSize) + " is: " + str(mean(f1ScoreList)))


def tuneNaiveBayesFeatureReduction():
    X, y = common.loadTrainingDataSet()        
    rus = RandomUnderSampler(random_state=42)
    
    X_res, y_res = rus.fit_resample(X, y)
    
    print("Counter(y_res) = " + str(Counter(y_res)))
    
    for j in common.getFeatureCountArray():
        reducer = feature_reduction.getChiSquared(X_res, y_res, j)
        #featureReducer = SelectKBest(chi2, k=j)
        #featureReducer.fit(X, y)
    
        X_new = feature_reduction.transform(reducer, X_res)
        
        f1 = getAvgF1Score(X_new, y_res)
        print("J = " + str(j) + ",     F1 =     " + str(f1))
    
    
    '''
    
    splitCount = 5
    
    
    
    kf = KFold(n_splits=splitCount, random_state=42, shuffle=True)
    avgSum = 0.0
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        
        
        
        
        nbClassifier = NaiveBayesClassifier()
        nbClassifier.fit(X_train, y_train)
        
        output = nbClassifier.predict(X_test)
        avgSum += f1_score(y_test, output)
    
    modelList = []
    for modelNum in range(11):
        
        
    result = common.predictCombined(X, modelList)           


    '''

def tuneNaiveBayesUpdated(featureSize, modelCount):
    X_raw, y = common.loadTrainingDataSet()
    
    
    reducer = SelectKBest(chi2, k=featureSize)
    X = reducer.fit_transform(X_raw, y).toarray()
    
    #print("Counter(y) = " + str(Counter(y)))
    
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    splitIndex = 0
    f1ScoreList = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        modelList = []
        
        for modelNum in range(modelCount):
            rs = 42 + modelNum
            rus = RandomUnderSampler(random_state=rs)
            X_model, y_model = rus.fit_resample(X_train, y_train) 
              

            
            nbClassifier = NaiveBayesClassifier()
            nbClassifier.fit(X_model, y_model)
    
            #X_test_2 = reducer.transform(X_test).toarray()
            #output = nbClassifier.predict(X_test_2)
            #modelScore = f1_score(y_test, output)
            
            #print("Split Index = " + str(splitIndex) + ", Model Num = " + str(modelNum) + ", F1 = " + str(modelScore))
            
            modelList.append(nbClassifier)
            print(".", end='')
        print()
        
        combinedModelOutput = common.predictCombinedSimple(X_test, modelList)
        combinedModelScore = f1_score(y_test, combinedModelOutput)
        f1ScoreList.append(combinedModelScore)
        print("Combined Model Score for split #" + str(splitIndex) + " = " + str(combinedModelScore))
       
        splitIndex += 1
    
    print("F1 Score for FR size = " + str(featureSize) + " is: " + str(mean(f1ScoreList)))

def tuneNaiveBayesUpdatedParams():
    sizes = [55, 105, 155, 205, 255, 305, 375, 505, 655, 805, 1005, 1305]
    
    for size in sizes:
        tuneNaiveBayesUpdated(size, 13)

def tuneNaiveBayesIgrFeatureSize(featureSizeList, modelCount):
    X_raw, y = common.loadTrainingDataSet()
    
    reducer = InformationGainReducer()
    reducer.fit(X_raw, y)
    
    for featureSize in featureSizeList:
        reducer.resize(featureSize)
        X = reducer.transform(X_raw).toarray()
        
        #print("Counter(y) = " + str(Counter(y)))
        
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        splitIndex = 0
        f1ScoreList = []
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            modelList = []
            
            for modelNum in range(modelCount):
                rs = 42 + modelNum
                rus = RandomUnderSampler(random_state=rs)
                X_model, y_model = rus.fit_resample(X_train, y_train) 
                  
    
                
                nbClassifier = NaiveBayesClassifier()
                nbClassifier.fit(X_model, y_model)
        
                #X_test_2 = reducer.transform(X_test).toarray()
                #output = nbClassifier.predict(X_test_2)
                #modelScore = f1_score(y_test, output)
                
                #print("Split Index = " + str(splitIndex) + ", Model Num = " + str(modelNum) + ", F1 = " + str(modelScore))
                
                modelList.append(nbClassifier)
                print(".", end='')
            print()
            
            combinedModelOutput = common.predictCombinedSimple(X_test, modelList)
            combinedModelScore = f1_score(y_test, combinedModelOutput)
            f1ScoreList.append(combinedModelScore)
            print("Combined Model Score for split #" + str(splitIndex) + " = " + str(combinedModelScore))
           
            splitIndex += 1
        
        print("F1 Score for FR size = " + str(featureSize) + " is: " + str(mean(f1ScoreList)))


def tuneIGR():
    sizes = [55, 105, 155, 205, 255, 305, 375, 505, 655, 805, 1005, 1305]
    
    tuneNaiveBayesIgrFeatureSize(sizes, 13)
    #for size in sizes:
    #    tuneNaiveBayesUpdated(size, 13)
    
if __name__ == '__main__':
    # Best submission so far
    #tuneNaiveBayesMultiModelWithSize()
    
    # Mediocre results
    #tuneNaiveBayesUpdatedParams() 
    
    
    #X, y = common.loadTrainingDataSet()
    #reducer = InformationGainReducer()
    #reducer.fit(X, y)
    
    tuneIGR()
    
    
         
        
        
        
        