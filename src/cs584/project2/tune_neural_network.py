'''
Created on Mar 23, 2020

@author: William
'''
from sympy.physics.units.dimensions import information
from cs584.project2.information_gain_reducer import InformationGainReducer

'''
Created on Mar 14, 2020

@author: William
'''

import cs584.project2.common as common
import cs584.project2.data_balancing as data_balancing
import cs584.project2.constants as constants
from sklearn.model_selection import cross_val_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import SparsePCA
from sklearn.metrics import f1_score

from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from statistics import mean
from sklearn.neural_network import MLPClassifier
import math


def tuneNeuralNetworkFeatureSizeChi2(featureSizes):
    # Some setup
    X_model_full, y_model = common.loadTrainingDataSet()
    
    #xBalanced, yBalanced = data_balancing.balanceDatasetWithRandomOversampling(xRawData, yRawData)
    #myCounter = Counter(yBalanced)
    #print("Finished loading and sampling. Data dist = " + str(myCounter))
    for featureSize in featureSizes:
        reducer = SelectKBest(chi2, k=featureSize)
        X_model = reducer.fit_transform(X_model_full, y_model).toarray()
                
        hiddenLayerSizes = (int(math.sqrt(featureSize)) + 1,)
        mc = MLPClassifier(solver='lbfgs', alpha=0.00001, hidden_layer_sizes=hiddenLayerSizes)
        cvFolds = 5
        cvScores = cross_val_score(estimator=mc, X=X_model, y=y_model, scoring='f1', cv=cvFolds)
        #print("Individual CV scores = " + str(cvScores))
        avg = sum(cvScores) / cvFolds
        print("Cross validation score for MLP Classifier with feature size = " + str(featureSize) + " is: "+ str(avg))
        
def tuneNeuralNetworkFeatureSizeIgr(featureSizes):
    # Some setup
    X_model_full, y_model = common.loadTrainingDataSet()
    
    reducer = InformationGainReducer()
    reducer.fit(X_model_full, y_model)
    #xBalanced, yBalanced = data_balancing.balanceDatasetWithRandomOversampling(xRawData, yRawData)
    #myCounter = Counter(yBalanced)
    #print("Finished loading and sampling. Data dist = " + str(myCounter))
    for featureSize in featureSizes:
        reducer.resize(featureSize)
        X_model = reducer.transform(X_model_full).todense()
                
        hiddenLayerSizes = (int(math.sqrt(featureSize)) + 1,)
        mc = MLPClassifier(solver='lbfgs', alpha=0.00001, hidden_layer_sizes=hiddenLayerSizes)
        cvFolds = 5
        cvScores = cross_val_score(estimator=mc, X=X_model, y=y_model, scoring='f1', cv=cvFolds)
        #print("Individual CV scores = " + str(cvScores))
        avg = sum(cvScores) / cvFolds
        print("Cross validation score for MLP Classifier with feature size = " + str(featureSize) + " is: " + str(avg))
        
def tuneNeuralNetworkFeatureSizeTruncatedSVD(featureSizes):
    X_model_full, y_model = common.loadTrainingDataSet()

    for featureSize in featureSizes:
        reducer = TruncatedSVD(n_components=featureSize, random_state=42)
        X_model = reducer.fit_transform(X_model_full)
                
        hiddenLayerSizes = (int(math.sqrt(featureSize)) + 1,)
        mc = MLPClassifier(solver='lbfgs', alpha=0.00001, hidden_layer_sizes=hiddenLayerSizes)
        cvFolds = 5
        cvScores = cross_val_score(estimator=mc, X=X_model, y=y_model, scoring='f1', cv=cvFolds)
        avg = sum(cvScores) / cvFolds
        print("Cross validation score for MLP Classifier with feature size = " + str(featureSize) + " is: "+ str(avg))           

'''
def tuneNeuralNetworkFeatureSizeSparsePCA(featureSizes):
    X_model_full, y_model = common.loadTrainingDataSet()

    for featureSize in featureSizes:
        reducer = SparsePCA(n_components=featureSize, random_state=42)
        X_model = reducer.fit_transform(X_model_full).todense()
                
        hiddenLayerSizes = (int(math.sqrt(featureSize)) + 1,)
        mc = MLPClassifier(solver='lbfgs', alpha=0.00001, hidden_layer_sizes=hiddenLayerSizes)
        cvFolds = 5
        cvScores = cross_val_score(estimator=mc, X=X_model, y=y_model, scoring='f1', cv=cvFolds)
        avg = sum(cvScores) / cvFolds
        print("Cross validation score for MLP Classifier with feature size = " + str(featureSize) + " is: "+ str(avg))  
'''
        
def tuneFeatureSizeAndAlphaIGR(featureSizes, alphaValues):
    X_model_full, y_model = common.loadTrainingDataSet()
    
    reducer = InformationGainReducer()
    reducer.fit(X_model_full, y_model)
    #xBalanced, yBalanced = data_balancing.balanceDatasetWithRandomOversampling(xRawData, yRawData)
    #myCounter = Counter(yBalanced)
    #print("Finished loading and sampling. Data dist = " + str(myCounter))
    for featureSize in featureSizes:
        reducer.resize(featureSize)
        X_model = reducer.transform(X_model_full).todense()
        
        for alphaValue in alphaValues:
            hiddenLayerSizes = (int(math.sqrt(featureSize)) + 1,)
            mc = MLPClassifier(solver='lbfgs', alpha=alphaValue, hidden_layer_sizes=hiddenLayerSizes)
            cvFolds = 5
            cvScores = cross_val_score(estimator=mc, X=X_model, y=y_model, scoring='f1', cv=cvFolds)
            #print("Individual CV scores = " + str(cvScores))
            avg = sum(cvScores) / cvFolds
            print("Cross validation score for MLP Classifier with alpha = " + str(alphaValue) + " and feature size = " + str(featureSize) + " is: " + str(avg))


def tuneBasicNeuralNetwork():
    # Some setup
    X_model_full, y_model = common.loadTrainingDataSet()
    
    #xBalanced, yBalanced = data_balancing.balanceDatasetWithRandomOversampling(xRawData, yRawData)
    #myCounter = Counter(yBalanced)
    #print("Finished loading and sampling. Data dist = " + str(myCounter))
    
    reducer = SelectKBest(chi2, k=128)
    X_model = reducer.fit_transform(X_model_full, y_model).toarray()
            

    
    mc = MLPClassifier(solver='lbfgs', alpha=0.00001, hidden_layer_sizes=(25,))
    #mc.fit(X_model)
    
    
    
    cvFolds = 5 #constants.crossValidationFoldCount
    cvScores = cross_val_score(estimator=mc, X=X_model, y=y_model, scoring='f1', cv=cvFolds)
    print("Individual CV scores = " + str(cvScores))
    avg = sum(cvScores) / cvFolds
    print("Cross validation score for decision tree = " + str(avg))

if __name__ == '__main__':
    sizes = [11, 15, 31, 47, 63, 95, 127, 191, 255, 383, 511, 767, 1023, 1535, 2047, 3071, 4095, 6143, 8191]
    alphas = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    # Best was about .716 @ size = 6143
    #tuneNeuralNetworkFeatureSizeChi2(sizes)
    
    # Best was 0.812 @ 6143
    #tuneNeuralNetworkFeatureSizeIgr(sizes)
    
    # Best was 0.686 @ 15. Slow
    # tuneNeuralNetworkFeatureSizeTruncatedSVD(sizes)
    
    # Max was 0.825 with features = 6143 and alpha = 0.001
    tuneFeatureSizeAndAlphaIGR([2047, 6143], alphas)
    
