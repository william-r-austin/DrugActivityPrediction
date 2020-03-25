'''
Created on Mar 14, 2020

@author: William
'''
from sympy.physics.units.dimensions import information
from cs584.project2.information_gain_reducer import InformationGainReducer
from cs584.project2.data_balancing import FeatureIndependentOversampler

import cs584.project2.common as common
import cs584.project2.data_balancing as data_balancing
import cs584.project2.constants as constants
from sklearn.model_selection import cross_val_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import SparsePCA
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from statistics import mean
from sklearn.neural_network import MLPClassifier
import math
import numpy as np


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


def tuneFeatureSizeAndAlphaIgrBalanced(featureSizes, alphaValues):
    X_model_full_imbalanced, y_model_imbalanced = common.loadTrainingDataSet()
    
    balancer = FeatureIndependentOversampler(random_state=42)
    X_model_full_raw, y_model_raw = balancer.fit_transform(X_model_full_imbalanced, y_model_imbalanced)
    
    X_model_full, y_model = shuffle(X_model_full_raw, y_model_raw, random_state=42) 
    
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

def testDataBalancing():
    x_test = np.random.choice(2, 25, p=[0.1, 0.9])
    print(x_test)
    X, y = common.loadTrainingDataSet()
    balancer = FeatureIndependentOversampler(random_state=42)
    X_new, y_new = balancer.fit_transform(X, y)
    print("Done!")

def tuneMultiModelChi2(featureSizes):
    X_raw, y_raw = common.loadTrainingDataSet()
    
    for featureSize in featureSizes:
        kf_rs = featureSize + 42
        kf = KFold(n_splits=5, random_state=kf_rs, shuffle=True)
        f1ScoreList = []
        
        for train_index, test_index in kf.split(X_raw):
            X_train, X_test = X_raw[train_index], X_raw[test_index]
            y_train, y_test = y_raw[train_index], y_raw[test_index]
        
            modelTransformerList = [] 
        
            for modelNum in range(13):
                rus_rs = 555 + modelNum
                rus = RandomUnderSampler(random_state=rus_rs)
                X_balanced, y_model = rus.fit_resample(X_train, y_train) 
                
                reducer = SelectKBest(chi2, k=featureSize)
                X_model = reducer.fit_transform(X_balanced, y_model).toarray()
            
                hiddenLayerSizes = (int(math.sqrt(featureSize)) + 1,)
                mc = MLPClassifier(solver='lbfgs', alpha=0.00005, hidden_layer_sizes=hiddenLayerSizes)
                mc.fit(X_model, y_model)
                
                modelTransformerList.append((mc, reducer)) 
            
            combinedModelOutput = common.predictCombined(X_test, modelTransformerList)
            combinedModelScore = f1_score(y_test, combinedModelOutput)
            f1ScoreList.append(combinedModelScore)
        
        meanF1Score = mean(f1ScoreList)
        print("F1 Score for NN with Chi2 and FR size = " + str(featureSize) + " is: " + str(meanF1Score))


def tuneNeuralNetworkSmote(featureSizes):
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
            reducer = SelectKBest(chi2, k=featureSize)
            reducer.fit(X_train, y_train)
            X_train_reduced = reducer.transform(X_train).toarray()
            
            ss_rs = 42+(featureSize*foldNumber)
            smoteSampler = SMOTE(random_state=ss_rs)

            X_model, y_model = smoteSampler.fit_resample(X_train_reduced, y_train)
            
            hiddenLayerSizes = (int(math.sqrt(featureSize)) + 1,)
            clf = MLPClassifier(solver='lbfgs', alpha=0.00005, hidden_layer_sizes=hiddenLayerSizes)    
            clf.fit(X_model, y_model)

            X_test_reduced = reducer.transform(X_test).toarray()
            output = clf.predict(X_test_reduced)
            combinedModelScore = f1_score(y_test, output)
            scoreMap[featureSize].append(combinedModelScore)
            
            print()
            print("Done with RF prediction for fold #" + str(foldNumber) + " for feature size = " + str(featureSize) + ". F1 = " + str(combinedModelScore))
        
        foldNumber += 1
        
    for featureSize in featureSizes:
        meanF1Score = mean(scoreMap[featureSize])
        print("F1 Score for RF with Chi2 and FR size = " + str(featureSize) + " is: " + str(meanF1Score))  


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
    #tuneFeatureSizeAndAlphaIGR([2047, 6143], alphas)
    
    # Seems to work
    # testDataBalancing()
    
    # Up to 99% + @ 8191 features. Prolly overfitting. .59 on miner
    # tuneFeatureSizeAndAlphaIgrBalanced(sizes, [0.00005])
    
    #tuneMultiModelChi2(sizes)
    tuneNeuralNetworkSmote(sizes)
