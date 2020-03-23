'''
Created on Mar 7, 2020

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

import numpy as np

from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import cross_val_score
from sklearn.utils import estimator_checks
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from audioop import avg
from nltk.tbl import feature
import cs584.project2.common as common
from collections import Counter

def runWithOversampling():
    #####################
    # Part 1. Balance the dataset
    #####################
    xRawData, yRawData = common.loadTrainingDataSet()
    
    xBalanced, yBalanced = data_balancing.balanceDatasetWithRandomOversampling(xRawData, yRawData)
    
    #####################
    # Part 2. Feature Reduction
    #####################
    featureReducer = SelectKBest(chi2, k=10000)
    featureReducer.fit(xBalanced, yBalanced)
    
    xReduced = featureReducer.transform(xBalanced).todense()
    
    nbClassifier = NaiveBayesClassifier()
    nbClassifier.fit(xReduced, yBalanced)
    
    rawTestData = common.loadTestDataSet()
    reducedTestData = featureReducer.transform(rawTestData).todense()
    
    resultsArray = nbClassifier.predict(reducedTestData)
    
    common.writeResultsFile(resultsArray)


def runBuiltInBernoulli():
    trainingDataMatrix, labelMatrix = common.loadTrainingDataSet()

       
    predictiveFeatures = feature_reduction.computePredictiveness(trainingDataMatrix, labelMatrix)
    
    #print("Performed feature selection. New shape is: " + str(trainingMatrix1.shape))
    
    bernoulliClf = BernoulliNB(alpha=constants.smoothingConstant, binarize=None, fit_prior=False)

    '''
    for j in range(5, 1001, 5):
        importantFeatures = [element[0] for element in predictiveFeatures[0:j]]
        #print("Important features = " + str(importantFeatures))
        importantFeaturesArray = np.array(importantFeatures)
        reducedDataSet = trainingDataMatrix[:, importantFeaturesArray]
    
        #print("Reduced data set shape = " + str(reducedDataSet.shape))
        cvScores = cross_val_score(estimator=bernoulliClf, X=reducedDataSet, y=labelMatrix, scoring='f1', cv=constants.crossValidationFoldCount)
    
        avg = sum(cvScores) / constants.crossValidationFoldCount
        print("My reducer. Feature Count = " + str(j) + "   Avg Score = " + str(avg))
    '''
    
    
    
    importantFeaturesArray = [element[0] for element in predictiveFeatures[0:205]]
    reducedTraining = trainingDataMatrix[:, importantFeaturesArray]
    
    bernoulliClf.fit(reducedTraining, labelMatrix)
    
    
    testDataMatrix = common.loadTestDataSet()
    reducedTesting = testDataMatrix[:, importantFeaturesArray]
    testPredictions = bernoulliClf.predict(reducedTesting)
    
    print("Test predictions shape = " + str(testPredictions.shape))
    print("Test Estimates = " + str(testPredictions))
    common.writeResultsFile(testPredictions)

def runBernoulliWithChiSquared():
    trainingDataMatrix, labelMatrix = common.loadTrainingDataSet()

       
    #predictiveFeatures = feature_reduction.computePredictiveness(trainingDataMatrix, labelMatrix)
    
    #print("Performed feature selection. New shape is: " + str(trainingMatrix1.shape))
    
    bernoulliClf = BernoulliNB(alpha=constants.smoothingConstant, binarize=None, fit_prior=False)
    
    '''
    maxAvg = 0
    maxK = -1
    
    for kVal in range(1025, 10000, 50):
        trainingMatrix1 = SelectKBest(chi2, k=kVal).fit_transform(trainingDataMatrix, labelMatrix)
        cvScores = cross_val_score(estimator=bernoulliClf, X=trainingMatrix1, y=labelMatrix, scoring='f1', cv=7)
        avg = sum(cvScores) / 7
        if avg > maxAvg:
            maxAvg = avg
            maxK = kVal
        
        print("k = " + str(kVal) + ", avg = " + str(avg))
    
    print("Best value is k = " + str(maxK) + ", " + str(maxAvg))
    '''
    featureReducer = SelectKBest(chi2, k=985)
    featureReducer.fit(trainingDataMatrix, labelMatrix)
    
    trainingMatrix1 = featureReducer.transform(trainingDataMatrix)

        
    cvScores = cross_val_score(estimator=bernoulliClf, X=trainingMatrix1, y=labelMatrix, scoring='f1', cv=7)
    avg = sum(cvScores) / 7
    print("k = 985, avg = " + str(avg))
    
    bernoulliClf.fit(trainingMatrix1, labelMatrix)
    
    '''
    estimateSet = trainingDataMatrix
    estimatePredictions = bernoulliClf.predict(estimateSet)
    print("estimates = " + str(estimatePredictions))
    
    results = np.zeros((2, 2), dtype=np.int)
    
    for i in range(len(trainDrugRecords)):
        actual = trainDrugRecords[i].label
        guess = int(estimatePredictions[i])
        #print("guess = " + str(guess) + ", actual = " + str(actual))
        results[guess, actual] += 1     
    
    print("results = " + str(results))
    '''
    
    
    testDataMatrix = common.loadTestDataSet()
    testMatrix1 = featureReducer.transform(testDataMatrix)
    testPredictions = bernoulliClf.predict(testMatrix1)
    
    print("Test predictions shape = " + str(testPredictions.shape))
    print("Test Estimates = " + str(testPredictions))
    common.writeResultsFile(testPredictions)

def runWithUndersampling():
    X, y = common.loadTrainingDataSet()
    
    print("Counter(y) = " + str(Counter(y)))
    
    rus = RandomUnderSampler(random_state=42)
    
    X_res, y_res = rus.fit_resample(X, y)
    
    print("Counter(y_res) = " + str(Counter(y_res)))
    
    reducer = feature_reduction.getChiSquared(X_res, y_res, 1331)
    #featureReducer = SelectKBest(chi2, k=j)
    #featureReducer.fit(X, y)
    
    X_new = feature_reduction.transform(reducer, X_res)
    
    nbClf = NaiveBayesClassifier()
    nbClf.fit(X_new, y_res)
        
    X_test = common.loadTestDataSet()
    X_test_new = feature_reduction.transform(reducer, X_test)
    testPredictions = nbClf.predict(X_test_new)
    
    print("Test predictions shape = " + str(testPredictions.shape))
    print("Test Estimates = " + str(testPredictions))
    common.writeResultsFile(testPredictions)
    print("Done!")

def runWithUndersamplingMutualInfo():
    X, y = common.loadTrainingDataSet()
    
    print("Counter(y) = " + str(Counter(y)))
    
    rus = RandomUnderSampler(random_state=42)
    
    X_res, y_res = rus.fit_resample(X, y)
    
    print("Counter(y_res) = " + str(Counter(y_res)))
    
    reducer = SelectKBest(mutual_info_classif, 300)
    X_new = reducer.fit_transform(X_res, y_res).toarray()
    
    print("Done with feature selection")
    
    #reducer = feature_reduction.getChiSquared(X_res, y_res, 1331)
    #featureReducer = SelectKBest(chi2, k=j)
    #featureReducer.fit(X, y)
    
    #X_new = feature_reduction.transform(reducer, X_res)
    
    nbClf = NaiveBayesClassifier()
    nbClf.fit(X_new, y_res)
        
    X_test = common.loadTestDataSet()
    X_test_new = reducer.transform(X_test).toarray()
    testPredictions = nbClf.predict(X_test_new)
    
    print("Test predictions shape = " + str(testPredictions.shape))
    print("Test Estimates = " + str(testPredictions))
    common.writeResultsFile(testPredictions)
    print("Done!")

def runWithMultiModel():
    modelTransformerList = []
    X, y = common.loadTrainingDataSet()
    
    for modelNum in range(9):
        rs = 42 + modelNum
        rus = RandomUnderSampler(random_state=rs)
        X_model_full, y_model = rus.fit_resample(X, y) 
          
        reducer = SelectKBest(chi2, k=105)
        X_model = reducer.fit_transform(X_model_full, y_model).toarray()
        
        nbClassifier = NaiveBayesClassifier()
        nbClassifier.fit(X_model, y_model)
        
        modelTransformerList.append((nbClassifier, reducer)) 
    
    X_test = common.loadTestDataSet()
    combinedModelOutput = common.predictCombined(X_test, modelTransformerList)
    common.writeResultsFile(combinedModelOutput)
    print("Done predicting with multi-model.")
    
if __name__ == '__main__':
    # This has an F1 score on Miner of 0.74
    runWithMultiModel()
    
    #runWithUndersamplingMutualInfo()()

    
    #naiveBayesModel = createNaiveBayesModel(trainDrugRecords)
    
    #for predictRecord in trainDrugRecords:
    #    naiveBayesModel.predict(predictRecord)
        
    # 1. Read the file
    # 2. Get the data structure.
    # 3. Parse the files to create a model.
    # 4. Run the model to get the prediction on the test set.
    #5. 