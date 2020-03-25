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
from sklearn.metrics import f1_score

from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from statistics import mean
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

def tuneBasicDecisionTree():
    # Some setup
    X_raw, y_raw = common.loadTrainingDataSet()
    
    #xBalanced, yBalanced = data_balancing.balanceDatasetWithRandomOversampling(xRawData, yRawData)
    #myCounter = Counter(yBalanced)
    #print("Finished loading and sampling. Data dist = " + str(myCounter))
    
    decisionTreeClassifier = DecisionTreeClassifier()
    cvFolds = 5 #constants.crossValidationFoldCount
    cvScores = cross_val_score(estimator=decisionTreeClassifier, X=X_raw, y=y_raw, scoring='f1', cv=cvFolds)
    print("Individual CV scores = " + str(cvScores))
    avg = sum(cvScores) / cvFolds
    print("Cross validation score for decision tree = " + str(avg))

def tuneReducedDecisionTree():
    X, y = common.loadTrainingDataSet()
    
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    splitIndex = 0
    f1ScoreList = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        totalF1 = 0.0
        numModels = 9
        for modelNum in range(numModels):
            rs = 42 + modelNum
            rus = RandomUnderSampler(random_state=rs)
            X_model_full, y_model = rus.fit_resample(X_train, y_train) 
              
            truncatedSvd = TruncatedSVD(n_components=331, n_iter=7, random_state=42)
            X_model = truncatedSvd.fit_transform(X_model_full, y_model)
            
            dtClassifier = DecisionTreeClassifier(ccp_alpha=0.015)
            dtClassifier.fit(X_model, y_model)
            
            X_model_test = truncatedSvd.transform(X_test)
            y_pred = dtClassifier.predict(X_model_test)
            #report = classification_report(y_test, y_pred)
            currentF1 = f1_score(y_test, y_pred) 
            print("Printing F1 for model #" + str(modelNum) + " = " + str(currentF1))
            #print(str(report))
            totalF1 += currentF1
        
        avgF1 = totalF1 / numModels
        print("f1 = " + str(avgF1))

def tuneReducedDecisionTreeWithFeatureSizeAndDepth(featureSize, depth):
    X, y = common.loadTrainingDataSet()
    
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    f1ScoreList = []
    
    foldNumber = 1
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        rus = RandomUnderSampler(random_state=foldNumber)
        X_model_full, y_model = rus.fit_resample(X_train, y_train) 
        
        
        reducer = SelectKBest(chi2, k=featureSize)
        X_model1 = reducer.fit_transform(X_model_full, y_model)
        X_model = X_model1.tocsc()
        #reducer = TruncatedSVD(n_components=featureSize, n_iter=7, random_state=42)
        #X_model = reducer.fit_transform(X_train, y_train)
        
        dtClassifier = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", 
                                              #min_samples_split=0.01, 
                                              min_samples_leaf=1,
                                              min_weight_fraction_leaf=0.01)
        dtClassifier.fit(X_model, y_model)
        
        X_model_test = reducer.transform(X_test).tocsr()
        y_pred = dtClassifier.predict(X_model_test)
        #report = classification_report(y_test, y_pred)
        currentF1 = f1_score(y_test, y_pred) 
        f1ScoreList.append(currentF1)
        
        foldNumber += 1
    
    #print("f1 Score list = " + str(f1ScoreList))
    print("Mean for feature and depth of (" + str(featureSize) + ", " + str(depth) + ") = " + str(mean(f1ScoreList)))


def tuneDecisionTreeMultiModel(featureSize):
    X, y = common.loadTrainingDataSet()
    
    #print("Counter(y) = " + str(Counter(y)))
    
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    splitIndex = 0
    f1ScoreList = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        modelTransformerList = []
        
        for modelNum in range(9):
            rs = 42 + modelNum
            rus = RandomUnderSampler(random_state=rs)
            X_model_full, y_model = rus.fit_resample(X_train, y_train) 
              
            reducer = SelectKBest(chi2, k=featureSize)
            X_model = reducer.fit_transform(X_model_full, y_model).toarray()
            
            dtClassifier = DecisionTreeClassifier()
            dtClassifier.fit(X_model, y_model)
    
            #X_test_2 = reducer.transform(X_test).toarray()
            #output = nbClassifier.predict(X_test_2)
            #modelScore = f1_score(y_test, output)
            
            #print("Split Index = " + str(splitIndex) + ", Model Num = " + str(modelNum) + ", F1 = " + str(modelScore))
            
            #modelTransformerList.append((nbClassifier, reducer)) 
        
        combinedModelOutput = common.predictCombined(X_test, modelTransformerList)
        #combinedModelScore = f1_score(y_test, combinedModelOutput)
        #f1ScoreList.append(combinedModelScore)
        #print("Combined Model Score = " + str(combinedModelScore))
       
        splitIndex += 1
    
    #print("F1 Score for FR size = " + str(featureSize) + " is: " + str(mean(f1ScoreList)))
    
def tuneDecisionTreeSmote(featureSizes):
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
                
            dtClassifier = DecisionTreeClassifier(max_depth=10)
            dtClassifier.fit(X_model, y_model)

            X_test_reduced = reducer.transform(X_test).toarray()
            output = dtClassifier.predict(X_test_reduced)
            combinedModelScore = f1_score(y_test, output)
            scoreMap[featureSize].append(combinedModelScore)
            
            print()
            print("Done with DT prediction for fold #" + str(foldNumber) + " for feature size = " + str(featureSize) + ". F1 = " + str(combinedModelScore))
        
        foldNumber += 1
        
    for featureSize in featureSizes:
        meanF1Score = mean(scoreMap[featureSize])
        print("F1 Score for DT with Chi2 and FR size = " + str(featureSize) + " is: " + str(meanF1Score))  
    

def tuneDecisionTreeFeatureReductionSize():
    featureSizes = [1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047]
    for featureSize in featureSizes:
        tuneReducedDecisionTreeWithFeatureSizeAndDepth(featureSize, 6)

if __name__ == '__main__':
    #tuneDecisionTreeFeatureReductionSize()
    
    sizes = [1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047]
    tuneDecisionTreeSmote(sizes)