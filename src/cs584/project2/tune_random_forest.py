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
    
def tuneRandomForestSmote(featureSizes):
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
                
            clf = RandomForestClassifier(max_depth=3)
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

def tuneRandomForestDepth(depths):
    X_raw, y_raw = common.loadTrainingDataSet()
    
    scoreMap = dict()
    for depth in depths:
        scoreMap[depth] = []
        
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    foldNumber = 0

    for train_index, test_index in kf.split(X_raw):
        X_train, X_test = X_raw[train_index], X_raw[test_index]
        y_train, y_test = y_raw[train_index], y_raw[test_index]
        
        for depth in depths:
            reducer = SelectKBest(chi2, k=127)
            reducer.fit(X_train, y_train)
            X_train_reduced = reducer.transform(X_train).toarray()
            
            ss_rs = 42+(depth*foldNumber)
            smoteSampler = SMOTE(random_state=ss_rs)

            X_model, y_model = smoteSampler.fit_resample(X_train_reduced, y_train)
                
            clf = RandomForestClassifier(max_depth=depth)
            clf.fit(X_model, y_model)

            X_test_reduced = reducer.transform(X_test).toarray()
            output = clf.predict(X_test_reduced)
            combinedModelScore = f1_score(y_test, output)
            scoreMap[depth].append(combinedModelScore)
            
            print()
            print("Done with RF prediction for fold #" + str(foldNumber) + " for depth = " + str(depth) + ". F1 = " + str(combinedModelScore))
        
        foldNumber += 1
        
    for depth in depths:
        meanF1Score = mean(scoreMap[depth])
        print("F1 Score for RF with Chi2 and depth = " + str(depth) + " is: " + str(meanF1Score))  

if __name__ == '__main__':
    #tuneDecisionTreeFeatureReductionSize()
    
    sizes = [1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047]
    
    # Got .68 CV
    tuneRandomForestSmote(sizes)
    
    #tuneRandomForestDepth([2, 3, 4, 5, 6, 7, 8, 9, 11, 13])
    