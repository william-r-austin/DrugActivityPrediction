'''
Created on Mar 15, 2020

@author: William
'''
import numpy as np
import cs584.project2.constants as constants
import cs584.project2.utilities as utilities
from cs584.project2.naive_bayes_classifier import NaiveBayesClassifier
from sklearn.utils.estimator_checks import check_estimator
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

def transform(reducer, X):
    X_new = reducer.transform(X).toarray()
    #print(X_new)
    #if X_new.issparse():
    #    X_new = X_new.todense()

    return X_new

def getChiSquared(X, y, newFeatureCount):
    reducer = SelectKBest(chi2, k=newFeatureCount)
    return reducer.fit(X, y)

def getTruncatedSVD(X, y, newFeatureCount):
    reducer = TruncatedSVD(n_components=newFeatureCount, n_iter=7)
    return reducer.fit(X, y)

def computePredictiveness(trainingSet, trainingLabels):
    activeIndexTuple = trainingLabels.nonzero()
    activeIndexValues = activeIndexTuple[0]
    print("Nonzero array = " + str(activeIndexValues))
    
    print("Nonzero array shape = " + str(activeIndexValues.shape))
    
    fs = frozenset(activeIndexValues)
    
    
    allIndices = [k for k in range(len(trainingLabels))]
    nonActiveIndices =  list(filter(lambda q: q not in fs, allIndices))
    
    inactiveIndexValues = np.array(nonActiveIndices, dtype=np.int64)
    print("Zero array = " + str(inactiveIndexValues))
    print("Zero array shape = " + str(inactiveIndexValues.shape))
    
    activeTraining = trainingSet[activeIndexValues,:]
    inactiveTraining = trainingSet[inactiveIndexValues,:]
    
    print("Active Shape = " + str(activeTraining.shape))
    print("Inactive Shape = " + str(inactiveTraining.shape)) 
    
    activeByFeature = np.sum(activeTraining, axis=0)
    inactiveByFeature = np.sum(inactiveTraining, axis=0)
    print("Shapes = " + str(activeByFeature.shape) + " and " + str(inactiveByFeature.shape))
    
    activeTotal = activeIndexValues.shape[0]
    inactiveTotal = inactiveIndexValues.shape[0]
    
    alpha = constants.smoothingConstant
    
    activeFractions = (activeByFeature + alpha) / (activeTotal + 2 * alpha)
    inactiveFractions = (inactiveByFeature + alpha) / (inactiveTotal + 2 * alpha)
    
    print("activeFractions 1 to 10 = " + str(activeFractions[0,0:10]))
    print("inactiveFractions 1 to 10 = " + str(inactiveFractions[0,0:10]))
    
    minValues = np.minimum(activeFractions, inactiveFractions)
    maxValues = np.maximum(activeFractions, inactiveFractions)
    
    print("minValues 1 to 10 = " + str(minValues[0,0:10]))
    print("maxValues 1 to 10 = " + str(maxValues[0,0:10]))
    
    predictiveness = minValues / maxValues
    
    featureList = []
    
    index = 0
    while index < predictiveness.shape[1]:
        featureTuple = (index, predictiveness[0, index])
        featureList.append(featureTuple)
        index += 1
    
    print("featureList length = " + str(len(featureList)))
    
    sortedFeatures = sorted(featureList, key = lambda k : k[1])
    
    print("Item 0 = " + str(sortedFeatures[0]))
    
    #print(sortedFeatures)
    
    return sortedFeatures