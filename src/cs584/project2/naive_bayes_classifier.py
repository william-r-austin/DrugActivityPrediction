'''
Created on Mar 14, 2020

@author: William
'''
import numpy as np
from cs584.project2 import utilities
from cs584.project2 import constants

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class NaiveBayesClassifier(BaseEstimator):
    '''
    classdocs
    '''

    def __init__(self, alpha=0.0001):
        '''
        Constructor
        '''
        self.alpha = alpha

     
    
    def fit(self, X, y):
        
        X, y = check_X_y(X, y)
        
        self.activeByFeature = None
        self.inactiveByFeature = None
        
        self.activeTotalCount = 0
        self.inactiveTotalCount = 0
        
        self.activeProbabilities = None
        self.inactiveProbabilities = None
        
        self.featureCount = X.shape[1]
        
        activeIndexTuple = y.nonzero()
        activeIndexValues = activeIndexTuple[0]

        
        self.activeTotalCount = activeIndexValues.shape[0] 
        
        fs = frozenset(activeIndexValues)
                
        allIndices = [k for k in range(len(y))]
        nonActiveIndices =  list(filter(lambda q: q not in fs, allIndices))
        
        inactiveIndexValues = np.array(nonActiveIndices, dtype=np.int64)
        self.inactiveTotalCount = inactiveIndexValues.shape[0]
        

        
        activeTraining = X[activeIndexValues,:]
        inactiveTraining = X[inactiveIndexValues,:]
        

        
        self.activeByFeature = np.sum(activeTraining, axis=0)
        self.inactiveByFeature = np.sum(inactiveTraining, axis=0)

        
        activeTotal = activeIndexValues.shape[0]
        inactiveTotal = inactiveIndexValues.shape[0]

        
        self.activeProbabilities = np.zeros((2, self.featureCount), dtype=np.float64)
        self.inactiveProbabilities = np.zeros((2, self.featureCount), dtype=np.float64)
        

        
        for q in range(self.featureCount):
            # Active Probabilities
            self.activeProbabilities[0, q] = utilities.getSmoothedLogEstimate(self.activeTotalCount - self.activeByFeature[q], self.activeTotalCount, self.alpha, 2)
            self.activeProbabilities[1, q] = utilities.getSmoothedLogEstimate(self.activeByFeature[q], self.activeTotalCount, self.alpha, 2)
            
            # Inactive Probabilities
            self.inactiveProbabilities[0, q] = utilities.getSmoothedLogEstimate(self.inactiveTotalCount - self.inactiveByFeature[q], self.inactiveTotalCount, self.alpha, 2)
            self.inactiveProbabilities[1, q] = utilities.getSmoothedLogEstimate(self.inactiveByFeature[q], self.inactiveTotalCount, self.alpha, 2)
        
        return self
        
        
    def predict(self, X):
        # Check is fit had been called
        #check_is_fitted(self, attributes = {'name': 'NaiveBayesClassifier'})
        
        # Input validation
        X = check_array(X)
        
        testSet = X
        inverseTestSet = 1 - testSet
                
        activeResult0 = np.multiply(inverseTestSet, self.activeProbabilities[0, :]).sum(axis = 1)
        activeResult1 = np.multiply(testSet, self.activeProbabilities[1, :]).sum(axis = 1)
        activeResultTotal = activeResult0 + activeResult1
                
        inactiveResult0 = np.multiply(inverseTestSet, self.inactiveProbabilities[0, :]).sum(axis = 1)
        inactiveResult1 = np.multiply(testSet, self.inactiveProbabilities[1, :]).sum(axis = 1)
        inactiveResultTotal = inactiveResult0 + inactiveResult1
                
        result = np.zeros((testSet.shape[0],), dtype=np.int64)
        
        for index, (activePrediction, inactivePrediction) in enumerate(zip(activeResultTotal, inactiveResultTotal)):
            if activePrediction > inactivePrediction:
                result[index] = 1
        
        return result
        