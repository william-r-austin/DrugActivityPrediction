'''
Created on Mar 5, 2020

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

class InformationGainReducer(object):
    '''
    classdocs
    '''

    def __init__(self, n_size=100):
        '''
        Constructor
        '''
        self.n_size = n_size
        self.fitted = False
    
    def resize(self, new_size):
        if not self.fitted:
            raise Exception("Reducer has not been fitted!") 
        self.n_size = new_size
        self.top_k = np.array(self.sorted_indices[0:self.n_size], dtype=np.int32)
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    
    def transform(self, X):
        if not self.fitted:
            raise Exception("Reducer has not been fitted!") 
        
        return X[:, self.top_k]
    
    def fit(self, trainingSet, trainingLabels):
        totalFeatures = trainingSet.shape[1]
        
        activeIndexTuple = trainingLabels.nonzero()
        activeIndexValues = activeIndexTuple[0]
        #print("Nonzero array = " + str(activeIndexValues))
        #print("Nonzero array shape = " + str(activeIndexValues.shape))
        
        fs = frozenset(activeIndexValues)
        
        
        allIndices = [k for k in range(len(trainingLabels))]
        nonActiveIndices =  list(filter(lambda q: q not in fs, allIndices))
        
        inactiveIndexValues = np.array(nonActiveIndices, dtype=np.int32)
        #print("Zero array = " + str(inactiveIndexValues))
        #print("Zero array shape = " + str(inactiveIndexValues.shape))
        
        activeTraining = trainingSet[activeIndexValues,:]
        inactiveTraining = trainingSet[inactiveIndexValues,:]
        
        #print("Active Shape = " + str(activeTraining.shape))
        #print("Inactive Shape = " + str(inactiveTraining.shape)) 
        
        activeByFeature = np.squeeze(np.asarray(np.sum(activeTraining, axis=0)))
        #activeByFeature = np.reshape(activeByFeature, (totalFeatures,))
        
        inactiveByFeature = np.squeeze(np.asarray(np.sum(inactiveTraining, axis=0)))
        #inactiveByFeature = np.reshape(inactiveByFeature, (totalFeatures,))
        #print("Shapes = " + str(activeByFeature.shape) + " and " + str(inactiveByFeature.shape))
        
        activeTotal = activeIndexValues.shape[0]
        inactiveTotal = inactiveIndexValues.shape[0]
        
        #print("activeTotal = " + str(activeTotal) + ", inactiveTotal = " + str(inactiveTotal))
        
        igArray = np.zeros((totalFeatures,), dtype=np.float32)
        #print("igArray shape = " + str(igArray.shape))
        
        #common.loadTrainingDataSet()
        #Ht = common.getEntropy([722/800, 78/800])
        #print(common.getEntropy([1/100, 99/100]))
        #print("H(T) = " + str(Ht))
        
        for index, (activeCount, inactiveCount) in enumerate(zip(activeByFeature, inactiveByFeature)):
            i1 = inactiveCount
            i0 = inactiveTotal - i1
            
            a1 = activeCount
            a0 = activeTotal - a1 
            #igArray[index] = Ht - common.getRemainder(i0, i1, a0, a1) 
            igArray[index] = common.getRemainder(i0, i1, a0, a1)
            #print(".", end='')
            #if index % 100 == 0:
            #    print()
            
        #print("Done")
        
        featureList = []
        
        index = 0
        while index < igArray.shape[0]:
            featureTuple = (index, igArray[index])
            featureList.append(featureTuple)
            index += 1
        
        #print("featureList length = " + str(len(featureList)))
        
        sortedTuples = sorted(featureList, key = lambda k : k[1])
        
        self.sorted_indices = [g[0] for g in sortedTuples]
        self.fitted = True
        
        self.top_k = np.array(self.sorted_indices[0:self.n_size], dtype=np.int32)
        
        
        #topk = 
        
        '''
        
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
        '''  