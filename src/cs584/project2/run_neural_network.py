'''
Created on Mar 16, 2020

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

def runWithBalancingAndIGR(featureSize, alphaValue):
    X_model_full_imbalanced, y_model_imbalanced = common.loadTrainingDataSet()
    
    balancer = FeatureIndependentOversampler(random_state=42)
    X_model_full_raw, y_model_raw = balancer.fit_transform(X_model_full_imbalanced, y_model_imbalanced)
    
    X_model_full, y_model = shuffle(X_model_full_raw, y_model_raw, random_state=42) 
    
    reducer = InformationGainReducer()
    reducer.fit(X_model_full, y_model)

    reducer.resize(featureSize)
    X_model = reducer.transform(X_model_full).todense()
    
    hiddenLayerSizes = (int(math.sqrt(featureSize)) + 1,)
    mc = MLPClassifier(solver='lbfgs', alpha=alphaValue, hidden_layer_sizes=hiddenLayerSizes)
    mc.fit(X_model, y_model)
    
    X_test_full = common.loadTestDataSet()
    X_test = reducer.transform(X_test_full)
    
    output = mc.predict(X_test)
    common.writeResultsFile(output)
    
    print("Done estimating with neural network for feature size = " + str(featureSize) + " and alpha = " + str(alphaValue))

if __name__ == '__main__':
    runWithBalancingAndIGR(1023, 0.00005)
    
