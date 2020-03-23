'''
Created on Mar 23, 2020

@author: William
'''

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
from sklearn.neural_network import MLPClassifier

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
    tuneBasicNeuralNetwork()
