'''
Created on Mar 7, 2020

@author: William
'''

import cs584.project2.common as common
import cs584.project2.data_balancing as data_balancing

from collections import Counter
from sklearn import tree

if __name__ == '__main__':
    # Some setup
    xRawData, yRawData = common.loadTrainingDataSet()
    
    xBalanced, yBalanced = data_balancing.balanceDatasetWithRandomOversampling(xRawData, yRawData)
    myCounter = Counter(yBalanced)
    print("Finished loading and sampling. Data dist = " + str(myCounter))
    
    decisionTreeClassifier = tree.DecisionTreeClassifier()
    
    decisionTreeClassifier.fit(xBalanced, yBalanced)

