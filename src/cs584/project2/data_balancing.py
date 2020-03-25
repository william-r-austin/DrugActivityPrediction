from imblearn.over_sampling import RandomOverSampler
import numpy as np
import cs584.project2.utilities as utilities
import cs584.project2.constants as constants
from scipy.sparse import lil_matrix, vstack

def balanceDatasetWithRandomOversampling(X, y):
    ros = RandomOverSampler(random_state = 55)
    X2, y2 = ros.fit_resample(X, y)
    return X2, y2

class FeatureIndependentOversampler(object):

    '''
    classdocs
    '''
    def __init__(self, random_state=0, alpha=constants.smoothingConstant):
        '''
        Constructor
        '''
        self.random_state = random_state
        self.alpha = alpha
        np.random.seed(self.random_state)
    
    def fit_transform(self, X, y):
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
        #self.inactiveProbabilities = np.zeros((2, self.featureCount), dtype=np.float64)
        
        for q in range(self.featureCount):
            # Active Probabilities
            self.activeProbabilities[0, q] = utilities.getSmoothedEstimate(self.activeTotalCount - self.activeByFeature[0, q], self.activeTotalCount, self.alpha, 2)
            self.activeProbabilities[1, q] = utilities.getSmoothedEstimate(self.activeByFeature[0, q], self.activeTotalCount, self.alpha, 2)
            
            # Inactive Probabilities
            #self.inactiveProbabilities[0, q] = utilities.getSmoothedEstimate(self.inactiveTotalCount - self.inactiveByFeature[0, q], self.inactiveTotalCount, self.alpha, 2)
            #self.inactiveProbabilities[1, q] = utilities.getSmoothedEstimate(self.inactiveByFeature[0, q], self.inactiveTotalCount, self.alpha, 2)
        
        generateCount = inactiveTotal - activeTotal
        
        X_new_samples = lil_matrix((generateCount, self.featureCount), dtype=np.int8)
        
        for q in range(self.featureCount):
            #p0 = self.activeProbabilities[0, q]
            #p1 = self.activeProbabilities[1, q]
            pArray = self.activeProbabilities[:, q]
            newSamplesColumn = np.random.choice(2, (generateCount, 1), p=pArray)
            #newSamplesColumn = np.reshape(newSamplesArray, (generateCount, 1))
            X_new_samples[:, q] = newSamplesColumn
        
        y_new_samples = np.full((generateCount,), 1, dtype=np.int8)
        X_new = vstack((X, X_new_samples))
        y_new = np.concatenate((y, y_new_samples), axis=0)
        
        print("Concatenated X. Shape = " + str(X_new.shape))
        print("Concatenated y. Shape = " + str(y_new.shape))
        
        return X_new, y_new
