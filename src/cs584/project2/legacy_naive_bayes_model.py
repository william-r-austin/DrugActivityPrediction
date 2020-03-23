'''
Created on Mar 7, 2020

@author: William
'''

import cs584.project2.constants as constants
import cs584.project2.utilities as utilities

class NaiveBayesSubModel():
    '''
    classdocs
    '''
    
    def __init__(self, paramDataGroup):
        self.dataGroup = paramDataGroup
        
        self.positiveFeatureCounts = dict()
        self.negativeFeatureCounts = dict()
        
        self.positiveCount = 0
        self.negativeCount = 0
    
    def incrementPositiveCount(self):
        self.positiveCount += 1
    
    def incrementNegativeCount(self):
        self.negativeCount += 1
    
    def incrementPositiveFeatureCount(self, featureValue):
        utilities.incrementDictionaryCount(self.positiveFeatureCounts, featureValue)
        
    def incrementNegativeFeatureCount(self, featureValue):
        utilities.incrementDictionaryCount(self.negativeFeatureCounts, featureValue)

class NaiveBayesModel(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.submodels = []
        self.splitSize = constants.crossValidationFoldCount
        
        for i in range(self.splitSize):
            currentSubmodel = NaiveBayesSubModel(i)
            self.submodels.append(currentSubmodel)
    
    def constructModel(self, trainingData):
        for currentRecord in trainingData:
            currentDataGroup = currentRecord.dataGroup
            currentModel = self.submodels[currentDataGroup]
            
            label = currentRecord.label
            if label == 0:
                currentModel.incrementNegativeCount()
                for value in currentRecord.values:
                    currentModel.incrementNegativeFeatureCount(value)
            elif label == 1:
                currentModel.incrementPositiveCount()
                for value in currentRecord.values:
                    currentModel.incrementPositiveFeatureCount(value)
            else:
                print("Error! Invalid Label: " + str(label) + ", Record details = ")
                currentRecord.printRecord()
    
    def predict(self, predictRecord):
        alpha = constants.smoothingConstant
        totalPositive = 0
        totalNegative = 0
        
        totalPositiveWithFeature = dict()
        totalNegativeWithFeature = dict()
        
        dataGroup = predictRecord.dataGroup
        
        for submodel in self.submodels:
            if dataGroup is not None and dataGroup != submodel.dataGroup:
                totalPositive += submodel.positiveCount
                totalNegative += submodel.negativeCount
                
                for featureKey in range(1, 100001):
                    submodelPositiveWithFeature = utilities.readDictionaryCount(submodel.positiveFeatureCounts, featureKey)
                    utilities.incrementDictionaryCountByAmount(totalPositiveWithFeature, featureKey, submodelPositiveWithFeature)
                    
                    submodelNegativeWithFeature = utilities.readDictionaryCount(submodel.negativeFeatureCounts, featureKey)
                    utilities.incrementDictionaryCountByAmount(totalNegativeWithFeature, featureKey, submodelNegativeWithFeature)
        
        totalSamples = totalPositive + totalNegative
        
        positiveValue = utilities.getSmoothedLogEstimate(totalPositive, totalSamples, alpha, 2)
        negativeValue = utilities.getSmoothedLogEstimate(totalNegative, totalSamples, alpha, 2)
        
        for featureKey in range(1, 100001):
            positiveCountForFeature = utilities.readDictionaryCount(totalPositiveWithFeature, featureKey)
            positiveFactor = utilities.getSmoothedLogEstimate(positiveCountForFeature, totalPositive, alpha, 2)
            positiveValue += positiveFactor
            
            negativeCountForFeature = utilities.readDictionaryCount(totalNegativeWithFeature, featureKey)
            negativeFactor = utilities.getSmoothedLogEstimate(negativeCountForFeature, totalNegative, alpha, 2)
            negativeValue += negativeFactor
        
        prediction = 1 if positiveValue > negativeValue else 0
        
        print("Predicting record: ")
        predictRecord.printRecord()
        print("Prediction = " + str(prediction) + ", Positive Value = " + str(positiveValue) + ", Negative Value = " + str(negativeValue))
