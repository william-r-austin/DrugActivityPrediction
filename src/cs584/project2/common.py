import cs584.project2.utilities as utilities
from cs584.project2.drug_record import DrugRecord
import cs584.project2.constants as constants
import os.path
from datetime import datetime
import numpy as np
from scipy.sparse import lil_matrix
import math

def parseDataFile(relativePath, tagged):
    records = []
    rootDirectory = utilities.getProjectRootDirectory()
    filePath = os.path.join(rootDirectory, relativePath)
    cvFolds = constants.crossValidationFoldCount
    
    index = 0
    dataFile = open(filePath, "r")
    for dataFileLine in dataFile:
        dataTag = None
        featureString = dataFileLine
        dataGroup = None
        if tagged:
            parts = dataFileLine.split("\t", 1)
            featureString = parts[1].strip()
            dataTag = int(parts[0].strip())
            dataGroup = index % cvFolds
        
        drugFeatures = [int(k) for k in featureString.split()]
                   
        currentRecord = DrugRecord(index + 1, dataFileLine, drugFeatures, dataTag, dataGroup)
        
        records.append(currentRecord)
        index += 1
        
    dataFile.close()
    
    return records

def writeResultsFile(resultsArray):
    rootDirectory = utilities.getProjectRootDirectory()
    x = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    relativePath = "resources/output/prediction_waustin_" + x + ".txt"
    fullPath = os.path.join(rootDirectory, relativePath)
    #print("Full Path = " + fullPath)

    arrayLength = resultsArray.shape[0]
    
    outputFile = open(fullPath, 'w', newline='')
    
    for index in range(arrayLength):
        intValue = int(resultsArray[index])
        outputLine = str(intValue) + "\n"
        outputFile.write(outputLine)
    
    outputFile.close()

def createDataMatrix(dataSet):
    sparseMatrix = lil_matrix((len(dataSet), 100001), dtype=np.int8)
    for index, record in enumerate(dataSet):
        columnIndexArray = np.array(record.values)
        nonZeroColumnCount = columnIndexArray.shape[0]
        rowIndexArray = np.array([index] * nonZeroColumnCount)
        valueArray = np.array([1] * nonZeroColumnCount) 
        sparseMatrix[rowIndexArray, columnIndexArray] = valueArray 
    
    return sparseMatrix

def createLabelMatrix(trainingSet):
    trainingSetSize = len(trainingSet)
    labelMatrix = np.zeros(trainingSetSize, dtype=np.int8)
    for index, record in enumerate(trainingSet):
        labelMatrix[index] = record.label
        
    #return labelMatrix.reshape((trainingSetSize, 1))
    return labelMatrix

def loadTestDataSet():
    testDrugRecords = parseDataFile(constants.testFileRelativePath, False)
    testDataMatrix = createDataMatrix(testDrugRecords)
    return testDataMatrix
    

def loadTrainingDataSet():
    trainDrugRecords = parseDataFile(constants.trainingFileRelativePath, True)
    trainingDataMatrix = createDataMatrix(trainDrugRecords)
    #print("Created training data matrix. Shape = " + str(trainingDataMatrix.shape))
    
    labelMatrix = createLabelMatrix(trainDrugRecords)
    #print("Created training lebel matrix. Shape = " + str(labelMatrix.shape))
    
    return trainingDataMatrix, labelMatrix

def getFeatureCountArray():
    featureCountList = [j for j in range(1, 20)]
    featureCountList += [j for j in range(20, 50, 2)]
    featureCountList += [j for j in range(50, 150, 5)]
    featureCountList += [j for j in range(150, 500, 10)]
    featureCountList += [j for j in range(500, 1000, 25)]
    featureCountList += [j for j in range(1000, 2000, 50)]
    featureCountList += [j for j in range(2000, 5000, 100)]
    featureCountList += [j for j in range(5000, 10001, 250)]
    featureCountList += [j for j in range(10000, 15001, 500)]
    
    return featureCountList

def getFeatureCountArray2():
    featureCountList = [j for j in range(50, 1001, 50)]
    return featureCountList

def predictCombinedSimple(X_test, modelList):
    total = np.zeros((X_test.shape[0],), dtype=np.int64)
    
    for classifier in modelList:
        y = classifier.predict(X_test)
        total = total + y
    
    predicter = lambda t: 0 if t < len(modelList) / 2 else 1
    result = np.array([predicter(xi) for xi in total], dtype=np.int8)
    return result

def predictCombined(X, modelTransformerList):
    total = np.zeros((X.shape[0],), dtype=np.int64)
    
    for classifier, reducer in modelTransformerList:
        X_reduced = reducer.transform(X).toarray()
        y = classifier.predict(X_reduced)
        total = total + y
    
    predicter = lambda t: 0 if t < len(modelTransformerList) / 2 else 1
    result = np.array([predicter(xi) for xi in total], dtype=np.int8)
    return result

def getEntropy(probabilities):
    runningSum = 0.0
    for p in probabilities:
        if p > 0.0:
            runningSum += (p * math.log2(p))
    #return -1 * sum([k * math.log2(k) for k in probabilities])
    return -1 * runningSum 

# Ex. [(50, 78), (100, 722)]    
def getRemainder(inactive_0, inactive_1, active_0, active_1):
    total = inactive_0 + inactive_1 + active_0 + active_1
    total_0 = inactive_0 + active_0
    total_1 = inactive_1 + active_1
    
    part_0 = 0.0
    if total_0 > 0:
        part_0 = (total_0 / total) * getEntropy([inactive_0 / total_0, active_0 / total_0])

    part_1 = 0.0
    if total_1 > 0:
        part_1 = (total_1 / total) * getEntropy([inactive_1 / total_1, active_1 / total_1])
                
    return part_0 + part_1
