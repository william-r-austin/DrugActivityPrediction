'''
Created on Mar 7, 2020

@author: William
'''

from os.path import dirname, realpath
from decimal import Decimal

def getProjectRootDirectory():
    dirPath = dirname(realpath(__file__))
    projectDirPath = dirname(dirname(dirname(dirPath)))
    return projectDirPath

def readDictionaryCount(dictObject, key):
    currentCount = 0
    
    if key in dictObject:
        currentCount = dictObject[key]
    
    return currentCount

def incrementDictionaryCountByAmount(dictObject, key, amount):
    currentCount = readDictionaryCount(dictObject, key) + amount
    dictObject[key] = currentCount
    
def incrementDictionaryCount(dictObject, key):
    incrementDictionaryCountByAmount(dictObject, key, 1)

def getSmoothedEstimate(numerator, denominator, alpha, classCount):
    newNumerator = numerator + alpha
    newDenominator = denominator + (alpha * classCount)
    result = Decimal(newNumerator) / Decimal(newDenominator)
    return result
    
def getSmoothedLogEstimate(numerator, denominator, alpha, classCount):
    smoothedEstimate = getSmoothedEstimate(numerator, denominator, alpha, classCount)
    return smoothedEstimate.ln()
        
