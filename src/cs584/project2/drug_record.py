'''
Created on Feb 29, 2020

@author: William
'''

class DrugRecord(object):
    '''
    classdocs
    '''

    def __init__(self, paramLineNum, paramInputLine, paramValues, paramLabel, paramDataGroup):
        '''
        Constructor
        '''
        self.lineNum = paramLineNum
        self.inputLine = paramInputLine
        self.values = paramValues
        self.label = paramLabel
        self.dataGroup = paramDataGroup
    
    def printRecord(self):
        print("Printing record for line #" + str(self.lineNum) + 
              ". Total features = " + str(len(self.values)) + 
              ". Label = " + str(self.label) + 
              ". Data Group = " + str(self.dataGroup))
        