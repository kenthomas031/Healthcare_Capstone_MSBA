# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:22:18 2019

@author: kat13g
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

def myOneHotMultiEncoder(mydf, colNameStr):
    '''Turns a categorical value with only two options into a binary column'''
    
    count=0
    dfNew = pd.get_dummies(mydf[colNameStr],drop_first=False)
    for colname in dfNew.columns.values:
        mydf[colNameStr + '_' + str(colname)]=dfNew.iloc[:,count]
        count+=1        
    mydf.drop(colNameStr,axis=1, inplace=True)
    
def myOneHotSingleEncoder(mydf, colNameStr):
    '''For categorical data, creates a column for each category option'''
    
    count=0
    dfNew = pd.get_dummies(mydf[colNameStr],drop_first=False)
    for colname in dfNew.columns.values:
        mydf[colNameStr + '_' + str(colname)]=dfNew.iloc[:,count]
        count+=1        
    mydf.drop([colNameStr,colNameStr + '_' + str(colname)],axis=1, inplace=True)



import pandas as pd

#import the data
hcdata = pd.read_csv('HCData.csv')
hcdata.head(5)

#drop non-predictive categories
hcdata.drop(['ICD','Procedure','DX','MaxPL','FVL','CSID'],axis=1,inplace=True)
#drop data with na values
hcdata.dropna(inplace=True)

#count by body part
hcdata.BRalpha.value_counts()


#drop date
#hcdata.drop(['Date'],axis=1,inplace=True)

#Splitting the date into year and month
yearMonth = hcdata['Date'].str.split('-',n=1,expand=True)
yearMonth.columns = ['Year','Month']
hcdata['Year'] = yearMonth['Year']
hcdata['Month'] = yearMonth['Month']
hcdata.drop('Date',inplace=True,axis=1)


#unique EMID's in a list
EMIDList = hcdata['EMID'].unique().tolist()
BRalphaList = hcdata['BRalpha'].unique().tolist()

#group data
dfGrouped = hcdata.groupby('EMID')
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
count6 = 0
count7 = 0
count8 = 0
count9up = 0


#Binning by body part amounts
for EM_ID in EMIDList:
    dummyGroup = dfGrouped.get_group(EM_ID)
    dummyList = []
    uniqueList = []
    for bodyPart in dummyGroup['BRalpha']:
        dummyList.append(bodyPart)
    for uniqueBodyPart in dummyList:
        if uniqueBodyPart not in uniqueList:
            uniqueList.append(uniqueBodyPart)
    uniqueList = sorted(uniqueList, key=str.lower)
    if len(uniqueList) == 1:
        count1 += 1
    if len(uniqueList) == 2:
        count2 += 1
    if len(uniqueList) == 3:
        count3 += 1
    if len(uniqueList) == 4:
        count4 += 1
    if len(uniqueList) == 5:
        count5 += 1
    if len(uniqueList) == 6:
        count6 += 1
    if len(uniqueList) == 7:
        count7 += 1
    if len(uniqueList) == 8:
        count8 += 1
    if len(uniqueList) >= 9:
        count9up += 1
print(count1,count2,count3,count4,count5,count6,count7,count8,count9up)


EMIDDf = pd.DataFrame(EMIDList)
EMIDDf.columns = ['EMID']

lengthList = []
for EM_ID in EMIDList:
    dummyGroup = dfGrouped.get_group(EM_ID)
    lengthList.append(len(dummyGroup))
EMIDDf['VisitCount'] = lengthList


#Maps the visit count to figure out if they visited once or more than once
EMIDDf['Returned'] = EMIDDf['VisitCount']
uniqueDict = {}
for i in EMIDDf['VisitCount']:
    if i == 1:
        uniqueDict.update({i:0})
    else:
        uniqueDict.update({i:1})
print(uniqueDict)

EMIDDf['Returned'] = EMIDDf['Returned'].map(uniqueDict)



#Maps whether or not a client visited for multiple injuries
lengthDict = {}
for EM_ID in EMIDList:
    dummyGroup = dfGrouped.get_group(EM_ID)
    dummyList = []
    uniqueList = []
    for bodyPart in dummyGroup['BRalpha']:
        dummyList.append(bodyPart)
    for uniqueBodyPart in dummyList:
        if uniqueBodyPart not in uniqueList:
            uniqueList.append(uniqueBodyPart)
    uniqueList = sorted(uniqueList, key=str.lower)
    if len(uniqueList) == 1:
        lengthDict.update({EM_ID:0})
    if len(uniqueList) >= 2:
        lengthDict.update({EM_ID:1})

EMIDDf['MultipleInjuries'] = EMIDDf['EMID']
EMIDDf['MultipleInjuries'] = EMIDDf['MultipleInjuries'].map(lengthDict)



#for EM_ID in EMIDList:
#    dummyGroup = dfGrouped.get_group(EM_ID)
    





def patientLength(yearStart,monthStart,yearEnd,monthEnd):
    '''This function will tell us how long a patient has been visiting 
    the clinic'''
    
    yearLength = (yearEnd - yearStart) *12
    if monthStart < monthEnd:
        totalTime = yearLength + monthEnd - monthStart
    elif monthStart > monthEnd:
        totalTime = yearLength - (monthStart - monthEnd)
    elif monthStart == monthEnd:
        if yearEnd != yearStart:
            totalTime = yearLength
        else:
            totalTime = 0
    return totalTime

print(patientLength(2012,6,2012,6))
        

#trying to get year and month
yearDict = {}
for EM_ID in EMIDList:
    dummyGroup = dfGrouped.get_group(EM_ID)
    minYear = min(dummyGroup['Year'])
    maxYear = max(dummyGroup['Year'])
    yearGroup = dummyGroup.groupby('Year')
    minYearGrouped = yearGroup.get_group(minYear)
    minMonth = min(minYearGrouped['Month'])
    maxYearGrouped = yearGroup.get_group(maxYear)
    maxMonth = max(maxYearGrouped['Month'])
    minYear = int(minYear)
    maxYear = int(maxYear)
    minMonth = int(minMonth)
    maxMonth = int(maxMonth)
    yearDict.update({EM_ID:patientLength(minYear,minMonth,maxYear,maxMonth)})
    
    
EMIDDf['MonthsAsPatient'] = EMIDDf['EMID']
EMIDDf['MonthsAsPatient'] = EMIDDf['MonthsAsPatient'].map(yearDict)
    





#Doug Code

dfFVLRisk = pd.read_csv('dfFVLRisk.csv')
hugeDf = pd.merge(hcdata,EMIDDf,how='inner')

# make a series of each unique patient and their risk change 
import numpy as np
Risk_Change = hugeDf.groupby('EMID')['FVrrisk'].agg(np.ptp)
Risk_Change = pd.DataFrame(Risk_Change)
Risk_Change = Risk_Change.reset_index()
Risk_Change = Risk_Change.rename(columns={'FVrrisk':'Risk_Change'})

# append risk change to hugeDf
hugeDf = hugeDf.merge(Risk_Change,left_on='EMID',right_on='EMID')

hugeDf.to_csv('analysisDf.csv')

# append risk to change to EMIDDf
EMIDDf = EMIDDf.merge(Risk_Change,left_on='EMID',right_on='EMID')






returnedDf = hugeDf.copy()
returnedDf = returnedDf.drop_duplicates(subset=['EMID'])


#Binning Data


def myBins(lower_bound, width, quantity):
 
    bins = []
    for low in range(lower_bound, 
                     lower_bound + quantity*width + 1, width):
        bins.append(low)
    return bins

def create_bins(lower_bound, width, quantity):
    """ create_bins returns an equal-width (distance) partitioning. 
        It returns an ascending list of tuples, representing the intervals.
        A tuple bins[i], i.e. (bins[i][0], bins[i][1])  with i > 0 
        and i < quantity, satisfies the following conditions:
            (1) bins[i][0] + width == bins[i][1]
            (2) bins[i-1][0] + width == bins[i][0] and
                bins[i-1][1] + width == bins[i][1]
    """
    
    bins = []
    for low in range(lower_bound, 
                     lower_bound + quantity*width + 1, width):
        bins.append((low, low+width))
        final = low+width
    bins.append((final,1000000))
    return bins

#Binning Page Numbers
lowBound = 0
binWidth = 5
numOfBins = 22

#Gets a list of all the bound values
bounds = myBins(lower_bound=lowBound,width=binWidth,quantity=numOfBins)
bounds

#Plotting frequencies of the bounds
import matplotlib.pyplot as plt
plt.hist(returnedDf['FVage'], bounds)  # arguments are passed to np.histogram
plt.title('Frequency of Patient Age')
plt.xlabel('Age')
plt.ylabel('Number of Peoples')
plt.show()


#Creating a string category for the bounds that can be put into the df for grouping
boundList = []
for bound in bounds:
    lowerBound = bound
    upperBound = bound + binWidth
    boundList.append(str(lowerBound)+'-'+str(upperBound))
boundList.append(str(upperBound)+'-max')
boundList

#Connecting the bin order to the string category 
boundDict = {}
count=0
for bound in boundList:
    count+=1
    boundDict.update({count:bound})
boundDict

#Creates numerical values for the bins
bins = create_bins(lowBound,binWidth,numOfBins)
bins


#Finds the bin the the value belongs to
valueDict = {}
for value in returnedDf['FVage']:
    counter=0
    for bound in bins:
        counter+=1
        if bound[0] <= value < bound[1]: 
            valueDict.update({value:counter})

#Creating a new binned column
returnedDf['Age Bin'] = returnedDf['FVage']
returnedDf['Age Bin'] = returnedDf['Age Bin'].map(valueDict)
returnedDf['Age Bin'] = returnedDf['Age Bin'].map(boundDict)



returnedDf.drop(['Allowed','FVage'],axis=1,inplace=True)





#onehotencode categorical variables
myOneHotSingleEncoder(returnedDf,'Gender')
myOneHotMultiEncoder(returnedDf,'BRalpha')
myOneHotMultiEncoder(returnedDf,'PLalias')
#myOneHotMultiEncoder(returnedDf,'Age Bin')
#myOneHotMultiEncoder(returnedDf,'Allowed Bin')




#hugeDf.to_csv('SPSShcData.csv')

dfY = returnedDf['Returned']
dfY2=returnedDf['MultipleInjuries']
dfY3=returnedDf['MonthsAsPatient']
dfX = returnedDf.drop(['Returned'],axis=1,inplace=False)

dfX = returnedDf.drop(['Returned','EMID','MultipleInjuries','VisitCount','MonthsAsPatient','Risk_Change','Month','Year'],axis=1,inplace=False)
dfX.drop(['FVrrisk','Age Bin'],axis=1,inplace=True)
#Converting date from obj to int
dfX.dtypes
#dfX['Year'] = dfX['Year'].astype(int)
#dfX['Month'] = dfX['Month'].astype(int)


#Running the Model




#Train/Test Split
from sklearn.model_selection import train_test_split
xTrain,xTest = train_test_split(dfX, test_size=0.3, random_state=56)
yTrain,yTest = train_test_split(dfY, test_size=0.3, random_state=56)


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=56, solver='lbfgs')
clf.fit(xTrain,yTrain)
clfResults = clf.predict(xTest)
clf.score(xTest,yTest)


from sklearn.model_selection import cross_val_score
scores=cross_val_score(clf,dfX,dfY,cv=10)
print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std()*2))



#Predict for whole dataset
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=56, solver='lbfgs')
clf.fit(xTrain,yTrain)
returnedResults = clf.predict(dfX)









#All Body Parts Prediction
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
import copy

def randForestResults(randForestBinResult):
    '''Puts the random forest results into a format that can be used
    to represent the probability a customer selects a certain rate code'''
    
    colNames = []
    dfRandForest = randForestBinResult[0][:,1]
    dfRandForest = pd.DataFrame(dfRandForest)
    for i in range(1,len(randForestBinResult)):
        dfRandForest[i] = randForestBinResult[i][:,1]
    for j in range(1,len(randForestBinResult)+1):
        colNames += ['Rate_Code'+str(j)]
    dfRandForest2 = pd.DataFrame(dfRandForest.values,columns=colNames)
    return dfRandForest2

dfX2 = dfX.drop(dfX.columns[5:25],axis=1)
dfY2 = dfX.iloc[:,5:25]
#dfX2.drop(['FVage'],axis=1,inplace=True)

xTrain2,xTest2 = train_test_split(dfX2, test_size=0.3, random_state=56)
yTrain2,yTest2 = train_test_split(dfY2, test_size=0.3, random_state=56)

dfX2.dtypes
yTrain2.dtypes
randForestClf = RandomForestClassifier(n_estimators=1000, criterion='entropy',max_depth=20,random_state=56) 
randForestClf.fit(xTrain2,yTrain2)
randForestBinResult = randForestClf.predict_proba(xTest2)


len(dfY2.columns)
#getting the random forest results
dfRandForest = randForestBinResult[0][:,1]
dfRandForest = pd.DataFrame(dfRandForest)
for i in range(1,len(dfY2.columns)):
    dfRandForest[i] = randForestBinResult[i][:,1]
dfRandForest.sum(axis=1)

randForestClf.score(xTest2,yTest2)
r2_score(yTest2, dfRandForest)

