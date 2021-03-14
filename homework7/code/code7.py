# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 23:54:27 2021

@author: ioasp

Created on Thu Feb 18 11:07:47 2021

Economics 7103
Homework 3
Ioanna Maria Spyrou
"""

# Clear all

from IPython import get_ipython
get_ipython().magic('reset -sf')

# Import packages--------------------------------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import ttest_ind
from scipy.optimize import minimize
from statsmodels.sandbox.regression.gmm import LinearIVGMM, IV2SLS
import statistics

# Set working directories and seed---------------------------------------------------------------------------------------------

outputpath = r'C:\Users\ioasp\Desktop\Ioanna\Economics\gatech courses\Spring 2021\Environment II\GitHub\phdee-2021-sim\homework7\output'

np.random.seed(6578103)

#  Import data ----------------------------------------------------------------------------------------------------------------

data = pd.read_csv('instrumentalvehicles.csv')

#Question 1--------------------------------------------------------------------------------------------------------------------
Y = data[['price']].to_numpy()
x1 = np.full((1000,1),1.)
x2 = data[['mpg']].to_numpy()
x3 = data[['car']].to_numpy()
X = np.concatenate((x1,x2,x3),axis=1)

model = sm.OLS(Y,X)
results = model.fit()
print(results.summary())


#Question 2--------------------------------------------------------------------------------------------------------------------
#(a)---------------------------------------------------------------------------------------------------------------------------
data['con'] = 1


results1 = sm.OLS(data['mpg'],data[['con','weight','car']], missing='drop').fit()
print(results1.summary())
f1 = results1.fvalue

data['mpghat'] = results1.predict()

results2 = sm.OLS(data['price'],data[['con','mpghat','car']]).fit()
print(results2.summary())
res2 = pd.Series(results2.params)
se2 = pd.Series(results2.bse)

iv = IV2SLS(data['price'],data[['con','car','mpg']],data[['con','car','weight']]).fit()
print(iv.summary())

gmm = LinearIVGMM(data['price'],data[['con','car','mpg']],data[['con','car','weight']]).fit()
print(gmm.summary())
t = gmm.params
t_e = gmm.bse

#(b)---------------------------------------------------------------------------------------------------------------------------

data['weightsquare'] = data['weight'].pow(2)

results3 = sm.OLS(data['mpg'],data[['con','weightsquare','car']], missing='drop').fit()
print(results3.summary())
f3 = results3.fvalue

data['mpghat1'] = results3.predict()

results4 = sm.OLS(data['price'],data[['con','mpghat1','car']]).fit()
print(results4.summary())
res4 = pd.Series(results4.params)
se4 = pd.Series(results4.bse)

#(c)----------------------------------------------------------------------------------------------------------------------------



results5 = sm.OLS(data['mpg'],data[['con','height','car']], missing='drop').fit()
print(results5.summary())
f5 = results5.fvalue

data['mpghat2'] = results5.predict()

results6 = sm.OLS(data['price'],data[['con','mpghat2','car']]).fit()
print(results6.summary())
res6 = pd.Series(results6.params)
se6 = pd.Series(results6.bse)

l_f1 = pd.Series([f1])
res2 = res2.append(l_f1)

l_f3 = pd.Series([f3])
res4 = res4.append(l_f3)

l_f5 = pd.Series([f5])
res6 = res6.append(l_f5)

#---------------------------------------------------------------------------------------------------------------------------
#Table
res2 = res2.map('{:.2f}'.format)
se2 = se2.map('({:.2f})'.format)
res4 = res4.map('{:.2f}'.format)
se4 = se4.map('({:.2f})'.format)
res6 = res6.map('{:.2f}'.format)
se6 = se6.map('({:.2f})'.format)

se2 = se2.append(pd.Series([' ']))
se4 = se4.append(pd.Series([' ']))
se6 = se6.append(pd.Series([' ']))

# Generate a table of coefficients and standard deviations for the observed variables 

## Set the row and column names
rownames1 = pd.concat([pd.Series(['constant','mpg','car','F-statistic']),pd.Series([' ',' ',' ',' '])],axis = 1).stack() 

## Align se under coefficients
col2 = pd.concat([res2,se2],axis = 1).stack()
col4 = pd.concat([res4,se4],axis = 1).stack()
col6 = pd.concat([res6,se6],axis = 1).stack()


col2 = pd.DataFrame(col2)
col2.index = rownames1
col4 = pd.DataFrame(col4)
col4.index = rownames1
col6 = pd.DataFrame(col6)
col6.index = rownames1


df = pd.concat([col2,col4,col6], axis=1, join='inner')
df.columns=['(a)','(b)','(c)']

## Output to LaTeX folder-----------------------------------------------------------------------------------------------------
os.chdir(outputpath) # Output directly to LaTeX folder

df.to_latex('coeftable.tex') 


#-------------------------------------------------------------------------------------------------------------------------------
#Question 4---------------------------------------------------------------------------------------------------------------------
