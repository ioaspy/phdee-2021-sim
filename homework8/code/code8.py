# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:04:13 2021

@author: ioasp
Homework 8
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
import warnings

# Set working directories and seed---------------------------------------------------------------------------------------------

outputpath = r'C:\Users\ioasp\Desktop\Ioanna\Economics\gatech courses\Spring 2021\Environment II\GitHub\phdee-2021-sim\homework8\output'

np.random.seed(6578103)

#Import data ----------------------------------------------------------------------------------------------------------------

data = pd.read_csv('instrumentalvehicles.csv')

#Question 2--------------------------------------------------------------------------------------------------------------------
sns.scatterplot(data=data, x='length', y='mpg')
plt.plot([225,225],[0,55],'k-',lw=2)
plt.savefig('figure1.eps',format='eps')
plt.show()

#Question 3-------------------------------------------------------------------------------------------------------------------

lmin = data['length'].min()
lmax = data['length'].max()

data.loc[data['length']<=225,'tech']=0
data.loc[data['length']>225,'tech']=1

l1x=[]
for i in range(1000):
    if data['tech'][i]==0:
        l1x.append(data['length'][i])
        
l1y=[]
for i in range(1000):
    if data['tech'][i]==0:
        l1y.append(data['mpg'][i])
        
x1 = np.array(l1x)
y1 = np.array(l1y)
z1 = np.poly1d(np.polyfit(x1,y1,1))
t = np.linspace(lmin,225,200)

l2x=[]
for i in range(1000):
    if data['tech'][i]==1:
        l2x.append(data['length'][i])
        
l2y=[]
for i in range(1000):
    if data['tech'][i]==1:
        l2y.append(data['mpg'][i])
        
x2 = np.array(l2x)
y2 = np.array(l2y)
z2 = np.poly1d(np.polyfit(x2,y2,1))
r = np.linspace(225,lmax,200)

plt.plot(x1,y1,'o',t,z1(t),'-')
plt.plot(x2,y2,'o',r,z2(r),'-')
plt.xlabel('length')
plt.ylabel('mpg')
plt.savefig('figure2.eps',format='eps')
plt.show()

#Treatment effect--------------------------------------------------------------

left1 = np.polyfit(x1,y1,1)[1] + np.polyfit(x1,y1,1)[0]*225
right1 = np.polyfit(x2,y2,1)[1] + np.polyfit(x2,y2,1)[0]*225

e1= left1 - right1

#Question 4-------------------------------------------------------------------------------------------------------------------

z3 = np.poly1d(np.polyfit(x1,y1,2))
z4 = np.poly1d(np.polyfit(x2,y2,2))
print(z3)

plt.plot(x1,y1,'o',t,z3(t),'-')
plt.plot(x2,y2,'o',r,z4(r),'-')
plt.xlabel('length')
plt.ylabel('mpg')
plt.savefig('figure3.eps',format='eps')
plt.show()

#Treatment effect-------------------------------------------------------------


left2 = np.polyfit(x1,y1,2)[2] + np.polyfit(x1,y1,2)[1]*225 + np.polyfit(x1,y1,2)[0]*pow(225,2)
right2 = np.polyfit(x2,y2,2)[2] + np.polyfit(x2,y2,2)[1]*225 + + np.polyfit(x2,y2,2)[0]*pow(225,2)

e2= left2 - right2

#Question 5-------------------------------------------------------------------------------------------------------------------

z5 = np.poly1d(np.polyfit(x1,y1,5))
z6 = np.poly1d(np.polyfit(x2,y2,5))

plt.plot(x1,y1,'o',t,z5(t),'-')
plt.plot(x2,y2,'o',r,z6(r),'-')
plt.xlabel('length')
plt.ylabel('mpg')
plt.savefig('figure4.eps',format='eps')
plt.show()

#Treatment effect-------------------------------------------------------------


left3 = np.polyfit(x1,y1,5)[5] + np.polyfit(x1,y1,5)[4]*225 + np.polyfit(x1,y1,5)[3]*pow(225,2)+ np.polyfit(x1,y1,5)[2]*pow(225,3)+ np.polyfit(x1,y1,5)[1]*pow(225,4)+ np.polyfit(x1,y1,5)[0]*pow(225,5)
right3 = np.polyfit(x2,y2,5)[5] + np.polyfit(x2,y2,5)[4]*225 + np.polyfit(x2,y2,5)[3]*pow(225,2) +np.polyfit(x2,y2,5)[2]*pow(225,3)+np.polyfit(x2,y2,5)[1]*pow(225,4)+np.polyfit(x2,y2,5)[0]*pow(225,5)

e3= left3 - right3

#Question 6----------------------------------------------------------------------------------------------------------------
data['instr']=0
for i in range(1000):
    if data['length'][i]<=225:
        data['instr'][i]= np.polyfit(x1,y1,2)[2] + np.polyfit(x1,y1,2)[1]*data['length'][i] + np.polyfit(x1,y1,2)[0]*pow(data['length'][i],2)
    else:
        data['instr'][i]= np.polyfit(x2,y2,2)[2] + np.polyfit(x2,y2,2)[1]*data['length'][i] + np.polyfit(x2,y2,2)[0]*pow(data['length'][i],2)
        
data['con'] = 1

results = sm.OLS(data['mpg'],data[['con','instr']], missing='drop').fit()
print(results.summary())
f1 = results.fvalue
print(f1)

data['mpghat'] = results.predict()

results2 = sm.OLS(data['price'],data[['con','mpghat','car']]).fit()
print(results2.summary())
res2 = pd.Series(results2.params)
se2 = pd.Series(results2.bse)        
    
#Table----------------------------------------------------------------------------------------------------------------
res2 = res2.map('{:.2f}'.format)
se2 = se2.map('({:.2f})'.format)

# Generate a table of coefficients and standard deviations for the observed variables 

## Set the row and column names
rownames1 = pd.concat([pd.Series(['constant','mpg','car']),pd.Series([' ',' ',' '])],axis = 1).stack() 

## Align se under coefficients
col2 = pd.concat([res2,se2],axis = 1).stack()


col2 = pd.DataFrame(col2)
col2.index = rownames1

df = pd.concat([col2], axis=1, join='inner')
df.columns=['(a)']

## Output to LaTeX folder-----------------------------------------------------------------------------------------------------
os.chdir(outputpath) # Output directly to LaTeX folder

df.to_latex('coeftable.tex')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    