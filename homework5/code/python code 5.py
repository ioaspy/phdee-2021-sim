# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 09:20:24 2021

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statistics
import statsmodels.formula.api as sm
import os
import seaborn as sns

# Set working directories and seed---------------------------------------------------------------------------------------------

outputpath = r'C:\Users\ioasp\Desktop\Ioanna\Economics\gatech courses\Spring 2021\Environment II\GitHub\phdee-2021-sim\homework5\output'

np.random.seed(6578103)

#  Import data ----------------------------------------------------------------------------------------------------------------

data = pd.read_csv('fishbycatch.csv')

#Question 1--------------------------------------------------------------------------------------------------------------------

data['id'] = data.index
a = data.iloc[:,0:3] 
b = data.iloc[:,75]
data1 = pd.concat([a,b],axis=1)
data2 = data.iloc[:,51:76]
data_new=pd.merge(data1,data2, on='id')
data_new_long = pd.wide_to_long(data_new, ['bycatch'], i='id',j='month' )

l=[]
for i in range (1200):
    l.append(i)
data_new_long = data_new_long.set_index([pd.Index(l)])

#New dataframe for treat
list_tre = []

for i in range(600):
    list_tre.append(0)

for i in range(600,1200):
    list_tre.append(1)
    
    
s_tre =pd.Series(list_tre)    

data_all= pd.concat([data_new_long,s_tre], axis=1)
data_all = data_all.rename(columns={0:'tre'})

l1=[]

for i in range(1200):
    l1.append(0)
    
l1_s=pd.Series(l1)

data_all= pd.concat([data_all,l1_s], axis=1)
data_all = data_all.rename(columns={0:'new'})

data_all.loc[(data_all['treated']==1) & (data_all['tre']==1), 'new'] = 1
#OLS
#Clustered s.e. at the firm level


results = sm.ols(formula='bycatch~new+tre+treated', data=data_all).fit(cov_type = 'cluster', cov_kwds={'groups':data_all['firm']}, use_t=True)
print(results.summary())
res=results.params
se=results.bse

res=res.drop(labels=['tre'])
se=se.drop(labels=['tre'])

## Format means and std devs to display to two decimal places
res = res.map('{:.2f}'.format)
se = se.map('({:.2f})'.format)

s1=pd.Series([' ',' ',' '], index=['Shrimp','Salmon','Firmsize'])
res=res.append(s1)
se=se.append(s1)


#Question 2------------------------------------------------------------------------------------------------------------------

data_long = pd.wide_to_long(data, ['shrimp','salmon','bycatch'], i='id',j='month' )

l1=[]
for i in range (1200):
    l1.append(i)
data_long = data_long.set_index([pd.Index(l1)])

#New dataframe for treat
list_tre_new = []

for i in range(600):
    list_tre_new.append(0)

for i in range(600,1200):
    list_tre_new.append(1)
    
    
s_tre_new =pd.Series(list_tre_new)    

data_all1= pd.concat([data_long,s_tre_new], axis=1)
data_all1 = data_all1.rename(columns={0:'tre'})

l2=[]

for i in range(1200):
    l2.append(0)
    
l2_s=pd.Series(l2)

data_all1 = pd.concat([data_all1,l2_s], axis=1)
data_all1 = data_all1.rename(columns={0:'new'})

data_all1.loc[(data_all1['treated']==1) & (data_all1['tre']==1), 'new'] = 1
#OLS
#Clustered s.e. at the firm level


results1 = sm.ols(formula='bycatch~new+tre+treated+shrimp+salmon+firmsize', data=data_all1).fit(cov_type = 'cluster', cov_kwds={'groups':data_all['firm']}, use_t=True)
print(results1.summary())
res1=results1.params
se1=results1.bse

res1=res1.drop(labels=['tre'])

se1=se1.drop(labels=['tre'])

## Format means and std devs to display to two decimal places
res1 = res1.map('{:.2f}'.format)
se1 = se1.map('({:.2f})'.format)

# Generate a table of coefficients and standard deviations for the observed variables 

## Set the row and column names
rownames1 = pd.concat([pd.Series(['constant','Treated','Treatment group','Shrimp','Salmon','Firmsize','Time indicator']),pd.Series([' ',' ',' ',' ',' ',' ',' '])],axis = 1).stack() 


## Align se under coefficients
col0 = pd.concat([res,se],axis = 1).stack()
col0_new = pd.concat([res1,se1],axis = 1).stack()

## Add column and row labels.  Convert to dataframe (helps when you export it)
col0 = pd.DataFrame(col0)
col0 = col0.append({0:'Y'}, ignore_index=True)
col0 = col0.append({0:' '}, ignore_index=True)
col0.index = rownames1


## Add column and row labels.  Convert to dataframe (helps when you export it)
col0_new = pd.DataFrame(col0_new)
col0_new = col0_new.append({0:'Y'}, ignore_index=True)
col0_new = col0_new.append({0:' '}, ignore_index=True)
col0_new.index = rownames1


df = pd.concat([col0,col0_new], axis=1, join='inner')
df.columns=['(1)','(2)']

## Output to LaTeX folder-----------------------------------------------------------------------------------------------------
os.chdir(outputpath) # Output directly to LaTeX folder

df.to_latex('coeftable.tex') 
