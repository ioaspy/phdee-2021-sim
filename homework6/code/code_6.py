# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:00:33 2021

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

import os
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Set working directories and seed---------------------------------------------------------------------------------------------

outputpath = r'C:\Users\ioasp\Desktop\Ioanna\Economics\gatech courses\Spring 2021\Environment II\GitHub\phdee-2021-sim\homework6\output'

np.random.seed(6578103)

#  Import data ----------------------------------------------------------------------------------------------------------------

data = pd.read_csv('fishbycatch.csv')

#Question 1--------------------------------------------------------------------------------------------------------------------

data['id'] = data.index
data_long = pd.wide_to_long(data, ['shrimp','salmon','bycatch'], i='id',j='month' )

l1=[]
for i in range (1200):
    l1.append(i)
data_long = data_long.set_index([pd.Index(l1)])

#New dataframe for treat months
list_tre_new = []

for i in range(600):
    list_tre_new.append(0)

for i in range(600,1200):
    list_tre_new.append(1)
    
    
s_tre_new =pd.Series(list_tre_new)  #series  

data_all1= pd.concat([data_long,s_tre_new], axis=1)
data_all1 = data_all1.rename(columns={0:'tre'})

#create treated variable
l2=[]

for i in range(1200):
    l2.append(0)
    
l2_s=pd.Series(l2)

data_all1 = pd.concat([data_all1,l2_s], axis=1)
data_all1 = data_all1.rename(columns={0:'new'})

data_all1.loc[(data_all1['treated']==1) & (data_all1['tre']==1), 'new'] = 1

#indicator variables for each firm
dummy = pd.get_dummies(data_all1['firm'])

Data = pd.concat([data_all1,dummy], axis=1, join='inner')

#get dataframes for ols 
Y=Data.iloc[:,5]
X=Data.drop(['bycatch','firm'], axis=1)
#dataframes to arrays
Y=Y.to_numpy()
X=X.to_numpy()


results1 = sm.OLS(Y,X).fit()
print(results1.summary())
res1=results1.params
se1=results1.bse

res1=[res1[0],res1[1],res1[2],res1[3],res1[5]]
se1=[se1[0],se1[1],se1[2],se1[3],se1[5]]

res1=pd.Series(res1)
se1=pd.Series(se1)

## Format means and std devs to display to two decimal places
res1 = res1.map('{:.2f}'.format)
se1 = se1.map('({:.2f})'.format)


#Question 2--------------------------------------------------------------------------------------------------------------

data1 = pd.read_csv('fishbycatch.csv')

data1['id'] = data1.index

data1['meanshrimp'] = data1.iloc[:,3:26].mean(axis=1)
data1['meansalmon'] = data1.iloc[:,27:50].mean(axis=1)
data1['meanbycatch'] = data1.iloc[:,51:74].mean(axis=1)
data_long1 = pd.wide_to_long(data1, ['shrimp','salmon','bycatch'], i='id',j='month' )

l3=[]
for i in range (1200):
    l3.append(i)
data_long1 = data_long1.set_index([pd.Index(l3)])

#New dataframe for treatment months
list_tre_new1 = []

for i in range(600):
    list_tre_new1.append(0)

for i in range(600,1200):
    list_tre_new1.append(1)
    
    
s_tre_new1 =pd.Series(list_tre_new1)   

data_all2= pd.concat([data_long1,s_tre_new1], axis=1)
data_all2 = data_all2.rename(columns={0:'Tre'})

l4=[]

for i in range(1200):
    l4.append(0)
    
l4_s=pd.Series(l4)

data_all2 = pd.concat([data_all2,l4_s], axis=1)
data_all2 = data_all2.rename(columns={0:'New'})

#variable for treated
data_all2.loc[(data_all2['treated']==1) & (data_all2['Tre']==1), 'New'] = 1

data_all2['Bycatch'] = data_all2['bycatch']-data_all2['meanbycatch']
data_all2['Shrimp'] = data_all2['shrimp']-data_all2['meanshrimp']
data_all2['Salmon'] = data_all2['salmon']-data_all2['meansalmon']

dummy1 = pd.get_dummies(data_all2['firm'])

Data2 = pd.concat([data_all2,dummy1], axis=1, join='inner')

Y1=Data2.iloc[:,11]
X1=Data2.drop(['meanbycatch','meansalmon','meanshrimp','Bycatch','firm','shrimp','salmon','bycatch'], axis=1)

Y1=Y1.to_numpy()
X1=X1.to_numpy()


results2 = sm.OLS(Y1,X1).fit()
print(results2.summary())
res2=results2.params
se2=results2.bse

res2=[res2[0],res2[1],res2[4],res2[5],res2[3]]
se2=[se2[0],se2[1],se2[4],se2[5],se2[3]]

res2=pd.Series(res2)
se2=pd.Series(se2)


## Format means and std devs to display to two decimal places
res2 = res2.map('{:.2f}'.format)
se2 = se2.map('({:.2f})'.format)


#Question3---------------------------------------------------------------------------------------------------------- 

# Generate a table of coefficients and standard errors for the observed variables 

## Set the row and column names
rownames1 = pd.concat([pd.Series(['Firmsize', 'Treatment group', 'Shrimp','Salmon','Treated']),pd.Series([' ',' ',' ',' ',' '])],axis = 1).stack() 


## Align se under coefficients
col0 = pd.concat([res1,se1],axis = 1).stack()
col0_new = pd.concat([res2,se2],axis = 1).stack()

## Add column and row labels.  Convert to dataframe (helps when you export it)
col0 = pd.DataFrame(col0)
col0.index = rownames1


## Add column and row labels.  Convert to dataframe (helps when you export it)
col0_new = pd.DataFrame(col0_new)
col0_new.index = rownames1


df = pd.concat([col0,col0_new], axis=1, join='inner')
df.columns=['(a)','(b)']

## Output to LaTeX folder-----------------------------------------------------------------------------------------------------
os.chdir(outputpath) # Output directly to LaTeX folder

df.to_latex('coeftable.tex') 