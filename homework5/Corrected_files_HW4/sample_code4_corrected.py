# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:07:47 2021

Economics 7103
Homework 4
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

outputpath = r'C:\Users\ioasp\Desktop\Ioanna\Economics\gatech courses\Spring 2021\Environment II\GitHub'

np.random.seed(6578103)

#  Import data ----------------------------------------------------------------------------------------------------------------

data = pd.read_csv('fishbycatch.csv')

#Question 1--------------------------------------------------------------------------------------------------------------------
#Take control and treated separately----------------------------------------------------------------------------------------------------------

data_con = data.loc[data['treated']==0]
data_tre = data.loc[data['treated']==1]

control_sum = data_con.sum()
treated_sum = data_tre.sum()
l_control = []
l_treated = []
for i in range(51,75):
    l_control.append(control_sum[i])
    l_treated.append(treated_sum[i])
    
month_list = []
for i in range(1,25):
    month_list.append(i)
    
plt.plot(month_list,l_control,label='control')
plt.plot(month_list,l_treated, label='treated')
plt.xlabel('Month')
plt.ylabel('Pounds of bycatch')
plt.legend()
plt.savefig('graph.eps',format='eps')

#--------------------------------------------------------------------------------------------------------------------------------

#Question 2

control_mean = data_con.mean()
treated_mean = data_tre.mean()

l_control_mean_pre = []
l_treated_mean_pre = []
l_control_mean_post = []
l_treated_mean_post = []

for i in range(51,63):
    l_control_mean_pre.append(control_mean[i])
    l_treated_mean_pre.append(treated_mean[i])
    
for i in range(63,75):
    l_control_mean_post.append(control_mean[i])
    l_treated_mean_post.append(treated_mean[i])

    
E_tre_post = statistics.mean(l_treated_mean_post)
E_tre_pre = statistics.mean(l_treated_mean_pre)
E_con_post = statistics.mean(l_control_mean_post)
E_con_pre = statistics.mean(l_control_mean_pre)

did = E_tre_post - E_tre_pre - E_con_post + E_con_pre

#--------------------------------------------------------------------------------------------------------------------------------
#Question 3

data['id'] = data.index
data_new = data[['firm','treated','bycatch12','bycatch13','id']]
data_new_long = pd.wide_to_long(data_new, ['bycatch'], i='id',j='month' )

l=[]
for i in range (100):
    l.append(i)
data_new_long = data_new_long.set_index([pd.Index(l)])

#New dataframe for treat
list_tre = []

for i in range(50):
    list_tre.append(0)

for i in range(50,100):
    list_tre.append(1)
    
    
s_tre =pd.Series(list_tre)    

data_all= pd.concat([data_new_long,s_tre], axis=1)
data_all = data_all.rename(columns={0:'tre'})

l1=[]

for i in range(100):
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


# Generate a table of coefficients and standard deviations for the observed variables 

## Set the row and column names
rownames = pd.concat([pd.Series(['constant','Treated','Pre-period','Treatment group']),pd.Series([' ',' ',' ',' '])],axis = 1).stack() 
colnames = [('Coefficients','(s.e.)')] # Two rows of column names

## Format means and std devs to display to two decimal places
res = res.map('{:.2f}'.format)
se = se.map('({:.2f})'.format)

## Align se under coefficients
col0 = pd.concat([res,se],axis = 1).stack()

## Add column and row labels.  Convert to dataframe (helps when you export it)
col0 = pd.DataFrame(col0)
col0.index = rownames
col0.columns = pd.MultiIndex.from_tuples(colnames)

## Output to LaTeX folder
os.chdir(outputpath) # Output directly to LaTeX folder

col0.to_latex('coef_table_corrected.tex') 

