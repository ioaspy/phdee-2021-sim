# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 22:39:37 2021

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


# Set working directories and seed---------------------------------------------------------------------------------------------

outputpath = r'C:\Users\ioasp\Desktop\Ioanna\Economics\gatech courses\Spring 2021\Environment II\GitHub\phdee-2021-sim\homework3\output'

np.random.seed(6578103)

#  Import data ----------------------------------------------------------------------------------------------------------------

data = pd.read_csv('kwh.csv')

##Question 1(e)

##lists
lY = data['electricity'].tolist()
#ln(Y)
lnY = np.log(lY).tolist()

lx2 = data['sqft'].tolist()
lx3 = data['retrofit'].tolist()
lx4 = data['temp'].tolist()

#ln(sqft) and ln(temp)
lnx2 = np.log(lx2).tolist()
lnx4 = np.log(lx4).tolist()

data_new = [lnY, lnx2, lx3, lnx4]

#create new DataFrame
data1 = pd.DataFrame(data_new)
data1 = data1.transpose()

#Give columns names
data1.columns = ['ln(electricity)','ln(sqft)','retrofit','ln(temp)']

# Fit a linear regression model to the data ----------------------------------

ols = sm.OLS(data1['ln(electricity)'],sm.add_constant(data1.drop('ln(electricity)',axis = 1))).fit()
betaols = ols.params.to_numpy() # save estimated parameters
params, = np.shape(betaols) # save number of estimated parameters
nobs3 = int(ols.nobs)

#Get average marginal effects estimates
l1 = []
l2 = []
for i in range(1000):
    y_z2 = betaols[1]*lnY[i]/lnx2[i]
    y_z4 = betaols[3]*lnY[i]/lnx4[i]
    l1.append(y_z2)
    l2.append(y_z4)
    
delta = np.exp(betaols[2])

l3 = []
for i in range(1000):
    a = np.power(delta, lx3[i])
    d_i = lnY[i]*(delta - 1)/a
    l3.append(d_i)

#Average marginal effect    
ame_sqft = np.mean(l1)
ame_temp = np.mean(l2)
ame_retrofit = np.mean(l3)

arr = np.array([ame_sqft, ame_retrofit, ame_temp]) # save average marginal effects estimates


# Bootstrap by hand and get confidence intervals -----------------------------
## Set values and initialize arrays to output to
breps = 1000 # number of bootstrap replications
olsbetablist = np.zeros((breps,params))

## Get an index of the data we will sample by sampling with replacement
bidx = np.random.choice(nobs3,(nobs3,breps)) # Generates random numbers on the interval [0,nobs3] and produces a nobs3 x breps sized array


## Sample with replacement to get the size of the sample on each iteration
for r in range(breps):
    ### Sample the data
    datab = data1.iloc[bidx[:,r]]
    
    ### Perform the estimation
    olsb = sm.OLS(datab['ln(electricity)'],sm.add_constant(datab.drop('ln(electricity)',axis = 1))).fit()
    
    ### Output the result
    olsbetablist[r,:] = olsb.params.to_numpy()

    

# ---------------------------------------------------------------------------------------------------






##Get mean values and CI 

#Create lists of coefficients
list_1 = []
list_2 = []
list_3 = []
list_4 = []
for i in range(breps):
    list_1.append(olsbetablist[i,0])
    list_2.append(olsbetablist[i,1]) 
    list_3.append(olsbetablist[i,2])
    list_4.append(olsbetablist[i,3])

#Mean values of coefficients
coef1_mean = np.mean(list_1)
coef2_mean = np.mean(list_2)
coef3_mean = np.mean(list_3)
coef4_mean = np.mean(list_4)

coeff_list = [coef1_mean, coef2_mean, coef3_mean, coef4_mean]
coeff_s = np.array(coeff_list)




l1_s = []
l2_s = []
l3_s = []
#Marginal effects estimates lists
for i in range(breps):    
    delta_s = np.exp(list_3[i])
    y_z2_s = list_2[i]*datab.iat[i,0]/datab.iat[i,1]
    y_z4_s = list_4[i]*datab.iat[i,0]/datab.iat[i,3]
    l1_s.append(y_z2_s) # sqft
    l2_s.append(y_z4_s) # temp
    a_s = np.power(delta_s, datab.iat[i,2])
    d_i_s = datab.iat[i,0]*(delta_s - 1)/a_s
    l3_s.append(d_i_s) # retrofit
    
    
list_all = [l1_s, l2_s, l3_s]
list_all = np.array(list_all)
list_all = np.transpose(list_all)
    
#Mean values of marginal effects estimates
ame_sqft_mean = np.mean(l1_s)
ame_temp_mean = np.mean(l2_s)
ame_retrofit_mean = np.mean(l3_s)

all_list = [coef1_mean, coef2_mean, coef3_mean, coef4_mean, ame_sqft_mean, ame_retrofit_mean, ame_temp_mean]
all_list = np.asarray(all_list)

allt = np.concatenate((olsbetablist, list_all), axis = 1)
## Extract 2.5th and 97.5th percentile
# for coefficients
lb_s = np.percentile(allt,2.5,axis = 0,interpolation = 'lower')
ub_s = np.percentile(allt,97.5,axis = 0,interpolation = 'higher')


#Table
## Format estimates and confidence intervals
all_list = np.round(all_list,2)

lb_s_P = pd.Series(np.round(lb_s,2)) # Round to two decimal places and get a Pandas Series version
ub_s_P = pd.Series(np.round(ub_s,2))
ci = '(' + lb_s_P.map(str) + ', ' + ub_s_P.map(str) + ')'


#--------------------------------------------------------------------------------------------------


## Get output in order
order1 = [1,2,3,0]
c_1_mean=all_list[0:4][order1]
c_1_ci=ci[0:4][order1]
c_2_ame=all_list[4:7]
c_2_ci=ci[4:7]
output1 = pd.DataFrame(np.column_stack([c_1_mean,c_1_ci]))
output2=pd.DataFrame(np.column_stack([c_2_ame,c_2_ci]))

## Row and column names
rownames = pd.concat([pd.Series(['sqft','retrofit','temp','constant']),pd.Series([' ',' ',' ',' '])],axis = 1).stack() # Note this stacks an empty list to make room for CIs
colname1 = ['Coefficient estimates']

## Append CIs, # Observations, row and column names
output1 = pd.DataFrame(output1.stack())
output1.index = [0,1,2,3,4,5,6,7]
output1.columns = colname1

#---------------------------------------------------------------------------------------------------

colname2=['Marginal Effect Estimates']
## Append CIs, # Observations, row and column names
output2 = pd.DataFrame(output2.stack())
output2.index = [0,1,2,3,4,5]
output2.columns = colname2

## Output directly to LaTeX
output=pd.concat([output1,output2],axis=1)
output.index=rownames
output.to_latex('sampleoutput.tex')

array_new = np.asarray([ame_sqft_mean, ame_temp_mean])
array1 =np.asarray(l1_s)
array2 =np.asarray(l2_s)
array_ci = np.vstack((array1, array2))
array_ci = np.transpose(array_ci)
lb_new = np.percentile(array_ci,2.5,axis = 0,interpolation = 'lower')
ub_new = np.percentile(array_ci,97.5,axis = 0,interpolation = 'higher')

# Plot regression output with error bars -------------------------------------
lowbar = np.array(array_new - lb_new)
highbar = np.array(ub_new - array_new)
plt.errorbar(y = array_new, x = np.arange(2), yerr = [lowbar,highbar], fmt = 'o', capsize = 5)
plt.ylabel('Estimate size')
plt.xticks(np.arange(2),['sqft', 'temp'])
plt.xlim((-0.5,2.5)) # Scales the figure more nicely
plt.axhline(linewidth=2, color='r')
plt.savefig('samplebars.eps',format='eps')
plt.show()
