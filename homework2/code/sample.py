
"""
Economics 7103
Homework 2
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

# Set working directories and seed---------------------------------------------------------------------------------------------

outputpath = r'C:\Users\ioasp\Desktop\Ioanna\Economics\gatech courses\Spring 2021\Environment II\GitHub\phdee-2021-sim\homework2\output'
np.random.seed(6578103)

#  Import data ----------------------------------------------------------------------------------------------------------------

data = pd.read_csv('kwh.csv')

#Question 1--------------------------------------------------------------------------------------------------------------------
# Split data to control and treatment------------------------------------------------------------------------------------------
control = data[data['retrofit']==0]
control_data = control[['electricity','sqft','temp']]
treatment = data[data['retrofit']==1]
treatment_data = treatment[['electricity','sqft','temp']]


# Generate a table of means, standard deviations and p-values-----------------------------------------------------------------
## Generate means-------------------------------------------------------------------------------------------------------------
means_control = control_data.mean()
means_treatment = treatment_data.mean()

## Generate standard deviations-----------------------------------------------------------------------------------------------
stdev_control = control_data.std()
stdev_treatment = treatment_data.std()

##Generate p-values-----------------------------------------------------------------------------------------------------------
stat, p = ttest_ind(control_data, treatment_data)
print('p = ' + str(p))
p_values = pd.Series(p)
blank = np.array([' ',' ',' '])
blank_s = pd.Series(blank)

## Set the row and column names-----------------------------------------------------------------------------------------------
rownames = pd.concat([pd.Series(['electricity','sqft','temp']),pd.Series([' ',' ',' '])],axis = 1).stack()
colnames = [('control mean','(s.d.)')]
colnames1 = [('treated mean','(s.d)')]
colnames2 = [('p-values',' ')] 

## Format means and std devs to display to two decimal places-----------------------------------------------------------------
means_control = means_control.map('{:.2f}'.format)
stdev_control = stdev_control.map('({:.2f})'.format)
means_treatment = means_treatment.map('{:.2f}'.format)
stdev_treatment = stdev_treatment.map('({:.2f})'.format)
p_values = p_values.map('({:.2f})'.format)

## Align std deviations under means and add observations----------------------------------------------------------------------
col0 = pd.concat([means_control,stdev_control],axis = 1).stack()
col1 = pd.concat([means_treatment,stdev_treatment],axis = 1).stack()
col2 = pd.concat([p_values,blank_s],axis = 1).stack()

## Add column and row labels and convert to dataframe------------------------------------------------------------------------- 
col0 = pd.DataFrame(col0)
col0.index = rownames
col0.columns = pd.MultiIndex.from_tuples(colnames)

col1 = pd.DataFrame(col1)
col1.index = rownames
col1.columns = pd.MultiIndex.from_tuples(colnames1)

col2 = pd.DataFrame(col2)
col2.index = rownames
col2.columns = pd.MultiIndex.from_tuples(colnames2)

#Table with means and p-values------------------------------------------------------------------------------------------------
df = pd.concat([col0,col1,col2], axis=1, join='inner')

## Output to LaTeX folder-----------------------------------------------------------------------------------------------------
os.chdir(outputpath) # Output directly to LaTeX folder

df.to_latex('meantable.tex') 


#-----------------------------------------------------------------------------------------------------------------------------

#Question 2------------------------------------------------------------------------------------------------------------------- 
#Dataframes of electricity for the two groups to plot-------------------------------------------------------------------------
electricity_control = control_data['electricity']
electricity_treated = treatment_data['electricity']

e_control = pd.Series.tolist(electricity_control)
e_treated = pd.Series.tolist(electricity_treated)

# Plot kernel density plot of electricity for the two groups -----------------------------------------------------------------

# We can use the following:
sns.distplot(e_control,hist =False,label = 'control')
sns.distplot(e_treated,hist =False,label = 'treated')
plt.xlabel('Electricity use')
plt.savefig('plot.eps',format='eps')
plt.title('Distribution of electricity use')
plt.legend() 
plt.show()


# Or we can use the following, but we lose two observations:
df_both = pd.DataFrame(list(zip(e_control,e_treated)), columns=['Control','Treated'])
figure = df_both.plot.kde()
plt.xlabel('electricity use')
plt.xlim(left=0.0)
plt.savefig('plot1.eps',format='eps')
plt.title('Distribution of electricity use')
plt.show()

#Question 3-------------------------------------------------------------------------------------------------------------------
##a)OLS by hand---------------------------------------------------------------------------------------------------------------
#Create arrays
Y = data[['electricity']].to_numpy()
x1 = np.full((1000,1),1.)
x2 = data[['sqft']].to_numpy()
x3 = data[['retrofit']].to_numpy()
x4 = data[['temp']].to_numpy()

#Calculate beta 
X = np.concatenate((x1,x2,x3,x4),axis=1)
X_T = X.transpose()
A = np.dot(X_T,X)
 
A_inverse = np.linalg.inv(A)
B = np.dot(X_T,Y)
beta_q3a = np.dot(A_inverse,B)

print("\n\nBeta coefficients-Question 3rd-part A:\n")
print("beta_0 = ", beta_q3a[0])
print("beta_sqft = ", beta_q3a[1])
print("beta_retrofit = ", beta_q3a[2])
print("beta_temp = ", beta_q3a[3],"\n\n")



##b)OLS by simulated least squares--------------------------------------------------------------------------------------------
def sumOLS(beta):
    ''' Input vector beta is expected to be (4,)
    where beta[0] is the constant coefficient
    beta[1] is the coefficient of sqft
    beta[2] is the coefficient of retrofit
    beta[3] is the coefficient of temp'''
    S=0.0
    for i in range(len(Y)):
        S+=(Y[i]-beta[0]-beta[1]*x2[i]-beta[2]*x3[i]-beta[3]*x4[i])**2 
        # Another way : S+=(Y[i]-np.dot(beta.transpose(),X[i,:]))**2
    # WITHOUT the for loop
    # i) Another way would be: S=(np.linalg.norm( Y - np.dot(X,beta)  ))**2
    # ii) Another way would also be: S=np.sum( np.square( Y - np.dot(X,beta) )  ))  )
    return S

# Initial guess of beta values
beta_q3b_0 = np.random.uniform(size=(4,))
res=minimize(sumOLS, beta_q3b_0)
beta_q3b = res.x

print(res.message)
print("\n\nBeta coefficients-Question 3rd-part B:\n")
print("beta_0 = ", beta_q3b[0])
print("beta_sqft = ", beta_q3b[1])
print("beta_retrofit = ", beta_q3b[2])
print("beta_temp = ", beta_q3b[3],"\n\n")



##c)OLS-----------------------------------------------------------------------------------------------------------------------
model = sm.OLS(Y,X)
results = model.fit()
print(results.summary())
