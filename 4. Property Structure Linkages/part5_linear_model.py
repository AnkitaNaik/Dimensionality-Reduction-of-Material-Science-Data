"""
@author: Aaditya
Code for using linear regression for setting p Structure-Property Linkages

Input Files : python_50_corr.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt

os.chdir("F:/Studies/DDP/Final Codes/Final Datasets/")
mydf = pd.read_csv('python_50_corr.csv',sep = ',')
a = mydf.iloc[:,0:50]
del mydf

# C11
k = range(15,32,1)
m = range(15,32,1)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['legend.numpoints'] = 1
plt.rcParams['legend.markerscale'] = 2

regr = linear_model.LinearRegression()
pc = 50
l = range(0,400,50)
c11 = []
colors = ['blue','green','red','cyan','magenta','yellow','black','purple']
classes = [1,2,3,4,5,6,7,8]
j = 0
for i in l:
    X = a.iloc[i:(i+50),0:pc]
    y = vec[i:(i+50),0]/1000.0
    y = y.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=63)
    mod_wls = regr.fit(X_train,y_train)
    res_wls = regr.predict(X_test)
    q = res_wls
    ax.plot(q,y_test,'o',markersize = 10,color =str(colors[j]),label = 'class'+str(classes[j]))
    print('c11',classes[j])
    c11 = np.concatenate([c11, q],axis = 0)
    j = j+1
ax.plot(k,m,color ='red')
ax.legend(loc='lower right')
ax.set_xlim([15,32])
ax.set_ylim([15,32])
plt.xlabel(r'Reduced Order Model $C_{11}$ (GPa)',fontsize = 18,fontweight = 'bold')
plt.ylabel(r'FFT $C_{11}$ (GPa)',fontsize = 18,fontweight = 'bold')
plt.tick_params(labelsize=14)
#plt.title('Accuracy of Reduced Order C11 Model')
plt.show()
#print(max(q))
#print(max(y_test))

#del [X,y,X_train,y_train,y_test,X_test,w,mod_wls,res_wls,z,q,i,k,m]

# yield stress
k = range(15,70,1)
m = range(15,70,1)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['legend.numpoints'] = 1
plt.rcParams['legend.markerscale'] = 2
ys = []
j = 0
for i in l:    
    X = a.iloc[i:(i+50),0:pc]
    y = vec[i:(i+50),1]
    y = y.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=63)
    mod_wls = regr.fit(X_train,y_train)
    res_wls = regr.predict(X_test)
    q = res_wls
    print('ys',classes[j])
    ys = np.concatenate([ys, q],axis = 0)
    ax.plot(q,y_test,'o',markersize = 10,color =str(colors[j]),label = 'class'+str(classes[j]))
    j = j+1
    
ax.plot(k,m,color ='red')
ax.legend(loc='lower right')
ax.set_xlim([15,70])
ax.set_ylim([15,70])
plt.xlabel('Reduced Order Model Yield Stress (GPa)',fontsize = 18,fontweight = 'bold')
plt.ylabel('FFT Yield Stress (GPa)',fontsize = 18,fontweight = 'bold')
plt.tick_params(labelsize=14)
#plt.title('Accuracy of Reduced Order Yield Stress Model')
plt.show()

#del [X,y,X_train,y_train,y_test,X_test,w,mod_wls,res_wls,z,q,i]