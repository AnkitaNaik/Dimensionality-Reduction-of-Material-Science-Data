"""
@author: Ankita
Code for using WLS for setting p Structure-Property Linkages

Input Files : python_50_corr.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

os.chdir("F:/Studies/DDP/Final Codes/Final Datasets/")
mydf = pd.read_csv('python_50_corr.csv',sep = ',')
a = mydf.iloc[:,0:50]
del mydf
s = s.reshape(400,1)
a = np.concatenate((a,s),axis = 1)
a = pd.DataFrame(a)

# C11
k = range(5,35,1)
m = range(5,35,1)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['legend.numpoints'] = 1
plt.rcParams['legend.markerscale'] = 2

pc = 50
l = range(0,400,50)
c11 = []
colors = ['blue','green','red','cyan','magenta','yellow','black','purple']
classes = [1,2,3,4,5,6,7,8]
j = 0

for i in l:
    X = a.iloc[i:(i+50),:]
    y = vec[i:(i+50),0]/1000.0
    y = y.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=63)
    w = X_train.iloc[:,50]
    X_train = X_train.iloc[:,0:49]
    X_test = X_test.iloc[:,0:49]
    mod_wls = sm.WLS(y_train,X_train,weights = w)
    res_wls = mod_wls.fit()
    z = res_wls.t_test(X_test)
    q = np.dot(X_test,res_wls.params)
    ax.plot(q,y_test,'o',markersize = 10,color =str(colors[j]),label = 'class'+str(classes[j]))
    print('c11',classes[j])
    c11 = np.concatenate([c11, q],axis = 0)
    j = j+1
ax.plot(k,m,color ='red')
ax.legend(loc='lower right')
plt.xlabel(r'Reduced Order Model $C_{11}$ (GPa)',fontsize = 18,fontweight = 'bold')
plt.ylabel(r'FFT $C_{11}$ (GPa)',fontsize = 18,fontweight = 'bold')
plt.tick_params(labelsize=14)
#plt.title('Accuracy of Reduced Order C11 Model')
plt.show()
print(max(y_test))

del [X,y,X_train,y_train,y_test,X_test,w,mod_wls,res_wls,z,q,i,k,m]

# yield stress
k = range(10,70,1)
m = range(10,70,1)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['legend.numpoints'] = 1
plt.rcParams['legend.markerscale'] = 2
ys = []
j = 0
for i in l:    
    X = a.iloc[i:(i+50),:]
    y = vec[i:(i+50),1]
    y = y.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=63)
    w = X_train.iloc[:,50]
    X_train = X_train.iloc[:,0:49]
    X_test = X_test.iloc[:,0:49]
    mod_wls = sm.WLS(y_train,X_train,weights = w)
    res_wls = mod_wls.fit()
    z = res_wls.t_test(X_test)
    q = np.dot(X_test,res_wls.params)
    print('ys',classes[j])
    ys = np.concatenate([ys, q],axis = 0)
    ax.plot(q,y_test,'o',markersize = 10,color =str(colors[j]),label = 'class'+str(classes[j]))
    j = j+1
    
ax.plot(k,m,color ='red')
ax.legend(loc='lower right')
plt.xlabel('Reduced Order Model Yield Stress (GPa)',fontsize = 18,fontweight = 'bold')
plt.ylabel('FFT Yield Stress (GPa)',fontsize = 18,fontweight = 'bold')
plt.tick_params(labelsize=14)
#plt.title('Accuracy of Reduced Order Yield Stress Model')
plt.show()

del [X,y,X_train,y_train,y_test,X_test,w,mod_wls,res_wls,z,q,i]
