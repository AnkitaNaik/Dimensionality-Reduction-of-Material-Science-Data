"""
author : Ankita
Code for using SVM linear kernel for setting up Structure-Property Linkages

Input Files : python_50_corr.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt


os.chdir("F:/Studies/DDP/Final Codes/Final Datasets/")
mydf = pd.read_csv('python_50_corr.csv',sep = ',')
a = mydf.iloc[:,0:50]
del mydf

tuned_parameters = [{'kernel': ['linear'], 'C': [0.000001,0.00001,0.0001,0.001,0.1,1,10,100,1000]}]
                    
# C11
k = range(15,32,1)
m = range(15,32,1)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['legend.numpoints'] = 1
plt.rcParams['legend.markerscale'] = 2

pc = 50
l = range(0,400,50)
c11 = []
y_axis =[]
colors = ['blue','green','red','cyan','magenta','yellow','black','purple']
classes = [1,2,3,4,5,6,7,8]
j = 0
for i in l:
    X = a.iloc[i:(i+50),0:pc]
    y = vec[i:(i+50),0]/1000.0
    #y = y.astype(float)
    w = s[0:50].T
    X = pd.DataFrame.as_matrix(X)    
    X = X*w
    X = pd.DataFrame(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)
    clf = GridSearchCV(SVR(), tuned_parameters, cv=3)
    clf.fit(X_train, y_train)
    print('Tuning Completed')
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print("Grid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    y_true, y_pred = y_test, clf.predict(X_test)
    #y_pred = y_pred.astype(int)
    ax.plot(y_pred,y_test,'o',markersize = 10,color =str(colors[j]),label = 'class'+str(classes[j]))
    print('c11',classes[j])
    c11 = np.concatenate([c11, y_pred],axis = 0)
    y_axis = np.concatenate([y_axis,y_true],axis = 0)
    j = j+1
ax.plot(k,m,color ='red')
#ax.plot(c11,y_axis,'^',markersize = 8,color ='black',label = 'actual')
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
k = range(25,70,1)
m = range(25,70,1)

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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)
    X = pd.DataFrame.as_matrix(X)    
    X = X*w
    X = pd.DataFrame(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=63)
    clf = GridSearchCV(SVR(), tuned_parameters, cv=3)
    clf.fit(X_train, y_train)
    print('Tuning Completed')
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print("Grid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    y_true, y_pred = y_test, clf.predict(X_test)
    ax.plot(y_pred,y_test,'o',markersize = 10,color =str(colors[j]),label = 'class'+str(classes[j]))
    print('c11',classes[j])
    c11 = np.concatenate([c11, y_pred],axis = 0)
    y_axis = np.concatenate([y_axis,y_true],axis = 0)
    j = j+1
    
ax.plot(k,m,color ='red')
ax.legend(loc='lower right')
ax.set_xlim([25,70])
ax.set_ylim([25,70])
plt.xlabel('Reduced Order Model Yield Stress (GPa)',fontsize = 18,fontweight = 'bold')
plt.ylabel('FFT Yield Stress (GPa)',fontsize = 18,fontweight = 'bold')
plt.tick_params(labelsize=14)
#plt.title('Accuracy of Reduced Order Yield Stress Model')
plt.show()

#del [X,y,X_train,y_train,y_test,X_test,w,mod_wls,res_wls,z,q,i]

