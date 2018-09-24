"""
@author: Ankita
Code for tuning the SVM and calculating accuracy at different number of PCs

Input File : Folder Corr_datasets_final_rect_classes
"""

import os
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
print('Libraries Imported')

os.chdir("/home/Aaditya/Desktop/Ankita/Datasets/Corr_datasets_final_rect_classes/")

l = [2,3,5,10,15,20,25,30,35,40,45,50,60,70,80,90,100]
l = list(l)
acc = []
train_acc = []
for i in l:
    print(i)
    file = 'python_'+str(i)+'_corr.csv'
    mydf = pd.read_csv(file,sep = ',')
    a = mydf    
    X = a.iloc[:,0:i]
    y = a.iloc[:,i]
    y = y.astype(int)
    y = y.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify= y)
    
    tuned_parameters = [{'kernel': ['linear'], 'C': 10**np.arange(-3.0,3.0,1.0)}]
                         
    scores = ['accuracy']
    ## Tuning SVM for acuracy measure    
    for score in scores:
        print('Tuning Started')
    
        clf = GridSearchCV(SVC(decision_function_shape = 'ovo'), tuned_parameters, cv=5,n_jobs = 5,return_train_score=True, scoring='%s' % score)
        clf.fit(X_train, y_train)
        print('Tuning Completed')
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)    
        
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        train_acc = np.append(train_acc,(means +stds))
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        
        y_true, y_pred = y_test, clf.predict(X_test)
        y_pred = y_pred.astype(int)
        print(classification_report(y_true, y_pred))
        print()
        
        acc_now = sum(np.diagonal(confusion_matrix(y_true, y_pred)))/100.0
        print("Accuracy:",sum(np.diagonal(confusion_matrix(y_true, y_pred)))/100.0)
        acc= np.append(acc,acc_now)