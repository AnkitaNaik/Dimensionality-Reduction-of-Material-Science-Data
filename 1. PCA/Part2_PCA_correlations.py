"""
@author: Ankita

Input File : correaltions_final.csv
"""
import pickle
import os
from sklearn import decomposition
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale

os.chdir("/home/Aaditya/Desktop/Ankita/")
Y_con = pd.read_csv('correlations_final.csv',index_col = False,header = None)
Y_con = Y_con.T
Y_con = Y_con.iloc[:,0:250047]
print('Loaded')
os.chdir("/home/Aaditya/Desktop/Ankita/Datasets/New_5/")
l = [2,3,5,10,15,20,25,30,35,40,45,50,60,70,80,90,100]
#l = [50]
l = list(l)
for i in l:
    print(i)
    pca = decomposition.PCA(n_components = i)
    pca.fit(Y_con)
    Z = pca.transform(Y_con)
    print('Transform Done')
    a = np.concatenate(([1]*50,[2]*50,[3]*50,[4]*50,[5]*50,[6]*50,[7]*50,[8]*50), axis = 0).reshape(400,1)
    mydf = np.concatenate((Z,a),axis = 1)
    mydf = pd.DataFrame(mydf)
    file = 'python_'+str(i)+'_corr.csv'
    mydf.to_csv(file, sep=',',index = False,index_label = False)
    print(i)