
"""
@author: Ankita
"""

import glob
import os
import pandas as pd
import numpy as np

## Class 1
X_con = np.zeros(shape = (1,262144,1,1))
os.chdir("/home/Aaditya/Desktop/Ankita/micros/10_10_10_1/")
i = 0
for file in glob.glob("*.in"):
    print(i)
    df = pd.read_csv(file,header = None)
    df['x'], df['y'],df['z'],df['values'] = zip(*df[0].map(lambda x: x.split(' ')))
    df.drop(df.columns[[0]], axis=1,inplace = True)
    p = df.applymap(int)
    p = p.iloc[:,3].reshape(1,262144,1,1)
    Y = p
    X_con = np.concatenate([X_con, Y],axis = 0)
    print(X_con.shape)
    i = i+1

## Class 2
os.chdir("/home/Aaditya/Desktop/Ankita/micros/15_15_15_2/")
for file in glob.glob("*.in"):
    print(i)
    df = pd.read_csv(file,header = None)
    df['x'], df['y'],df['z'],df['values'] = zip(*df[0].map(lambda x: x.split(' ')))
    df.drop(df.columns[[0]], axis=1,inplace = True)
    p = df.applymap(int)
    p = p.iloc[:,3].reshape(1,262144,1,1)
    Y = p
    X_con = np.concatenate([X_con, Y],axis = 0)
    print(X_con.shape)
    i = i+1

## Class 3
os.chdir("/home/Aaditya/Desktop/Ankita/micros/20_5_2.5_7/")
for file in glob.glob("*.in"):
    print(i)
    df = pd.read_csv(file,header = None)
    df['x'], df['y'],df['z'],df['values'] = zip(*df[0].map(lambda x: x.split(' ')))
    df.drop(df.columns[[0]], axis=1,inplace = True)
    p = df.applymap(int)
    p = p.iloc[:,3].reshape(1,262144,1,1)
    Y = p
    X_con = np.concatenate([X_con, Y],axis = 0)
    print(X_con.shape)
    i = i+1

## Class 4
os.chdir("/home/Aaditya/Desktop/Ankita/micros/20_10_10_3/")
for file in glob.glob("*.in"):
    print(i)
    df = pd.read_csv(file,header = None)
    df['x'], df['y'],df['z'],df['values'] = zip(*df[0].map(lambda x: x.split(' ')))
    df.drop(df.columns[[0]], axis=1,inplace = True)
    p = df.applymap(int)
    p = p.iloc[:,3].reshape(1,262144,1,1)
    Y = p
    X_con = np.concatenate([X_con, Y],axis = 0)
    print(X_con.shape)
    i = i+1

## Class 5
os.chdir("/home/Aaditya/Desktop/Ankita/micros/20_20_5_8/")
for file in glob.glob("*.in"):
    print(i)
    df = pd.read_csv(file,header = None)
    df['x'], df['y'],df['z'],df['values'] = zip(*df[0].map(lambda x: x.split(' ')))
    df.drop(df.columns[[0]], axis=1,inplace = True)
    p = df.applymap(int)
    p = p.iloc[:,3].reshape(1,262144,1,1)
    Y = p
    X_con = np.concatenate([X_con, Y],axis = 0)
    print(X_con.shape)
    i = i+1

## Class 6
os.chdir("/home/Aaditya/Desktop/Ankita/micros/20_20_10_4/")
for file in glob.glob("*.in"):
    print(i)
    df = pd.read_csv(file,header = None)
    df['x'], df['y'],df['z'],df['values'] = zip(*df[0].map(lambda x: x.split(' ')))
    df.drop(df.columns[[0]], axis=1,inplace = True)
    p = df.applymap(int)
    p = p.iloc[:,3].reshape(1,262144,1,1)
    Y = p
    X_con = np.concatenate([X_con, Y],axis = 0)
    print(X_con.shape)
    i = i+1

## Class 7
os.chdir("/home/Aaditya/Desktop/Ankita/micros/30_10_5_9/")
for file in glob.glob("*.in"):
    print(i)
    df = pd.read_csv(file,header = None)
    df['x'], df['y'],df['z'],df['values'] = zip(*df[0].map(lambda x: x.split(' ')))
    df.drop(df.columns[[0]], axis=1,inplace = True)
    p = df.applymap(int)
    p = p.iloc[:,3].reshape(1,262144,1,1)
    Y = p
    X_con = np.concatenate([X_con, Y],axis = 0)
    print(X_con.shape)
    i = i+1

## Class 8
os.chdir("/home/Aaditya/Desktop/Ankita/micros/50_5_5_6/")
for file in glob.glob("*.in"):
    print(i)
    df = pd.read_csv(file,header = None)
    df['x'], df['y'],df['z'],df['values'] = zip(*df[0].map(lambda x: x.split(' ')))
    df.drop(df.columns[[0]], axis=1,inplace = True)
    p = df.applymap(int)
    p = p.iloc[:,3].reshape(1,262144,1,1)
    Y = p
    X_con = np.concatenate([X_con, Y],axis = 0)
    print(X_con.shape)
    i = i+1
    
os.chdir("/home/Aaditya/Desktop/Ankita/Codes/")
    
X_con = X_con[1:]
print(X_con.shape)
Y = X_con.reshape(X_con.shape[0],X_con.shape[1])