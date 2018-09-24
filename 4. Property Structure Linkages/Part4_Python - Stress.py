"""
@author: Ankita
Code for reading C11 and ys values from files

Input Files : Folder yield_stress_data
"""

import glob
import os
import pandas as pd
import numpy as np
from pymks.stats import correlate
from pymks import PrimitiveBasis

## Class 1
vec = np.zeros(shape = (1,2))
os.chdir("F:/Studies/DDP/yield_stress_data/10_10_10_1_data/")
i = 0
for file in glob.glob("*.dat"):
    print(i)
    df = pd.read_csv(file,header = None)
    df['empty'],df['yield_stress'],df['empty1'],df['empty2'], df['c11'] = df[0][0].split(" ")
    df.drop(df.columns[[0,1,3,4]],axis =1, inplace = True)
    df[['yield_stress','c11']] = df[['yield_stress','c11']].apply(pd.to_numeric)
    vec = np.concatenate([vec, df],axis = 0)
    i = i+1

## Class 2
os.chdir("F:/Studies/DDP/yield_stress_data/15_15_15_2_data/")
for file in glob.glob("*.dat"):
    print(i)
    df = pd.read_csv(file,header = None)
    df['empty'],df['yield_stress'],df['empty1'],df['empty2'], df['c11'] = df[0][0].split(" ")
    df.drop(df.columns[[0,1,3,4]],axis =1, inplace = True)
    df[['yield_stress','c11']] = df[['yield_stress','c11']].apply(pd.to_numeric)
    vec = np.concatenate([vec, df],axis = 0)
    i = i+1

## Class 3 
os.chdir("F:/Studies/DDP/yield_stress_data/20_10_10_3_data/")
for file in glob.glob("*.dat"):
    print(i)
    df = pd.read_csv(file,header = None)
    df['empty'],df['yield_stress'],df['empty1'],df['empty2'], df['c11'] = df[0][0].split(" ")
    df.drop(df.columns[[0,1,3,4]],axis =1, inplace = True)
    df[['yield_stress','c11']] = df[['yield_stress','c11']].apply(pd.to_numeric)
    vec = np.concatenate([vec, df],axis = 0)
    i = i+1

## Class 4
os.chdir("F:/Studies/DDP/yield_stress_data/20_20_10_4_data/")
for file in glob.glob("*.dat"):
    print(i)
    df = pd.read_csv(file,header = None)
    df['empty'],df['yield_stress'],df['empty1'],df['empty2'], df['c11'] = df[0][0].split(" ")
    df.drop(df.columns[[0,1,3,4]],axis =1, inplace = True)
    df[['yield_stress','c11']] = df[['yield_stress','c11']].apply(pd.to_numeric)
    vec = np.concatenate([vec, df],axis = 0)
    i = i+1

## Class 5
os.chdir("F:/Studies/DDP/yield_stress_data/50_5_5_5_data/")
for file in glob.glob("*.dat"):
    print(i)
    df = pd.read_csv(file,header = None)
    df['empty'],df['yield_stress'],df['empty1'],df['empty2'], df['c11'] = df[0][0].split(" ")
    df.drop(df.columns[[0,1,3,4]],axis =1, inplace = True)
    df[['yield_stress','c11']] = df[['yield_stress','c11']].apply(pd.to_numeric)
    vec = np.concatenate([vec, df],axis = 0)
    i = i+1

## Class 6
os.chdir("F:/Studies/DDP/yield_stress_data/20_5_2.5_6_data/")
for file in glob.glob("*.dat"):
    print(i)
    df = pd.read_csv(file,header = None)
    df['empty'],df['yield_stress'],df['empty1'],df['empty2'], df['c11'] = df[0][0].split(" ")
    df.drop(df.columns[[0,1,3,4]],axis =1, inplace = True)
    df[['yield_stress','c11']] = df[['yield_stress','c11']].apply(pd.to_numeric)
    vec = np.concatenate([vec, df],axis = 0)
    i = i+1

## Class 7 
os.chdir("F:/Studies/DDP/yield_stress_data/20_20_5_7_data/")
for file in glob.glob("*.dat"):
    print(i)
    df = pd.read_csv(file,header = None)
    df['empty'],df['yield_stress'],df['empty1'],df['empty2'], df['c11'] = df[0][0].split(" ")
    df.drop(df.columns[[0,1,3,4]],axis =1, inplace = True)
    df[['yield_stress','c11']] = df[['yield_stress','c11']].apply(pd.to_numeric)
    vec = np.concatenate([vec, df],axis = 0)
    i = i+1

## Class 8
os.chdir("F:/Studies/DDP/yield_stress_data/30_10_5_8_data/")
for file in glob.glob("*.dat"):
    print(i)
    df = pd.read_csv(file,header = None)
    df['empty'],df['yield_stress'],df['empty1'],df['empty2'], df['c11'] = df[0][0].split(" ")
    df.drop(df.columns[[0,1,3,4]],axis =1, inplace = True)
    df[['yield_stress','c11']] = df[['yield_stress','c11']].apply(pd.to_numeric)
    vec = np.concatenate([vec, df],axis = 0)
    i = i+1

os.chdir("F:/Studies/DDP/Final Codes/")
    
vec = vec[1:]
print(vec.shape)
del file
del i
del df