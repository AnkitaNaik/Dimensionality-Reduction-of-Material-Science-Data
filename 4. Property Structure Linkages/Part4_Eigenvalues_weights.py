"""
@author: Ankita
Code for getting the eigen values to be used as weights in weighted least square method

Input File : correlations_final.csv
"""

import os
import pandas as pd
import numpy as np
from math import sqrt

os.chdir("F:/Studies/DDP/Final Codes/Final Datasets/")
Y_con = pd.read_csv('correlations_final.csv',index_col = False,header = None)
Y_con = Y_con.T

Y_con = Y_con.iloc[:,0:250047].T
print('Loaded')

# cov_mat = np.cov(Y_con)  

s = np.linalg.svd(Y_con,compute_uv = False,full_matrices=True)
del Y_con