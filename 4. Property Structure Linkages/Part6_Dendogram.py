__author__ = 'Ankita'
'''
Code for constructing the dendogram.

Input Files : Folder Corr_datasets_final_rect_classes
'''

import os
import pandas as pd
import numpy as np

## Number of PC's should be less than the number of classes which constructing the dendogram
# Calculating the centroid
pc = 5
os.chdir("F:/Studies/DDP/Final Codes/Final Datasets/Corr_datasets_final_rect_classes/")
mydf = pd.read_csv('python_'+str(pc)+'_corr.csv',sep = ',')
a = mydf.iloc[:,0:pc]
del mydf
a = a.T
l = range(0,400,50)
X = np.zeros((pc,1))

for i in l:
    cls = a.iloc[0:50,i:(i+50)]
    X = np.concatenate((X,(np.mean(cls,axis = 1)).reshape(pc,1)),axis = 1)
X = X[:,1:9]
X = X.T    
    

##############################################################################
## Constructing the dendogram
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

Z = linkage(X, 'complete','mahalanobis')

plt.figure(figsize=(25, 10))
plt.title('Cluster Tree Hierarchy',fontsize = 20,fontweight = 'bold')
plt.xlabel('Class Index',fontsize = 20,fontweight = 'bold')
plt.ylabel('Expected Distance',fontsize = 20,fontweight = 'bold')
dendrogram(Z,leaf_font_size=16.,labels = ['1','2','3','4','5','6','7','8'])
plt.tick_params(labelsize=16)
plt.show()