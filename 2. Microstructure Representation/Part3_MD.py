"""
@author: Ankita
Code for plotting the inter and intra class projection onto the microstructure space

Input File : correlations_final.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale,StandardScaler, normalize
from sklearn import decomposition

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from scipy.spatial import ConvexHull

os.chdir("/home/Aaditya/Desktop/Ankita/")
X_con = pd.read_csv('correlations_final.csv',index_col = False,header = None)
X_con = X_con.T
Y = X_con
Y = pd.DataFrame.as_matrix(X_con)

#=============================================================================
## 3-D non-transformed representation

pca = decomposition.PCA(n_components = 3, svd_solver = 'auto')

class1 = Y[0:50,:]
class2 = Y[50:100,:]
class3 = Y[100:150,:]
class4 = Y[150:200,:]
class5 = Y[200:250,:]
class6 = Y[250:300,:]
class7 = Y[300:350,:]
class8 = Y[350:400,:]

pca.fit(class1)
class1 = pca.transform(class1)

pca.fit(class2)
class2 = pca.transform(class2)

pca.fit(class3)
class3 = pca.transform(class3)

pca.fit(class4)
class4 = pca.transform(class4)

pca.fit(class5)
class5 = pca.transform(class5)

pca.fit(class6)
class6 = pca.transform(class6)

pca.fit(class7)
class7 = pca.transform(class7)

pca.fit(class8)
class8 = pca.transform(class8)

fig = plt.figure(figsize=(30,30))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['legend.markerscale'] = 2

ax.plot(class1[:,0], class1[:,1], class1[:,2],'o', markersize=6, color='blue', alpha=0.5, label='class1')

ax.plot(class2[:,0], class2[:,1], class2[:,2],'o', markersize=6, color='red', alpha=0.5, label='class2')

ax.plot(class3[:,0], class3[:,1], class3[:,2],'o', markersize=6, color='cyan', alpha=0.5, label='class3')

ax.plot(class4[:,0], class4[:,1], class4[:,2],'o', markersize=6, color='green', alpha=0.5, label='class4')

ax.plot(class5[:,0], class5[:,1], class5[:,2],'o', markersize=6, color='magenta', alpha=0.5, label='class5')

ax.plot(class6[:,0], class6[:,1], class6[:,2],'o', markersize=6, color='yellow', alpha=0.5, label='class6')

ax.plot(class7[:,0], class7[:,1], class7[:,2],'o', markersize=6, color='black', alpha=0.5, label='class7')

ax.plot(class8[:,0], class8[:,1], class8[:,2],'o', markersize=6, color='purple', alpha=0.5, label='class8')

#plt.title('Three-dimensional visualization space for reduced order representation of 2-point statistics with Eucledian distance measure')
ax.set_xlabel(r'$\alpha_1$',fontsize = 18,fontweight = 'bold')
ax.set_ylabel(r'$\alpha_2$',fontsize = 18,fontweight = 'bold')
ax.set_zlabel(r'$\alpha_3$',fontsize = 18,fontweight = 'bold')
ax.legend(loc='lower right')
plt.tick_params(labelsize=12)
plt.show()
#=============================================================================

#=============================================================================
## PCA without transformation
class1 = Y[0:50,:]
class2 = Y[50:100,:]
class3 = Y[100:150,:]
class4 = Y[150:200,:]
class5 = Y[200:250,:]
class6 = Y[250:300,:]
class7 = Y[300:350,:]
class8 = Y[350:400,:]

from sklearn import decomposition
pca = decomposition.PCA(n_components = 2, svd_solver = 'auto')

pca.fit(class1)
class1 = pca.transform(class1)

pca.fit(class2)
class2 = pca.transform(class2)

pca.fit(class3)
class3 = pca.transform(class3)

pca.fit(class4)
class4 = pca.transform(class4)

pca.fit(class5)
class5 = pca.transform(class5)

pca.fit(class6)
class6 = pca.transform(class6)

pca.fit(class7)
class7 = pca.transform(class7)

pca.fit(class8)
class8 = pca.transform(class8)

X2 = np.zeros(shape = (50,2,1))
for i in range(1,9):
    X2 = np.concatenate((X2,locals()["class"+str(i)].reshape(50,2,1)),axis = 2)
#=============================================================================

#=============================================================================
## Plot Convex - Hull for transposed data
X2 = X2[:,:,1:9]
transformed_A = np.transpose(X2,axes = (1,0,2))
fig = plt.figure(figsize=(30,30))
ax = fig.add_subplot(111)
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['legend.markerscale'] = 2

ax.plot(transformed_A[0,:,0], transformed_A[1,:,0],'o', markersize=6, color='blue', alpha=0.5, label='class1')
trans = transformed_A[:,:,0].T
hull = ConvexHull(trans)
plt.fill(trans[hull.vertices,0],trans[hull.vertices,1],'blue',alpha = 0.4)

ax.plot(transformed_A[0,:,1], transformed_A[1,:,1],'o', markersize=6, color='red', alpha=0.5, label='class2')
trans = transformed_A[:,:,1].T
hull = ConvexHull(trans)
plt.fill(trans[hull.vertices,0],trans[hull.vertices,1],'red',alpha = 0.4)

ax.plot(transformed_A[0,:,2], transformed_A[1,:,2],'o', markersize=6, color='cyan', alpha=0.5, label='class3')
trans = transformed_A[:,:,2].T
hull = ConvexHull(trans)
plt.fill(trans[hull.vertices,0],trans[hull.vertices,1],'cyan',alpha = 0.4)

ax.plot(transformed_A[0,:,3], transformed_A[1,:,3],'o', markersize=6, color='green', alpha=0.5, label='class4')
trans = transformed_A[:,:,3].T
hull = ConvexHull(trans)
plt.fill(trans[hull.vertices,0],trans[hull.vertices,1],'green',alpha = 0.4)

ax.plot(transformed_A[0,:,4], transformed_A[1,:,4],'o', markersize=6, color='magenta', alpha=0.5, label='class5')
trans = transformed_A[:,:,4].T
hull = ConvexHull(trans)
plt.fill(trans[hull.vertices,0],trans[hull.vertices,1],'magenta',alpha = 0.4)

ax.plot(transformed_A[0,:,5], transformed_A[1,:,5],'o', markersize=6, color='yellow', alpha=0.5, label='class6')
trans = transformed_A[:,:,5].T
hull = ConvexHull(trans)
plt.fill(trans[hull.vertices,0],trans[hull.vertices,1],'yellow',alpha = 0.4)

ax.plot(transformed_A[0,:,6], transformed_A[1,:,6],'o', markersize=6, color='black', alpha=0.5, label='class7')
trans = transformed_A[:,:,6].T
hull = ConvexHull(trans)
plt.fill(trans[hull.vertices,0],trans[hull.vertices,1],'black',alpha = 0.4)

ax.plot(transformed_A[0,:,7], transformed_A[1,:,7],'o', markersize=6, color= 'purple', alpha=0.5, label='class8')
trans = transformed_A[:,:,7].T
hull = ConvexHull(trans)
plt.fill(trans[hull.vertices,0],trans[hull.vertices,1],'purple',alpha = 0.4)

ax.legend(loc='upper right')
ax.set_xlabel(r'$\alpha_1$',fontsize = 18,fontweight = 'bold')
ax.set_ylabel(r'$\alpha_2$',fontsize = 18,fontweight = 'bold')
plt.tick_params(labelsize=12)
plt.show()
#=============================================================================

pc = 50

from sklearn import decomposition
pca = decomposition.PCA(n_components = pc, svd_solver = 'auto')
pca.fit(X_con)
X_con = pca.transform(X_con)


#X_con = X_con.values
#X_con = X_con[:,0:pc]
X_con = X_con.T

b = np.arange(0,400,50)
x_new = np.zeros(shape = (50,pc,1))
for i in b:
    x_new = np.concatenate((x_new,(X_con[:,i:(i+50)].T).reshape(50,pc,1)),axis = 2)

x_new = x_new[:,:,1:9]
#x = np.transpose(x, axes=(2, 0, 1))
A = x_new

## Constructing W
k = 8
p = 50
w = np.zeros(shape= (pc,pc,1))
for i in range(k):
    for j in range(p):
        #print(A[j,:,i] - (np.mean(A[:,:,i],axis = 0)))
        scaled_A = (A[j,:,i] - (np.mean(A[:,:,i],axis = 0))).reshape(pc,1)
        #print(scaled_A)
        w_T = (scaled_A*scaled_A.T).reshape(pc,pc,1)
        w = np.concatenate((w,w_T),axis = 2)
        
w_final = np.sum(w,axis = 2)
print('W', w_final.shape)

## Constructing B
b = np.zeros(shape= (pc,pc,1))    
A_mean = np.sum(np.sum(A,axis = 0),axis = 1)/400.0
for i in range(k):
    scaled_A_mean = ((np.mean(A[:,:,i],axis = 0))-A_mean).reshape(pc,1)
    b_T = (50**k)*((scaled_A_mean*scaled_A_mean.T).reshape(pc,pc,1))
    b = np.concatenate((b,b_T),axis = 2)
        
b_final = np.sum(b,axis = 2)
print('b',b_final.shape)

middle_vec = np.linalg.inv(w_final)*b_final
print('inv',middle_vec.shape)
l = np.linalg.eigvals(middle_vec)
(u,s,v) = np.linalg.svd(middle_vec)

n = 2
transformed_A = np.zeros(shape = (n,50,1))
for i in range(k):
    transformed_A  = np.concatenate((transformed_A,(np.dot(v[0:n,:],(A[:,:,i].T))).reshape(n,50,1)),axis = 2)

transformed_A = transformed_A[:,:,1:9]
print('transformed_mat',transformed_A.shape)

## 2-D Plot

fig = plt.figure(figsize=(30,30))
ax = fig.add_subplot(111)
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['legend.markerscale'] = 2

ax.plot(transformed_A[0,:,0], transformed_A[1,:,0],'o', markersize=6, color='blue', alpha=0.5, label='class1')
trans = transformed_A[:,:,0].T
hull = ConvexHull(trans)
plt.fill(trans[hull.vertices,0],trans[hull.vertices,1],'blue',alpha = 0.4)

ax.plot(transformed_A[0,:,1], transformed_A[1,:,1],'o', markersize=6, color='red', alpha=0.5, label='class2')
trans = transformed_A[:,:,1].T
hull = ConvexHull(trans)
plt.fill(trans[hull.vertices,0],trans[hull.vertices,1],'red',alpha = 0.4)

ax.plot(transformed_A[0,:,2], transformed_A[1,:,2],'o', markersize=6, color='cyan', alpha=0.5, label='class3')
trans = transformed_A[:,:,2].T
hull = ConvexHull(trans)
plt.fill(trans[hull.vertices,0],trans[hull.vertices,1],'cyan',alpha = 0.4)

ax.plot(transformed_A[0,:,3], transformed_A[1,:,3],'o', markersize=6, color='green', alpha=0.5, label='class4')
trans = transformed_A[:,:,3].T
hull = ConvexHull(trans)
plt.fill(trans[hull.vertices,0],trans[hull.vertices,1],'green',alpha = 0.4)

ax.plot(transformed_A[0,:,4], transformed_A[1,:,4],'o', markersize=6, color='magenta', alpha=0.5, label='class5')
trans = transformed_A[:,:,4].T
hull = ConvexHull(trans)
plt.fill(trans[hull.vertices,0],trans[hull.vertices,1],'magenta',alpha = 0.4)

ax.plot(transformed_A[0,:,5], transformed_A[1,:,5],'o', markersize=6, color='yellow', alpha=0.5, label='class6')
trans = transformed_A[:,:,5].T
hull = ConvexHull(trans)
plt.fill(trans[hull.vertices,0],trans[hull.vertices,1],'yellow',alpha = 0.4)

ax.plot(transformed_A[0,:,6], transformed_A[1,:,6],'o', markersize=6, color='black', alpha=0.5, label='class7')
trans = transformed_A[:,:,6].T
hull = ConvexHull(trans)
plt.fill(trans[hull.vertices,0],trans[hull.vertices,1],'black',alpha = 0.4)

ax.plot(transformed_A[0,:,7], transformed_A[1,:,7],'o', markersize=6, color= 'purple', alpha=0.5, label='class8')
trans = transformed_A[:,:,7].T
hull = ConvexHull(trans)
plt.fill(trans[hull.vertices,0],trans[hull.vertices,1],'purple',alpha = 0.4)

#plt.title('Two-dimensional visualization space for reduced order representation of 2-point statistics with Mahalanobis distance measure')
ax.set_xlabel(r'$\alpha_1$',fontsize = 18,fontweight = 'bold')
ax.set_ylabel(r'$\alpha_2$',fontsize = 18,fontweight = 'bold')
plt.tick_params(labelsize=12)
ax.legend(loc='lower right')

plt.show()


## 3-D plot

n = 3
transformed_A = np.zeros(shape = (n,50,1))
for i in range(k):
    transformed_A  = np.concatenate((transformed_A,(np.dot(v[0:n,:],(A[:,:,i].T))).reshape(n,50,1)),axis = 2)
    print(transformed_A.shape)

transformed_A = transformed_A[:,:,1:9]

fig = plt.figure(figsize=(30,30))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['legend.markerscale'] = 2

ax.plot(transformed_A[0,:,0], transformed_A[1,:,0], transformed_A[2,:,0],'o', markersize=6, color='blue', alpha=0.5, label='class1')

ax.plot(transformed_A[0,:,1], transformed_A[1,:,1], transformed_A[2,:,1],'o', markersize=6, color='red', alpha=0.5, label='class2')

ax.plot(transformed_A[0,:,2], transformed_A[1,:,2], transformed_A[2,:,2],'o', markersize=6, color='cyan', alpha=0.5, label='class3')

ax.plot(transformed_A[0,:,3], transformed_A[1,:,3], transformed_A[2,:,3],'o', markersize=6, color='green', alpha=0.5, label='class4')

ax.plot(transformed_A[0,:,4], transformed_A[1,:,4], transformed_A[2,:,4],'o', markersize=6, color='magenta', alpha=0.5, label='class5')

ax.plot(transformed_A[0,:,5], transformed_A[1,:,5], transformed_A[2,:,5],'o', markersize=6, color='yellow', alpha=0.5, label='class6')

ax.plot(transformed_A[0,:,6], transformed_A[1,:,6], transformed_A[2,:,6],'o', markersize=6, color='black', alpha=0.5, label='class7')

ax.plot(transformed_A[0,:,7], transformed_A[1,:,7], transformed_A[2,:,7],'o', markersize=6, color='purple', alpha=0.5, label='class8')

#plt.title('Three-dimensional visualization space for reduced order representation of 2-point statistics with Mahalanobis distance measure')
ax.set_xlabel(r'$\alpha_1$',fontsize = 18,fontweight = 'bold')
ax.set_ylabel(r'$\alpha_2$',fontsize = 18,fontweight = 'bold')
ax.set_zlabel(r'$\alpha_3$',fontsize = 18,fontweight = 'bold',rotation = 90)
plt.tick_params(labelsize=12)
ax.legend(loc='lower right')

plt.show()