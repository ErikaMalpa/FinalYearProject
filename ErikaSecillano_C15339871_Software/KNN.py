
# coding: utf-8

# # KNN

# In[1]:


from sklearn.decomposition import PCA
import csv
import random
import math
import operator
import pandas as pd  
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pylab as pl


# In[2]:


#read data
df = pd.read_csv('./data/joinedData.csv', sep=r'\s*(?:\||\#|\,)\s*',
                 engine='python')


# In[3]:


df.describe()


# In[4]:


#change the 5 tumour types to numbers
Class = {'LUAD': 0,'BRCA': 1,'KIRC': 2,'PRAD': 3,'COAD': 4} 

#this is where we add the class to the table
df.Class = [Class[item] for item in df.Class]

#drop the 2 unnamed table because we do not need them
df = df.drop('Unnamed: 0',1)
df = df.drop('Unnamed: 0.1',1)
df


# In[5]:


#Split the X and y
X = df.drop('Class', axis=1).values
y = df['Class'].values
y = np.asarray(y)

#Standarize using min and max
X = (X - X.mean()) / (X.max() - X.min())

#Split the data set with 80 to traina dn20 to test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  
X_train


# In[6]:


#using k=n^(1/2) where n = columns, therefore it's 143.2 
K = 143


# In[7]:


clf = KNeighborsClassifier(n_neighbors=K, weights='distance')


# In[8]:


clf.fit(X_train, y_train)


# In[9]:


accuracy = clf.score(X_test, y_test)
print(accuracy)


# In[10]:


##Predict
df2 = df.drop('Class',1)
test = df2.iloc[[800]]
test


# In[11]:


print ("The test is: ")
print(clf.predict(test))


# In[12]:


print ("The right answer is: ")
check = df.iloc[[800]]
check


# In[13]:


pca = PCA(n_components=2).fit(X_train)
pcaX_train = pca.transform(X_train)


# In[14]:


h = .02 # step size in the mesh

# the color maps
cmap_light = ListedColormap(['#DDA0DD', '#FFFF00','#40E0D0','#DC143C','#7CFC00'])
clf.fit(pca_2d, y_train)

# Plot the decision boundary. For that, we will asign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = pcaX_train[:,0].min() - 1, pcaX_train[:,0].max() + 1
y_min, y_max = pcaX_train[:,1].min() - 1, pcaX_train[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


# In[15]:


Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])


# In[16]:


# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

for i in range(0, pcaX_train.shape[0]):
    if y_train[i] == 0:
        c1 = pl.scatter(pcaX_train[i,0],pcaX_train[i,1],c='#800080',s=10)
    elif y_train[i] == 1:
        c2 = pl.scatter(pcaX_train[i,0],pcaX_train[i,1],c='#8B8B00',s=10)
    elif y_train[i] == 2:
        c3 = pl.scatter(pcaX_train[i,0],pcaX_train[i,1],c='#00868B',s=10)
    elif y_train[i] == 3:
        c4 = pl.scatter(pcaX_train[i,0],pcaX_train[i,1],c='#B22222',s=10)
    elif y_train[i] == 4:
        c5 = pl.scatter(pcaX_train[i,0],pcaX_train[i,1],c='#3CB371',s=10)

# Plot also the training points
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("5-Class classification (k = %i)" % (K))
plt.show()


# ### References 
# #### The code was obtained by using sklearn
# #### http://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/auto_examples/tutorial/plot_knn_iris.html
# #### https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# #### https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# 
# ### References in Vancouver
# #### 1. sklearn.neighbors.KNeighborsClassifier — scikit-learn 0.20.1 documentation [Internet]. Scikit-learn.org. [cited 4 December 2018]. Available from: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# #### 2. sklearn.decomposition.PCA — scikit-learn 0.20.1 documentation [Internet]. Scikit-learn.org. [cited 4 December 2018]. Available from: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# #### 3. KNN (k-nearest neighbors) classification example — scikit-learn 0.11-git documentation [Internet]. Ogrisel.github.io. [cited 4 December 2018]. Available from: http://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/auto_examples/tutorial/plot_knn_iris.html
