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
import pickle
import requests
import json

#read the data
df = pd.read_csv('../upload/joinedData.csv', sep=r'\s*(?:\||\#|\,)\s*',
                 engine='python')

#change the classes to numbers
Class = {'LUAD': 0,'BRCA': 1,'KIRC': 2,'PRAD': 3,'COAD': 4} 
df.Class = [Class[item] for item in df.Class] 
df = df.drop('Unnamed: 0',1)
df = df.drop('Unnamed: 0.1',1)

df = df[['Class','gene_219', 'gene_220', 'gene_450', 'gene_1858', 'gene_3439',
       'gene_3737', 'gene_3921', 'gene_6733', 'gene_7421', 'gene_7896',
       'gene_7964', 'gene_9175', 'gene_9176', 'gene_13818', 'gene_14114',
       'gene_15895', 'gene_15898', 'gene_16169', 'gene_16392', 'gene_18135']]


#Split the dataframe into X and y, y containing only the class column
X = df.drop('Class', axis=1).values
y = df['Class'].values
y = np.asarray(y)
X

#Standardize data
X = (X - X.mean()) / (X.max() - X.min())

#Split the training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  
X_train

#using k=n^(1/2) where n = columns, therefore it's 143.2 
K = 143

clf = KNeighborsClassifier(n_neighbors=K, weights='distance')

clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

import pickle

df2 = df.drop('Class',1)
test = df2.iloc[[4]]

# Saving model 
with open('KNN_pickle','wb') as file:
    pickle.dump(clf,file)

with open('KNN_pickle','rb') as file:
    mp = pickle.load(file)

res = mp.predict(test)
print(res)