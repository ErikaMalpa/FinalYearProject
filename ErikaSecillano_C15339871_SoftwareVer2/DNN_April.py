#!/usr/bin/env python
# coding: utf-8

# # DNN Tensorflow

# In[1]:


import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn import svm
import pylab as pl
import numpy as np
import pandas as pd  
from sklearn.model_selection import train_test_split


# In[9]:


#read the data
df = pd.read_csv('./data/joinedData.csv', sep=r'\s*(?:\||\#|\,)\s*',
                 engine='python')

#change the classes to numbers
Class = {'LUAD': 0,'BRCA': 1,'KIRC': 2,'PRAD': 3,'COAD': 4} 
df.Class = [Class[item] for item in df.Class] 
df = df.drop('Unnamed: 0',1)
df = df.drop('Unnamed: 0.1',1)
df.head()


# In[10]:


df = df[['Class','gene_219', 'gene_220', 'gene_450', 'gene_1858', 'gene_3439',
       'gene_3737', 'gene_3921', 'gene_6733', 'gene_7421', 'gene_7896',
       'gene_7964', 'gene_9175', 'gene_9176', 'gene_13818', 'gene_14114',
       'gene_15895', 'gene_15898', 'gene_16169', 'gene_16392', 'gene_18135']]


# In[11]:


#Split the dataframe into X and y, y containing only the class column
X = df.drop('Class', axis=1).values
y = df['Class'].values
y = np.asarray(y)
X

#Standardize data
X = (X - X.mean()) / (X.max() - X.min())


# In[12]:


#Split the training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  
X_train


# In[13]:


#get the estimator
estimator = tf.estimator.DNNClassifier(
    feature_columns=[tf.feature_column.numeric_column('x', shape=X_train.shape[1:])],
    hidden_units=[1000, 500, 250], 
    optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=0.01,
      l1_regularization_strength=0.001
    ), #optimizer was used to improve the estimator
    n_classes=5) #the number of label classes, we have 5


# In[14]:


# defining the training inputs
train = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    batch_size=X_test.shape[0],
    num_epochs=None,
    shuffle=False,
    num_threads=1
    ) 


# In[15]:


estimator.train(input_fn = train,steps=1000)


# In[16]:


# defining the test inputs
input_fn2 = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=y_test, 
    shuffle=False,
    batch_size=X_test.shape[0],
    num_epochs=None)


# In[17]:


#evaluate the estimator
estimator.evaluate(input_fn2,steps=1000) 


# In[18]:


#evaluate the estimator
estimator.evaluate(input_fn2,steps=1000) 


# In[20]:


pred_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    shuffle=False)


# In[21]:


#predict the results
pred_results = estimator.predict(input_fn=pred_input_fn)


# In[22]:


#print the results, example of the result is array([b'2'], dtype=object) 
#b = 2 = KIRC
for i in pred_results:
    print (i)


# In[23]:


y_test


# ### References 
# #### The code was obtained by using tensorlfow
# 
# ### https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNClassifier
# 
# ### References using vancouver
# #### 1. tf.contrib.learn.DNNClassifier  |  TensorFlow [Internet]. TensorFlow. [cited 4 December 2018]. Available from: https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNClassifier
# 

# In[ ]:




