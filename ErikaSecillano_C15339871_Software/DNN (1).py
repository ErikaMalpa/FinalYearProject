
# coding: utf-8

# # NEURAL NETWORKS

# In[1]:


import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn import svm
import pylab as pl
import numpy as np
import pandas as pd  
from sklearn.model_selection import train_test_split


# In[2]:


#read the data
df = pd.read_csv('./data/joinedData.csv', sep=r'\s*(?:\||\#|\,)\s*',
                 engine='python')

#change the classes to numbers
Class = {'LUAD': 0,'BRCA': 1,'KIRC': 2,'PRAD': 3,'COAD': 4} 
df.Class = [Class[item] for item in df.Class] 
df = df.drop('Unnamed: 0',1)
df = df.drop('Unnamed: 0.1',1)
df

#separate the data into X and y
X = df.drop('Class', axis=1).values
y = df['Class'].values
y = np.asarray(y)

#standardize the data
X = (X - X.mean()) / (X.max() - X.min())

#Split the data set to test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  
X_train


# In[3]:


#get the estimator
estimator = tf.estimator.DNNClassifier(
    feature_columns=[tf.feature_column.numeric_column('x', shape=X_train.shape[1:])],
    hidden_units=[1000, 500, 250], 
    optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=0.01,
      l1_regularization_strength=0.001
    ), #optimizer was used to improve the estimator
    n_classes=5) #the number of label classes, we have 5


# In[4]:


# defining the training inputs
train = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    batch_size=X_test.shape[0],
    num_epochs=None,
    shuffle=False,
    num_threads=1
    ) 


# In[5]:


estimator.train(input_fn = train,steps=1000)


# In[6]:


# defining the test inputs
input_fn2 = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=y_test, 
    shuffle=False,
    batch_size=X_test.shape[0],
    num_epochs=None)


# In[7]:


#evaluate the estimator
estimator.evaluate(input_fn2,steps=1000) 


# In[8]:


#predict input
pred_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    shuffle=False)


# In[9]:


#predict the results
pred_results = estimator.predict(input_fn=pred_input_fn)


# In[10]:


#print the results, example of the result is array([b'2'], dtype=object) 
#b = 2 = KIRC
for i in pred_results:
    print (i)


# In[11]:


y_test


# ### References 
# #### The code was obtained by using sklearn
# 
# ##### https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNClassifier
# 
# ### References using vancouver
# #### 1. tf.contrib.learn.DNNClassifier  |  TensorFlow [Internet]. TensorFlow. [cited 4 December 2018]. Available from: https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNClassifier
