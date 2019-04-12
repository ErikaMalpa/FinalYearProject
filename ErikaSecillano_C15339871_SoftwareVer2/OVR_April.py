#!/usr/bin/env python
# coding: utf-8

# # One VS Rest

# In[1]:


import pylab as pl
import numpy as np
import pandas as pd  
from sklearn.model_selection import train_test_split
import numpy as np


# In[2]:


#read the data set
df = pd.read_csv('./data/joinedData.csv', sep=r'\s*(?:\||\#|\,)\s*',
                 engine='python')


# In[3]:


#change the 5 tumour types to numbers
Class = {'LUAD': 0,'BRCA': 1,'KIRC': 2,'PRAD': 3,'COAD': 4} 
#this is where we add the class to the table
df.Class = [Class[item] for item in df.Class]
#drop the 2 unnamed table because we do not need them
df = df.drop('Unnamed: 0',1)
df = df.drop('Unnamed: 0.1',1)
df


# In[4]:


X = df.drop('Class', axis=1).values
y = df['Class'].values
y = np.asarray(y)


# In[5]:


#Standardize data
X = (X - X.mean()) / (X.max() - X.min())


# In[6]:


df = df[['Class','gene_219', 'gene_220', 'gene_450', 'gene_1858', 'gene_3439',
       'gene_3737', 'gene_3921', 'gene_6733', 'gene_7421', 'gene_7896',
       'gene_7964', 'gene_9175', 'gene_9176', 'gene_13818', 'gene_14114',
       'gene_15895', 'gene_15898', 'gene_16169', 'gene_16392', 'gene_18135']]


# In[7]:


df.head()


# In[8]:


X = df.drop('Class', axis=1).values
y = df['Class'].values
y = np.asarray(y)
X


# In[9]:


#Split the training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  
X_train


# In[11]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

#MultiClassOnevsRestClassifier

clf5 = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)
print("MultiClassOnevsRestClassifier prediction :",clf5.predict(X_test))
lrTest = clf5.predict(X_test)
print("MultiClassOnevsRestClassifier score :", accuracy_score(y_test,lrTest)) 


# In[12]:


res = clf5.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,res))  
print(classification_report(y_test,res))


# In[13]:


cm = confusion_matrix(y_test, res)
pl.matshow(cm)
pl.title('Confusion matrix of the classifier')
pl.colorbar()
pl.show()


# In[14]:


from sklearn.metrics import log_loss
from matplotlib import pyplot
#loss = log_loss(y_test, knnresd)
yhat = [x*0.01 for x in range(0, 401)]
# evaluate predictions for a 0 true value
losses_0 = [log_loss([0], [x], labels=[0,4]) for x in yhat]
# evaluate predictions for a 1 true value
losses_1 = [log_loss([1], [x], labels=[0,4]) for x in yhat]
losses_2 = [log_loss([2], [x], labels=[0,4]) for x in yhat]
losses_3 = [log_loss([3], [x], labels=[0,4]) for x in yhat]
losses_4 = [log_loss([4], [x], labels=[0,4]) for x in yhat]
# plot input to loss
pyplot.plot(yhat, losses_0, label='true=0')
pyplot.plot(yhat, losses_1, label='true=1')
pyplot.plot(yhat, losses_2, label='true=2')
pyplot.plot(yhat, losses_3, label='true=3')
pyplot.plot(yhat, losses_4, label='true=4')
pyplot.legend()
pyplot.show()


# # References
# 
# #### https://scikit-learn.org/stable/modules/multiclass.html
# #### https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
# #### https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier
# #### https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
# #### https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
# #### https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
# 
# ###  References in Vancouver
# #### 1. 1.12. Multiclass and multilabel algorithms — scikit-learn 0.20.1 documentation [Internet]. Scikit-learn.org. [cited 4 December 2018]. Available from: https://scikit-learn.org/stable/modules/multiclass.html
# #### 2. sklearn.svm.LinearSVC — scikit-learn 0.20.1 documentation [Internet]. Scikit-learn.org. [cited 4 December 2018]. Available from: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
# #### 3. sklearn.gaussian_process.GaussianProcessClassifier — scikit-learn 0.20.1 documentation [Internet]. Scikit-learn.org. [cited 4 December 2018]. Available from: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier
# #### 4. sklearn.linear_model.LogisticRegression — scikit-learn 0.20.1 documentation [Internet]. Scikit-learn.org. [cited 4 December 2018]. Available from: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
# #### 5. 3.2.4.1.5. sklearn.linear_model.LogisticRegressionCV — scikit-learn 0.20.1 documentation [Internet]. Scikit-learn.org. [cited 4 December 2018]. Available from: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV

# In[ ]:




