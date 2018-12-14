
# coding: utf-8

# # One vs Rest

# In[1]:


import pylab as pl
import numpy as np
import pandas as pd  
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression


# In[2]:


#read the data set
df = pd.read_csv('./data/joinedData.csv', sep=r'\s*(?:\||\#|\,)\s*',
                 engine='python')


# In[3]:


from sklearn.feature_selection import RFE

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


# feature extraction
#model = LogisticRegression()
#rfe = RFE(model, 3)
#fit = rfe.fit(X, y)
#print("Num Features: %d") % fit.n_features_
#print("Selected Features: %s") % fit.support_
#print("Feature Ranking: %s") % fit.ranking_


# In[6]:


#Standardize data
X = (X - X.mean()) / (X.max() - X.min())

#Split the training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  
X_train


# In[7]:


##Predict
df2 = df.drop('Class',1)
test = df2.iloc[[800]]
test


# In[8]:


from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score, classification_report


# In[9]:


##Logistic Regression
clf1 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(X_train, y_train)
print("Logistic Regression prediction :",clf1.predict(X_test))
print("Logistic Regression probability :",clf1.predict_proba(test))
lrTest = clf1.predict(X_test)
print("Logistic Regression score :", accuracy_score(y_test,lrTest))


# In[10]:


##Logistic Regression CV
clf2 = LogisticRegressionCV(cv=5, random_state=0, multi_class='ovr').fit(X_train, y_train)
print("Logistic Regression CV prediction :",clf2.predict(X_test))
print("Logistic Regression CV probability :",clf2.predict_proba(test))
lrTest = clf2.predict(X_test)
print("Logistic Regression score CV :", accuracy_score(y_test,lrTest))


# In[11]:


##Gaussian Process Classifier
kernel = 1.0 * RBF(1.0)
clf3 = GaussianProcessClassifier(multi_class = 'one_vs_rest',kernel=kernel,random_state=0).fit(X_train, y_train)
print("Gaussian Process Classifier prediction :",clf3.predict(X_test))
print("Gaussian Process Classifier probability :",clf3.predict_proba(test))
lrTest = clf3.predict(X_test)
print("Gaussian Process Classifier score :", accuracy_score(y_test,lrTest))


# In[12]:


##Linear SVC
kernel = 1.0 * RBF(1.0)
clf4 = LinearSVC(multi_class='ovr',random_state=0, tol=1e-5).fit(X_train, y_train)
print("Linear SVC prediction :",clf4.predict(X_test))
lrTest = clf4.predict(X_test)
print("Linear SVC score :", accuracy_score(y_test,lrTest))


# In[13]:


from sklearn.multiclass import OneVsRestClassifier

#MultiClassOnevsRestClassifier

clf5 = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)
print("MultiClassOnevsRestClassifier prediction :",clf5.predict(X_test))
lrTest = clf5.predict(X_test)
print("MultiClassOnevsRestClassifier score :", accuracy_score(y_test,lrTest)) 


# ### References
# 
# #### https://scikit-learn.org/stable/modules/multiclass.html
# #### https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
# #### https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier
# #### https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
# #### https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
# 
# ###  References in Vancouver
# #### 1. 1.12. Multiclass and multilabel algorithms — scikit-learn 0.20.1 documentation [Internet]. Scikit-learn.org. [cited 4 December 2018]. Available from: https://scikit-learn.org/stable/modules/multiclass.html
# #### 2. sklearn.svm.LinearSVC — scikit-learn 0.20.1 documentation [Internet]. Scikit-learn.org. [cited 4 December 2018]. Available from: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
# #### 3. sklearn.gaussian_process.GaussianProcessClassifier — scikit-learn 0.20.1 documentation [Internet]. Scikit-learn.org. [cited 4 December 2018]. Available from: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier
# #### 4. sklearn.linear_model.LogisticRegression — scikit-learn 0.20.1 documentation [Internet]. Scikit-learn.org. [cited 4 December 2018]. Available from: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
# #### 5. 3.2.4.1.5. sklearn.linear_model.LogisticRegressionCV — scikit-learn 0.20.1 documentation [Internet]. Scikit-learn.org. [cited 4 December 2018]. Available from: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV

# In[ ]:


##Logistic Regression
clf1 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')
rfe = rfe = RFE(clf1, 3)
rfe.fit(X_train, y_train)
#print("Logistic Regression prediction :",clf1.predict(X_test))
#print("Logistic Regression probability :",clf1.predict_proba(test))
#lrTest = clf1.predict(X_test)
#print("Logistic Regression score :", accuracy_score(y_test,lrTest))
# print summaries for the selection of attributes
print(rfe.support_)
print(rfe.ranking_)

