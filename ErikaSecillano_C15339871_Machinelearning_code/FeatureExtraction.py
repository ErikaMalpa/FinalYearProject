#!/usr/bin/env python
# coding: utf-8

# ### Feature Extraction  

# In[1]:


import pylab as pl
import numpy as np
import pandas as pd  
from sklearn.model_selection import train_test_split
import numpy as np


# In[5]:


#read the data set
df = pd.read_csv('./data/joinedData.csv', sep=r'\s*(?:\||\#|\,)\s*',
                 engine='python')


# In[6]:


#change the 5 tumour types to numbers
Class = {'LUAD': 0,'BRCA': 1,'KIRC': 2,'PRAD': 3,'COAD': 4} 
#this is where we add the class to the table
df.Class = [Class[item] for item in df.Class]
#drop the 2 unnamed table because we do not need them
df = df.drop('Unnamed: 0',1)
df = df.drop('Unnamed: 0.1',1)
df.head()


# In[7]:


X = df.drop('Class', axis=1).values
y = df['Class'].values
y = np.asarray(y)

#Standardize data
X = (X - X.mean()) / (X.max() - X.min())


# In[8]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# Feature extraction
X_new = SelectKBest(score_func=f_classif, k=20)
X_new2 = X_new.fit_transform(X, y)


# In[10]:


#To get the columns for the new dataframe
features_dataframe = list(df.columns.values)
features_dataframe


# In[11]:


#To get the columns for the new dataframe
df3 = pd.DataFrame(columns = list(df.columns.values))
df3


# In[12]:


#We drop the class since it makes an error but this will be returned later
df4 = df3.drop(columns="Class")


# In[14]:


mask = X_new.get_support()
new_features = df4.columns[mask]
#Prints the relevant genes
new_features


# In[15]:


#Creating the new dataframe
df5 = df[['Class','gene_219', 'gene_220', 'gene_450', 'gene_1858', 'gene_3439',
       'gene_3737', 'gene_3921', 'gene_6733', 'gene_7421', 'gene_7896',
       'gene_7964', 'gene_9175', 'gene_9176', 'gene_13818', 'gene_14114',
       'gene_15895', 'gene_15898', 'gene_16169', 'gene_16392', 'gene_18135']]


# In[16]:


df5


# In[ ]:




