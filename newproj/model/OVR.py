import pylab as pl
import numpy as np
import pandas as pd  
from sklearn.model_selection import train_test_split
import numpy as np


# In[2]:


#read the data set
df = pd.read_csv('../upload/joinedData.csv', sep=r'\s*(?:\||\#|\,)\s*',
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

import pickle

# Saving model 
pickle.dump(clf5, open('OVR.pkl','wb'))

with open('OVR.pkl','rb') as file:
    mp = pickle.load(file)

df2 = df.drop('Class',1)
test = df2.iloc[[4]]

res = mp.predict(test)
print(res)