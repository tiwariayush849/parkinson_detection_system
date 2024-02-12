#!/usr/bin/env python
# coding: utf-8

# In[1]:


#parkinson disease is a progressive nervous system disorder that affects movement leading to shaking, stiffings, and difficulty with walking, balance, and coordination.


# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[2]:


parkinson_data= pd.read_csv('parkinsons.data')
parkinson_data.head()


# In[3]:


parkinson_data.shape


# In[4]:


parkinson_data.info()


# In[5]:


parkinson_data.isnull().sum()


# In[6]:


parkinson_data.describe()


# In[7]:


parkinson_data['status'].value_counts()#1-->parkinson,0->healthy


# In[8]:


parkinson_data.groupby('status').mean()


# data preprocessing

# In[9]:


x= parkinson_data.drop(columns=['name','status'], axis=1)
y=parkinson_data['status']


# In[10]:


print(x)


# In[11]:


print(y)


# splitting the data into train and test data

# In[12]:


x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[13]:


print(x.shape,x_train.shape, x_test.shape)


# data standardization

# In[14]:


scaler= StandardScaler()


# In[15]:


scaler.fit(x_train)


# In[16]:


x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


# In[17]:


print(x_train)


# model training

# In[18]:


model=svm.SVC(kernel='linear')


# In[19]:


#training svm model with training data
model.fit(x_train, y_train)


# MODEL EVALUATION
# 
# 
# 
# Accuracy Score

# In[20]:


# accuracy score on training data
x_train_prediction= model.predict(x_train)
training_data_accuracy= accuracy_score(y_train,x_train_prediction)


# In[21]:


print('accuracy score of training data:', training_data_accuracy)


# In[22]:


x_test_prediction= model.predict(x_test)
testing_data_accuracy= accuracy_score(y_test,x_test_prediction)


# In[23]:


print('accuracy score of testing data:', testing_data_accuracy)


# building a predictive system

# In[26]:


input_data = (120.55200,131.16200,113.78700,0.00968,0.00008,0.00463,0.00750,0.01388,0.04701,0.45600,0.02328,0.03526,0.03243,0.06985,0.01222,21.37800,0.415564,0.825069,-4.242867,0.299111,2.187560,0.357775)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)
std_data= scaler.transform(input_data_reshaped)
prediction= model.predict(std_data)
print(prediction)
if(prediction[0]==0):
    print("Person does not have Parkinson")
    
else:
    print("Person has Parkinson")

