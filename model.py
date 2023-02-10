#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import joblib
import pickle


# In[77]:


df_train = pd.read_csv('Train.csv')


# In[78]:


train_length = len(df_train)
df_test = pd.read_csv("Test.csv")
df_test


# In[83]:


test_length = len(df_test)
df_train.rename(columns = {'Personality (Class label)':'Personality'}, inplace = True) 
df_test.rename(columns = {'Personality (class label)':'Personality'}, inplace = True)
#inpace = True indicates that the dataframe is updated but it will not return anything
# df.rename(columns = {'Personality (Class label)':'Personality'}, inplace = False)
df_train


# In[84]:


df = pd.concat([df_train,df_test])
df


# In[85]:


df.head(10)


# In[86]:


# Label Encoding converts labels into numeric form so as to convert them into machine readable format.
label = LabelEncoder() # Targets values between 0 to n_classes-1
df['Gender'] = label.fit_transform(df['Gender'])
df.head(10) 


# In[87]:


# Standard Scaler standaradizes the a feature by subtracting the mean and then scaling to unit variance
# Unit variance means dividing all values by standard deviation
scaler = StandardScaler(with_std=1)
#default it takes std = 1 and mean = 0 when with_std and with_mean are False
input_columns = ['Gender', 'Age', 'openness', 'neuroticism','conscientiousness', 'agreeableness','extraversion']
output_columns = ['Personality']
df[input_columns] = scaler.fit_transform(df[input_columns])


# In[88]:


df.head()


# In[89]:


df_train = df[:train_length]
df_test = df[train_length:]
X = df_train[input_columns]
Y = df_train[output_columns]


# In[90]:


X


# In[91]:


Y


# In[108]:


# Assumption : We assume that data is identically distributed [Check]
df_train['Personality'].value_counts()


# In[109]:


df_test['Personality'].value_counts()


# In[100]:


model = LogisticRegression(multi_class='multinomial',solver='newton-cg',max_iter=10000)
model.fit(X,Y)


# In[101]:


X_test = df_test[input_columns]
Y_test = df_test[output_columns]


# In[102]:


X_test


# In[103]:


Y_test


# In[104]:


Predicted_class = model.predict(X_test)
len(Predicted_class)


# In[105]:


Predicted_class


# In[106]:


print("Accuracy: ",metrics.accuracy_score(Y_test, Predicted_class)*100)
print("Confusion Matrix: ")
print(metrics.confusion_matrix(Y_test, Predicted_class))
print("Classification Report: ")
print(metrics.classification_report(Y_test,Predicted_class))
# print("Precision: ",metrics.precision_score(Y_test, Predicted_class))
# print("Recall: ",metrics.recall_score(Y_test, Predicted_class))
# print("F - Measure: ",metrics.f1_score(Y_test,Predicted_class))


# In[107]:


# Linking model to application
joblib.dump(model, 'train_model.pkl')

