#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from flask import Flask, render_template,request
from sklearn.preprocessing import StandardScaler
import joblib


# In[2]:


app = Flask(__name__) #Naming the application
model = joblib.load("train_model.pkl")
scaler = StandardScaler()


# In[3]:


@app.route('/') # Mapping url to perform specific functions that will handle logic [ root url ]
def home():
    return render_template('index.html') # used to generate output from the default template file


# In[4]:


@app.route('/submit',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        gender = request.form['gender']
        if(gender == "Female"):
            gender_no = 1
        else:
            gender_no = 2
        age = request.form['age']
        openness = request.form['openness']
        neuroticism = request.form['neuroticism']
        conscientiousness = request.form['conscientiousness']
        agreeableness = request.form['agreeableness']
        extraversion = request.form['extraversion']
        result = np.array([gender_no, age, openness,neuroticism, conscientiousness, agreeableness, extraversion], ndmin = 2)
        final = scaler.fit_transform(result)
        personality = str(model.predict(final)[0])
        return render_template("submit.html",answer = personality)


# In[5]:


if __name__ ==  '__main__':
    app.run(debug=True)


# In[ ]:


# In post - large amount of data can be sent because data is in body
# In get - limited amount of data can be send because data is in header


# In[ ]:


get_ipython().run_line_magic('tb', '')


# In[ ]:




