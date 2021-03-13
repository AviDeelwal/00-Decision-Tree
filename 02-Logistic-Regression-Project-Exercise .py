#!/usr/bin/env python
# coding: utf-8

# ## Imports
# 
# **TASK: Run the cell below to import the necessary libraries.**

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ----
# 
# **TASK: Run the cell below to read in the data.**

# In[2]:


df = pd.read_csv('heart.csv')


# In[3]:


df.head()


# In[4]:


df['target'].unique()


# ### Exploratory Data Analysis and Visualization
# 
# Feel free to explore the data further on your own.
# 
# **TASK: Explore if the dataset has any missing data points and create a statistical summary of the numerical features as shown below.**

# In[5]:


# CODE HERE


# In[5]:


df.info()


# In[7]:


# CODE HERE


# In[6]:


df.describe().transpose()


# ### Visualization Tasks
# 
# **TASK: Create a bar plot that shows the total counts per target value.**

# In[9]:


# CODE HERE!


# In[7]:


sns.countplot(x='target',data=df)


# **TASK: Create a pairplot that displays the relationships between the following columns:**
# 
#     ['age','trestbps', 'chol','thalach','target']
#    
# *Note: Running a pairplot on everything can take a very long time due to the number of features*

# In[11]:


# CODE HERE


# In[8]:


sns.pairplot(df[['age','trestbps', 'chol','thalach','target']],hue='target')


# **TASK: Create a heatmap that displays the correlation between all the columns.**

# In[11]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),cmap='viridis',annot=True)


# ----
# ----
# 
# # Machine Learning
# 
# ## Train | Test Split and Scaling
# 
# **TASK: Separate the features from the labels into 2 objects, X and y.**

# In[12]:


X = df.drop('target',axis=1)
y = df['target']


# **TASK: Perform a train test split on the data, with the test size of 10% and a random_state of 101.**

# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)


# **TASK: Create a StandardScaler object and normalize the X train and test set feature data. Make sure you only fit to the training data to avoid data leakage (data knowledge leaking from the test set).**

# In[15]:


scaler = StandardScaler()


# In[16]:


scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)


# ## Logistic Regression Model
# 
# **TASK: Create a Logistic Regression model and use Cross-Validation to find a well-performing C value for the hyper-parameter search. You have two options here, use *LogisticRegressionCV* OR use a combination of *LogisticRegression* and *GridSearchCV*. The choice is up to you.**

# In[17]:


from sklearn.linear_model import LogisticRegression


# In[18]:


log_model = LogisticRegression()


# In[19]:


log_model.fit(scaled_X_train,y_train)


# **TASK: Report back your search's optimal parameters, specifically the C value.** 
# 
# *Note: You may get a different value than what is shown here depending on how you conducted your search.*

# In[23]:


log_model.get_params()


# ### Coeffecients
# 
# **TASK: Report back the model's coefficients.**

# In[24]:


log_model.coef_


# **BONUS TASK: We didn't show this in the lecture notebooks, but you have the skills to do this! Create a visualization of the coefficients by using a barplot of their values. Even more bonus points if you can figure out how to sort the plot! If you get stuck on this, feel free to quickly view the solutions notebook for hints, there are many ways to do this, the solutions use a combination of pandas and seaborn.**

# In[33]:


#CODE HERE


# In[25]:


coefs = pd.Series(index=X.columns,data=log_model.coef_[0])


# In[26]:


coefs = coefs.sort_values()


# In[27]:


plt.figure(figsize=(10,6))
sns.barplot(x=coefs.index,y=coefs.values);


# ---------
# 
# ## Model Performance Evaluation

# **TASK: Let's now evaluate your model on the remaining 10% of the data, the test set.**
# 
# **TASK: Create the following evaluations:**
# * Confusion Matrix Array
# * Confusion Matrix Plot
# * Classification Report

# In[53]:


# CODE HERE


# In[28]:


from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix


# In[29]:


y_pred = log_model.predict(scaled_X_test)


# In[30]:


confusion_matrix(y_test,y_pred)


# In[31]:


plot_confusion_matrix(log_model,scaled_X_test,y_test)


# In[32]:


print(classification_report(y_test,y_pred))


# ### Performance Curves
# 
# **TASK: Create both the precision recall curve and the ROC Curve.**

# In[33]:


from sklearn.metrics import plot_precision_recall_curve,plot_roc_curve


# In[34]:


plot_precision_recall_curve(log_model,scaled_X_test,y_test)


# In[35]:


plot_roc_curve(log_model,scaled_X_test,y_test)


# **Final Task: A patient with the following features has come into the medical office:**
# 
#     age          48.0
#     sex           0.0
#     cp            2.0
#     trestbps    130.0
#     chol        275.0
#     fbs           0.0
#     restecg       1.0
#     thalach     139.0
#     exang         0.0
#     oldpeak       0.2
#     slope         2.0
#     ca            0.0
#     thal          2.0

# **TASK: What does your model predict for this patient? Do they have heart disease? How "sure" is your model of this prediction?**
# 
# *For convience, we created an array of the features for the patient above*

# In[39]:


patient = [[ 54. ,   1. ,   0. , 122. , 286. ,   0. ,   0. , 116. ,   1. ,
          3.2,   1. ,   2. ,   2. ]]


# In[36]:


X_test.iloc[-1]


# In[37]:


y_test.iloc[-1]


# In[40]:


log_model.predict(patient)


# In[41]:


log_model.predict_proba(patient)


# ----
# 
# ## Great Job!
