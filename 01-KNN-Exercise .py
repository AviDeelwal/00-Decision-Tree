#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('sonar.all-data.csv')


# In[3]:


df.head()


# ## Data Exploration
# 
# **TASK: Create a heatmap of the correlation between the difference frequency responses.**

# In[9]:


plt.figure(figsize=(8,6))
sns.heatmap(df.corr(),cmap='coolwarm')


# **TASK: What are the top 5 correlated frequencies with the target\label?**
# 
# *Note: You many need to map the label to 0s and 1s.*
# 
# *Additional Note: We're looking for **absolute** correlation values.*

# In[11]:


df['Target'] = df['Label'].map({'R':0,'M':1})


# In[12]:


np.abs(df.corr()['Target']).sort_values().tail(6)


# ## Train | Test Split
# 
# Our approach here will be one of using Cross Validation on 90% of the dataset, and then judging our results on a final test set of 10% to evaluate our model.
# 
# **TASK: Split the data into features and labels, and then split into a training set and test set, with 90% for Cross-Validation training, and 10% for a final test set.**
# 
# *Note: The solution uses a random_state=42*

# In[13]:


from sklearn.model_selection import train_test_split


# In[15]:


X = df.drop(['Target','Label'],axis = 1)
y = df['Label']


# In[18]:


X_cv, X_test, y_cv, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# **TASK: Create a PipeLine that contains both a StandardScaler and a KNN model**

# In[16]:


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# In[17]:


scaler = StandardScaler()


# In[19]:


knn = KNeighborsClassifier()


# In[20]:


operations = [('scaler',scaler),('knn',knn)]


# In[21]:


from sklearn.pipeline import Pipeline


# In[22]:


pipe = Pipeline(operations)


# **TASK: Perform a grid-search with the pipeline to test various values of k and report back the best performing parameters.**

# In[23]:


from sklearn.model_selection import GridSearchCV


# In[24]:


k_values = list(range(1,30))


# In[25]:


param_grid = {'knn__n_neighbors': k_values}


# In[26]:


full_cv_classifier = GridSearchCV(pipe,param_grid,cv=5,scoring='accuracy')


# In[27]:


full_cv_classifier.fit(X_cv,y_cv)


# In[28]:


full_cv_classifier.best_estimator_.get_params()


# **(HARD) TASK: Using the .cv_results_ dictionary, see if you can create a plot of the mean test scores per K value.**

# In[29]:


full_cv_classifier.cv_results_['mean_test_score']


# In[30]:


scores = full_cv_classifier.cv_results_['mean_test_score']
plt.plot(k_values,scores,'o-')
plt.xlabel("K")
plt.ylabel("Accuracy")


# ### Final Model Evaluation
# 
# **TASK: Using the grid classifier object from the previous step, get a final performance classification report and confusion matrix.**

# In[32]:


pred = full_cv_classifier.predict(X_test)


# In[33]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[34]:


confusion_matrix(y_test,pred)


# In[35]:


print(classification_report(y_test,pred))


# ### Great Job!
