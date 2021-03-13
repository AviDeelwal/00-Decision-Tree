#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("../DATA/mouse_viral_study.csv")


# In[3]:


df.head()


# In[4]:


df.columns


# ## Classes

# In[5]:


sns.scatterplot(x='Med_1_mL',y='Med_2_mL',hue='Virus Present',
                data=df,palette='seismic')


# ## Separating Hyperplane
# 
# Our goal with SVM is to create the best separating hyperplane. In 2 dimensions, this is simply a line.

# In[6]:


sns.scatterplot(x='Med_1_mL',y='Med_2_mL',hue='Virus Present',palette='seismic',data=df)

# We want to somehow automatically create a separating hyperplane ( a line in 2D)

x = np.linspace(0,10,100)
m = -1
b = 11
y = m*x + b
plt.plot(x,y,'k')


# ## SVM - Support Vector Machine

# In[7]:


from sklearn.svm import SVC # Supprt Vector Classifier


# In[8]:


help(SVC)


# **NOTE: For this example, we will explore the algorithm, so we'll skip any scaling or even a train\test split for now**

# In[9]:


y = df['Virus Present']
X = df.drop('Virus Present',axis=1) 


# In[10]:


model = SVC(kernel='linear', C=1000)
model.fit(X, y)


# In[11]:


# This is imported from the supplemental .py file
# https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html
from svm_margin_plot import plot_svm_boundary


# In[12]:


plot_svm_boundary(model,X,y)


# ## Hyper Parameters
# 
# ### C
# 
# Regularization parameter. The strength of the regularization is **inversely** proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.
# 
# *Note: If you are following along with the equations, specifically the value of C as described in ISLR, C in scikit-learn is **inversely** proportional to this value.*

# In[13]:


model = SVC(kernel='linear', C=0.05)
model.fit(X, y)


# In[14]:


plot_svm_boundary(model,X,y)


# In[15]:


model = SVC(kernel='rbf', C=1)
model.fit(X, y)
plot_svm_boundary(model,X,y)


# In[16]:


model = SVC(kernel='sigmoid')
model.fit(X, y)
plot_svm_boundary(model,X,y)


# #### Degree (poly kernels only)
# 
# Degree of the polynomial kernel function ('poly').
# Ignored by all other kernels.

# In[17]:


model = SVC(kernel='poly', C=1,degree=1)
model.fit(X, y)
plot_svm_boundary(model,X,y)


# In[18]:


model = SVC(kernel='poly', C=1,degree=2)
model.fit(X, y)
plot_svm_boundary(model,X,y)


# ### gamma
# 
# gamma : {'scale', 'auto'} or float, default='scale'
#     Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
# 
#     - if ``gamma='scale'`` (default) is passed then it uses
#       1 / (n_features * X.var()) as value of gamma,
#     - if 'auto', uses 1 / n_features.

# In[19]:


model = SVC(kernel='rbf', C=1,gamma=0.01)
model.fit(X, y)
plot_svm_boundary(model,X,y)


# ## Grid Search
# 
# Keep in mind, for this simple example, we saw the classes were easily separated, which means each variation of model could easily get 100% accuracy, meaning a grid search is "useless".

# In[20]:


from sklearn.model_selection import GridSearchCV


# In[21]:


svm = SVC()
param_grid = {'C':[0.01,0.1,1],'kernel':['linear','rbf']}
grid = GridSearchCV(svm,param_grid)


# In[22]:


# Note again we didn't split Train|Test
grid.fit(X,y)


# In[23]:


# 100% accuracy (as expected)
grid.best_score_


# In[24]:


grid.best_params_


# This is more to review the grid search process, recall in a real situation such as your exercise, you will perform a train|test split and get final evaluation metrics.
