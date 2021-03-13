#!/usr/bin/env python
# coding: utf-8

# ## Complete the Tasks in bold
# 
# **TASK: Run the cells below to import the libraries and load the dataset.**

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("wine_fraud.csv")


# In[3]:


df.head()


# **TASK: What are the unique variables in the target column we are trying to predict (quality)?**

# In[4]:


df['quality'].unique()


# **TASK: Create a countplot that displays the count per category of Legit vs Fraud. Is the label/target balanced or unbalanced?**

# In[6]:


sns.countplot(x='quality',data=df)


# **TASK: Let's find out if there is a difference between red and white wine when it comes to fraud. Create a countplot that has the wine *type* on the x axis with the hue separating columns by Fraud vs Legit.**

# In[7]:


sns.countplot(x='type',hue='quality',data=df)


# **TASK: What percentage of red wines are Fraud? What percentage of white wines are fraud?**

# In[8]:


reds = df[df['type']=='red']


# In[9]:


whites = df[df['type']=='white']


# In[10]:


print("Percentage of fraud in Red Wines:")
print(100* (len(reds[reds['quality']=='Fraud'])/len(reds)))


# In[11]:


print("Percentage of fraud in White Wines:")
print(100* (len(whites[whites['quality']=='Fraud'])/len(whites)))


# **TASK: Calculate the correlation between the various features and the "quality" column. To do this you may need to map the column to 0 and 1 instead of a string.**

# In[14]:


df['Fraud'] = df['quality'].map({'Legit':0,'Fraud':1})


# In[15]:


df.corr()['Fraud']


# **TASK: Create a bar plot of the correlation values to Fraudlent wine.**

# In[16]:


df.corr()['Fraud'][:-1].sort_values().plot(kind='bar')


# **TASK: Create a clustermap with seaborn to explore the relationships between variables.**

# In[17]:


sns.clustermap(df.corr(),cmap='viridis')


# ----
# ## Machine Learning Model
# 
# **TASK: Convert the categorical column "type" from a string or "red" or "white" to dummy variables:**

# In[18]:


df['type'] = pd.get_dummies(df['type'],drop_first=True)


# In[19]:


df = df.drop('Fraud',axis=1)


# **TASK: Separate out the data into X features and y target label ("quality" column)**

# In[20]:


X = df.drop('quality',axis=1)
y = df['quality']


# **TASK: Perform a Train|Test split on the data, with a 10% test size. Note: The solution uses a random state of 101**

# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)


# **TASK: Scale the X train and X test data.**

# In[23]:


from sklearn.preprocessing import StandardScaler


# In[24]:


scaler = StandardScaler()


# In[25]:


scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)


# **TASK: Create an instance of a Support Vector Machine classifier. Previously we have left this model "blank", (e.g. with no parameters). However, we already know that the classes are unbalanced, in an attempt to help alleviate this issue, we can automatically adjust weights inversely proportional to class frequencies in the input data with a argument call in the SVC() call. Check out the [documentation for SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) online and look up what the argument\parameter is.**

# In[26]:


from sklearn.svm import SVC


# In[34]:


svc = SVC(class_weight='balanced')


# **TASK: Use a GridSearchCV to run a grid search for the best C and gamma parameters.**

# In[35]:


from sklearn.model_selection import GridSearchCV


# In[36]:


param_grid = {'C':[0.001,0.01,0.1,0.5,1],'gamma':['scale','auto']}
grid = GridSearchCV(svc,param_grid)


# In[39]:


grid.fit(scaled_X_train,y_train)


# In[40]:


grid.best_params_


# **TASK: Display the confusion matrix and classification report for your model.**

# In[41]:


from sklearn.metrics import confusion_matrix,classification_report


# In[42]:


grid_pred = grid.predict(scaled_X_test)


# In[43]:


confusion_matrix(grid_pred,y_test)


# In[45]:


print(classification_report(y_test,grid_pred))

