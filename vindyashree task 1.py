#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing of necessary packages and loading dataset
import pandas as pd
import numpy as np
import sys
d = pd.read_csv("Iris.csv")


# In[3]:


d.head()


# In[4]:


# delete a column
d = d.drop(columns = ['Id'])
d.head()


# In[5]:


# to display stats about data in the dataset
d.describe()


# In[6]:


# to know the info about datatypes
d.info()


# In[8]:


#to know the shape
d.shape


# In[9]:


# to display no. of samples on each class
d['Species'].value_counts()


# Preprocessing the Dataset
# 

# In[10]:


#Checking for null values
d.isnull().sum()


# Exploratory data analysis
# 

# In[11]:


#importing necessary packages for plots
import seaborn as sns
import matplotlib.pyplot as plt


# In[12]:


# countplot
sns.countplot(x='Species',data=d,)
plt.show()


# In[13]:


#scatter plot for Sepal length and width
sns.scatterplot(x='SepalLengthCm',y='SepalWidthCm',hue='Species',data=d,)

plt.legend(bbox_to_anchor=(1,1),loc=1)
plt.show()


# In[14]:


# scatter plot for Petal
sns.scatterplot(x='PetalLengthCm',y='PetalWidthCm',hue='Species',data=d,)

plt.legend(bbox_to_anchor=(1,1),loc=2)
plt.show()


# In[18]:


#pair plot
sns.pairplot(d,hue='Species',height=2)


# In[19]:


#Histogram
fig, axes = plt.subplots(2, 2, figsize=(10,10))

axes[0,0].set_title("Sepal Length")
axes[0,0].hist(d['SepalLengthCm'], bins=7)
 
axes[0,1].set_title("Sepal Width")
axes[0,1].hist(d['SepalWidthCm'], bins=5);
 
axes[1,0].set_title("Petal Length")
axes[1,0].hist(d['PetalLengthCm'], bins=6);
 
axes[1,1].set_title("Petal Width")
axes[1,1].hist(d['PetalWidthCm'], bins=6);


# In[21]:


#Histogram with Distplot plot
plot=sns.FacetGrid(d,hue="Species")
plot.map(sns.distplot,"SepalLengthCm").add_legend()

plot=sns.FacetGrid(d,hue="Species")
plot.map(sns.distplot,"SepalWidthCm").add_legend()

plot=sns.FacetGrid(d,hue="Species")
plot.map(sns.distplot,"PetalLengthCm").add_legend()

plot=sns.FacetGrid(d,hue="Species")
plot.map(sns.distplot,"PetalWidthCm").add_legend()

plt.show()


# Correlation Matrix

# In[22]:


# display the correlation matrix
d.corr()


# In[25]:


# plot the heat map
corr = d.corr()

fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')


# Label Encoder

# In[28]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# transform the string labels to integer
d['Species'] = le.fit_transform(d['Species'])
d.head()


# Training and Testing

# In[29]:


from sklearn.model_selection import train_test_split
## train - 70%
## test - 30%

X = d.drop(columns=['Species'])    # input data
# output data
Y = d['Species']   #output data

# split the data for train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)


# In[30]:


#Importing some models and train

# 1. logistic regression 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[31]:


# model training
model.fit(x_train, y_train)


# In[32]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[33]:


# 2. knn - k-nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[34]:


model.fit(x_train, y_train)


# In[35]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[36]:


# 3. decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[37]:


model.fit(x_train, y_train)


# In[39]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[41]:


# 4. SVM
from sklearn import svm
model = svm.SVC()


# In[42]:


model.fit(x_train, y_train)


# In[43]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# finally ,I got 100% accuracy for Logistic Regression
