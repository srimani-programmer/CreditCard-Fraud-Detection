#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Required Modules for the Project

import sys
import numpy
import pandas
import matplotlib
import seaborn
import sklearn

print("System Version: {}".format(sys.version))
print("Numpy Version: {}".format(numpy.__version__))
print("Pandas Version: {}".format(pandas.__version__))
print("Matplotlib Version: {}".format(matplotlib.__version__))
print("Seaborn Version: {}".format(seaborn.__version__))
print("Scikit Learn Version: {}".format(sklearn.__version__))
print("Done..!")


# In[6]:


# Importing the required modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


# Loading the complete dataset.

data_set = pd.read_csv('creditcard.csv')

# Printing the shape of the dataset
print(data_set.shape)


# In[9]:


# Extracting the coloumns from the dataset

print(data_set.columns)


# In[11]:


# Describing the Dataset
print(data_set.describe())


# In[16]:


# Visualizing the dataset with histograms

data_set.hist(figsize=(20,20))
plt.show()


# In[17]:


# Since the data is too large we are taking the sample data to predict our results. this is doing to
# improve the computational power.

sample_data = data_set.sample(frac=0.15, random_state=1)

# Shape of the sample data.

print(sample_data.shape)


# In[19]:


# Finding the number of Faurd and Valid cases in the dataset

faurd_cases = sample_data[sample_data['Class'] == 1]
vaild_cases = sample_data[sample_data['Class'] == 0]

print("Number of Valid cases: {}".format(len(faurd_cases)))
print('Number of Faurd Cases: {}'.format(len(vaild_cases)))


# In[20]:


# Finding the outerline fraction value to predict the overall case result
outerline_fraction = len(faurd_cases) / len(vaild_cases)

print("The Outerline Fraction value is: {}".format(outerline_fraction))


# In[29]:


# Constructing a Correlation Matrix and visualizing it

correlationMatrix = sample_data.corr()
fig = plt.figure(figsize=(13,10))
sns.heatmap(correlationMatrix, vmax=0.8, square=True)
plt.show()


# In[32]:


# Creating the Flexible data for our project by removing unnecessary columns

# Creating a column list

column_list = sample_data.columns.tolist()

# Separating the class column from the dataset

for col in column_list:
    if 'Class' not in column_list:
        column_list.append(col)
        
target = "Class"

# In class contains 0 and 1
# 0 means valid transaction
# 1 means faurd transaction

# Divind the datasets into two separate parts

X = sample_data[column_list]
Y = sample_data[target]


# In[33]:


# Shape of the new datasets will be--

print('Shape of X: {}'.format(X.shape))
print('Shape of Y: {}'.format(Y.shape))


# ### Applying Machine Learning Algorithms on Dataset.

# In[35]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# define random states
state = 1

# define outlier detection tools to be compared
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination=outerline_fraction,
                                        random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=20,
        contamination=outerline_fraction)}


# In[37]:


# Fit the model
plt.figure(figsize=(9, 7))
n_outliers = len(faurd_cases)


for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    # fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
    
    # Reshape the prediction values to 0 for valid, 1 for fraud. 
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y).sum()
    
    # Run classification metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))


# In[ ]:




