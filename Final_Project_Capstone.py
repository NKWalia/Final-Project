
# coding: utf-8

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation


import warnings
warnings.filterwarnings('ignore')
get_ipython().magic('matplotlib inline')
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile


# In[4]:


df_train = pd.read_csv("train.csv")


# In[5]:

df_train.columns


# In[6]:

#descriptive statistics summary
df_train['SalePrice'].describe()


# In[7]:

#Distribution of Sale Price of houses as histogram
sns.distplot(df_train['SalePrice']);


# In[8]:

'''Skewness is usually described as a measure of a dataset’s symmetry – or lack of symmetry.   
A perfectly symmetrical data set will have a skewness of 0.   The normal distribution has a skewness of 0.
So, when is the skewness too much?  The rule of thumb seems to be:
If the skewness is between -0.5 and 0.5, the data are fairly symmetrical
If the skewness is between -1 and – 0.5 or between 0.5 and 1, the data are moderately skewed
If the skewness is less than -1 or greater than 1, the data are highly skewed
Kurtosis is the degree of peakedness of a distribution” – Wolfram MathWorld
“We use kurtosis as a measure of peakedness (or flatness)” – Real Statistics Using Excel'''

#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())


# In[9]:

#scatter plot for Great living area vs saleprice.  It shows linear pattern.
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[10]:

#Plot between Total Basement Square ft and Sale Price shows linear pattern.
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[11]:

#Year built of house has linear relation with Sale price of the house.
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[12]:

#Remodeled yr of house has linear relation with Sale price of the house.
var = 'YearRemodAdd'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[13]:

#Linear relationship between 1st floor sq ft and sale price.
var = '1stFlrSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000)); 


# In[14]:

#Linear relationship between 2st floor sq ft and sale price.
var = '2ndFlrSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000)); 


# In[15]:

#Range of house price as per their year of built
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);


# In[16]:

#Compute pairwise correlation of columns, excluding NA/null values
#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[17]:

#Correlation of numerical variables with Sales Price
corrmat.sort_values(["SalePrice"], ascending = False, inplace = True)
print(corrmat.SalePrice)


# In[18]:

df_test = pd.read_csv("test.csv")


# In[19]:

#Missing Values


missing = df_train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()


# In[20]:

#Number of missing values and its percentage
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[21]:

#Missing Values for test data

missing = df_test.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()


# In[22]:

#Number of missing values and its percentage
total = df_test.isnull().sum().sort_values(ascending=False)
percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[23]:

#Following columns are dropped and rest are kept as they are not acctually missing.
#dealing with missing data
df_train = df_train.drop(['Id','LotFrontage','MasVnrType','MasVnrArea','GarageYrBlt' ],1)

#df_test.head()
df_test = df_test.drop(['Id','LotFrontage','MasVnrType','MasVnrArea','GarageYrBlt'],1)


# In[24]:

df_test.dropna(subset=['MSZoning','BsmtHalfBath'])
df_test=df_test.dropna(subset=['BsmtFullBath','GarageCars','GarageArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF'])


# In[25]:

#df_train.head()
df_train.dropna(subset=['Electrical'])


# In[26]:

df_train.head()


# In[27]:

#Converting Categorical variables into binary
#one_hot_encoded_training_predictors2 = pd.get_dummies(df_train)
#Add new data set for houses that are not for sale but would like to predict its values


#one_hot_encoded_training_predictors2 = pd.get_dummies(df_train)

one_hot_encoded_training_predictors = pd.get_dummies(df_train)
one_hot_encoded_test_predictors = pd.get_dummies(df_test)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left', 
                                                                    axis=1)


# In[28]:

final_test.head()


# In[29]:

final_train.head()


# In[30]:

final_test=final_test.drop(['SalePrice'],1)


# In[31]:

final_test.head()

#fill missing values in final_test with 0 since those features are not available
final_test.fillna(0, inplace = True)


# In[32]:

#saleprice correlation matrix
corrmat = final_train.corr()

k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(final_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[33]:

final_train.corr()


# In[34]:

#Split Data into Train and test.

X = final_train.drop(['SalePrice'],1)
y = final_train['SalePrice']
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)


# In[35]:

#Linear Regression Model
lm = LinearRegression()

#Train the model
lm.fit(X, y)

#predict test data
#pred=lm.predict(X_test)
#lm.score(X_test, y_test)


# In[ ]:




# In[ ]:




# In[40]:

#Linear regression using cross validation
score =cross_validation.cross_val_score(lm,X,y,cv=5, scoring='mean_squared_error')
mse_score = -score
rmse_score = np.sqrt(mse_score)
rmse_score.mean()


# In[ ]:




# In[42]:

#Random Forest regression
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X, y)
score1 =cross_validation.cross_val_score(rf,X,y,cv=5, scoring='mean_squared_error')
mse_score1 = -score1
rmse_score1 = np.sqrt(mse_score1)
rmse_score1.mean()


# In[43]:

prediction=rf.predict(final_test)


# In[44]:

df_test['Predicted_SalePrice'] = prediction


# In[45]:

df_test.head()


# In[265]:

df_test.to_csv("test_prediction.csv", index=True, header=True)


# In[ ]:



