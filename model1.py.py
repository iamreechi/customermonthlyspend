#!/usr/bin/env python
# coding: utf-8

# ### Business Problem 
# Predicting Customer's average monthly spend from demographic features, Income Buying behaviour on a Bike ETC

# This data is about a company's customers, including demographic features and information about purchases they have made. The company is particularly interested in analyzing customer data to determine any apparent relationships between demographic features known about the customers and the purchase behaviour. 

# In[1]:


#importing necessary reading, processing and visualization library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Importing all neccessary model for model evaluation 
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost import XGBRegressor


# ### Importing the Data

# In[3]:


#Reading of all dataset
train = pd.read_csv('Competition 1- train_technidus.csv')
test = pd.read_csv('Competition 1- test_technidus.csv')
train['source'] = 'train'
test['source'] = 'test'
#concatinating train and test dataset
full_data = pd.concat([train, test], axis =0 )
print("shape of train data is: {}".format(train.shape))
print("shape of test data is: {}".format(test.shape))
print("shape of full_data data is: {}".format(full_data.shape))
full_data.head()


# In[4]:


#Data Type for each Attribute
types = full_data.dtypes
types


# In[5]:


#identifying all column in dataset 
full_data.columns


# ### Descriptive Statistics

# In[6]:


#checking all Statistical attribute in datase for numerical variable
from pandas import set_option
set_option('display.width', 100)
set_option('precision', 3)
full_data.describe().T


# In[7]:


#checking all Statistical attribute in datase for categorical variable
full_data.describe(include='object').T


# ### Class Distribution (Classification Only)

# In[8]:


class_counts = full_data.groupby('AveMonthSpend').size().plot(kind='hist')
print(class_counts)


# From the above, more of the data skewed to the Right between 1 - 25 spending habits 

# #### Corelations Between Attributes

# In[9]:


correlations = full_data.corr(method='pearson')
print(correlations)


# ### Understanding Data with Visualization

# ##### Univariate plots

# In[10]:


#Univariate Histogram
full_data.hist(figsize=(8,8))
plt.show()


# In[11]:


#Univeriate Density Plots
full_data.plot(kind='density', subplots=True, layout=(3,3), sharex=False, figsize=(9,9))
plt.show()


# In[12]:


#Box ans Whisker Plots
full_data.plot(kind='box',subplots=True, layout=(3,3), sharex=False, figsize=(9,9))
plt.show()


# #### Multivariate Plots

# In[13]:


correlations = full_data.corr()
names = [ 
         
        'BirthDate',
       'HomeOwnerFlag',
       'NumberCarsOwned', 'NumberChildrenAtHome', 'TotalChildren',
       'YearlyIncome', 'AveMonthSpend', 'BikeBuyer']


fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
 
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels((names), rotation=45)
ax.set_yticklabels(names)
plt.show()


# In[14]:


#scatterplot Matrix
from pandas.plotting import scatter_matrix
scatter_matrix(full_data,alpha=0.5, figsize=(20, 20))
plt.show()


# ## Data Pre-Processing

# #### Checking for missing Value

# In[15]:


for i in full_data.columns:
    print(i + ": "+str(sum(full_data[i].isnull()))+" missing value")


# #### Droping of uninformative columns
# This is as a result of the following 
# 1. The column has only 30% of the data or less
# 2. irrelevant in prediting AveMonthSpend

# In[16]:


#Print out all the columns that has less than 30% null values
nn_cols=[col for col in full_data.columns if train[col].count()<=0.7*len(train)]
print(nn_cols)


# In[17]:


uninformative=['FirstName','LastName','CustomerID','PhoneNumber','PostalCode', 'Title', 
               'MiddleName', 'Suffix','AddressLine1', 'AddressLine2']
full_data.drop(uninformative,axis=1,inplace=True)
print("shape of full_data is: {}".format(full_data.shape))


# ##### Treating AveMonthSpend missing value 

# In[19]:


#Treating AveMontSpend missing value 
gr = full_data.groupby(['Occupation', 'Education'])
gr.first()


# In[41]:


#full_data.loc[full_data['CountryRegionName'] == 'United States']
#droping nan rows and replacing with computed countries mean of AvemonthSpend
full_data.dropna(inplace=True)
joint = [full_data, USA, Austraila, Germany, UK, France, Canada]
full_data = pd.concat(joint)


# In[52]:


#checking for missing value in Avemonthspend
for i in full_data.columns:
    print(i + ": "+str(sum(full_data[i].isnull()))+" missing value")


# In[41]:


#converting YearlyIncome to MonthlyIncome since we are predicting AveMonthSpend
#full_data['MonthlyIncom'] = full_data['YearlyIncome']/12
# full_data.drop('YearlyIncome',axis=1,inplace=True)


# #### Data preprocessing Continues 

# We will extract the years from *BirthDate* column and create a new column called *Year_of_Birth*. We will use this column to create bins used in our analysis.

# In[20]:


#onvdrting BirthDate to year 
full_data['Year_of_Birth'] = pd.DatetimeIndex(full_data['BirthDate']).year
full_data.drop("BirthDate", axis=1, inplace=True)
full_data.head()


# In[21]:


print("Year of birth of customers range from {} to {}".format(full_data["Year_of_Birth"].min(),
                                                              full_data["Year_of_Birth"].max()))


# In[22]:


#creating an array of bin used fo scaling 
bins = np.array([1911, 1960, 1980])
bins


# In[23]:


group_names = ["Retired", "Not_Retired"]


# In[24]:


#grouping the Year_of_birth into retired or not which range for 60 years and above Retired and others Not retird
full_data["Year_of_birth_binned"] = pd.cut(full_data["Year_of_Birth"], bins, labels=group_names, include_lowest=True)
full_data[["Year_of_Birth", "Year_of_birth_binned"]].head(20)


# In[25]:


#visualising the Retired and Not retired in respect to AveMonthSpend
full_data.groupby(["Year_of_birth_binned"]).AveMonthSpend.sum().plot(kind="bar", color="darkblue")
plt.title("Customers Total Average Monthly Spend by Age bracket")
plt.ylabel("Customers Monthly spend")
plt.show()


# ### $ Exploratory Data Analysis

# In[26]:


full_data.hist(figsize=(8,8))
plt.show()


# A few observations can be made based on the information and histograms for numerical features:
# 
# 1. Year_of_birth is a slightly left-skewed normal distribution with the bulk of the staff born between 1960 and 1970.Data transformation methods may be required to approach a normal distribution prior to fitting a model to the data.
# 2. Target feature is also not normally distributed 

# In[27]:


#finding correlation between Numerical column
corrmat = full_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True, annot=True);


# **Assumption** 
# 
# 1. *TotalChildren* is highly correlated with *NumberChildrenAtHome* which is not good for our regression model
# 2. AveMonthSpend is positivly correlated with HomeOwnerFlag, NumberCarsOwned, NumberChildrenAtHome,TotalChildren, BikeBuyer and YearlyIncome
# 3. AveMonthSpend is negatively correlated with Year_of_Birth

# In[28]:


#Observation of the distriution of AveMonthSpend
sns.distplot(full_data[full_data['source'] == 'train'].AveMonthSpend) 


# In[29]:


#plot to count gender 
sns.countplot(x='Gender',data=full_data)


# In[30]:


#observation of MonthlySpending  and Yearly income for male and female
var = 'YearlyIncome'
data = pd.concat([full_data['AveMonthSpend'], full_data[var]], axis=1)
sns.lmplot(x=var,y='AveMonthSpend',data=full_data,hue='Gender')


# It appears the Male's earns more than the female counterpart. Just like the law of abnormal demand thats states, the higher the price the higher the quantity demanded. Thus the higher the income the higher the Monthly spending. There was also a very high concerntration between 80 - 60. 

# In[31]:


#plot to show the various Occupation, their Income and spending count
sns.lmplot(x='YearlyIncome',y='AveMonthSpend',data=full_data,col='Occupation')


# The Target Occupation here are Professional's and Management which had a high income and spend the most among others 

# In[32]:


#plot to show the Ave monthly spending for genders as to total number of childern 
sns.lmplot(x="YearlyIncome", y="AveMonthSpend", row="Gender", col="TotalChildren",data=full_data)


# In[33]:


#plot to show the various customer Educational qualification, their Income and spending count
sns.lmplot(x='YearlyIncome',y='AveMonthSpend',data=full_data,col='Education')


# In[34]:


#plot showing maritalStatus and thir number of children at home aainst their spending monthly 
sns.lmplot(x="YearlyIncome", y="AveMonthSpend", row="MaritalStatus", col="NumberChildrenAtHome",data=full_data)


# In[35]:


#plot showing countries from customers and their monthly and monthly spending
sns.lmplot(x='YearlyIncome',y='AveMonthSpend',data=full_data,col='CountryRegionName')


# It is interesting to know customers from UK have more high income earners but customers from Australia spends more

# In[36]:


#plot showing customer based on country and their gender against their spending monthly 
sns.stripplot(x="CountryRegionName", y="AveMonthSpend", data=full_data,hue='Gender',palette='Set1')


# ## Model

# In[37]:


full_data.head()


# In[38]:


#droping correlated column and some irrelevant column
full_data.drop(['City','TotalChildren','StateProvinceName'], axis=1, inplace=True)


# In[39]:


new_train = full_data[full_data.source == 'train']
new_test = full_data[full_data.source== 'test']
print("shape of new_train data is: {}".format(new_train.shape))
print("shape of new_test data is: {}".format(new_test.shape))


# In[40]:


new_train = new_train.copy()
new_test = new_train.copy()


# In[41]:


new_train.drop(['source'],axis=1, inplace=True)
new_test.drop(['source'],axis=1,inplace=True)


# In[43]:


new_test.shape, new_train.shape


# #### Cloumn Transformation

# In[46]:


# determine categorical and numerical features
numerical_ix = X.select_dtypes(include=['int64', 'float64']).columns
categorical_ix = X.select_dtypes(include=['object', 'bool']).columns


# In[69]:


numerical_ix,categorical_ix


# In[85]:


import category_encoders as ce

#Label encoding
cat_cols = ['Gender', 'MaritalStatus', 'CountryRegionName', 'Education', 'Occupation','Year_of_birth_binned'] 

encoder = ce.OrdinalEncoder(cols=cat_cols)
new_train = encoder.fit_transform(new_train)


# In[89]:


correlations = new_train.corr()['AveMonthSpend'].sort_values()
print('Most Positive Correlations: \n', correlations.tail(10))
print('\nMost Negative Correlations: \n', correlations.head(3))


# In[91]:


drop = ['Year_of_birth_binned','AveMonthSpend' ]
X = new_train.drop(drop, axis= 1)
y = new_train["AveMonthSpend"]


# In[92]:


X.shape,y.shape


# In[93]:


X


# ## Model Evaluation and Optimization

# In[94]:


from numpy import mean
from numpy import std
from numpy import absolute
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.dummy import DummyRegressor
#checking baseline model with an algorithm 
x_ = X
y_ = y
naive = DummyRegressor(strategy='median')
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(naive, x_, y_, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
n_scores = absolute(n_scores)
print('Baseline: %.3f (%.3f)' % (np.sqrt(n_scores.mean()), std(n_scores.mean())))
# evaluate model
model = XGBRegressor(objective='reg:squarederror')
m_scores = cross_val_score(model, x_, y_, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
m_scores = absolute(m_scores)
print('Good: %.3f (%.3f)' % (np.sqrt(m_scores.mean()), std(m_scores.mean())))


# In[95]:


from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
ss = StandardScaler()
rs = RobustScaler()
qt = QuantileTransformer(output_distribution='normal',n_quantiles=3864)
yj = PowerTransformer(method = 'yeo-johnson')
bc = PowerTransformer(method = 'box-cox')


# In[96]:


ss_data = pd.DataFrame(ss.fit_transform(x_), columns = x_.columns)
rs_data = pd.DataFrame(rs.fit_transform(x_), columns = x_.columns)
qt_data = pd.DataFrame(qt.fit_transform(x_), columns = x_.columns)
yj_data = pd.DataFrame(yj.fit_transform(x_), columns = x_.columns)


# In[97]:


naive = DummyRegressor(strategy='median')
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(naive, ss_data, y_, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
n_scores = absolute(n_scores)
print('Baseline for ss: %.3f (%.3f)' % (np.sqrt(n_scores.mean()), std(n_scores.mean())))
# evaluate model
model = XGBRegressor(objective='reg:squarederror')
m_scores = cross_val_score(model, ss_data, y_, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
m_scores = absolute(m_scores)
print('Good: %.3f (%.3f)' % (np.sqrt(m_scores.mean()), std(m_scores.mean())))

naive = DummyRegressor(strategy='median')
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(naive, rs_data, y_, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
n_scores = absolute(n_scores)
print('Baseline for rs: %.3f (%.3f)' % (np.sqrt(n_scores.mean()), std(n_scores.mean())))
# evaluate model
model = XGBRegressor(objective='reg:squarederror')
m_scores = cross_val_score(model, rs_data, y_, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
m_scores = absolute(m_scores)
print('Good: %.3f (%.3f)' % (np.sqrt(m_scores.mean()), std(m_scores.mean())))

naive = DummyRegressor(strategy='median')
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(naive, qt_data, y_, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
n_scores = absolute(n_scores)
print('Baseline for qt: %.3f (%.3f)' % (np.sqrt(n_scores.mean()), std(n_scores.mean())))
# evaluate model
model = XGBRegressor(objective='reg:squarederror')
m_scores = cross_val_score(model, qt_data, y_, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
m_scores = absolute(m_scores)
print('Good: %.3f (%.3f)' % (np.sqrt(m_scores.mean()), std(m_scores.mean())))

naive = DummyRegressor(strategy='median')
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(naive, yj_data, y_, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
n_scores = absolute(n_scores)
print('Baseline for yj: %.3f (%.3f)' % (np.sqrt(n_scores.mean()), std(n_scores.mean())))
# evaluate model
model = XGBRegressor(objective='reg:squarederror')
m_scores = cross_val_score(model, yj_data, y_, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
m_scores = absolute(m_scores)
print('Good: %.3f (%.3f)' % (np.sqrt(m_scores.mean()), std(m_scores.mean())))


# In[98]:


validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y,
test_size=validation_size,shuffle=True, random_state=seed)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", Y_train.shape)
print("Number transactions X_validation dataset: ", X_validation.shape)
print("Number transactions y_validation dataset: ", Y_validation.shape)


# In[112]:


#creating a function to calculate RMSE
def evaluate_rmse(y, pred):
    results = mean_squared_error(y, pred)
    return np.sqrt(results)
num_folds = 10
scoring = 'neg_mean_squared_error'

models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))
models.append(('XGB', XGBRegressor(objective= 'reg:squarederror')))
models.append(('RGF', RandomForestRegressor()))
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, np.sqrt(np.abs(cv_results.mean())), np.sqrt(cv_results.std()))
    print(msg)


# In[113]:


# Compare Algorithms
fig = plt.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[114]:


# prepare the model
#using
model_base0 =  XGBRegressor(objective= 'reg:squarederror')
model_base0.fit(X_train.values, Y_train.values)
pred_train = model_base0.predict(X_train.values)
pred_test = model_base0.predict(X_validation.values)
print('Train RMSE: ', evaluate_rmse(Y_train, pred_train))
#print('Test_clean RMSE: ' ,evaluate_rmse(test.price, pred_test))
print('Test RMSE: ' ,evaluate_rmse(Y_validation, pred_test))


# In[129]:


# prepare the model
#using
model_base0 =  XGBRegressor(objective= 'reg:squarederror')
model_base0.fit(X_train, Y_train)
pred_train = model_base0.predict(X_train)
pred_test = model_base0.predict(X_validation)
print('Train RMSE: ', evaluate_rmse(Y_train, pred_train))
#print('Test_clean RMSE: ' ,evaluate_rmse(test.price, pred_test))
print('Test RMSE: ' ,evaluate_rmse(Y_validation, pred_test))


# In[123]:


X_train.columns, new_test.columns


# In[118]:


import category_encoders as ce

#Label encoding
cat_cols = ['Gender', 'MaritalStatus', 'CountryRegionName', 'Education', 'Occupation','Year_of_birth_binned'] 

encoder = ce.OrdinalEncoder(cols=cat_cols)
new_test = encoder.fit_transform(new_test)


# In[124]:


newtest = new_test.drop(drop, axis=1)


# In[125]:


#predicting actual test dataset
test_predictions = model_base0.predict(newtest.values)
test_predictions[:5]


# In[126]:


#plotting our prediction and  test prediction 
sns.distplot(test_predictions, hist=False, rug=True)
sns.distplot(pred_train, hist=False, rug=True)


# In[131]:


# Saving model to disk
import pickle
pickle.dump(model_base0, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict(newtest))

