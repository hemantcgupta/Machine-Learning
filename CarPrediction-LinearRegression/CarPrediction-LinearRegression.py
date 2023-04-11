# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 21:13:14 2023

@author: Hemant
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Data Collection, EDA and Data Processing
# =============================================================================
df = pd.read_csv(r'Car Price Data.csv')

df.shape
df.info()
df.isnull().sum()
df.describe()

# Now Seprate the numerical features of the data set and categorical Fetures
numerical_feature = [feature for feature in df.columns if df[feature].dtypes not in ['O','object']]
df[numerical_feature].head()
# Now get infomation about the our target feature Price of the cars
plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
plt.title('car Price Distribution Plot')
sns.distplot(df['price'])

plt.subplot(1,2,2)
plt.title("car Price spread")
sns.boxplot(y=df["price"])
plt.show()

# Now We are going to see the relastionship between dependent and independet Features
for feature in numerical_feature:
    if feature not in ["car_ID"]:
        plt.scatter(y=df['price'], x=df[feature])
        plt.ylabel('Car Price')
        plt.xlabel(feature)
        plt.show()

# Ploting the heat map to see the relastionship between other features
# And we also see the correlation betwwen them
plt.figure(figsize=(18,7))
sns.heatmap(df[numerical_feature].corr(), annot =True, fmt = '0.2f', 
            annot_kws = {'size':15}, linewidth=2, linecolor='orange')
plt.show()

# Finding the distribution and outliear from each numerical features 
index = 1
for feature in numerical_feature:
    if feature not in ['car_ID']:
        plt.figure(figsize=(20,12))
        # First Plot
        plt.subplot(index,2,1)
        plt.title(feature+' Distribution Plot')
        sns.distplot(df[feature])
        # Second Plot
        plt.subplot(index,2,2)
        plt.title(feature+' Box Plot')
        sns.boxplot(y=df[feature])
        plt.show()
        index=index + 1

# Now We are gonna to work on the categorical Features
categorical_feature = [feature for feature in df.columns if df[feature].dtypes in ['O', 'object']]
df[categorical_feature].head()

# Coutnter and outlier Plot of the categorical values 
plt.figure(figsize=(16,30))
plotnumber = 1
for i in range(1, len(categorical_feature)):
    if plotnumber <= 20:
        # 1st plot count
        ax = plt.subplot(18,2,plotnumber)
        sns.countplot(x=categorical_feature[i], data =df, ax=ax)
        plotnumber += 1
        # 2nd plot box plot
        ax = plt.subplot(18,2,plotnumber)
        sns.boxplot(x=categorical_feature[i], y = df['price'], data = df, ax=ax)
        plotnumber += 1 
plt.tight_layout()
plt.show()

# Univeriant Analysis
df['CarName'].count()
df['CarName'].unique()
df['CarName'].value_counts()
# finding the car compnay name from the carname columns 
CompanyName = df['CarName'].apply(lambda x:x.split(' ')[0])
df.insert(3, "CompanyName", CompanyName)
df.drop(['CarName'], axis=1, inplace=True)
df.head()

# Data Cleanning in CompanyName
df['CompanyName'] = df['CompanyName'].str.lower()

def replace_name(a, b):
    df['CompanyName'].replace(a, b, inplace=True)

replace_name('maxda', 'mazda')
replace_name('porcshce', 'porsche')
replace_name('toyouta', 'toyota')
replace_name('vokswagen', 'volkswagen')
replace_name('vw', 'volkswagen')

df['CompanyName'].unique()

# Now, we need to check the duplicated value in dataframe before applying Ml model 
df.loc[df.duplicated()]

# Checking the Hitogram Plot by unique value Couts
# For CompanyName, Fueltype, carbody
plt.figure(figsize=(25, 6))
plt.subplot(1,3,1)
plt1 = df['CompanyName'].value_counts().plot(kind='bar')
plt.title('car Company Histogram')
plt1.set(xlabel='Car', ylabel='Frequency of car')

plt.subplot(1,3,2)
plt1 = df['fueltype'].value_counts().plot(kind='bar')
plt.title('Fule Type Hitogram')
plt1.set(xlabel='Fule Type', ylabel='Frequency of fule type')

plt.subplot(1,3,3)
plt1 = df['carbody'].value_counts().plot(kind='bar')
plt.title('Car Type Histogram')
plt1.set(xlabel='Car Type', ylabel='Frequency of Car type')
plt.show()

# Now, we are see the other columns features of the categorical columns
cat_columns = ['aspiration', 'doornumber', 'drivewheel', 'enginelocation', 
               'enginetype', 'cylindernumber', 'fuelsystem']
for feature in cat_columns:
    plt.figure(figsize=(20, 8))
    # 1st Plot
    plt.subplot(1,2,1)
    plt.title(feature.title()+'Histogram')
    sns.countplot(df[feature], palette=('Blues_d'))
    # 2nd plot
    plt.subplot(1,2,2)
    plt.title(feature.title()+ 'vs Price')
    sns.boxplot(x=df[feature], y=df['price'], palette=('PuBuGn'))
    plt.show()

# Bivariate and multivariate analysis
sns.pairplot(df)
plt.show()

# =============================================================================
# Data Processing 
# =============================================================================
from words2num import words2num
df.drop(columns=['car_ID'], axis = 1, inplace=True)
df['doornumber'] = df['doornumber'].apply(lambda x: words2num(x))
df['cylindernumber'] = df['cylindernumber'].apply(lambda x: words2num(x))

# creating feature and label variable
X = df.drop(columns='price', axis=1)
y = df['price']

# encoding categorical columns
X = pd.get_dummies(X, drop_first=True)
X.head()

# Checking fopr multicollinearity using VIF and correleation matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
vif['Features'] = X.columns
# Dropping curbweight because of high VIF value. (shows that curbweight has high multicollinearity.)
# The higher the VIF, the higher the possibility that multicollinearity exists, and further research is required. When VIF is higher than 10, there is significant multicollinearity that needs to be corrected.

X = X.drop(['CompanyName_subaru', 'enginelocation_rear', 'enginetype_ohcf'], axis=1)

# =============================================================================
# Splitting data into traing and testing set
# =============================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Traing Data Shape of x and y repectively: ", X_train.shape, y_train.shape)
print("Testing Data Shape of x and y repectively: ", X_test.shape, y_test.shape)

# =============================================================================
# Model Building 
# =============================================================================
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression().fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

# =============================================================================
# Model Evalution 
# =============================================================================
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
mse = mean_squared_error(y_pred, y_test)
r2_score = r2_score(y_pred, y_test)
lr_model.score(X_test, y_test)
rmse = np.sqrt(mse)




