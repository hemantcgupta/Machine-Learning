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
# Data Collection, EDA and Dat Processing
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
# fing the car compnay name from the carname columns 
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

plt.figure(figsize=(25, 6))










