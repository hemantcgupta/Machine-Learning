# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 23:41:24 2023

@author: Hemant
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib.inline
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 60)
pd.set_option('expand_frame_repr', True)

dataset = pd.read_csv(r'C:\Users\heman\Desktop\Machine-Learning\Loan Eligibility Prediction\loan-train.csv')
dataset.shape
# =============================================================================
#  EDA And data engineering
# =============================================================================
dataset.info()
dataset.describe()
pd.crosstab(dataset['Credit_History'], dataset['Loan_Status'], margins=True)

dataset.boxplot(column='ApplicantIncome')
dataset['ApplicantIncome'].hist(bins=20)

dataset['CoapplicantIncome'].hist(bins=20)

dataset.boxplot(column='ApplicantIncome', by='Education')

dataset.boxplot(column='LoanAmount')
dataset['LoanAmount'].hist(bins=20)
dataset['LoanAmount_log'] = np.log(dataset['LoanAmount'])
dataset['LoanAmount_log'].hist(bins=20) # normalizein the data 
# =============================================================================
#  filling the null values
# =============================================================================
dataset.isnull().sum()
dataset['Gender'].fillna(dataset['Gender'].mode()[0], inplace=True)
dataset['Married'].fillna(dataset['Married'].mode()[0], inplace=True) 
dataset['Dependents'].fillna(dataset['Dependents'].mode()[0], inplace=True)
dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0], inplace=True)
dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0], inplace=True)
dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0], inplace=True)
dataset['LoanAmount'] = dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean())
dataset['LoanAmount_log'] = dataset['LoanAmount_log'].fillna(dataset['LoanAmount_log'].mean())

# =============================================================================
# now we have to normalized the applicate and coapplicat income but we have take sum both incomes
# =============================================================================
dataset['TotalIncome'] = dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
dataset['TotalIncome'].hist(bins=20)
dataset['TotalIncome_log'] = np.log(dataset['TotalIncome'])
dataset['TotalIncome_log'].hist(bins=20)
dataset.head()

# =============================================================================
# Splitting 80 and 20
# =============================================================================
X = dataset.iloc[:,np.r_[1:5, 9:11,13:15]].values
y = dataset.iloc[:,12].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train)

#  now convert the the categrical values into numeric formate
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()

for i in range(0, 5):
    X_train[:,i] = labelencoder_X.fit_transform(X_train[:,i])
    
X_train[:,7] = labelencoder_X.fit_transform(X_train[:,7])
X_train    

labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)
y_train 
#  for test dataset
for i in range(0, 5):
    X_test[:,i] = labelencoder_X.fit_transform(X_test[:,i])
X_test[:,7] = labelencoder_X.fit_transform(X_test[:,7])
 
# labelencoder_y = LabelEncoder()
y_test = labelencoder_y.fit_transform(y_test)

# =============================================================================
# Now Scale the dataset
#  becasue need prediction accuretly
# =============================================================================
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

# =============================================================================
# Noe applying the DecisionTreeClassifier algorithm
# =============================================================================
from sklearn.tree import DecisionTreeClassifier
DTClassifier = DecisionTreeClassifier(criterion ='entropy', random_state=0)
DTClassifier.fit(X_train, y_train)
# noe prdicting the value using test data

y_pred = DTClassifier.predict(X_test)
y_pred

# Now checking the accuracy of model
from sklearn import metrics
print('The accuracy of the decision tree is: ', metrics.accuracy_score(y_pred, y_test))
# from sklearn.metrics import mean_absolute_percentage_error
# print('The accuracy of the decision tree is: ', mean_absolute_percentage_error(y_pred, y_test))

# =============================================================================
# applying the another algorithms which is naive bayes
# =============================================================================
from sklearn.naive_bayes import GaussianNB
NBClassifier = GaussianNB()
NBClassifier.fit(X_train, y_train)
y_pred = NBClassifier.predict(X_test)
y_pred
print('The accuracy of the Naive bayes is: ', metrics.accuracy_score(y_pred, y_test))

# =============================================================================
# Applying SVM model
# =============================================================================
from sklearn.svm import SVC
# Create an SVM classifier object
clf = SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('The accuracy of the SVM is: ', metrics.accuracy_score(y_pred, y_test))

# =============================================================================
# applying randomforest model
# =============================================================================
from sklearn.ensemble import RandomForestClassifier
# Create a random forest classifier object
clf = RandomForestClassifier(n_estimators=200, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('The accuracy of the RandomForest is: ', metrics.accuracy_score(y_pred, y_test))

# =============================================================================
# Auto select the estimators in randomforest
# =============================================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
# Create a random forest classifier object
clf = RandomForestClassifier(random_state=0)
# Define the parameter distribution
param_dist = {'n_estimators': [50, 100, 200, 300, 400, 500],
              'max_depth': [1, 5, 10, 20, 30, 40, 50, None]}
# Create a RandomizedSearchCV object
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=10, cv=5)
# Fit the RandomizedSearchCV object to the training data
random_search.fit(X_train, y_train)
# Print the best parameters and best score
print("Best parameters: ", random_search.best_params_)
print("Best score: {:.2f}%".format(random_search.best_score_*100))
# Make predictions on the test set
y_pred = random_search.predict(X_test)
# Calculate the accuracy of the model
print('The accuracy of the RandomForest is: ', metrics.accuracy_score(y_pred, y_test))

# =============================================================================
# using Grid search in randomforest
# =============================================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Create a random forest classifier object
clf = RandomForestClassifier(random_state=0)
# Define the parameter grid
param_grid = {'n_estimators': [50, 100, 200, 300, 400, 500],
              'max_depth': [1, 5, 10, 20, 30, 40, 50, None]}
# Create a GridSearchCV object
grid_search = GridSearchCV(clf, param_grid, cv=5)
# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)
# Print the best parameters and best score
print("Best parameters: ", grid_search.best_params_)
print("Best score: {:.2f}%".format(grid_search.best_score_*100))
# Make predictions on the test set
y_pred = grid_search.predict(X_test)
# Calculate the accuracy of the model
print('The accuracy of the RandomForest is: ', metrics.accuracy_score(y_pred, y_test))

# =============================================================================
# LogisticRegression model
# =============================================================================
from sklearn.linear_model import LogisticRegression
# Create a logistic regression classifier object
clf = LogisticRegression()
# Fit the classifier to the training data
clf.fit(X_train, y_train)
# Make predictions on the test set
y_pred = clf.predict(X_test)
# Calculate the accuracy of the model
print('The accuracy of the Logistic Regression is: ', metrics.accuracy_score(y_pred, y_test))

# =============================================================================
# Tenserfolow With Keras
# =============================================================================
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Initializing Neural Network
clf = Sequential()
# Adding Input layer and hidden layer
clf.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))
# Adding Output Layer
clf.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Compiling Neural Network
clf.compile(optimizer = Adam(learning_rate=0.01), loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting our model 
clf.fit(X_train, y_train, batch_size = 10, epochs = 100)
# Make predictions on the test set
y_pred = clf.predict(X_test)
y_pred = (y_pred > 0.5)
# Calculate the accuracy of the model
print('The accuracy of the Neural Network is: ', metrics.accuracy_score(y_pred, y_test))

# =============================================================================
# Tenserfolow With Keras with automatically select the best hyperparameters for a neural network
# =============================================================================
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def create_model(optimizer='adam', hidden_layers=1, hidden_units=6, learning_rate=0.01):
    # Initializing Neural Network
    model = Sequential()
    # Adding Input layer and hidden layer
    model.add(Dense(units=hidden_units, kernel_initializer='uniform', activation='relu', input_dim=X_train.shape[1]))
    for _ in range(hidden_layers - 1):
        model.add(Dense(units=hidden_units, kernel_initializer='uniform', activation='relu'))
    # Adding Output Layer
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    # Compiling Neural Network
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model
# create the model
clf = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10)
# define the grid search parameters
param_grid = {'hidden_layers': [1, 2, 3],
              'hidden_units': [6, 8, 10],
              'learning_rate': [0.01, 0.02, 0.05],
              'optimizer': ['adam', 'rmsprop']}
# create grid search
grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
# fit grid search
grid_result = grid.fit(X_train, y_train)
# print best parameter after tuning 
print("Best parameters found: ",grid_result.best_params_)
print("Best score: {:.2f}%".format(grid_result.best_score_*100))
# Make predictions on the test set
y_pred = grid.predict(X_test)
y_pred = (y_pred > 0.5)
print('The accuracy of the Neural Network is: ', metrics.accuracy_score(y_pred, y_test))

# =============================================================================
# Tenserfolow With Keras with RandomizedSearchCV
# =============================================================================
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

def create_model(optimizer='adam', hidden_layers=1, hidden_units=6, learning_rate=0.01):
    # Initializing Neural Network
    model = Sequential()
    # Adding Input layer and hidden layer
    model.add(Dense(units=hidden_units, kernel_initializer='uniform', activation='relu', input_dim=X_train.shape[1]))
    for _ in range(hidden_layers - 1):
        model.add(Dense(units=hidden_units, kernel_initializer='uniform', activation='relu'))
    # Adding Output Layer
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    # Compiling Neural Network
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model
# create the model
clf = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10)
# define the randomized search parameters
param_grid = {'hidden_layers': sp_randint(1, 4),
              'hidden_units': sp_randint(6, 11),
              'learning_rate': [0.01, 0.02, 0.05],
              'optimizer': ['adam', 'rmsprop']}
# create randomized search
random_search = RandomizedSearchCV(clf, param_grid, n_iter=20, cv=5)
# fit randomized search
random_search_result = random_search.fit(X_train, y_train)
# print best parameter after tuning 
print("Best parameters found: ",random_search_result.best_params_)
print("Best score: {:.2f}%".format(grid_result.best_score_*100))
# Make predictions on the test set
y_pred = random_search.predict(X_test)
y_pred = (y_pred > 0.5)
print('The accuracy of the Neural Network is: ', metrics.accuracy_score(y_pred, y_test))


# =============================================================================
# KNeighborsClassifier model
# =============================================================================
from sklearn.neighbors import KNeighborsClassifier

# Create a KNN classifier object
clf = KNeighborsClassifier()
# Fit the classifier to the training data
clf.fit(X_train, y_train)
# Make predictions on the test set
y_pred = clf.predict(X_test)
print('The accuracy of the KNeighborsClassifier is: ', metrics.accuracy_score(y_pred, y_test))
                                            

# =============================================================================
# KNeighborsClassifier model with grid serch
# =============================================================================
from sklearn.model_selection import GridSearchCV

# define the grid search parameters
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
# create grid search
grid = GridSearchCV(clf, param_grid, cv=5)
# fit grid search
grid_result = grid.fit(X_train, y_train)
# print best parameter after tuning 
print("Best parameters found: ",grid_result.best_params_)
print("Best score: {:.2f}%".format(grid_result.best_score_*100))
# Make predictions on the test set
y_pred = grid.predict(X_test)
print('The accuracy of the KNeighborsClassifier is: ', metrics.accuracy_score(y_pred, y_test))


# =============================================================================
# Now final data set importing where the data not having the status of the cutomer who eligble of the loan or not 
# This real data  that we need find the coutomer is eligible or not
# =============================================================================
testdata = pd.read_csv(r'C:\Users\heman\Desktop\Loan Eligibility Prediction\Original Data\loan-test.csv')
testdata.head()
# data engineering test data the come from applicanth those need loan 
testdata.info()
testdata.isnull().sum()
# Now filling the missing values 
testdata['Gender'].fillna(testdata['Gender'].mode()[0], inplace=True)
testdata['Married'].fillna(testdata['Married'].mode()[0], inplace=True) 
testdata['Dependents'].fillna(testdata['Dependents'].mode()[0], inplace=True)
testdata['Self_Employed'].fillna(testdata['Self_Employed'].mode()[0], inplace=True)
testdata['Loan_Amount_Term'].fillna(testdata['Loan_Amount_Term'].mode()[0], inplace=True)
testdata['Credit_History'].fillna(testdata['Credit_History'].mode()[0], inplace=True)

testdata.boxplot(column='LoanAmount')
testdata.boxplot(column='ApplicantIncome')

testdata['LoanAmount'] = testdata['LoanAmount'].fillna(testdata['LoanAmount'].mean())
testdata['LoanAmount_log'] = np.log(testdata['LoanAmount'])

testdata['TotalIncome'] = testdata['ApplicantIncome'] + testdata['CoapplicantIncome']
testdata['TotalIncome_log'] = np.log(testdata['TotalIncome'])
testdata.head()


#  now convert the the categrical values into numeric formate
test = testdata.iloc[:,np.r_[1:5,9:11,13:15]].values
for i in range(0,5):
    test[:,i] = labelencoder_X.fit_transform(test[:,i])
    
test[:,7] = labelencoder_X.fit_transform(test[:,7])
test
#  now scale the data 
test = ss.fit_transform(test)
pred = NBClassifier.predict(test)
pred


# Note: we can also apply the SVM and Randomforest algorithms for checking accuracy of the model 


