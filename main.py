# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 13:13:22 2020

@author: Eric
"""

# -*- coding: utf-8 -*-
"""
Eric Born
Date: 27 June 2020
Predicting churn in telcom data
utilizing various machine learning algorithms
https://www.kaggle.com/jpacse/datasets-for-churn-telecom
"""

import os
import time
import operator
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sys import exit
from sklearn import svm
from sklearn import tree
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, recall_score,\
                            classification_report

# Set display options for dataframes
#pd.set_option('display.max_rows', 100)
#pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 60)

# set seaborn to dark backgrounds
sns.set_style("darkgrid")

################
# Start data import and cleanup
################
timings_list = []

# start timing
timings_list.append(['Global duration:', time.time()])
timings_list.append(['Data clean up duration:', time.time()])

# setup input directory and filename
data = 'celldata'
input_dir = r'C:\Code projects\git projects\ML-Pipeline'
csv_file = os.path.join(input_dir, data + '.csv')

# read csv file into dataframe
try:
    main_df = pd.read_csv(csv_file)
    print('opened data file: ', data, '\n')

except Exception as e:
    print(e)
    exit('Failed to read data from: '+ str(data)+'.csv')

# describe the total rows and columns
print('The total length of the dataframe is', main_df.shape[0], 'rows',
      'and the width is', main_df.shape[1], 'columns')

# total Churn/No Churn
print('The churn stats of the dataframe is\n', 
      'No:', main_df.Churn.value_counts()[0], '\n',
      'Yes:', main_df.Churn.value_counts()[1], '\n',
      'Ratio of', round(main_df.Churn.value_counts()[0]/
                  main_df.Churn.value_counts()[1], 2),': 1')

# find count of columns with null and 0 values
nulls = {}
zero_values = {}
for col in main_df.columns:
    if (main_df[col] == 0).sum() > 0:
        zero_values.update({col:(main_df[col] == 0).sum()})
    if main_df[col].isnull().sum() > 0:
        nulls.update({col:main_df[col].isnull().sum()})

# output all null columns and counts
nulls

# output all columns and counts with a 0 value
zero_values

# print highest null value divided by total rows
print('Highest percentage of null values:',
      100*round(max(nulls.items(), key=operator.itemgetter(1))[1] / 
            main_df.shape[0],4))

# create a class label using 0 or 1 to indicate churnning team
# 0 = no churn
# 1 = churn
main_df['Churn'] = main_df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Create an area code column out of the ServiceArea column
main_df['AreaCode'] = main_df['ServiceArea'].str[-3:]

# convery categorical data to numerical
main_df['ChildrenInHH'] = main_df['ChildrenInHH'].apply(lambda x: 1 \
                                                        if x == 'Yes' else 0)

main_df['HandsetRefurbished'] = main_df['HandsetRefurbished'].apply(lambda x: \
                                                        1 if x == 'Yes' else 0)

main_df['HandsetWebCapable'] = main_df['HandsetWebCapable'].apply(lambda x: \
                                                        1 if x == 'Yes' else 0)

main_df['TruckOwner'] = main_df['TruckOwner'].apply(lambda x: 1\
                                                    if x == 'Yes' else 0)

main_df['RVOwner'] = main_df['RVOwner'].apply(lambda x: 1\
                                                    if x == 'Yes' else 0)

main_df['Homeownership'] = main_df['Homeownership'].apply(lambda x: 1\
                                                    if x == 'Known' else 0)

main_df['BuysViaMailOrder'] = main_df['BuysViaMailOrder'].apply(lambda x: 1\
                                                    if x == 'Yes' else 0)

main_df['RespondsToMailOffers'] = main_df['RespondsToMailOffers'].apply(\
                                                                  lambda x: 1\
                                                          if x == 'Yes' else 0)

main_df['OptOutMailings'] = main_df['OptOutMailings'].apply(lambda x: 1\
                                                    if x == 'Yes' else 0)

main_df['NonUSTravel'] = main_df['NonUSTravel'].apply(lambda x: 1\
                                                    if x == 'Yes' else 0)

main_df['OwnsComputer'] = main_df['OwnsComputer'].apply(lambda x: 1\
                                                    if x == 'Yes' else 0)

main_df['HasCreditCard'] = main_df['HasCreditCard'].apply(lambda x: 1\
                                                    if x == 'Yes' else 0)

main_df['NewCellphoneUser'] = main_df['NewCellphoneUser'].apply(lambda x: 1\
                                                    if x == 'Yes' else 0)

main_df['NotNewCellphoneUser'] = main_df['NotNewCellphoneUser'].apply(
                                              lambda x: 1 if x == 'Yes' else 0)

main_df['OwnsMotorcycle'] = main_df['OwnsMotorcycle'].apply(lambda x: 1\
                                                    if x == 'Yes' else 0)

main_df['MadeCallToRetentionTeam'] = main_df['MadeCallToRetentionTeam'].apply(\
                                              lambda x: 1 if x == 'Yes' else 0)
  
main_df.loc[main_df['HandsetPrice'] == 'Unknown', 'HandsetPrice'] = 0
                                              
# 1-Highest
# 2-High
# 3-Good
# 4-Medium
# 5-Low
# 6-VeryLow
# 7-Lowest
main_df['CreditRating'] = main_df['CreditRating'].str[:1]


# Town = 1
# Suburban = 2
# Rural = 3
# Other = 4
main_df.loc[main_df['PrizmCode'] == 'Town', 'PrizmCode'] = 1
main_df.loc[main_df['PrizmCode'] == 'Suburban', 'PrizmCode'] = 2
main_df.loc[main_df['PrizmCode'] == 'Rural', 'PrizmCode'] = 3
main_df.loc[main_df['PrizmCode'] == 'Other', 'PrizmCode'] = 4


# Clerical = 1
# Crafts = 2
# Homemaker = 3
# Other = 4
# Professional = 5
# Retired = 6
# Self = 7
# Student = 8
main_df.loc[main_df['Occupation'] == 'Clerical', 'Occupation'] = 1
main_df.loc[main_df['Occupation'] == 'Crafts', 'Occupation'] = 2
main_df.loc[main_df['Occupation'] == 'Homemaker', 'Occupation'] = 3
main_df.loc[main_df['Occupation'] == 'Other', 'Occupation'] = 4
main_df.loc[main_df['Occupation'] == 'Professional', 'Occupation'] = 5
main_df.loc[main_df['Occupation'] == 'Retired', 'Occupation'] = 6
main_df.loc[main_df['Occupation'] == 'Self', 'Occupation'] = 7
main_df.loc[main_df['Occupation'] == 'Student', 'Occupation'] = 8

# yes = 1
# no = 2
# unknown = 3
main_df.loc[main_df['MaritalStatus'] == 'Yes', 'MaritalStatus'] = 1
main_df.loc[main_df['MaritalStatus'] == 'No', 'MaritalStatus'] = 2
main_df.loc[main_df['MaritalStatus'] == 'Unknown', 'MaritalStatus'] = 3


# replaces all NaN with 0
main_df = main_df.fillna(0)

# convert object dtypes to int
main_df['HandsetPrice'] = main_df['HandsetPrice'].astype('int64')
main_df['CreditRating'] = main_df['CreditRating'].astype('int64')
main_df['AreaCode'] = main_df['AreaCode'].astype('int64')

# remove columns customerId and ServiceArea
main_df.drop(main_df.columns[[0,26]], axis = 1, inplace = True)

# x stores all columns except for the Churn column
main_x = main_df.drop('Churn', 1)

# replaces all NaN with 0
#main_x = main_x.fillna(0)

# y stores only the Churn column since its used as a predictor
main_y = main_df['Churn']

# setup empty list to store all of the models accuracy
global_accuracy = []

# end timing
timings_list.append(['clean time end', time.time()])

def plot_churn_values(main_df, column=''):
    plt.figure(figsize=(7,7))
    plt.grid(True)
    plt.bar(main_df[column][main_df.Churn==1].value_counts().index, 
            main_df[column][main_df.Churn==1].value_counts().values)
    plt.title(f'{column}')
    plt.xticks(rotation=-90)

plot_churn_values(main_df, column='CreditRating')

################
# End data import and cleanup
################

################
# Start attribute selection with various methods
################
# start timing
timings_list.append(['Features selection duration:', time.time()])

#######
# Start Pearsons corerelation
#######

# create a correlation object
cor = main_df.corr()

# correlation with output variable
cor_target = abs(cor['Churn'])

# selecting features correlated greater than 0.5
relevant_features = cor_target[cor_target > 0.1]

# create a dataframe from the highest correlated item
pearsons_df = main_df[['CurrentEquipmentDays']]

#########################
# results for the top 5 and top 10 attributes
print('Pearsons top attribute:')
print(list(pearsons_df.columns), '\n')

# Create correlation matrix for > 0.5 correlation
fig = plt.figure(figsize=(10,10))
sns.heatmap(main_df[['Churn', 'CurrentEquipmentDays']].corr(), 
            annot=True, vmin=-1, vmax=1, center=0, cmap="coolwarm", fmt='.2f',
            linewidths=2, linecolor='black')
plt.title('Correlation matrix for > 0.5 correlation', y=1.1)

#######
# Start Ordinary Least Squares
#######

# creates a list of column names
cols = list(main_x.columns)
# sets a max value
pmax = 1

# while loop that calculates the p values of each attribute using the 
# OLS model and eliminiates the highest value from the list of columns
# loop breaks if all columns remaining have less than 0.05 p value
# or all columns are removed
try:
    while (len(cols)>0):
        p = []
        ols_x1 = sm.add_constant(main_x[cols].values)
        model = sm.OLS(main_y,ols_x1).fit()
        p = pd.Series(model.pvalues.values[1:], index = cols)
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if(pmax > 0.05):
            cols.remove(feature_with_p_max)
        else:
            break
except Exception as e:
    print(e)
    exit('Failed to reduce features for ols dataset')
    
# sets and prints the remaining unremoved features
selected_features_BE = cols
print(selected_features_BE)

# creates a dataframe with the ols selected columns
ols_df = main_df[selected_features_BE]

######
# End Ordinary Least Squares
######

######
# Start Recursive Feature Elimination
######

#####
# only used to determine optimum number of attributes
#####

# Total number of features
nof_list = np.arange(1,57)            
high_score = 0

# Variable to store the optimum features
nof = 0           
score_list = []
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(main_x, main_y, 
                                            test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score > high_score):
        high_score = score
        nof = nof_list[n]

# 55 features score of 0.033387
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))

#####
# only used to determine optimum number of attributes
#####

# setup column list and regression model
cols = list(main_x.columns)
model = LinearRegression()

#Initializing RFE model with 55 features
rfe = RFE(model, 55)   
          
#Transforming data using RFE
X_rfe = rfe.fit_transform(main_x,main_y)  

#Fitting the data to model
model.fit(X_rfe,main_y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index

# output the selected features
print(selected_features_rfe)

# creates a dataframe with the rfe selected columns
rfe_df = main_df[selected_features_rfe]

print('Total features:', len(selected_features_rfe),
    '\nOrdinary Least Squares features:\n', selected_features_rfe)

#######
# End Recursive Feature Elimination
#######

#######
# Start lasso method
#######

# build the model with a cross validation set to 5
reg = LassoCV(cv = 5)

# fit the model
reg.fit(main_x, main_y)

# build the coefficients between each attribute
coef = pd.Series(reg.coef_, index = main_x.columns)

# creates a dataframe based on the 32 columns selected from lasso
lasso_df = main_df[coef[coef.values != 0].index]

# output total attributes chosen and discarded
print("Total Features: " + str(sum(coef != 0)),
      '\nLasso features:\n',list(lasso_df.columns))

#######
# End lasso method
#######

# end timing
timings_list.append(['features time end', time.time()])

# output min, max and mean values for the lasso features
lasso_df.min()
lasso_df.max()
lasso_df.mean()

################
# End attribute selection with various methods
################

################
# Start five number summary setup 
################

# create an empty list for the columns from each feature selection
full_feature_list = []

# append the columns from each feature selection
full_feature_list.append(list(pearsons_df.columns))
full_feature_list.append(list(ols_df.columns))
full_feature_list.append(list(rfe_df.columns))
full_feature_list.append(list(lasso_df.columns))

# flatten the list of lists into a single list
flat_list = []
for sublist in full_feature_list:
    for item in sublist:
        flat_list.append(item)

# convert the list to a set to keep only unique values
full_feature_set = set(flat_list)

# use describe to view the 5 number summary for each of the columns
main_df[full_feature_set].describe()


main_df[full_feature_set].min()
main_df[full_feature_set].max()
main_df[full_feature_set].mean()

################
# End five number summary setup 
################

################
# Start building scaled dataframes
################
# start timing
timings_list.append(['Dataframe build duration:', time.time()])

# Setup scalers X datasets
scaler = StandardScaler()
scaler.fit(pearsons_df)
pearsons_df_scaled = scaler.transform(pearsons_df)

# pear_five split dataset into 33% test 66% training
(pearsons_scaled_df_train_x, pearsons_scaled_df_test_x, 
 pearsons_scaled_df_train_y, pearsons_scaled_df_test_y) = (
        train_test_split(pearsons_df_scaled, main_y, test_size = 0.333, 
                         random_state=1337))

#pearsons_scaled_df_train_x
#pearsons_scaled_df_test_x
#pearsons_scaled_df_train_y
#pearsons_scaled_df_test_y

# Setup scalers X dataset
scaler = StandardScaler()
scaler.fit(ols_df)
ols_df_scaled = scaler.transform(ols_df)

# ols_df split dataset into 33% test 66% training
(ols_scaled_df_train_x, ols_scaled_df_test_x, ols_scaled_df_train_y,
 ols_scaled_df_test_y) = (
        train_test_split(ols_df_scaled, main_y, test_size = 0.333, 
                         random_state=1337))

#ols_scaled_df_train_x
#ols_scaled_df_test_x
#ols_scaled_df_train_y
#ols_scaled_df_test_y

# Setup scalers X dataset
scaler = StandardScaler()
scaler.fit(rfe_df)
rfe_df_scaled = scaler.transform(rfe_df)

# ols_df split dataset into 33% test 66% training
(rfe_scaled_df_train_x, rfe_scaled_df_test_x, 
 rfe_scaled_df_train_y, rfe_scaled_df_test_y) = (
        train_test_split(rfe_df_scaled, main_y, test_size = 0.333, 
                         random_state=1337))
#rfe_scaled_df_train_x
#rfe_scaled_df_test_x
#rfe_scaled_df_train_y
#rfe_scaled_df_test_y

# Setup scalers X dataset
scaler = StandardScaler()
scaler.fit(lasso_df)
lasso_df_scaled = scaler.transform(lasso_df)

# lasso split dataset into 33% test 66% training
(lasso_scaled_df_train_x, lasso_scaled_df_test_x, lasso_scaled_df_train_y, 
 lasso_scaled_df_test_y) = (train_test_split(lasso_df_scaled, main_y, 
                                             test_size = 0.333, 
                                             random_state=1337))
#lasso_scaled_df_train_x
#lasso_scaled_df_test_x
#lasso_scaled_df_train_y
#lasso_scaled_df_test_y

################
# End building scaled dataframes
################

################
# Start building test/train datasets with various attribute selections
################

# dataframes with selected attribtues from 5 methods for attribute eliminiation
#pearsons_df
#pear_ten_df
#ols_df
#rfe_df
#lasso_df

# pear_five split dataset into 33% test 66% training
(pearsons_df_train_x, pearsons_df_test_x, 
 pearsons_df_train_y, pearsons_df_test_y) = (
        train_test_split(pearsons_df, main_y, test_size = 0.333, 
                         random_state=1337))

#pearsons_df_train_x
#pearsons_df_test_x
#pearsons_df_train_y
#pearsons_df_test_y

# ols_df split dataset into 33% test 66% training
ols_df_train_x, ols_df_test_x, ols_df_train_y, ols_df_test_y = (
        train_test_split(ols_df, main_y, test_size = 0.333, 
                         random_state=1337))

#ols_df_train_x
#ols_df_test_x
#ols_df_train_y
#ols_df_test_y

# ols_df split dataset into 33% test 66% training
rfe_df_train_x, rfe_df_test_x, rfe_df_train_y, rfe_df_test_y = (
        train_test_split(rfe_df, main_y, test_size = 0.333, 
                         random_state=1337))

#rfe_df_train_x
#rfe_df_test_x
#rfe_df_train_y
#rfe_df_test_y

# ols_df split dataset into 33% test 66% training
lasso_df_train_x, lasso_df_test_x, lasso_df_train_y, lasso_df_test_y = (
        train_test_split(lasso_df, main_y, test_size = 0.333, 
                         random_state=1337))

#lasso_df_train_x
#lasso_df_test_x
#lasso_df_train_y
#lasso_df_test_y

# ols_df split dataset into 33% test 66% training
full_df_train_x, full_df_test_x, full_df_train_y, full_df_test_y = (
        train_test_split(main_x, main_y, test_size = 0.333, 
                         random_state=1337))

#full_df_train_x
#full_df_test_x
#full_df_train_y
#full_df_test_y

# end timing
timings_list.append(['frame build time end', time.time()])

################
# End building test/train datasets with various attribute selections
################

########
# Start counts of the churn totals
########

# store number in variable for churn
churn = sum(main_df.Churn == 1)
pearsons_train_churn = sum(pearsons_df_train_y == 1)
pearsons_test_churn  = sum(pearsons_df_test_y  == 1)
ols_train_churn       = sum(ols_df_train_y       == 1)
ols_test_churn        = sum(ols_df_test_y        == 1)
rfe_train_churn       = sum(rfe_df_train_y       == 1)
rfe_test_churn        = sum(rfe_df_test_y        == 1)
lasso_train_churn     = sum(lasso_df_train_y     == 1)
lasso_test_churn      = sum(lasso_df_test_y      == 1)

# store number in variable for no churn
no_churn = sum(main_df.Churn == 0)
pearsons_train_no_churn = sum(pearsons_df_train_y == 0)
pearsons_test_no_churn  = sum(pearsons_df_test_y  == 0)
ols_train_no_churn       = sum(ols_df_train_y       == 0)
ols_test_no_churn        = sum(ols_df_test_y        == 0)
rfe_train_no_churn       = sum(rfe_df_train_y       == 0)
rfe_test_no_churn        = sum(rfe_df_test_y        == 0)
lasso_train_no_churn     = sum(lasso_df_train_y     == 0)
lasso_test_no_churn      = sum(lasso_df_test_y      == 0)

# create a ratio of the churns
ratio = round(churn / no_churn, 4)

pear_five_train_ratio = round(pearsons_train_churn / \
                              pearsons_train_no_churn, 4)
pear_five_test_ratio  = round(pearsons_test_churn / \
                              pearsons_test_no_churn, 4)

ols_train_ratio = round(ols_train_churn / ols_train_no_churn, 4)
ols_test_ratio  = round(ols_test_churn / ols_test_no_churn, 4)

rfe_train_ratio = round(rfe_train_churn / rfe_train_no_churn, 4)
rfe_test_ratio  = round(rfe_test_churn / rfe_test_no_churn, 4)

lasso_train_ratio = round(lasso_train_churn / lasso_train_no_churn, 4)
lasso_test_ratio  = round(lasso_test_churn / lasso_test_no_churn, 4)

# Print churn ratios
print('\nOriginal dataset churn ratios\n','churn : no churn\n', str(ratio)+
      ' :   1')
print('\nPearson five training churn ratios\n','churn : no churn\n', 
      str(pear_five_train_ratio)+' :   1')
print('\nPearson five test churn ratios\n','churn : no churn\n', 
      str(pear_five_test_ratio)+' :   1')
print('\nOls training churn ratios\n','churn : no churn\n', 
      str(ols_train_ratio)+' :   1')
print('\nOls test churn ratios\n','churn : no churn\n', 
      str(ols_test_ratio)+' :   1')
print('\nRfe training churn ratios\n','churn : no churn\n', 
      str(rfe_train_ratio)+' :   1')
print('\nRfe test churn ratios\n','churn : no churn\n', 
      str(rfe_test_ratio)+' :   1')
print('\nLasso training churn ratios\n','churn : no churn\n', 
      str(lasso_train_ratio)+' :   1')
print('\nLasso test churn ratios\n','churn : no churn\n', 
      str(lasso_test_ratio)+' :   1')

########
# End counts of the churn totals
########

# Create list just for the algorithm durations
algorithm_duration_list = []

################
# Start building non-scaled algorithms
################
# start timing
timings_list.append(['Non-scaled duration:', time.time()])
timings_list.append(['Decision tree duration:', time.time()])

#######
# Start decision tree
#######

# start time
algorithm_duration_list.append(time.time()) 

# Create a decisions tree classifier
pearsons_tree_clf = tree.DecisionTreeClassifier(criterion = 'entropy')

# Train the classifier on pearsons top 5 attributes
pearsons_tree_clf = pearsons_tree_clf.fit(pearsons_df_train_x, 
                                            pearsons_df_train_y)

# Predict on pearsons top 5 attributes
pearsons_tree_clf_prediction = pearsons_tree_clf.predict(pearsons_df_test_x)

# end time
algorithm_duration_list.append(time.time()) 

####
# End five
####

####
# Start ols_df
####

# start time
algorithm_duration_list.append(time.time()) 

# Create a decisions tree classifier
ols_df_tree_clf = tree.DecisionTreeClassifier(criterion = 'entropy')

# Train the classifier on ols attributes
ols_df_tree_clf = ols_df_tree_clf.fit(ols_df_train_x, 
                                      ols_df_train_y)

# Predict on ols attributes
ols_df_prediction = ols_df_tree_clf.predict(ols_df_test_x)

# end time
algorithm_duration_list.append(time.time()) 


####
# End ols_df
####

####
# Start rfe_df
####

# start time
algorithm_duration_list.append(time.time()) 

# Create a decisions tree classifier
rfe_df_tree_clf = tree.DecisionTreeClassifier(criterion = 'entropy')

# Train the classifier on rfe attributes
rfe_df_tree_clf = rfe_df_tree_clf.fit(rfe_df_train_x, 
                                      rfe_df_train_y)

# Predict on rfe attributes
rfe_df_prediction = rfe_df_tree_clf.predict(rfe_df_test_x)

# end time
algorithm_duration_list.append(time.time()) 

####
# End rfe_df
####

####
# Start lasso_df
####

# start time
algorithm_duration_list.append(time.time()) 

# Create a decisions tree classifier
lasso_df_tree_clf = tree.DecisionTreeClassifier(criterion = 'entropy')

# Train the classifier on lasso attributes
lasso_df_tree_clf = lasso_df_tree_clf.fit(lasso_df_train_x, 
                                          lasso_df_train_y)

# Predict on lasso attributes
lasso_df_prediction = lasso_df_tree_clf.predict(lasso_df_test_x)

# end time
algorithm_duration_list.append(time.time()) 

####
# End lasso_df
####

# Store predictions
global_accuracy.append(100-(round(np.mean(pearsons_tree_clf_prediction 
                                          != pearsons_df_test_y) * 100, 2)))
global_accuracy.append(100-(round(np.mean(ols_df_prediction 
                                          != ols_df_test_y) * 100, 2)))
global_accuracy.append(100-(round(np.mean(rfe_df_prediction 
                                          != rfe_df_test_y) * 100, 2)))
global_accuracy.append(100-(round(np.mean(lasso_df_prediction 
                                          != lasso_df_test_y) * 100, 2)))
#######
# End decision tree
#######

# end timing
timings_list.append(['tree time end', time.time()])

#######
# Start naive bayes
#######

# start timing
timings_list.append(['Naive Bayes duration:', time.time()])
algorithm_duration_list.append(time.time()) 

# Create a naive bayes classifier
pearsons_gnb_clf = GaussianNB()

# Train the classifier on pearsons top 5 attributes
pearsons_gnb_clf = pearsons_gnb_clf.fit(pearsons_df_train_x, 
                                          pearsons_df_train_y)

# Predict on pearsons top 5 attributes
pearsons_gnb_clf_prediction = pearsons_gnb_clf.predict(pearsons_df_test_x)

# end time
algorithm_duration_list.append(time.time()) 

####
# End five
####

####
# Start ols_df
####

# start timing
algorithm_duration_list.append(time.time())

# Create a naive bayes classifier
ols_df_gnb_clf = GaussianNB()

# Train the classifier on ols attributes
ols_df_gnb_clf = ols_df_gnb_clf.fit(ols_df_train_x, 
                                    ols_df_train_y)

# Predict on ols attributes
ols_df_prediction = ols_df_gnb_clf.predict(ols_df_test_x)

# end time
algorithm_duration_list.append(time.time()) 

####
# End ols_df
####

####
# Start rfe_df
####

# start timing
algorithm_duration_list.append(time.time()) 

# Create a naive bayes classifier
rfe_df_gnb_clf = GaussianNB()

# Train the classifier on rfe attributes
rfe_df_gnb_clf = rfe_df_gnb_clf.fit(rfe_df_train_x, 
                                    rfe_df_train_y)

# Predict on rfe attributes
rfe_df_prediction = rfe_df_gnb_clf.predict(rfe_df_test_x)

# end time
algorithm_duration_list.append(time.time()) 

####
# End rfe_df
####

####
# Start lasso_df
####

# start timing
algorithm_duration_list.append(time.time()) 

# Create a naive bayes classifier
lasso_df_gnb_clf = GaussianNB()

# Train the classifier on lasso attributes
lasso_df_gnb_clf = lasso_df_gnb_clf.fit(lasso_df_train_x, 
                                        lasso_df_train_y)

# Predict on lasso attributes
lasso_df_prediction = lasso_df_gnb_clf.predict(lasso_df_test_x)

# end time
algorithm_duration_list.append(time.time()) 

####
# End lasso_df
####

# Store predictions
global_accuracy.append(100-(round(np.mean(pearsons_gnb_clf_prediction 
                                          != pearsons_df_test_y) * 100, 2)))
global_accuracy.append(100-(round(np.mean(ols_df_prediction 
                                          != ols_df_test_y) * 100, 2)))
global_accuracy.append(100-(round(np.mean(rfe_df_prediction 
                                          != rfe_df_test_y) * 100, 2)))
global_accuracy.append(100-(round(np.mean(lasso_df_prediction 
                                          != lasso_df_test_y) * 100, 2)))

#######
# End naive bayes
#######

# end timing
timings_list.append(['naive time end', time.time()])

#######
# Start Random Forest
#######

#####
## Start RF accuracy tests
#####
#
## Random forest classifiers using a range from
## 1 to 25 trees and from 1 to 10 depth of each tree
## Set random state to 1337 for repeatability
#
## Create a list to store the optimal tree and depth values 
## for each random forest classifier
#trees_depth = []
#
## create an empty list to store the rf accuracy at various settings
#rf_accuracy = []
#
## setup an empty dataframe for the rf tests
#rf_accuracy_df = pd.DataFrame()
#
#####
## Start pear five dataSet
#####
#
#pred_list = []
#
## RF with iterator
#for trees in range(1, 26):
#    for depth in range(1, 11):
#        rf_clf_test = RandomForestClassifier(n_estimators = trees, 
#                                    max_depth = depth, criterion ='entropy',
#                                    random_state = 1337)
#        rf_clf_test.fit(pearsons_df_train_x, pearsons_df_train_y)
#        pred_list.append([trees, depth, 
#                    round(np.mean(rf_clf_test.predict(pearsons_df_test_x) 
#                    == pearsons_df_test_y) 
#                    * 100, 2), 'pearsons'])
#         
##create a dataframe from the classifer data
#forest_df_1 = pd.DataFrame(pred_list, columns = ['Estimators', 'Depth',
#                                                 'Accuracy', 'Set'])
#
## append forest 1 to full df
#rf_accuracy_df = rf_accuracy_df.append(forest_df_1)
#
## store the lowest error rate value from the classifier
#ind = forest_df_1.loc[forest_df_1['Accuracy'] == 
#                      max(forest_df_1.Accuracy)].values
#
## pull out the number of trees and depth
#trees_depth.append(int(ind.item(0)))
#trees_depth.append(int(ind.item(1)))
#
## append the models accruacy to the accuracy list
#rf_accuracy.append(round(ind.item(2), 2))
#
#print('Pearsons:\nOptimal trees:', trees_depth[0],
#      '\nOptimal depth:', trees_depth[1])
#
#####
## End pear five dataSet
#####
#
#####
## Start ols dataSet
#####
#
#pred_list = []
#
#for trees in range(1, 26):
#    for depth in range(1, 11):
#        rf_clf_test = RandomForestClassifier(n_estimators = trees, 
#                                    max_depth = depth, criterion ='entropy',
#                                    random_state = 1337)
#        rf_clf_test.fit(ols_df_train_x, ols_df_train_y)
#        pred_list.append([trees, depth, 
#                    round(np.mean(rf_clf_test.predict(ols_df_test_x) 
#                    == ols_df_test_y) 
#                    * 100, 2), 'ols'])
#
## create a dataframe from the classifer data
#forest_df_3 = pd.DataFrame(pred_list, columns = ['Estimators', 'Depth',
#                                                 'Accuracy', 'Set'])
#
## append forest 3 to full df
#rf_accuracy_df = rf_accuracy_df.append(forest_df_3)
#
## store the lowest error rate value from the classifier
#ind = forest_df_3.loc[forest_df_3['Accuracy'] == 
#                      max(forest_df_3.Accuracy)].values
#                      
## pull out the number of trees and depth
#trees_depth.append(int(ind.item(0)))
#trees_depth.append(int(ind.item(1)))
#
## append the models accruacy to the accuracy list
#rf_accuracy.append(round(ind.item(2), 2))
#
#print('OLS:\nOptimal trees:', trees_depth[2],
#      '\nOptimal depth:', trees_depth[3])
#
#
#####
## End ols dataSet
#####
#
#####
## Start rfe dataSet
#####
#
#pred_list = []
#
#for trees in range(1, 26):
#    for depth in range(1, 11):
#        rf_clf_test = RandomForestClassifier(n_estimators = trees, 
#                                    max_depth = depth, criterion ='entropy',
#                                    random_state = 1337)
#        rf_clf_test.fit(rfe_df_train_x, rfe_df_train_y)
#        pred_list.append([trees, depth, 
#                    round(np.mean(rf_clf_test.predict(rfe_df_test_x) 
#                    == rfe_df_test_y) 
#                    * 100, 2), 'rfe'])
#
## create a dataframe from the classifer data
#forest_df_4 = pd.DataFrame(pred_list, columns = ['Estimators', 'Depth',
#                                                 'Accuracy', 'Set'])
#
## append forest 4 to full df
#rf_accuracy_df = rf_accuracy_df.append(forest_df_4)
#
## store the lowest error rate value from the classifier
#ind = forest_df_4.loc[forest_df_4['Accuracy'] == 
#                      max(forest_df_4.Accuracy)].values
#                      
## pull out the number of trees and depth
#trees_depth.append(int(ind.item(0)))
#trees_depth.append(int(ind.item(1)))
#
## append the models accruacy to the accuracy list
#rf_accuracy.append(round(ind.item(2), 2))
#
#print('RFE:\nOptimal trees:', trees_depth[4],
#      '\nOptimal depth:', trees_depth[5])
#
#####
## End rfe dataSet
#####
#
#####
## Start lasso dataSet
#####
#
#pred_list = []
#
#for trees in range(1, 26):
#    for depth in range(1, 11):
#        rf_clf_test = RandomForestClassifier(n_estimators = trees, 
#                                    max_depth = depth, criterion ='entropy',
#                                    random_state = 1337)
#        rf_clf_test.fit(lasso_df_train_x, lasso_df_train_y)
#        pred_list.append([trees, depth, 
#                    round(np.mean(rf_clf_test.predict(lasso_df_test_x) 
#                    == lasso_df_test_y) 
#                    * 100, 2), 'lasso'])
#
## create a dataframe from the classifer data
#forest_df_5 = pd.DataFrame(pred_list, columns = ['Estimators', 'Depth',
#                                                 'Accuracy', 'Set'])
#
## append forest 5 to full df
#rf_accuracy_df = rf_accuracy_df.append(forest_df_5)
#
## store the lowest error rate value from the classifier
#ind = forest_df_5.loc[forest_df_5['Accuracy'] == 
#                      max(forest_df_5.Accuracy)].values
#                      
## pull out the number of trees and depth
#trees_depth.append(int(ind.item(0)))
#trees_depth.append(int(ind.item(1)))
#
## append the models accruacy to the accuracy list
#rf_accuracy.append(round(ind.item(2), 2))
#
#print('Lasso:\nOptimal trees:', trees_depth[6],
#      '\nOptimal depth:', trees_depth[7])
#
#####
## End lasso dataSet
#####
#
#####
## Start full dataSet
#####
#
#pred_list = []
#
#for trees in range(1, 26):
#    for depth in range(1, 11):
#        rf_clf_test = RandomForestClassifier(n_estimators = trees, 
#                                    max_depth = depth, criterion ='entropy',
#                                    random_state = 1337)
#        rf_clf_test.fit(full_df_train_x, full_df_train_y)
#        pred_list.append([trees, depth, 
#                    round(np.mean(rf_clf_test.predict(full_df_test_x) 
#                    == full_df_test_y) 
#                    * 100, 2), 'full'])
#
## create a dataframe from the classifer data
#forest_df_6 = pd.DataFrame(pred_list, columns = ['Estimators', 'Depth',
#                                                 'Accuracy', 'Set'])
#
## append forest 6 to full df
#rf_accuracy_df = rf_accuracy_df.append(forest_df_6)
#    
## store the lowest error rate value from the classifier
#ind = forest_df_6.loc[forest_df_6['Accuracy'] == 
#                      max(forest_df_6.Accuracy)].values
#                      
## pull out the number of trees and depth
#trees_depth.append(int(ind.item(0)))
#trees_depth.append(int(ind.item(1)))
#
## append the models accruacy to the accuracy list
#rf_accuracy.append(round(ind.item(2), 2))
#
#print('Full:\nOptimal trees:', trees_depth[8],
#      '\nOptimal depth:', trees_depth[9])
#
#####
## End full dataSet
#####
#
## Create palette
#palette = dict(zip(rf_accuracy_df.Depth.unique(),
#                   sns.color_palette("tab10", 10)))
#
## Plot
#sns.relplot(x="Estimators", y="Accuracy",
#            hue="Depth", col="Set",
#            palette=palette, col_wrap=3,
#            height=3, aspect=1, facet_kws=dict(sharex=False),
#            kind="line", legend="full", data=rf_accuracy_df)
#
#
#####
## End RF accuracy tests
#####

####
# Start fixed value RFs
####

# start timing
timings_list.append(['Random forest duration:', time.time()])

# Random forest classifiers previously configured using a range from
# 1 to 25 trees and from 1 to 10 depth of each tree. 
# Optimal values for each dataset used below
# set random state to 1337 for repeatability

# Create a list to store the optimal tree and depth values 
# for each random forest classifier

####
# Start pearsons dataset
####

# start time
algorithm_duration_list.append(time.time())

#singular RF 
#rf_clf = RandomForestClassifier(n_estimators = 12, 
#                                max_depth = 9, criterion ='entropy',
#                                random_state = 1337)

rf_clf = RandomForestClassifier(n_estimators = 10, 
                                max_depth = 6, criterion ='entropy',
                                random_state = 1337)
rf_clf.fit(pearsons_df_train_x, pearsons_df_train_y)

# store predictions
rf_pearsons_pred = rf_clf.predict(pearsons_df_test_x)

# store accuracy
global_accuracy.append(100-(round(np.mean(rf_pearsons_pred
                                  != pearsons_df_test_y),2)))

# end time
algorithm_duration_list.append(time.time())
 
####
# End pearsons dataset
####

####
# Start ols dataset
####

# start time
algorithm_duration_list.append(time.time())

#singular RF 
rf_clf = RandomForestClassifier(n_estimators = 19, 
                                    max_depth = 10, criterion ='entropy',
                                    random_state = 1337)
rf_clf.fit(ols_df_train_x, ols_df_train_y)

# store predictions
rf_ols_pred = rf_clf.predict(ols_df_test_x)

# store accuracy
global_accuracy.append(100-(round(np.mean(rf_ols_pred != ols_df_test_y),2)))
                                  

# end time
algorithm_duration_list.append(time.time())

####
# End ols dataset
####

####
# Start rfe dataset
####

# start time
algorithm_duration_list.append(time.time())

#singular RF 
rf_clf = RandomForestClassifier(n_estimators = 25, 
                                    max_depth = 8, criterion ='entropy',
                                    random_state = 1337)
rf_clf.fit(rfe_df_train_x, rfe_df_train_y)

# store predictions
rf_rfe_pred = rf_clf.predict(rfe_df_test_x)

# store accuracy
global_accuracy.append(100-(round(np.mean(rf_rfe_pred != rfe_df_test_y),2)))
                                  
# end time
algorithm_duration_list.append(time.time())

####
# End rfe dataset
####

####
# Start lasso dataset
####

# start time
algorithm_duration_list.append(time.time())

#singular RF 
rf_clf = RandomForestClassifier(n_estimators = 21, 
                                    max_depth = 10, criterion ='entropy',
                                    random_state = 1337)
rf_clf.fit(lasso_df_train_x, lasso_df_train_y)

# store predictions
rf_lasso_pred = rf_clf.predict(lasso_df_test_x)


# store accuracy
global_accuracy.append(100-(round(np.mean(rf_lasso_pred
                                  != lasso_df_test_y),2)))

# end time
algorithm_duration_list.append(time.time())

####
# End lasso dataset
####

####
# Start full dataset
####

# start time
algorithm_duration_list.append(time.time())


#singular RF 
rf_clf = RandomForestClassifier(n_estimators = 16, 
                                    max_depth = 10, criterion ='entropy',
                                    random_state = 1337)
rf_clf.fit(full_df_train_x, full_df_train_y)

# store predictions
rf_full_pred = rf_clf.predict(full_df_test_x)

# store accuracy
global_accuracy.append(100-(round(np.mean(rf_full_pred != full_df_test_y),2)))                  

# end time
algorithm_duration_list.append(time.time())

####
# End full dataset
####

####
# End fixed value RFs
####

#############
# End Random Forest
#############

# end timing
timings_list.append(['forest time end', time.time()])
timings_list.append(['non scaled time end', time.time()])

################
# End building non-scaled algorithms
################

################
# Start building scaled algorithms
################

# start timing
timings_list.append(['Scaled time', time.time()])
timings_list.append(['Knn duration:', time.time()])

#######
# Start KNN
#######

########
## Start KNN evaluation
########
#
#####
## Start pearsons dataset
#####
#
## Create empty lists to store the models error rate and 
## accuracy across various K's
#error_rate = []
#accuracy = []
#k_value = []
#
## For loop to test the model using various neighbor sizes
## with k neighbors set to 3 to 25
#try:
#    for k in range (3, 105, 2):
#        # Create the classifier with neighbors set to k from the loop
#        knn_classifier = KNeighborsClassifier(n_neighbors = k)
#       
#        # Train the classifier
#        knn_classifier.fit(pearsons_scaled_df_train_x, 
#                           pearsons_scaled_df_train_y)
#        
#        # Perform predictions
#        pred_k = knn_classifier.predict(pearsons_scaled_df_test_x)
#        
#        # Store error rate and accuracy for particular K value
#        k_value.append(k)
#        error_rate.append(round(np.mean(
#                pred_k != pearsons_scaled_df_test_y) * 100, 2))
#        
#        accuracy.append(round(sum(
#                pred_k == pearsons_scaled_df_test_y) / len(pred_k) * 100, 2))
#        
#except Exception as e:
#    print(e)
#    print('Failed to build the KNN classifier.')
#
#for i in range(len(accuracy)):
#    print('The accuracy on the pearsons data when K =', k_value[i], 
#          'is:', accuracy[i])
#  
## create a plot to display the accuracy of the model across K
#fig = plt.figure(figsize=(10, 4))
#ax = plt.gca()
#plt.plot(range(3, 105, 2), accuracy, color ='blue',
#         marker = 'o', markerfacecolor = 'black', markersize = 10)
#plt.title('Accuracy vs. k for the pearsons dataset')
#plt.xlabel('Number of neighbors: k')
#plt.ylabel('Accuracy')
#
## find the index of the highest accuracy
#max_index = accuracy.index(max(accuracy))
#
## store the accuracy value
#highest_accuracy = accuracy[max_index]
#
## append to accuracy list
#global_accuracy.append(accuracy[max_index])
#
## store the best k value
#best_k = [k_value[max_index]]
#
#print('The most accurate k for the pearsons attribute selection is', 
#      k_value[max_index], 'at', accuracy[max_index],'%')
#
#####
## end pearsons dataset
#####
#
#####
## Start ols dataset
#####
#
## Create empty lists to store the models error rate and 
## accuracy across various K's
#error_rate = []
#accuracy = []
#k_value = []
#
## For loop to test the model using various neighbor sizes
## with k neighbors set to 3 to 25
#try:
#    for k in range (9, 31, 2):
#        # Create the classifier with neighbors set to k from the loop
#        knn_classifier = KNeighborsClassifier(n_neighbors = k)
#       
#        # Train the classifier
#        knn_classifier.fit(ols_scaled_df_train_x, 
#                           ols_scaled_df_train_y)
#        
#        # Perform predictions
#        pred_k = knn_classifier.predict(ols_scaled_df_test_x)
#        
#        # Store error rate and accuracy for particular K value
#        k_value.append(k)
#        error_rate.append(round(np.mean(
#                pred_k != ols_scaled_df_test_y) * 100, 2))
#        
#        accuracy.append(round(sum(
#                pred_k == ols_scaled_df_test_y) / len(pred_k) * 100, 2))
#        
#except Exception as e:
#    print(e)
#    print('failed to build the KNN classifier.')
#
#for i in range(len(accuracy)):
#    print('The accuracy on the ols data when K =', k_value[i], 
#          'is:', accuracy[i])
#    
## create a plot to display the accuracy of the model across K
#fig = plt.figure(figsize=(10, 4))
#ax = plt.gca()
#plt.plot(range(9, 31, 2), accuracy, color ='blue',
#         marker = 'o', markerfacecolor = 'black', markersize = 10)
#plt.title('Accuracy vs. k for the ols dataset')
#plt.xlabel('Number of neighbors: k')
#plt.ylabel('Accuracy')
#
## find the index of the highest accuracy
#max_index = accuracy.index(max(accuracy))
#
## store the accuracy value
#highest_accuracy = accuracy[max_index]
#
## append to accuracy list
#global_accuracy.append(accuracy[max_index])
#
## store the best k value
#best_k = [k_value[max_index]]
#
#print('The most accurate k for the ols attribute selection is', 
#      k_value[max_index], 'at', accuracy[max_index],'%')
#
#####
## end ols dataset
#####
#    
#####
## Start rfe dataset
#####
#
## Create empty lists to store the models error rate and 
## accuracy across various K's
#error_rate = []
#accuracy = []
#k_value = []
#
## For loop to test the model using various neighbor sizes
## with k neighbors set to 3 to 25
#try:
#    for k in range (9, 31, 2):
#        # Create the classifier with neighbors set to k from the loop
#        knn_classifier = KNeighborsClassifier(n_neighbors = k)
#       
#        # Train the classifier
#        knn_classifier.fit(rfe_scaled_df_train_x, 
#                           rfe_scaled_df_train_y)
#        
#        # Perform predictions
#        pred_k = knn_classifier.predict(rfe_scaled_df_test_x)
#        
#        # Store error rate and accuracy for particular K value
#        k_value.append(k)
#        error_rate.append(round(np.mean(
#                pred_k != rfe_scaled_df_test_y) * 100, 2))
#        
#        accuracy.append(round(sum(
#                pred_k == rfe_scaled_df_test_y) / len(pred_k) * 100, 2))
#        
#except Exception as e:
#    print(e)
#    print('failed to build the KNN classifier.')
#
#for i in range(len(accuracy)):
#    print('The accuracy on the rfe data when K =', k_value[i], 
#          'is:', accuracy[i])
#    
## create a plot to display the accuracy of the model across K
#fig = plt.figure(figsize=(10, 4))
#ax = plt.gca()
#plt.plot(range(9, 31, 2), accuracy, color ='blue',
#         marker = 'o', markerfacecolor = 'black', markersize = 10)
#plt.title('Accuracy vs. k for the rfe dataset')
#plt.xlabel('Number of neighbors: k')
#plt.ylabel('Accuracy')
#
## find the index of the highest accuracy
#max_index = accuracy.index(max(accuracy))
#
## store the accuracy value
#highest_accuracy = accuracy[max_index]
#
## append to accuracy list
#global_accuracy.append(accuracy[max_index])
#
## store the best k value
#best_k = [k_value[max_index]]
#
##best_set = 'pearson five'
#
#print('The most accurate k for the rfe selection is', 
#      k_value[max_index], 'at', accuracy[max_index],'%')
#
#####
## end rfe dataset
#####
#    
#####
## Start lasso dataset
#####
#
## Create empty lists to store the models error rate and 
## accuracy across various K's
#error_rate = []
#accuracy = []
#k_value = []
#
## For loop to test the model using various neighbor sizes
## with k neighbors set to 3 to 25
#try:
#    for k in range (9, 31, 2):
#        # Create the classifier with neighbors set to k from the loop
#        knn_classifier = KNeighborsClassifier(n_neighbors = k)
#       
#        # Train the classifier
#        knn_classifier.fit(lasso_scaled_df_train_x, 
#                           lasso_scaled_df_train_y)
#        
#        # Perform predictions
#        pred_k = knn_classifier.predict(lasso_scaled_df_test_x)
#        
#        # Store error rate and accuracy for particular K value
#        k_value.append(k)
#        error_rate.append(round(np.mean(
#                pred_k != lasso_scaled_df_test_y) * 100, 2))
#        
#        accuracy.append(round(sum(
#                pred_k == lasso_scaled_df_test_y) / len(pred_k) * 100, 2))
#        
#except Exception as e:
#    print(e)
#    print('failed to build the KNN classifier.')
#
#for i in range(len(accuracy)):
#    print('The accuracy on the lasso data when K =', k_value[i], 
#          'is:', accuracy[i])
#    
## create a plot to display the accuracy of the model across K
#fig = plt.figure(figsize=(10, 4))
#ax = plt.gca()
#plt.plot(range(9, 31, 2), accuracy, color ='blue',
#         marker = 'o', markerfacecolor = 'black', markersize = 10)
#plt.title('Accuracy vs. k for the lasso dataset')
#plt.xlabel('Number of neighbors: k')
#plt.ylabel('Accuracy')
#
## find the index of the highest accuracy
#max_index = accuracy.index(max(accuracy))
#
## store the accuracy value
#highest_accuracy = accuracy[max_index]
#
## append to accuracy list
#global_accuracy.append(accuracy[max_index])
#
## store the best k value
#best_k = [k_value[max_index]]
#
##best_set = 'pearson five'
#
#print('The most accurate k for the lasso attribute selection is', 
#      k_value[max_index], 'at', accuracy[max_index],'%')
#
#####
## end lasso dataset
#####
#
########
## end KNN evaluation
########


####
# Start pearsons dataset
####

# start time
algorithm_duration_list.append(time.time())

# initalize knn
knn_classifier = KNeighborsClassifier(n_neighbors = 83)

# Train the classifier
knn_classifier.fit(pearsons_scaled_df_train_x, 
                   pearsons_scaled_df_train_y)

# store predictions
knn_pearsons_pred = knn_classifier.predict(pearsons_scaled_df_test_x)

# store accuracy
global_accuracy.append(100-(round(np.mean(knn_pearsons_pred
                                          != pearsons_scaled_df_test_y),2)))

# end time
algorithm_duration_list.append(time.time())

####
# end pear-five dataset
####

####
# Start ols dataset
####

# start time
algorithm_duration_list.append(time.time())

# initalize knn
knn_classifier = KNeighborsClassifier(n_neighbors = 29)

# Train the classifier
knn_classifier.fit(ols_scaled_df_train_x, 
                   ols_scaled_df_train_y)

# store predictions
knn_ols_pred = knn_classifier.predict(ols_scaled_df_test_x)
        
# store accuracy
global_accuracy.append(100-(round(np.mean(knn_ols_pred
                                          != ols_scaled_df_test_y),2)))

# end time
algorithm_duration_list.append(time.time()) 

####
# end ols dataset
####
    
####
# Start rfe dataset
####

# start time
algorithm_duration_list.append(time.time())

# initalize knn
knn_classifier = KNeighborsClassifier(n_neighbors = 27)

# Train the classifier
knn_classifier.fit(rfe_scaled_df_train_x, 
                   rfe_scaled_df_train_y)

# store predictions
knn_rfe_pred = knn_classifier.predict(rfe_scaled_df_test_x) 
       
# store accuracy
global_accuracy.append(100-(round(np.mean(knn_rfe_pred
                                          != rfe_scaled_df_test_y),2)))

# end time
algorithm_duration_list.append(time.time()) 

####
# end rfe dataset
####
    
####
# Start lasso dataset
####

# start time
algorithm_duration_list.append(time.time())

# initalize knn
knn_classifier = KNeighborsClassifier(n_neighbors = 29)

# Train the classifier
knn_classifier.fit(lasso_scaled_df_train_x, 
                   lasso_scaled_df_train_y)
 
# store predictions
knn_lasso_pred = knn_classifier.predict(lasso_scaled_df_test_x)
       
# store accuracy
global_accuracy.append(100-(round(np.mean(knn_lasso_pred
                                          != rfe_scaled_df_test_y),2)))

####!!!!!WHY IS THIS IMPLEMENTED DIFFERENTLY ???!!!!
## Store accuracy
#global_accuracy.append(round(sum(pred_k == lasso_scaled_df_test_y) 
#                              / len(pred_k) * 100, 2))

# end time
algorithm_duration_list.append(time.time()) 

####
# end lasso dataset
####

#######
# End KNN
#######

# end timing
timings_list.append(['knn time end', time.time()])

#######
# Start linear SVM
#######

# start timing
timings_list.append(['SVM linear duration:', time.time()])

####
# Start pear five dataset
####

# start time
algorithm_duration_list.append(time.time())

# create a linear SVM classifier
svm_classifier_linear = svm.SVC(kernel = 'linear')

# fit the classifier on training data
svm_classifier_linear.fit(pearsons_scaled_df_train_x, 
                          pearsons_scaled_df_train_y)

# Predict using 2018 feature data
prediction_linear = svm_classifier_linear.predict(pearsons_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_linear != 
                                   pearsons_scaled_df_test_y) * 100, 2)))

# end time
algorithm_duration_list.append(time.time()) 

####
# End pear five dataset
####

####
# Start ols ten dataset
####

# start time
algorithm_duration_list.append(time.time())

# create a linear SVM classifier
svm_classifier_linear = svm.SVC(kernel = 'linear')

# fit the classifier on training data
svm_classifier_linear.fit(ols_scaled_df_train_x, 
                          ols_scaled_df_train_y)

# Predict using 2018 feature data
prediction_linear = svm_classifier_linear.predict(ols_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_linear != 
                                         ols_scaled_df_test_y) * 100, 2)))

# end time
algorithm_duration_list.append(time.time())

####
# End ols ten dataset
####

####
# Start rfe ten dataset
####

# start time
algorithm_duration_list.append(time.time())

# create a linear SVM classifier
svm_classifier_linear = svm.SVC(kernel = 'linear')

# fit the classifier on training data
svm_classifier_linear.fit(rfe_scaled_df_train_x, 
                          rfe_scaled_df_train_y)

# Predict using 2018 feature data
prediction_linear = svm_classifier_linear.predict(rfe_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_linear != 
                                         rfe_scaled_df_test_y) * 100, 2)))

# end time
algorithm_duration_list.append(time.time())

####
# End rfe ten dataset
####

####
# Start lasso ten dataset
####

# start time
algorithm_duration_list.append(time.time())

# create a linear SVM classifier
svm_classifier_linear = svm.SVC(kernel = 'linear')

# fit the classifier on training data
svm_classifier_linear.fit(lasso_scaled_df_train_x, 
                          lasso_scaled_df_train_y)

# Predict using 2018 feature data
prediction_linear = svm_classifier_linear.predict(lasso_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_linear != 
                                         lasso_scaled_df_test_y) * 100, 2)))

# end time
algorithm_duration_list.append(time.time())

####
# End lasso ten dataset
####

#######
# End linear SVM
#######

# end timing
timings_list.append(['svm linear time end', time.time()])

#######
# Start rbf SVM
#######

# start timing
timings_list.append(['SVM Gaussian duration:', time.time()])

####
# Start pear five dataset
####

# start time
algorithm_duration_list.append(time.time())

# create a rbf SVM classifier
svm_classifier_rbf = svm.SVC(kernel = 'rbf')

# fit the classifier on training data
svm_classifier_rbf.fit(pearsons_scaled_df_train_x, 
                       pearsons_scaled_df_train_y)

# Predict using 2018 feature data
prediction_rbf = svm_classifier_rbf.predict(pearsons_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_rbf != 
                                   pearsons_scaled_df_test_y) * 100, 2)))

# end time
algorithm_duration_list.append(time.time())

####
# End pear five dataset
####

####
# Start ols dataset
####

# start time
algorithm_duration_list.append(time.time())

# create a rbf SVM classifier
svm_classifier_rbf = svm.SVC(kernel = 'rbf')

# fit the classifier on training data
svm_classifier_rbf.fit(ols_scaled_df_train_x, 
                       ols_scaled_df_train_y)

# Predict using 2018 feature data
prediction_rbf = svm_classifier_rbf.predict(ols_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_rbf != 
                                         ols_scaled_df_test_y) * 100, 2)))

# end time
algorithm_duration_list.append(time.time())

####
# End ols dataset
####

####
# Start rfe dataset
####

# start time
algorithm_duration_list.append(time.time())

# create a rbf SVM classifier
svm_classifier_rbf = svm.SVC(kernel = 'rbf')

# fit the classifier on training data
svm_classifier_rbf.fit(rfe_scaled_df_train_x, 
                       rfe_scaled_df_train_y)

# Predict using 2018 feature data
prediction_rbf = svm_classifier_rbf.predict(rfe_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_rbf != 
                                         rfe_scaled_df_test_y) * 100, 2)))

# end time
algorithm_duration_list.append(time.time())

####
# End rfe dataset
####

####
# Start lasso dataset
####

# start time
algorithm_duration_list.append(time.time())

# create a rbf SVM classifier
svm_classifier_rbf = svm.SVC(kernel = 'rbf')

# fit the classifier on training data
svm_classifier_rbf.fit(lasso_scaled_df_train_x, 
                       lasso_scaled_df_train_y)

# Predict using 2018 feature data
prediction_rbf = svm_classifier_rbf.predict(lasso_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_rbf != 
                                         lasso_scaled_df_test_y) * 100, 2)))

# end time
algorithm_duration_list.append(time.time())

####
# End lasso dataset
####

#######
# End rbf SVM
#######

# end timing
timings_list.append(['svm rbf time end', time.time()])

#######
# Start poly SVM
#######

# start timing
timings_list.append(['SVM poly duration:', time.time()])

####
# Start pear five dataset
####

# start time
algorithm_duration_list.append(time.time())

# create a poly SVM classifier
svm_classifier_poly = svm.SVC(kernel = 'poly')

# fit the classifier on training data
svm_classifier_poly.fit(pearsons_scaled_df_train_x, 
                        pearsons_scaled_df_train_y)

# Predict using 2018 feature data
prediction_poly = svm_classifier_poly.predict(pearsons_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_poly != 
                                   pearsons_scaled_df_test_y) * 100, 2)))

# end time
algorithm_duration_list.append(time.time())

####
# End pear five dataset
####

####
# Start ols dataset
####

# start time
algorithm_duration_list.append(time.time())

# create a poly SVM classifier
svm_classifier_poly = svm.SVC(kernel = 'poly')

# fit the classifier on training data
svm_classifier_poly.fit(ols_scaled_df_train_x, 
                        ols_scaled_df_train_y)

# Predict using 2018 feature data
prediction_poly = svm_classifier_poly.predict(ols_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_poly != 
                                         ols_scaled_df_test_y) * 100, 2)))

# end time
algorithm_duration_list.append(time.time())

####
# End ols dataset
####

####
# Start rfe dataset
####

# start time
algorithm_duration_list.append(time.time())

# create a poly SVM classifier
svm_classifier_poly = svm.SVC(kernel = 'poly')

# fit the classifier on training data
svm_classifier_poly.fit(rfe_scaled_df_train_x, 
                        rfe_scaled_df_train_y)

# Predict using 2018 feature data
prediction_poly = svm_classifier_poly.predict(rfe_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_poly != 
                                         rfe_scaled_df_test_y) * 100, 2)))

# end time
algorithm_duration_list.append(time.time())

####
# End rfe dataset
####

####
# Start lasso dataset
####

# start time
algorithm_duration_list.append(time.time())

# create a poly SVM classifier
svm_classifier_poly = svm.SVC(kernel = 'poly')

# fit the classifier on training data
svm_classifier_poly.fit(lasso_scaled_df_train_x, 
                        lasso_scaled_df_train_y)

# Predict using 2018 feature data
prediction_poly = svm_classifier_poly.predict(lasso_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_poly != 
                                         lasso_scaled_df_test_y) * 100, 2)))

# end time
algorithm_duration_list.append(time.time())

####
# End lasso dataset
####

#######
# End poly SVM
#######

#############
# End SVM
#############

# end timing
timings_list.append(['svm poly time end', time.time()])

#############
# Start log regression liblinear solver
#############

# start timing
timings_list.append(['Logistic Regression liblinear duration:', time.time()])

####
# Start pear five dataset
####

# start time
algorithm_duration_list.append(time.time())

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'liblinear')

# Train the classifier on 2017 data
log_reg_classifier.fit(pearsons_scaled_df_train_x, 
                       pearsons_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(pearsons_scaled_df_test_x)

# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              pearsons_scaled_df_test_y) * 100, 2))

# end time
algorithm_duration_list.append(time.time())

####
# End pear five dataset
####

####
# Start ols dataset
####

# start time
algorithm_duration_list.append(time.time())

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'liblinear')

# Train the classifier on 2017 data
log_reg_classifier.fit(ols_scaled_df_train_x, 
                       ols_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(ols_scaled_df_test_x)


# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              ols_scaled_df_test_y) * 100, 2))

# end time
algorithm_duration_list.append(time.time())

####
# End ols dataset
####

####
# Start rfe dataset
####

# start time
algorithm_duration_list.append(time.time())

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'liblinear')

# Train the classifier on 2017 data
log_reg_classifier.fit(rfe_scaled_df_train_x, 
                       rfe_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(rfe_scaled_df_test_x)

# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              rfe_scaled_df_test_y) * 100, 2))

# end time
algorithm_duration_list.append(time.time())

####
# End rfe dataset
####

####
# Start lasso dataset
####

# start time
algorithm_duration_list.append(time.time())

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'liblinear')

# Train the classifier on 2017 data
log_reg_classifier.fit(lasso_scaled_df_train_x, 
                       lasso_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(lasso_scaled_df_test_x)

# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              lasso_scaled_df_test_y) * 100, 2))

# end time
algorithm_duration_list.append(time.time())

####
# End lasso dataset
####

#############
# End log regression liblinear solver
#############

# end timing
timings_list.append(['log lib time end', time.time()])

#############
# Start log regression sag solver
#############

# start timing
timings_list.append(['Logistic Regression sag duration:', time.time()])

####
# Start pear five dataset
####

# start time
algorithm_duration_list.append(time.time())

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'sag')

# Train the classifier on 2017 data
log_reg_classifier.fit(pearsons_scaled_df_train_x, 
                       pearsons_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(pearsons_scaled_df_test_x)

# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              pearsons_scaled_df_test_y) * 100, 2))

# end time
algorithm_duration_list.append(time.time())

####
# End pear five dataset
####

####
# Start ols dataset
####

# start time
algorithm_duration_list.append(time.time())

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'sag')

# Train the classifier on 2017 data
log_reg_classifier.fit(ols_scaled_df_train_x, 
                       ols_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(ols_scaled_df_test_x)

# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              ols_scaled_df_test_y) * 100, 2))

# end time
algorithm_duration_list.append(time.time())

####
# End ols dataset
####

####
# Start rfe dataset
####

# start time
algorithm_duration_list.append(time.time())

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'sag')

# Train the classifier on 2017 data
log_reg_classifier.fit(rfe_scaled_df_train_x, 
                       rfe_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(rfe_scaled_df_test_x)

# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              rfe_scaled_df_test_y) * 100, 2))

# end time
algorithm_duration_list.append(time.time())

####
# End rfe dataset
####

####
# Start lasso dataset
####

# start time
algorithm_duration_list.append(time.time())

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'sag')

# Train the classifier on 2017 data
log_reg_classifier.fit(lasso_scaled_df_train_x, 
                       lasso_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(lasso_scaled_df_test_x)

# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              lasso_scaled_df_test_y) * 100, 2))

# end time
algorithm_duration_list.append(time.time())

####
# End lasso dataset
####

#############
# End log regression sag solver
#############

# end timing
timings_list.append(['log sag time end', time.time()])

#############
# Start log regression newton-cg solver
#############

# start timing
timings_list.append(['Logistic Regression newton-cg duration:', time.time()])

####
# Start pear five dataset
####

# start time
algorithm_duration_list.append(time.time())

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'newton-cg')

# Train the classifier on 2017 data
log_reg_classifier.fit(pearsons_scaled_df_train_x, 
                       pearsons_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(pearsons_scaled_df_test_x)

# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              pearsons_scaled_df_test_y) * 100, 2))

# end time
algorithm_duration_list.append(time.time())

####
# End pear five dataset
####

####
# Start ols dataset
####

# start time
algorithm_duration_list.append(time.time())

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'newton-cg')

# Train the classifier on 2017 data
log_reg_classifier.fit(ols_scaled_df_train_x, 
                       ols_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(ols_scaled_df_test_x)

# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              ols_scaled_df_test_y) * 100, 2))

# end time
algorithm_duration_list.append(time.time())

####
# End ols dataset
####

####
# Start rfe dataset
####

# start time
algorithm_duration_list.append(time.time())

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'newton-cg')

# Train the classifier on 2017 data
log_reg_classifier.fit(rfe_scaled_df_train_x, 
                       rfe_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(rfe_scaled_df_test_x)

# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              rfe_scaled_df_test_y) * 100, 2))

# end time
algorithm_duration_list.append(time.time())

####
# End rfe dataset
####

####
# Start lasso dataset
####

# start time
algorithm_duration_list.append(time.time())

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'newton-cg')

# Train the classifier on 2017 data
log_reg_classifier.fit(lasso_scaled_df_train_x, 
                       lasso_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(lasso_scaled_df_test_x)

# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              lasso_scaled_df_test_y) * 100, 2))

# end time
algorithm_duration_list.append(time.time())

####
# End lasso dataset
####

#############
# End log regression newton-cg solver
#############

# end timing
timings_list.append(['log newt time end', time.time()])

################
# End building scaled algorithms
################
timings_list.append(['scaled time end', time.time()])

timings_list.append(['global time end', time.time()])

####
# Start prediction prints
####

# create a list containing the attribute reduction methods
attributes = ['Pearsons', 'OLS', 'RFE', 'Lasso']

# create a list containing the classifier names
classifiers = ['Decision Tree', 'Naive Bayes', 'Random Forest', 'KNN', 'SVM', 
               'SVM', 'SVM', 'Logistic Regression', 'Logistic Regression',
               'Logistic Regression']

# Creates a dataframe containing information about the classifiers and accuracy
prediction_df = pd.DataFrame(columns = ['classifier', 'details', 'attributes',
                                        'accuracy', 'time'])

# Build out a dataframe to store the classifiers and their accuracy
for i in range(0, len(classifiers)):
    for k in range(0, len(attributes)):
        prediction_df = prediction_df.append({'classifier' : classifiers[i],
                                      'details' : 'None',
                                      'attributes' : attributes[k],
                                      'accuracy' : 0,
                                      'time' : 0}, 
                                      ignore_index=True)

# Move indexes down 1 starting at 12 to add Random forest with full dataset
prediction_df.index = (prediction_df.index[:12].tolist() + 
                      (prediction_df.index[12:] + 1).tolist())

# Adds Random forest with full dataset to the dataframe
prediction_df.loc[12] = ['Random Forest', 'None', 'Full', 0, 0]

# reorders the indexes after the insert
prediction_df = prediction_df.sort_index()

# Updates the accuracy
prediction_df['accuracy'] = global_accuracy

# Manually set algorithm details
# decision tree
prediction_df['details'].iloc[0:4] = 'Entropy'

# Naive Bayes
prediction_df['details'].iloc[4:7] = 'Gaussian'

# Random Forest tree/depth
prediction_df['details'].iloc[8] = '10 trees, 6 depth'
prediction_df['details'].iloc[9] = '19 trees, 10 depth'
prediction_df['details'].iloc[10] = '25 trees, 8 depth'
prediction_df['details'].iloc[11] = '21 trees, 10 depth'
prediction_df['details'].iloc[12] = '16 trees, 10 depth'
    
# knn k value
prediction_df['details'].iloc[13] = 'K - 83'
prediction_df['details'].iloc[14] = 'K - 29'
prediction_df['details'].iloc[15] = 'K - 27'
prediction_df['details'].iloc[16] = 'K - 29'

# log liblinear solver
prediction_df['details'].iloc[17:21] = 'Linear'

# log sag solver
prediction_df['details'].iloc[21:25] = 'Gaussian'

# log newton-cg solver
prediction_df['details'].iloc[25:29] = 'Poly'

# log liblinear solver
prediction_df['details'].iloc[29:33] = 'Liblinear'

# log sag solver
prediction_df['details'].iloc[33:37] = 'Sag'

# log newton-cg solver
prediction_df['details'].iloc[37:41] = 'Newton-cg'

# creates a duration list
dur_list = []

# stores the durations of each algorithm from start to finish
for dur in range(0, len(algorithm_duration_list), 2):
    dur_list.append(algorithm_duration_list[dur + 1] - 
                    algorithm_duration_list[dur])

# stores the durations into the prediction_df
prediction_df['time'] = dur_list

# Print durations 
print('Durations for each of the major steps in the process,', 
      'measured in seconds\n', timings_list[1][0],
int(timings_list[2][1]) - int(timings_list[1][1]),'\n',

timings_list[3][0],
int(timings_list[4][1]) - int(timings_list[3][1]),'\n',

timings_list[5][0],
int(timings_list[6][1]) - int(timings_list[5][1]),'\n',

timings_list[8][0],
int(timings_list[9][1]) - int(timings_list[8][1]),'\n',

timings_list[10][0],
int(timings_list[11][1]) - int(timings_list[10][1]),'\n',

timings_list[12][0],
int(timings_list[13][1]) - int(timings_list[12][1]),'\n',

timings_list[7][0],
int(timings_list[14][1]) - int(timings_list[7][1]),'\n',

timings_list[16][0],
int(timings_list[17][1]) - int(timings_list[16][1]),'\n',

timings_list[18][0],
int(timings_list[19][1]) - int(timings_list[18][1]),'\n',

timings_list[20][0],
int(timings_list[21][1]) - int(timings_list[20][1]),'\n',

timings_list[22][0],
int(timings_list[23][1]) - int(timings_list[22][1]),'\n',

timings_list[24][0],
int(timings_list[25][1]) - int(timings_list[24][1]),'\n',

timings_list[26][0],
int(timings_list[27][1]) - int(timings_list[26][1]),'\n',

timings_list[28][0],
int(timings_list[29][1]) - int(timings_list[28][1]),'\n',

timings_list[15][0],
int(timings_list[30][1]) - int(timings_list[15][1]),'\n',

timings_list[0][0],
int(timings_list[-1][1])- int(timings_list[0][1]))


# Finds the most and least accurate algorithms
accurate = [prediction_df[prediction_df['accuracy'] == 
              max(prediction_df.accuracy)].values[0]]

least_accurate = [prediction_df[prediction_df['accuracy'] == 
              min(prediction_df.accuracy)].values[0]]


print('The most accurate classifier was', accurate[0][0], 'using', 
      accurate[0][1], 'on the', accurate[0][2], 
      'attribute set with an accuracy of', accurate[0][3],'%')

print('The least accurate classifier was', least_accurate[0][0], 'using', 
      least_accurate[0][1], 'on the', least_accurate[0][2], 
      'attribute set with an accuracy of', least_accurate[0][3],'%')

####
# Start rf pearsons prediction prints
####

# confusion matrix for random forest pearsons
cm_one = confusion_matrix(pearsons_df_test_y, rf_pearsons_pred)
tn, fp, fn, tp  = confusion_matrix(pearsons_df_test_y, \
                                   rf_pearsons_pred).ravel()

# Create confusion matrix heatmap
# setup class names and tick marks
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Create heatmap and labels
sns.heatmap(pd.DataFrame(cm_one), annot=True, cmap="summer", fmt='g')
ax.xaxis.set_label_position("top")
ax.set_ylim([0,2])
plt.tight_layout()
plt.title('Confusion matrix for Random Forest', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# print precision and recall values
print(classification_report(pearsons_df_test_y, rf_pearsons_pred))

# TPR/TNR rates
print('The TPR is:', str(tp) + '/' + str(tp + fn) + ',',
      round(recall_score(pearsons_df_test_y, rf_pearsons_pred) * 100, 2),'%')
print('The TNR is:', str(tn) + '/' + str(tn + fp) + ',',
    round(tn / (tn + fp) * 100, 2),'%')

####
# End rf pearsons prediction prints
####

####
# Start rf ols prediction prints
####

# confusion matrix for random forest 8/8
cm_one = confusion_matrix(ols_df_test_y, rf_ols_pred)
tn, fp, fn, tp  = confusion_matrix(ols_df_test_y, \
                                   rf_ols_pred).ravel()

# Create confusion matrix heatmap
# setup class names and tick marks
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Create heatmap and labels
sns.heatmap(pd.DataFrame(cm_one), annot=True, cmap="summer", fmt='g')
ax.xaxis.set_label_position("top")
ax.set_ylim([0,2])
plt.tight_layout()
plt.title('Confusion matrix for Random Forest', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# print precision and recall values
print(classification_report(ols_df_test_y, rf_ols_pred))

# TPR/TNR rates
print('The TPR is:', str(tp) + '/' + str(tp + fn) + ',',
      round(recall_score(ols_df_test_y, rf_ols_pred) * 100, 2),'%')
print('The TNR is:', str(tn) + '/' + str(tn + fp) + ',',
    round(tn / (tn + fp) * 100, 2),'%')

####
# End rf ols prediction prints
####

####
# Start rf rfe prediction prints
####

# confusion matrix for random forest 8/8
cm_one = confusion_matrix(rfe_df_test_y, rf_rfe_pred)
tn, fp, fn, tp  = confusion_matrix(rfe_df_test_y, \
                                   rf_rfe_pred).ravel()

# Create confusion matrix heatmap
# setup class names and tick marks
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Create heatmap and labels
sns.heatmap(pd.DataFrame(cm_one), annot=True, cmap="summer", fmt='g')
ax.xaxis.set_label_position("top")
ax.set_ylim([0,2])
plt.tight_layout()
plt.title('Confusion matrix for Random Forest', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# print precision and recall values
print(classification_report(rfe_df_test_y, rf_rfe_pred))

# TPR/TNR rates
print('The TPR is:', str(tp) + '/' + str(tp + fn) + ',',
      round(recall_score(rfe_df_test_y, rf_rfe_pred) * 100, 2),'%')
print('The TNR is:', str(tn) + '/' + str(tn + fp) + ',',
    round(tn / (tn + fp) * 100, 2),'%')

####
# End rf rfe prediction prints
####

####
# Start rf lasso prediction prints
####

# confusion matrix for random forest 8/8
cm_one = confusion_matrix(lasso_df_test_y, rf_lasso_pred)
tn, fp, fn, tp  = confusion_matrix(lasso_df_test_y, \
                                   rf_lasso_pred).ravel()

# Create confusion matrix heatmap
# setup class names and tick marks
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Create heatmap and labels
sns.heatmap(pd.DataFrame(cm_one), annot=True, cmap="summer", fmt='g')
ax.xaxis.set_label_position("top")
ax.set_ylim([0,2])
plt.tight_layout()
plt.title('Confusion matrix for Random Forest', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# print precision and recall values
print(classification_report(lasso_df_test_y, rf_lasso_pred))

# TPR/TNR rates
print('The TPR is:', str(tp) + '/' + str(tp + fn) + ',',
      round(recall_score(lasso_df_test_y, rf_lasso_pred) * 100, 2),'%')
print('The TNR is:', str(tn) + '/' + str(tn + fp) + ',',
    round(tn / (tn + fp) * 100, 2),'%')

####
# End rf lasso prediction prints
####

####
# Start rf lasso prediction prints
####

# confusion matrix for random forest 8/8
cm_one = confusion_matrix(full_df_test_y, rf_full_pred)
tn, fp, fn, tp  = confusion_matrix(full_df_test_y, \
                                   rf_full_pred).ravel()

# Create confusion matrix heatmap
# setup class names and tick marks
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Create heatmap and labels
sns.heatmap(pd.DataFrame(cm_one), annot=True, cmap="summer", fmt='g')
ax.xaxis.set_label_position("top")
ax.set_ylim([0,2])
plt.tight_layout()
plt.title('Confusion matrix for Random Forest', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# print precision and recall values
print(classification_report(full_df_test_y, rf_full_pred))

# TPR/TNR rates
print('The TPR is:', str(tp) + '/' + str(tp + fn) + ',',
      round(recall_score(full_df_test_y, rf_full_pred) * 100, 2),'%')
print('The TNR is:', str(tn) + '/' + str(tn + fp) + ',',
    round(tn / (tn + fp) * 100, 2),'%')

####
# End rf lasso prediction prints
####

####
# Start knn pearsons prediction prints
####

# confusion matrix for random forest 8/8
cm_one = confusion_matrix(pearsons_scaled_df_test_y, knn_pearsons_pred)
tn, fp, fn, tp  = confusion_matrix(pearsons_scaled_df_test_y, \
                                   knn_pearsons_pred).ravel()

# Create confusion matrix heatmap
# setup class names and tick marks
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Create heatmap and labels
sns.heatmap(pd.DataFrame(cm_one), annot=True, cmap="summer", fmt='g')
ax.xaxis.set_label_position("top")
ax.set_ylim([0,2])
plt.tight_layout()
plt.title('Confusion matrix for KNN Pearsons', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# print precision and recall values
print(classification_report(pearsons_scaled_df_test_y, knn_pearsons_pred))

# TPR/TNR rates
print('The TPR is:', str(tp) + '/' + str(tp + fn) + ',',
      round(recall_score(pearsons_scaled_df_test_y, knn_pearsons_pred) * 100, 2),'%')
print('The TNR is:', str(tn) + '/' + str(tn + fp) + ',',
    round(tn / (tn + fp) * 100, 2),'%')

####
# End knn pearsons prediction prints
####

####
# Start knn ols prediction prints
####

# confusion matrix for random forest 8/8
cm_one = confusion_matrix(ols_scaled_df_test_y, knn_ols_pred)
tn, fp, fn, tp  = confusion_matrix(ols_scaled_df_test_y, \
                                   knn_ols_pred).ravel()

# Create confusion matrix heatmap
# setup class names and tick marks
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Create heatmap and labels
sns.heatmap(pd.DataFrame(cm_one), annot=True, cmap="summer", fmt='g')
ax.xaxis.set_label_position("top")
ax.set_ylim([0,2])
plt.tight_layout()
plt.title('Confusion matrix for KNN Pearsons', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# print precision and recall values
print(classification_report(ols_scaled_df_test_y, knn_ols_pred))

# TPR/TNR rates
print('The TPR is:', str(tp) + '/' + str(tp + fn) + ',',
      round(recall_score(ols_scaled_df_test_y, knn_ols_pred) * 100, 2),'%')
print('The TNR is:', str(tn) + '/' + str(tn + fp) + ',',
    round(tn / (tn + fp) * 100, 2),'%')

####
# End knn ols prediction prints
####

####
# Start knn rfe prediction prints
####

# confusion matrix for random forest 8/8
cm_one = confusion_matrix(rfe_scaled_df_test_y, knn_rfe_pred)
tn, fp, fn, tp  = confusion_matrix(rfe_scaled_df_test_y, \
                                   knn_rfe_pred).ravel()

# Create confusion matrix heatmap
# setup class names and tick marks
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Create heatmap and labels
sns.heatmap(pd.DataFrame(cm_one), annot=True, cmap="summer", fmt='g')
ax.xaxis.set_label_position("top")
ax.set_ylim([0,2])
plt.tight_layout()
plt.title('Confusion matrix for KNN Pearsons', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# print precision and recall values
print(classification_report(rfe_scaled_df_test_y, knn_rfe_pred))

# TPR/TNR rates
print('The TPR is:', str(tp) + '/' + str(tp + fn) + ',',
      round(recall_score(rfe_scaled_df_test_y, knn_rfe_pred) * 100, 2),'%')
print('The TNR is:', str(tn) + '/' + str(tn + fp) + ',',
    round(tn / (tn + fp) * 100, 2),'%')

####
# End knn rfe prediction prints
####

####
# Start knn lasso prediction prints
####

# confusion matrix for random forest 8/8
cm_one = confusion_matrix(lasso_scaled_df_test_y, knn_lasso_pred)
tn, fp, fn, tp  = confusion_matrix(lasso_scaled_df_test_y, \
                                   knn_lasso_pred).ravel()

# Create confusion matrix heatmap
# setup class names and tick marks
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Create heatmap and labels
sns.heatmap(pd.DataFrame(cm_one), annot=True, cmap="summer", fmt='g')
ax.xaxis.set_label_position("top")
ax.set_ylim([0,2])
plt.tight_layout()
plt.title('Confusion matrix for KNN Pearsons', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# print precision and recall values
print(classification_report(lasso_scaled_df_test_y, knn_lasso_pred))

# TPR/TNR rates
print('The TPR is:', str(tp) + '/' + str(tp + fn) + ',',
      round(recall_score(lasso_scaled_df_test_y, knn_lasso_pred) * 100, 2),'%')
print('The TNR is:', str(tn) + '/' + str(tn + fp) + ',',
    round(tn / (tn + fp) * 100, 2),'%')

####
# End knn lasso prediction prints
####