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
from sys import exit

import numpy as np
import pandas as pd

import lightgbm as lgb
import statsmodels.api as sm
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import svm, tree, preprocessing as pp
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LassoCV, LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, recall_score, roc_curve, auc,\
                            classification_report, roc_auc_score, \
                            accuracy_score

# Set display options for pandas
#pd.set_option('display.max_rows', 100)
#pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 60)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# set seaborn to dark backgrounds, matplot to use seaborn
sns.set_style("darkgrid")
mpl.style.use('seaborn')

# numpy print precision
np.set_printoptions(precision=4, suppress=True)

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
print('The total length of the dataframe is', main_df.shape[0], 'rows'8
      'and the width is', main_df.shape[1], 'columns')

# calculate the percentage of churn
churn_percentage = (main_df[main_df['Churn'] == 'Yes'].shape[0]/ \
                       main_df.shape[0] * 100)

# total Churn/No Churn
print('The churn stats of the dataframe is\n', 
      'No:', main_df.Churn.value_counts()[0], '\n',
      'Yes:', main_df.Churn.value_counts()[1], '\n',
      '{:.2f}% of the data are churn examples (highly skewed).'.format(\
       churn_percentage))

# plot Yes vs No churn
ax = sns.countplot(data=main_df, x = 'Churn')
ax

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

# Other = 0
# Rural = 1
# Suburban = 2
# Town = 3

# setup label encoder
le = pp.LabelEncoder()

# fit values to categories
le.fit(main_df['PrizmCode'])

# apply values
main_df['PrizmCode'] = le.transform(main_df['PrizmCode'])

# Clerical = 0
# Crafts = 1
# Homemaker = 2
# Other = 3
# Professional = 4
# Retired = 5
# Self = 6
# Student = 7
le = pp.LabelEncoder()
le.fit(main_df['Occupation'])
#list(le.classes_)
main_df['Occupation'] = le.transform(main_df['Occupation'])

# yes = 1
# no = 2
# unknown = 3
le = pp.LabelEncoder()
le.fit(main_df['MaritalStatus'])
#list(le.classes_)
main_df['MaritalStatus'] = le.transform(main_df['MaritalStatus'])

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

# plot churn sliced by credit raiting
#def plot_churn_values(main_df, column=''):
#    plt.figure(figsize=(7,7))
#    plt.grid(True)
#    plt.bar(main_df[column][main_df.Churn==1].value_counts().index, 
#            main_df[column][main_df.Churn==1].value_counts().values)
#    plt.title(f'{column}')
#    plt.xticks(rotation=-90)
#
#plot_churn_values(main_df, column='CreditRating')

################
# End data import and cleanup
################

# creates a list of column names
cols = list(main_x.columns)

# copy the main df for scaling
main_x_scaled = main_x

# Setup scaler for the first 28 columns which contain non-catagorical data
scaler = StandardScaler()
main_x_scaled[cols[0:29]] = scaler.fit_transform(main_x_scaled[cols[0:29]])

# Split dataset into 33% test 66% training
(main_scaled_df_train_x, main_scaled_df_test_x, 
 main_scaled_df_train_y, main_scaled_df_test_y) = (
        train_test_split(main_x_scaled, main_y, test_size = 0.333, 
                         random_state = 1337))

################
# Start LGBM
# model setup and parameters found here:
# https://www.kaggle.com/avanwyk/a-lightgbm-overview
################

# setup lgb training and test datasets
lgb_train = lgb.Dataset(main_scaled_df_train_x, main_scaled_df_train_y, \
                        free_raw_data=False)

lgb_test = lgb.Dataset(main_scaled_df_test_x, main_scaled_df_test_y, \
                      reference=lgb_train, free_raw_data=False)

# set parameters to be used for LGB
# gradient boosted decision tree
# optimization object is binary
# learning rate controls the step size
# number of leaves in one tree
# number of processor threads
# area under curve (auc) for calculating during validation
core_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'nthread': 6,
    'metric': 'auc' 
}

advanced_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    
    'learning_rate': 0.01,
    # more leaves increases accuracy, but may lead to overfitting
    'num_leaves': 75,
    
    # the maximum tree depth. Shallower trees reduce overfitting.
    'max_depth': 5,

    # minimal loss gain to perform a split
    'min_split_gain': 0,

    # or min_data_in_leaf: specifies the minimum samples per leaf node.
    'min_child_samples': 21,

    # minimal sum hessian in one leaf. Controls overfitting.
    'min_child_weight': 5,

    # L1 and L2 regularization
    'lambda_l1': 0.5, 
    'lambda_l2': 0.5,
    
    # randomly select a fraction of the features before building each tree.
    'feature_fraction': 0.5, 
    
    # Speeds up training and controls overfitting.
    # allows for bagging or subsampling of data to speed up training.
    'bagging_fraction': 0.5,
    
    # perform bagging on every Kth iteration, disabled if 0.
    'bagging_freq': 0,
    
    # add a weight to the positive class examples (compensates for imbalance).
    'scale_pos_weight': 140, 
    
    # amount of data to sample to determine histogram bins
    'subsample_for_bin': 200000,
    
    # the maximum number of bins to bucket feature values in.
    # LightGBM autocompresses memory based on this value. 
    # Larger bins improves accuracy.
    'max_bin': 1250, 
    
    # set state for repeatability
    'random_state' : [1337],
    
    # number of threads to use for LightGBM, 
    # best set to number of actual cores.
    'nthread': 6,
}

test_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    
    'learning_rate': 0.01,
    # more leaves increases accuracy, but may lead to overfitting
    'num_leaves': 60,
    
    # the maximum tree depth. Shallower trees reduce overfitting.
    'max_depth': 5,

    # minimal loss gain to perform a split
    'min_split_gain': 0,

    # or min_data_in_leaf: specifies the minimum samples per leaf node.
    'min_child_samples': 15,

    # minimal sum hessian in one leaf. Controls overfitting.
    'min_child_weight': 5,

    # L1 and L2 regularization
    'lambda_l1': 0.1, 
    'lambda_l2': 0.1,
    
    # randomly select a fraction of the features before building each tree.
    'feature_fraction': 0.7, 
    
    # Speeds up training and controls overfitting.
    # allows for bagging or subsampling of data to speed up training.
    'bagging_fraction': 0.5,
    
    # perform bagging on every Kth iteration, disabled if 0.
    'bagging_freq': 0,
    
    # add a weight to the positive class examples (compensates for imbalance).
    'scale_pos_weight': 120, 
    
    # amount of data to sample to determine histogram bins
    'subsample_for_bin': 200000,
    
    # the maximum number of bins to bucket feature values in.
    # LightGBM autocompresses memory based on this value. 
    # Larger bins improves accuracy.
    'max_bin': 500, 
    
    # set state for repeatability
    'random_state' : [1337],
    
    # number of threads to use for LightGBM, 
    # best set to number of actual cores.
    'nthread': 6,
}

# create a gradient boosted decision tree using LGBM
# boost_rounds = number of iterations
# stop training if none of the metrics improves on any validation data
def train_gbm(params, training_set, testing_set, init_gbm=None, 
              boost_rounds=100, early_stopping_rounds=0, metric='auc'):
    evals_result = {} 

    # uses the core_params dictionary for settings
    gbm = lgb.train(params, 
                    training_set,
                    init_model = init_gbm,
                    num_boost_round = boost_rounds, 
                    early_stopping_rounds = early_stopping_rounds,
                    valid_sets = training_set,
                    evals_result = evals_result,
                    verbose_eval = False)
    
#    y_true = training_set.label
#    y_pred = gbm.predict(training_set.data)
#    fpr, tpr, threshold = roc_curve(y_true, y_pred)
#    roc_auc = auc(fpr, tpr)
#    
#    plt.title("ROC Curve. Area under Curve: {:.3f}".format(roc_auc))
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('True Positive Rate')
#    _ = plt.plot(fpr, tpr, 'r')
    
    return gbm, evals_result

####
# Start building 1
####

# build the advanced parameters model
model, evals = train_gbm(advanced_params, lgb_train, lgb_test, \
                         boost_rounds = 1000)

# predicted value of y
model_pred_y = model.predict(main_scaled_df_test_x)

# store the true and prodicted values
model_test_results = pd.DataFrame({'trueValue': main_scaled_df_test_y, \
                                   'predictedValue': model_pred_y})
  
# find median values for churn and no churn predction
# -1.299 and -1.390
median_no_churn = np.median(1/np.log(model_test_results.predictedValue[model_test_results.trueValue == 0]))
median_churn = np.median(1/np.log(model_test_results.predictedValue[model_test_results.trueValue == 1]))

# transform results from predicted value
model_test_results['y_transformed'] = 1/np.log(model_test_results['predictedValue'])


# converts all predictions to 1 if < median_no_churn value otherwise 0
model_pred_y_mnc = np.where(model_test_results.y_transformed \
                                   < median_no_churn, 1, 0)
# same but median_churn value
model_pred_y_mc = np.where(model_test_results.y_transformed \
                                   < median_churn, 1, 0)

# find the customers within area under curve
auc_based = model_test_results[model_test_results.y_transformed <= \
                               median_no_churn]

# print customers AUC
auc_based.trueValue.value_counts()

# 9k customers, 36% possible to churn
auc_based.trueValue.value_counts()/auc_based.shape[0]

# find the customers within area under curve
bus_based = model_test_results[model_test_results.y_transformed <= \
                               median_churn]

# print customers AUC
bus_based.trueValue.value_counts()

# 6k customers, 40.1% possible to churn
bus_based.trueValue.value_counts()/bus_based.shape[0]

# store accuracy
global_accuracy.append(100-(round(np.mean(model_pred_y
                                  != main_scaled_df_test_y),2)))

# setup a list to save the roc scores
roc_scores = {}
roc_scores.update({'GBM1': roc_auc_score(model_test_results.trueValue, \
                                model_test_results.predictedValue)})
   
# setup plots  
#Print accuracy
acc_lgbm = accuracy_score(main_scaled_df_test_y, model_pred_y_mc)
print('Overall accuracy of Light GBM model:', acc_lgbm)

# print precision and recall values
print(classification_report(main_scaled_df_test_y, model_pred_y_mc))

# TPR/TNR rates
tn, fp, fn, tp  = confusion_matrix(main_scaled_df_test_y, \
                                   model_pred_y_mc).ravel()
# TPR/TNR rates
print('The TPR is:', str(tp) + '/' + str(tp + fn) + ',',
      round(recall_score(main_scaled_df_test_y, model_pred_y_mc) * 100, 2),'%')
print('The TNR is:', str(tn) + '/' + str(tn + fp) + ',',
    round(tn / (tn + fp) * 100, 2),'%')

#Print Area Under Curve
plt.figure()
false_positive_rate, recall, thresholds = roc_curve(main_scaled_df_test_y, 
                                                    model_pred_y)
roc_auc = auc(false_positive_rate, recall)
plt.title('Receiver Operating Characteristic (ROC)')
plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out (1-Specificity)')
plt.show()

print('AUC score:', roc_auc)

#Print Confusion Matrix
plt.figure()
cm = confusion_matrix(main_scaled_df_test_y, model_pred_y_mc)
labels = ['Churn', 'No Churn']
plt.figure(figsize=(10,10))
sns.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, \
            fmt='d', cmap="summer", vmin = 0.2);
plt.title('Confusion Matrix for LGBM')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()

############
optimal_index = np.argmax(recall - false_positive_rate)
optimal_threshhold = thresholds[optimal_index]

################
plt.figure(figsize=(8,8))
plt.grid(True)
sns.distplot(1/np.log(model_test_results.predictedValue[model_test_results.trueValue == 0]), color = 'green')
sns.distplot(1/np.log(model_test_results.predictedValue[model_test_results.trueValue == 1]), color = 'red')
plt.plot([median_no_churn, median_no_churn], [0, 2.4], 'bo--', linewidth=2.5)
plt.plot([median_churn, median_churn], [0, 2.4], 'go--', linewidth=2.5)
################

#Print Area Under Curve
def plot_roc_curve(test_res, threshold = -1.39):
    ns_probs = [0 for _ in range(len(test_res))]
    fpr, tpr, threshold = roc_curve(model_test_results.trueValue, 
                                    np.where(model_test_results.y_transformed \
                                             < threshold, 1, 0))
    _fpr_, _tpr_, _threshold_ = roc_curve(model_test_results.trueValue, \
                                          ns_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10,10))
    plt.grid(True)
    plt.title('ROC Curve. Area Under Curve: {:.3f}'.format(roc_auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    _ = plt.plot(fpr, tpr, 'r')
    __ = plt.plot(_fpr_, _tpr_, 'b', ls = '--')

# plot roc with both median adjustments
plot_roc_curve(model_test_results, median_churn)
plot_roc_curve(model_test_results, median_no_churn)


def plot_conf_mat(cm):
    """
    Helper function to plot confusion matrix.
    With text centerred.
    """
    plt.figure(figsize=(8,8))
    ax = sns.heatmap(cm, annot=True,fmt="d",annot_kws={"size": 16})
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    
# setup confusion matrix with adjusted churn values
cm_auc = confusion_matrix(model_test_results.trueValue, \
                          np.where(model_test_results.y_transformed \
                                   < median_churn, 1, 0), labels=[0, 1])    
  
# setup confusion matrix with adjusted no churn values
cm_bus = confusion_matrix(model_test_results.trueValue, \
                          np.where(model_test_results.y_transformed \
                                   < median_no_churn, 1, 0), labels=[0, 1])  

# plot confusion matrix's with adjusted values  
plot_conf_mat(cm_auc)
plot_conf_mat(cm_bus)

####
# End building 1
####

####
# Start building 2
####
# build another advanced model based on the first model
model2, evals2 = train_gbm(advanced_params, lgb_train, lgb_test, \
                         init_gbm=model, boost_rounds = 1000)

# predicted value of y
model_pred_y2 = model2.predict(main_scaled_df_test_x)

# save model to txt file
#model.save_model('cell_churn.txt')

# store the true and prodicted values
model_test_results2 = pd.DataFrame({'trueValue': main_scaled_df_test_y, \
                                   'predictedValue': model_pred_y2})

# find median values for churn and no churn predction
# -1.299 and -1.390
median_no_churn2 = np.median(1/np.log(model_test_results2.predictedValue[model_test_results2.trueValue == 0]))
median_churn2 = np.median(1/np.log(model_test_results2.predictedValue[model_test_results2.trueValue == 1]))

# converts all predictions to 1 if < median_no_churn value otherwise 0
model_pred_y_mnc2 = np.where(model_test_results2.y_transformed \
                                   < median_no_churn2, 1, 0)
# same but median_churn value
model_pred_y_mc2 = np.where(model_test_results2.y_transformed \
                                   < median_churn2, 1, 0)

# transform results from predicted value
model_test_results2['y_transformed'] = 1/np.log(\
                                        model_test_results2['predictedValue'])

 
# store accuracy
global_accuracy.append(100-(round(np.mean(model_pred_y2
                                  != main_scaled_df_test_y),2)))

# store roc score
roc_scores.update({'GBM2': roc_auc_score(model_test_results2.trueValue, \
                                model_test_results2.predictedValue)})

# setup plots
#Print accuracy
acc_lgbm = accuracy_score(main_scaled_df_test_y, model_pred_y_mc2)
print('Overall accuracy of Light GBM model:', acc_lgbm)

#Print Area Under Curve
plt.figure()
false_positive_rate, recall, thresholds = roc_curve(main_scaled_df_test_y, 
                                                    model_pred_y2)
roc_auc = auc(false_positive_rate, recall)
plt.title('Receiver Operating Characteristic (ROC)')
plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out (1-Specificity)')
plt.show()

print('AUC score:', roc_auc)

#Print Confusion Matrix
plt.figure()
cm = confusion_matrix(main_scaled_df_test_y, model_pred_y_mc2)
labels = ['No Churn', 'Churn']
plt.figure(figsize=(8,6))
sns.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, \
            fmt='d', cmap="Blues", vmin = 0.2);
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()
####
# End building 2
####

####
# Start building 3
####
# build another advanced model based on the first model
model3, evals3 = train_gbm(test_params, lgb_train, lgb_test, \
                           boost_rounds = 1000)

# predicted value of y
model_pred_y3 = model3.predict(main_scaled_df_test_x)

# converts all predictions to 1 if > 0.5 otherwise 0
model_pred_y3_01 = np.where(model_pred_y3 > 0.5, 1, 0)

# save model to txt file
#model.save_model('cell_churn.txt')

model_test_results3 = pd.DataFrame({'trueValue': main_scaled_df_test_y, \
                                   'predictedValue': model_pred_y3})

# store accuracy
global_accuracy.append(100-(round(np.mean(model_pred_y3
                                  != main_scaled_df_test_y),2)))
    

roc_scores.update({'GBM2': roc_auc_score(model_test_results3.trueValue, \
                                model_test_results3.predictedValue)})

# setup plots
#Print accuracy
acc_lgbm = accuracy_score(main_scaled_df_test_y, model_pred_y3_01)
print('Overall accuracy of Light GBM model:', acc_lgbm)

#Print Area Under Curve
plt.figure()
false_positive_rate, recall, thresholds = roc_curve(main_scaled_df_test_y, 
                                                    model_pred_y3)
roc_auc = auc(false_positive_rate, recall)
plt.title('Receiver Operating Characteristic (ROC)')
plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out (1-Specificity)')
plt.show()

print('AUC score:', roc_auc)

#Print Confusion Matrix
plt.figure()
cm = confusion_matrix(main_scaled_df_test_y, model_pred_y3_01)
labels = ['No Churn', 'Churn']
plt.figure(figsize=(8,6))
sns.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, \
            fmt='d', cmap="Blues", vmin = 0.2);
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()
####
# End building 3
####

###########
# Start gbm with grid search
###########

##Select Hyper-Parameters
#params = {    
#        'boosting_type': 'gbdt',
#        'objective': 'binary',
#        'metric': 'auc',
#        'learning_rate': 0.01,
#        'num_leaves': 41,
#        'max_depth': 5,
#        'min_split_gain': 0,
#        'min_child_samples': 21,
#        'min_child_weight': 5,
#        'lambda_l1': 0.5, 
#        'lambda_l2': 0.5,
#        'feature_fraction': 0.5, 
#        'bagging_fraction': 0.5,
#        'bagging_freq': 0,
#        'scale_pos_weight': 140, 
#        'subsample_for_bin': 200000,
#        'max_bin': 1000, 
#        'random_state' : [1337],
#        'nthread': 6,
#          }
#
## Create parameters to search
#gridParams = {
#    'learning_rate': [0.01, 0.025, 0.05],
#    'num_leaves': [20, 40, 60],
#    'min_child_samples': [15, 21, 26],
#    'min_child_weight': [5, 10, 15],
#    'lambda_l1': [0.1, 0.5, 1],
#    'lambda_l2': [0.1, 0.5, 1],
#    'scale_pos_weight': [120, 140, 160],
#    'feature_fraction': [0.1, 0.5, 0.7],
#    'max_bin': [500, 1000, 1500],
#    }
#
## Create classifier to use
#mdl = lgb.LGBMClassifier(
#          boosting_type= 'gbdt',
#          objective = 'binary',
#          silent = True,
#          learning_rate = params['learning_rate'],
#          num_leaves = params['num_leaves'],
#          min_child_samples = params['min_child_samples'],
#          min_child_weight = params['min_child_weight'],
#          lambda_l1 = params['lambda_l1'],
#          lambda_l2 = params['lambda_l2'],
#          scale_pos_weight = params['scale_pos_weight'],
#          feature_fraction = params['feature_fraction'],
#          max_bin = params['max_bin'])
#
## View the default model params:
#mdl.get_params().keys()
#
## Create the grid
#grid = GridSearchCV(mdl, gridParams, verbose=2, cv=4, n_jobs=-1)
#
## Run the grid
#grid.fit(main_scaled_df_train_x, main_scaled_df_train_y)
#
## Print the best parameters found
#print(grid.best_params_)
#print(grid.best_score_)
#
## Using parameters already set above, replace in the best from the grid search
#params['colsample_bytree'] = grid.best_params_['colsample_bytree']
#params['learning_rate'] = grid.best_params_['learning_rate']
#params['n_estimators'] = grid.best_params_['n_estimators']
#params['num_leaves'] = grid.best_params_['num_leaves']
#params['subsample'] = grid.best_params_['subsample']
#
##Train model on selected parameters and number of iterations
#grid_lgbm = lgb.train(params, lgb_train,
#                 init_model = None,
#                 early_stopping_rounds = 0,
#                 valid_sets = lgb_test,
#                 verbose_eval = False)
#
##Predict on test set
#predictions_lgbm_prob = grid_lgbm.predict(main_scaled_df_test_x)
#predictions_lgbm_01 = np.where(predictions_lgbm_prob > 0.5, 1, 0) #Turn probability to 0-1 binary output
#
#model_test_results3 = pd.DataFrame({'trueValue': main_scaled_df_test_y, \
#                                   'predictedValue': predictions_lgbm_prob})
#
## store accuracy
#global_accuracy.append(100-(round(np.mean(predictions_lgbm_prob
#                                  != main_scaled_df_test_y),2)))
#    
#
#roc_scores.update({'GBM3': roc_auc_score(model_test_results3.trueValue, \
#                                model_test_results3.predictedValue)})
#
## setup plots
##Print accuracy
#acc_lgbm = accuracy_score(main_scaled_df_test_y, predictions_lgbm_01)
#print('Overall accuracy of Light GBM model:', acc_lgbm)
#
##Print Area Under Curve
#plt.figure()
#false_positive_rate, recall, thresholds = roc_curve(main_scaled_df_test_y, 
#                                                    predictions_lgbm_01)
#roc_auc = auc(false_positive_rate, recall)
#plt.title('Receiver Operating Characteristic (ROC)')
#plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
#plt.legend(loc='lower right')
#plt.plot([0,1], [0,1], 'r--')
#plt.xlim([0.0,1.0])
#plt.ylim([0.0,1.0])
#plt.ylabel('Recall')
#plt.xlabel('Fall-out (1-Specificity)')
#plt.show()
#
#print('AUC score:', roc_auc)
#
##Print Confusion Matrix
#plt.figure()
#cm = confusion_matrix(main_scaled_df_test_y, predictions_lgbm_01)
#labels = ['No Churn', 'Churn']
#plt.figure(figsize=(8,6))
#sns.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, \
#            fmt='d', cmap="Blues", vmin = 0.2);
#plt.title('Confusion Matrix')
#plt.ylabel('True Class')
#plt.xlabel('Predicted Class')
#plt.show()
#
#############
### End building gbm model
#############

####
# setup some plots
####

# plot feature importance, has too many features to be useful
#lgb.plot_importance(model)

# create feature importance dictionary
feat_importance = {}
for feat in range(len(model.feature_importance())):
    feat_importance.update({model.feature_name()[feat] : \
                            model.feature_importance()[feat]})

# sort dict from largest to smallest
feat_importance = {k: v for k, v in sorted(feat_importance.items(), \
                                           key=lambda item: item[1], \
                                           reverse = True)}   
   
# create a top 10 list of features
feat_importance_ten = {}
for i in range(10):
    feat_importance_ten.update({list(feat_importance.keys())[i] : \
                               (list(feat_importance.values())[i])})

feat_importance_twenty = {}
for i in range(20):
    feat_importance_twenty.update({list(feat_importance.keys())[i] : \
                               (list(feat_importance.values())[i])})

# another way to plot the importance while limiting features
#lgb.plot_importance(model, max_num_features=10, importance_type='split')
    
# Create feature important plot 
import_plot = sns.barplot(y=list(feat_importance_ten.keys()), \
            x=list(feat_importance_ten.values()), palette='Blues_d')
ax.xaxis.set_label_position("top")
plt.title('Feature Importance')
plt.ylabel('Features')
plt.xlabel('Importance Value')

# create lists of the features
features_list_ten = list(feat_importance_ten.keys())
features_list_twenty = list(feat_importance_twenty.keys())

################
# Start attribute selection with various methods
################
# start timing
timings_list.append(['Features selection duration:', time.time()])

#######
# Start Pearsons corerelation
#######

# create a dataframe from the highest correlated item
pearsons_df = main_df[['CurrentEquipmentDays']]

#######
# Start Ordinary Least Squares
#######

# sets and prints the remaining unremoved features
selected_features_BE = \
['MonthlyRevenue',
 'MonthlyMinutes',
 'TotalRecurringCharge',
 'OverageMinutes',
 'PercChangeMinutes',
 'PercChangeRevenues',
 'DroppedCalls',
 'BlockedCalls',
 'CustomerCareCalls',
 'ThreewayCalls',
 'PeakCallsInOut',
 'MonthsInService',
 'UniqueSubs',
 'ActiveSubs',
 'Handsets',
 'CurrentEquipmentDays',
 'AgeHH1',
 'ChildrenInHH',
 'HandsetRefurbished',
 'HandsetWebCapable',
 'RespondsToMailOffers',
 'HasCreditCard',
 'RetentionCalls',
 'RetentionOffersAccepted',
 'IncomeGroup',
 'AdjustmentsToCreditRating',
 'HandsetPrice',
 'MadeCallToRetentionTeam',
 'CreditRating',
 'MaritalStatus']

# creates a dataframe with the ols selected columns
ols_df = main_df[selected_features_BE]

######
# End Ordinary Least Squares
######

######
# Start Recursive Feature Elimination
######

selected_features_rfe = \
      ['MonthlyRevenue', 'MonthlyMinutes', 'TotalRecurringCharge',
       'DirectorAssistedCalls', 'OverageMinutes', 'RoamingCalls',
       'PercChangeMinutes', 'PercChangeRevenues', 'DroppedCalls',
       'BlockedCalls', 'UnansweredCalls', 'CustomerCareCalls', 'ThreewayCalls',
       'ReceivedCalls', 'OutboundCalls', 'InboundCalls', 'PeakCallsInOut',
       'OffPeakCallsInOut', 'DroppedBlockedCalls', 'CallForwardingCalls',
       'CallWaitingCalls', 'MonthsInService', 'UniqueSubs', 'ActiveSubs',
       'Handsets', 'HandsetModels', 'CurrentEquipmentDays', 'AgeHH1', 'AgeHH2',
       'ChildrenInHH', 'HandsetRefurbished', 'HandsetWebCapable', 'TruckOwner',
       'RVOwner', 'Homeownership', 'BuysViaMailOrder', 'RespondsToMailOffers',
       'OptOutMailings', 'NonUSTravel', 'OwnsComputer', 'HasCreditCard',
       'RetentionCalls', 'RetentionOffersAccepted', 'NewCellphoneUser',
       'NotNewCellphoneUser', 'ReferralsMadeBySubscriber', 'IncomeGroup',
       'OwnsMotorcycle', 'AdjustmentsToCreditRating', 'HandsetPrice',
       'MadeCallToRetentionTeam', 'CreditRating', 'PrizmCode', 'Occupation',
       'MaritalStatus']

# creates a dataframe with the rfe selected columns
rfe_df = main_df[selected_features_rfe]

#######
# End Recursive Feature Elimination
#######

#######
# Start lasso method
#######

lasso_columns = ['MonthlyRevenue', 'MonthlyMinutes', 'TotalRecurringCharge',
       'DirectorAssistedCalls', 'OverageMinutes', 'RoamingCalls',
       'PercChangeMinutes', 'PercChangeRevenues', 'DroppedCalls',
       'UnansweredCalls', 'CustomerCareCalls', 'ThreewayCalls',
       'ReceivedCalls', 'OutboundCalls', 'InboundCalls', 'PeakCallsInOut',
       'OffPeakCallsInOut', 'DroppedBlockedCalls', 'CallWaitingCalls',
       'MonthsInService', 'UniqueSubs', 'ActiveSubs', 'Handsets',
       'CurrentEquipmentDays', 'AgeHH1', 'AgeHH2', 'ChildrenInHH',
       'HandsetRefurbished', 'HandsetWebCapable', 'RespondsToMailOffers',
       'HasCreditCard', 'RetentionCalls', 'NewCellphoneUser',
       'NotNewCellphoneUser', 'ReferralsMadeBySubscriber', 'IncomeGroup',
       'AdjustmentsToCreditRating', 'HandsetPrice', 'MadeCallToRetentionTeam',
       'CreditRating', 'Occupation', 'MaritalStatus', 'AreaCode']

# creates a dataframe based on the 32 columns selected from lasso
lasso_df = main_df[lasso_columns]

#######
# End lasso method
#######

# end timing
timings_list.append(['features time end', time.time()])

# output min, max and mean values for the lasso features
#lasso_df.min()
#lasso_df.max()
#lasso_df.mean()

################
# End attribute selection with various methods
################

ten_features_scaled = main_df[features_list_ten]
ten_features_scaled[features_list_ten[0:9]] = scaler.fit_transform(\
                                              ten_features_scaled[\
                                              features_list_ten[0:9]])

    # Split dataset into 33% test 66% training
(ten_scaled_df_train_x, ten_scaled_df_test_x, 
 ten_scaled_df_train_y, ten_scaled_df_test_y) = (
        train_test_split(ten_features_scaled, main_y, test_size = 0.333, 
                         random_state = 1337))

twenty_features_scaled = main_df[features_list_twenty]

# move non-scaled columns to the end
twenty_features_scaled['ac'] = twenty_features_scaled['AreaCode']
twenty_features_scaled['cr'] = twenty_features_scaled['CreditRating']

# drop credit and area code from df
twenty_features_scaled.drop('CreditRating', axis = 1, inplace=True)
twenty_features_scaled.drop('AreaCode', axis = 1, inplace=True)

# reindex
twenty_features_scaled = twenty_features_scaled.sort_index()

# scale first 18 columns
twenty_features_scaled.iloc[:,0:18] = scaler.fit_transform(
                                          twenty_features_scaled.iloc[:,0:18])

# Split dataset into 33% test 66% training
(twenty_scaled_df_train_x, twenty_scaled_df_test_x, 
 twenty_scaled_df_train_y, twenty_scaled_df_test_y) = (
        train_test_split(twenty_features_scaled, main_y, test_size = 0.333, 
                         random_state = 1337))


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

# Create list just for the algorithm durations
algorithm_duration_list = []

################
# Start building non-scaled algorithms
################
# start timing
timings_list.append(['Non-scaled duration:', time.time()])

####
# Start fixed value RFs
####

# start timing
timings_list.append(['Random forest duration:', time.time()])

####
# Start pearsons dataset
####

# start time
algorithm_duration_list.append(time.time())


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

# store true vs predicted value
rf_pearsons_df = pd.DataFrame({'trueValue': pearsons_df_test_y, \
                                   'predictedValue': rf_pearsons_pred})
# calculate the roc auc score
roc_scores.update({'RF-Pearsons': roc_auc_score(rf_pearsons_df.trueValue, \
                                                rf_pearsons_df.predictedValue)})
    
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
                                  
# store true vs predicted value
rf_ols_df = pd.DataFrame({'trueValue': ols_df_test_y, \
                          'predictedValue': rf_ols_pred})

# calculate the roc auc score
roc_scores.update({'RF-ols': roc_auc_score(rf_ols_df.trueValue, \
                                           rf_ols_df.predictedValue)})

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

# store true vs predicted value
rf_rfe_df = pd.DataFrame({'trueValue': rfe_df_test_y, \
                          'predictedValue': rf_rfe_pred})
# calculate the roc auc score
roc_scores.update({'RF-rfe': roc_auc_score(rf_rfe_df.trueValue, \
                                           rf_rfe_df.predictedValue)})

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

# store true vs predicted value
rf_lasso_df = pd.DataFrame({'trueValue': lasso_df_test_y, \
                          'predictedValue': rf_lasso_pred})
# calculate the roc auc score
roc_scores.update({'RF-lasso': roc_auc_score(rf_lasso_df.trueValue, \
                                             rf_lasso_df.predictedValue)})

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

# store true vs predicted value
rf_full_df = pd.DataFrame({'trueValue': full_df_test_y, \
                          'predictedValue': rf_full_pred})
# calculate the roc auc score
roc_scores.update({'RF-full': roc_auc_score(rf_full_df.trueValue, \
                                           rf_full_df.predictedValue)})  

# end time
algorithm_duration_list.append(time.time())

####
# End full dataset
####

####
# Start ten dataset
####

# start time
algorithm_duration_list.append(time.time())

#singular RF 
rf_clf = RandomForestClassifier(n_estimators = 25, 
                                max_depth = 10, criterion ='entropy',
                                random_state = 1337)
rf_clf.fit(ten_scaled_df_train_x, ten_scaled_df_train_y)

# store predictions
rf_ten_pred = rf_clf.predict(ten_scaled_df_test_x)

# store accuracy
global_accuracy.append(100-(round(np.mean(rf_ten_pred != \
                                          ten_scaled_df_test_y),2)))                  

# store true vs predicted value
rf_ten_df = pd.DataFrame({'trueValue': ten_scaled_df_test_y, \
                          'predictedValue': rf_ten_pred})
# calculate the roc auc score
roc_scores.update({'RF-ten': roc_auc_score(rf_ten_df.trueValue, \
                                           rf_ten_df.predictedValue)})

# end time
algorithm_duration_list.append(time.time())

####
# End ten dataset
####

####
# Start twenty dataset
####

# start time
algorithm_duration_list.append(time.time())


#singular RF 
rf_clf = RandomForestClassifier(n_estimators = 7, 
                                max_depth = 9, criterion ='entropy',
                                random_state = 1337)
rf_clf.fit(twenty_scaled_df_train_x, twenty_scaled_df_train_y)

# store predictions
rf_twenty_pred = rf_clf.predict(twenty_scaled_df_test_x)

# store accuracy
global_accuracy.append(100-(round(np.mean(rf_twenty_pred != twenty_scaled_df_test_y),2)))                  

# store true vs predicted value
rf_twenty_df = pd.DataFrame({'trueValue': twenty_scaled_df_test_y, \
                          'predictedValue': rf_twenty_pred})
# calculate the roc auc score
roc_scores.update({'RF-twenty': roc_auc_score(rf_twenty_df.trueValue, \
                                              rf_twenty_df.predictedValue)})
    
# end time
algorithm_duration_list.append(time.time())

####
# End twenty dataset
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

# store true vs predicted value
knn_pearsons_df = pd.DataFrame({'trueValue': pearsons_scaled_df_test_y, \
                          'predictedValue': knn_pearsons_pred})
# calculate the roc auc score
roc_scores.update({'KNN-pearsons': roc_auc_score(knn_pearsons_df.trueValue, \
                                           knn_pearsons_df.predictedValue)})
    
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

# store true vs predicted value
knn_ols_df = pd.DataFrame({'trueValue': ols_scaled_df_test_y, \
                          'predictedValue': knn_ols_pred})
# calculate the roc auc score
roc_scores.update({'KNN-ols': roc_auc_score(knn_ols_df.trueValue, \
                                           knn_ols_df.predictedValue)})
    
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

# store true vs predicted value
knn_rfe_df = pd.DataFrame({'trueValue': rfe_scaled_df_test_y, \
                          'predictedValue': knn_rfe_pred})
# calculate the roc auc score
roc_scores.update({'KNN-rfe': roc_auc_score(knn_rfe_df.trueValue, \
                                           knn_rfe_df.predictedValue)})

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
                                          != lasso_scaled_df_test_y),2)))

# store true vs predicted value
knn_lasso_df = pd.DataFrame({'trueValue': lasso_scaled_df_test_y, \
                          'predictedValue': knn_lasso_pred})
# calculate the roc auc score
roc_scores.update({'KNN-lasso': roc_auc_score(knn_lasso_df.trueValue, \
                                           knn_lasso_df.predictedValue)})
    
# end time
algorithm_duration_list.append(time.time()) 

####
# end lasso dataset
####

####
# Start ten dataset
####

# start time
algorithm_duration_list.append(time.time())

# initalize knn
knn_classifier = KNeighborsClassifier(n_neighbors = 47)

# Train the classifier
knn_classifier.fit(ten_scaled_df_train_x, 
                   ten_scaled_df_train_y)
 
# store predictions
knn_ten_pred = knn_classifier.predict(ten_scaled_df_test_x)
       
# store accuracy
global_accuracy.append(100-(round(np.mean(knn_ten_pred
                                          != ten_scaled_df_test_y),2)))

# store true vs predicted value
knn_ten_df = pd.DataFrame({'trueValue': ten_scaled_df_test_y, \
                          'predictedValue': knn_ten_pred})
# calculate the roc auc score
roc_scores.update({'KNN-ten': roc_auc_score(knn_ten_df.trueValue, \
                                           knn_ten_df.predictedValue)})

# end time
algorithm_duration_list.append(time.time()) 

####
# end ten dataset
####

####
# Start twenty dataset
####

# start time
algorithm_duration_list.append(time.time())

# initalize knn
knn_classifier = KNeighborsClassifier(n_neighbors = 61)

# Train the classifier
knn_classifier.fit(twenty_scaled_df_train_x, 
                   twenty_scaled_df_train_y)
 
# store predictions
knn_twenty_pred = knn_classifier.predict(twenty_scaled_df_test_x)
       
# store accuracy
global_accuracy.append(100-(round(np.mean(knn_twenty_pred
                                          != twenty_scaled_df_test_y),2)))

# store true vs predicted value
knn_twenty_df = pd.DataFrame({'trueValue': twenty_scaled_df_test_y, \
                          'predictedValue': knn_twenty_pred})
# calculate the roc auc score
roc_scores.update({'KNN-twenty': roc_auc_score(knn_twenty_df.trueValue, \
                                           knn_twenty_df.predictedValue)})
    
# end time
algorithm_duration_list.append(time.time()) 

####
# end twenty dataset
####

#######
# End KNN
#######

# end timing
timings_list.append(['knn time end', time.time()])

####
# Start prediction prints
####

# reorder roc scores from highest to lowest
roc_scores = {k: v for k, v in sorted(roc_scores.items(), \
                                      key=lambda item: item[1], \
                                      reverse = True)}

# setup data for algorithm comparisons
algorithms = ['GBM', 'Random Forest', 'KNN'] 
accuracy = [99, 99.72, 99.71]
roc = ['67.26%', '52.34%', '51.46%']
duration = ['<1','<1','<1','<1']

# create a dataframe
prediction_df = pd.DataFrame(list(zip(algorithms, accuracy, roc, duration)),
                             columns = ['Algorithm', 'Accuracy', 'ROC Score',\
                                        'Duration'])
    
acc_gbm = accuracy_score(main_scaled_df_test_y, model_pred_y)

    
####
# Start GMB prediction prints
####
    
# confusion matrix for random forest 8/8
cm_one = confusion_matrix(main_scaled_df_test_y, model_pred_y_01)
tn, fp, fn, tp  = confusion_matrix(main_scaled_df_test_y, \
                                   model_pred_y_01).ravel()

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
print(classification_report(main_scaled_df_test_y, model_pred_y_01))

# TPR/TNR rates
print('The TPR is:', str(tp) + '/' + str(tp + fn) + ',',
      round(recall_score(main_scaled_df_test_y, model_pred_y_01) * 100, 2),'%')
print('The TNR is:', str(tn) + '/' + str(tn + fp) + ',',
    round(tn / (tn + fp) * 100, 2),'%')

####
# End GMB prediction prints
####
    
    
####
# Start rf ten prediction prints
####

# confusion matrix for random forest pearsons
cm_one = confusion_matrix(ten_scaled_df_test_y, rf_ten_pred)
tn, fp, fn, tp  = confusion_matrix(ten_scaled_df_test_y, \
                                   rf_ten_pred).ravel()

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
print(classification_report(ten_scaled_df_test_y, rf_ten_pred))

# TPR/TNR rates
print('The TPR is:', str(tp) + '/' + str(tp + fn) + ',',
      round(recall_score(ten_scaled_df_test_y, rf_ten_pred) * 100, 2),'%')
print('The TNR is:', str(tn) + '/' + str(tn + fp) + ',',
    round(tn / (tn + fp) * 100, 2),'%')

####
# End rf ten prediction prints
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