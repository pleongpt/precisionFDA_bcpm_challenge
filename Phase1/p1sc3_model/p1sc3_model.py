#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Common imports
import os
import argparse
import sys
import platform
import time
import datetime

import numpy as np
import pandas as pd
import numpy.random as rnd
import matplotlib.pyplot as plt

import sklearn
# Data preparation
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Evaluation tools
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
import xgboost
from joblib import dump, load

"""
@authors: Patrick Leong and Nick Wang

Model for Phase 1 sub-challenge 3.

Input: RNA_DNA_CN expression matrix, Clinical data
Output: Predicted survival

Observations:
1. RNA_DNA_CN data has 20146 attributes but only 166 samples.
2. Clinical data has asymmetric distributions in each attribute
   (sex, race, WHO grading, cancer type). Also lots of NaN.

Algorithm:
Among the 166 samples, separate them into training set and test set.
Need to preserve the asymmetric distributions into the training
and test sets.

Normalize/Standardize/One-Hot Encode data where appropriate.

Try multiple learning algorithms that can reduce attributes:
1. Logistic Regression with Lasso
2. Elastic Net
3. Linear SVM
4. Gaussian SVM with RBF kernel
5. Random Forest
6. XGBoost
7. Soft Voting

Also, bagging can be optionally enabled or disabled.

Evaluate the performance of each algorithm.

Output topN most important features.
"""

# Random seed value
rnd_seed = 1234

# test set vs train set samples ratio
test_set_ratio = 0.2

# Include phenotype data in modeling
bAnalyze_Phenotype = False

# TopN attributes
topN = 20

# Classifier model flags
FLAG_LOG_REG      = 0x1  # Logistic regression
FLAG_ELASTIC      = 0x2  # Elastic net
FLAG_LINEAR_SVM   = 0x4  # Linear Support Vector Machine
FLAG_GAUSSIAN_RBF_SVM = 0x8  # Support Venctor Machine with Gaussian RBF kernel
FLAG_RF           = 0x10 # Random Forest
FLAG_XGBOOST      = 0x20 # XGBoost
# Ensemble learning algorrithm flags
FLAG_SOFT_VOTING  = 0x100 # Soft voting
FLAG_BAGGING      = 0x200 # Bagging
FLAG_ALL_CLASSIFIERS = (FLAG_LOG_REG + FLAG_ELASTIC + FLAG_LINEAR_SVM + FLAG_GAUSSIAN_RBF_SVM + FLAG_RF + FLAG_XGBOOST)
FLAG_ALL = (FLAG_LOG_REG + FLAG_ELASTIC + FLAG_LINEAR_SVM + FLAG_GAUSSIAN_RBF_SVM + FLAG_RF + FLAG_XGBOOST + FLAG_SOFT_VOTING)

model_flags = FLAG_ALL # Default

# Bagging
bBagging = False
bagging_max_sample_ratio = 0.7
bagging_n_estimators = 20

# Smote--to synthesize more minority samples
bSmote = True
# smote_ratio is number of minority samples / number of majority samples
smote_ratio = 0.8

# Apply attribute pre-filters from P1SC1 and P1SC2 on raw data
bPreFilter = True
default_rna_prefilter_file = 'p1sc1_topN_xgboost.txt'
rna_prefilter_filepath = None
default_dna_CN_prefilter_file = 'p1sc2_topN_xgboost.txt'
dna_CN_prefilter_filepath = None

# Predict based on trained model
bPredictMode = False
default_trained_model_filename = 'p1sc3_sv_clf.joblib'

# For plotting pretty figures
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Interactive blocking when a plot is made
bPlot_blocking = False # Default

PROJECT_ROOT_DIR = "."

def data_filepath(filename):
    return os.path.join(PROJECT_ROOT_DIR, "data", filename)

def alternative_data_filepath(filename):
    data_dir = os.getcwd().replace('scripts', 'data')
    data_dir = os.path.dirname(data_dir)
    return os.path.join(data_dir, filename)

def topN_filepath(filename):
    return os.path.join(PROJECT_ROOT_DIR, "outputs", "topN", filename)

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "outputs", "plots", fig_id)

def save_fig(fig_id, tight_layout=True, ext='png'):
    print("Saving figure %s" % fig_id)
    if tight_layout:
        plt.tight_layout()
    filepath = image_path(fig_id) + '.' + ext
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(filepath, format=ext, dpi=300)

def create_xgboost_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

def load_input_data():
    # RNA_DNA_CN_gene_expresion data
    filepath = data_filepath('sc3_Phase1_CN_GE_FeatureMatrix.tsv')
    if not os.path.isfile(filepath):
        filepath = alternative_data_filepath('sc3_Phase1_CN_GE_FeatureMatrix.tsv')
        if not os.path.isfile(filepath):
            print("Data file sc3_Phase1_CN_GE_FeatureMatrix.tsv is missing.")
            return None, None, None

    data_rna_dna_CN = pd.read_csv(filepath, index_col='PATIENTID',
                           sep='\t', skipinitialspace=True)

    # Check the percentage of cells in data_rna_dna_CN missing (NaN):
    data_rna_dna_CN_missing_values_count = data_rna_dna_CN.isnull().sum()
    data_rna_dna_CN_total_cells = np.product(data_rna_dna_CN.shape)
    total_missing = data_rna_dna_CN_missing_values_count.sum()

    print("Shape of RNA_DNA_CN data: %s, max=%.4f, min=%.4f, NaN=%.2f%%" % (data_rna_dna_CN.shape, max(data_rna_dna_CN.max()),
                                                                     min(data_rna_dna_CN.min()),
                                                                     (total_missing/data_rna_dna_CN_total_cells) * 100))

    # Clinical Phenotypes (SEX, RACE, WHO_GRADING, CANCER_TYPE)
    filepath = data_filepath('sc3_Phase1_CN_GE_Phenotype.tsv')
    if not os.path.isfile(filepath):
        filepath = alternative_data_filepath('sc3_Phase1_CN_GE_Phenotype.tsv')
        if not os.path.isfile(filepath):
            print("Data file sc3_Phase1_CN_GE_Phenotype.tsv is missing.")
            return None, None, None

    data_phenotype = pd.read_csv(filepath, index_col='PATIENTID',
                                 sep='\t', skipinitialspace=True)

    print("Shape of clinical phenotype data: ", end='')
    print(data_phenotype.shape)

    # Outcome (survival status)
    filepath = data_filepath('sc3_Phase1_CN_GE_Outcome.tsv')
    if not os.path.isfile(filepath):
        filepath = alternative_data_filepath('sc3_Phase1_CN_GE_Outcome.tsv')
        if not os.path.isfile(filepath):
            print("Data file sc3_Phase1_CN_GE_Outcome.tsv is missing.")
            return None, None, None

    data_survival = pd.read_csv(filepath, index_col='PATIENTID',
                                sep='\t', skipinitialspace=True)

    print("Shape of survival status data: ", end='')
    print(data_survival.shape)
    print()

    # Check if all dataframes have unique patent IDs
    flag = len(set(list(data_rna_dna_CN.index.values))) == len(list(data_rna_dna_CN.index.values))
    if flag:
        print("All PATIENTID entries in data_rna_dna_CN are unique.")
    else:
        print("Warning: Some PATIENTID entries in data_rna_dna_CN are duplicates!")

    flag = len(set(list(data_phenotype.index.values))) == len(list(data_phenotype.index.values))
    if flag:
        print("All PATIENTID entries in data_phenotype are unique.")
    else:
        print("Warning: Some PATIENTID entries in data_phenotype are duplicates!")

    flag = len(set(list(data_survival.index.values))) == len(list(data_survival.index.values))
    if flag:
        print("All PATIENTID entries in data_survival are unique.")
    else:
        print("Warning: Some PATIENTID entries in data_survival are duplicates!")
    print()

    # Check if all dataframes have the same patent IDs. (The rows
    # may be in different orders.)
    if (list(data_rna_dna_CN.index.values).sort() == list(data_phenotype.index.values).sort()):
        print("PATIENTID entries in data_rna_dna_CN are the same as in data_phenotype.")
    else:
        print("Warning: PATIENTID entries in data_rna_dna_CN are NOT the same as in data_phenotype!")

    if (list(data_rna_dna_CN.index.values).sort() == list(data_survival.index.values).sort()):
        print("PATIENTID entries in data_rna_dna_CN are the same as in data_survival.")
    else:
        print("Warning: PATIENTID entries in data_rna_dna_CN are NOT the same as in data_survival!")

    print()

    return data_rna_dna_CN, data_phenotype, data_survival

# ROC curve
def plot_roc_curve(fpr, tpr, title, auc, **options):
    plt.plot(fpr, tpr, linewidth=2, marker="o", **options)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.02, 1.02, -0.02, 1.02])
    auc_string = "AUC = %0.2f" % auc
    plt.text(0.6, 0.2, auc_string)
    plt.title(title)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

# Precision-Recall curve
def plot_precision_vs_recall(precisions, recalls, title):
    plt.plot(recalls, precisions, "bo--", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([-0.02, 1.02, -0.02, 1.02])
    plt.title(title)

# Check performance of classifier
def check_classifier_performance(classifier_pipeline, clf_name, train_data,
                                 train_labels, test_data, test_labels,
                                 bagging_oob_score=1.0):
    global bPlot_blocking
    global bBagging

    print("==============================")
    print("On training set (%s)" % clf_name)
    print("==============================")
    # Bagging score, if applicable
    if bBagging and clf_name != "RF":
        print("Training Set: Bagging OOB score = %0.2f" % bagging_oob_score)

    # Use k-fold cross-validation on training set to check accuracy of the classifier_pipeline. Here, k=5.
    # scores holds the k scores where each time a subset of the training data is randomly selected
    # as the validation set. n_jobs=-1 means use all CPU cores available.
    scores = cross_val_score(classifier_pipeline, train_data, train_labels, cv=5, scoring="accuracy", n_jobs=-1)
    print("Training set: Accuracy via cross_val_score(k=5): %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))
    #print(scores)

    # Repeat using F1 score
    scores = cross_val_score(classifier_pipeline, train_data, train_labels, cv=5, scoring="f1", n_jobs=-1)
    print("F1 score via cross_val_score(k=5): %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))
    #print(scores)

    # Track predictions based on the training set
    train_labels_pred = cross_val_predict(classifier_pipeline, train_data, train_labels, cv=5, n_jobs=-1)

    # Accuracy
    print("Training Set: Accuracy via cross_val_predict(k=5) = %0.2f" % (np.sum(train_labels_pred == train_labels) / len(train_labels)))

    # Balanced accuracy
    # The balanced accuracy in binary and multiclass classification problems to deal
    # with imbalanced datasets. It is defined as the average of recall obtained on each class.
    # The best value is 1 and the worst value is 0 when adjusted=False.
    print("Training Set: Balanced Accuracy = %0.2f" % balanced_accuracy_score(train_labels, train_labels_pred))

    # Cohen's Kappa score
    # The kappa score (see docstring) is a number between -1 and 1.
    # Scores above .8 are generally considered good agreement;
    # zero or lower means no agreement (practically random labels)
    print("Training Set: Cohen's Kappa score = %0.2f" % cohen_kappa_score(train_labels, train_labels_pred))

    # Confusion matrix
    # confusion_matrix(train_labels, train_labels_pred)
    # Use pd.crosstab instead of confusion_matrix.  crosstab creates a cross-tabulation of train_labels
    # against train_labels_pred, also prints the labels and the sum for each category.  This is essentially
    # the confusion matrix.
    print("Training Set: Confusion Matrix via cross_val_predict:")
    cm = pd.crosstab(train_labels, train_labels_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print(cm.to_string())

    # Precision
    print("\nTraining Set cross_val_predict: Precision = %0.2f" % (precision_score(train_labels, train_labels_pred)))

    # Recall (TPR) = TP / (TP + FN)
    print("Training Set cross_val_predict: Recall = %0.2f" % (recall_score(train_labels, train_labels_pred)))

    # F1-score = 2 * precision * recall / (precision + recall)
    print("Training Set cross_val_predict: F1 = %0.2f" % (f1_score(train_labels, train_labels_pred)))

    train_labels_probas = cross_val_predict(classifier_pipeline, train_data, train_labels, cv=5, method="predict_proba", n_jobs=-1)
    train_labels_scores = train_labels_probas[:, 1]  # score = probabilities of positive label
    fpr, tpr, thresholds = roc_curve(train_labels, train_labels_scores)

    # AUC
    auc_score = roc_auc_score(train_labels, train_labels_scores)
    print("Training Set: cross_val_predict AUC = %0.2f" % auc_score)

    print()
    plt.figure(figsize=(8, 6))
    plot_roc_curve(fpr, tpr, clf_name + ": ROC on Train Data", auc_score)
    filename = clf_name + "_ROC_train_data_plot"
    save_fig(filename)
    plt.show(block=bPlot_blocking)

    # Precision and Recall
    precisions, recalls, thresholds = precision_recall_curve(train_labels, train_labels_scores)

    plt.figure(figsize=(8, 6))
    plot_precision_vs_recall(precisions, recalls, clf_name + ": Precision-Recall on Train Data")
    filename = clf_name + "_precision_vs_recall_train_data_plot"
    save_fig(filename)
    plt.show(block=bPlot_blocking)

    # ===================================================================
    # ================ Check performance on test data set ===============
    # ===================================================================
    test_labels_pred = classifier_pipeline.predict(test_data)

    # Accuracy
    print()
    print("==============================")
    print("On test set (%s)" % clf_name)
    print("==============================")
    print("Test Set: Accuracy = %0.2f" % (np.sum(test_labels_pred == test_labels) / len(test_labels)))

    # Balanced accuracy
    print("Test Set: Balanced Accuracy = %0.2f" % balanced_accuracy_score(test_labels, test_labels_pred))

    # Cohen's Kappa score
    print("Test Set: Cohen's Kappa score = %0.2f" % cohen_kappa_score(test_labels, test_labels_pred))

    # confusion_matrix(test_labels, test_labels_pred)
    # Use pd.crosstab instead of confusion_matrix because the former prints out the labels and the sum for each category as well.
    print("Test Set: Confusion Matrix:")
    cm = pd.crosstab(test_labels, test_labels_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print(cm.to_string())

    # Precision
    print("\nTest Set: Precision = %0.2f" % (precision_score(test_labels, test_labels_pred)))

    # Recall
    print("Test Set: Recall = %0.2f" % (recall_score(test_labels, test_labels_pred)))

    # F1-score
    print("Test Set: F1 = %0.2f" % (f1_score(test_labels, test_labels_pred)))

    # The following returns the probabilities in the predictions.  Column 0 for the negative label and column 1 for the positive label.
    test_labels_probas = classifier_pipeline.predict_proba(test_data)
    test_labels_scores = test_labels_probas[:, 1]  # Probabilities of the positive label

    # ROC curve
    fpr, tpr, thresholds = roc_curve(test_labels, test_labels_scores)

    # AUC
    auc_score = roc_auc_score(test_labels, test_labels_scores)
    print("Test Set: AUC = %0.2f" % auc_score)

    print()
    plt.figure(figsize=(8, 6))
    plot_roc_curve(fpr, tpr, clf_name + ": ROC on Test Data", auc_score)
    filename = clf_name + "_ROC_test_data_plot"
    save_fig(filename)
    plt.show(block=bPlot_blocking)

    # Precision-Recall Curve
    precisions, recalls, thresholds = precision_recall_curve(test_labels, test_labels_scores)

    plt.figure(figsize=(8, 6))
    plot_precision_vs_recall(precisions, recalls, clf_name + ": Precision-Recall on Test Data")
    filename = clf_name + "_precision_vs_recall_test_data_plot"
    save_fig(filename)
    plt.show(block=bPlot_blocking)

    # Close all existing figures
    plt.close('all')

    # Relative importance of variables
    # Create a data frame with the first column holding the names of the variables and the second column the corresponding importance score.
    # importance_scores = classifier.feature_importances_
    # variables_names = list(test_data)
    # importances = pd.DataFrame({'Name': variables_names, 'Gini Importance': importance_scores})
    # importances = importances.sort_values('Gini Importance', ascending=False).set_index('Name')
    # # Show the top 20 variables
    # importances.head(topN)
    # # importances.head(topN).plot(kind='bar', rot=90)
    #
    # variables_topN = list(importances.head(topN).index)
    # importances_topN = importances.head(topN)['Gini Importance']
    #
    # # Plot topN variables
    # plt.figure(figsize=(8, 6))
    # plot_variables_importances(variables_topN, importances_topN)
    # save_fig("RF_Gini_importances_key_variables")
    # plt.show(block=bPlot_blocking)
    #
    # # Plot all variables
    # plt.figure(figsize=(8, 6))
    # variables = list(importances.index)
    # scores = importances['Gini Importance']
    # plot_variables_importances(variables, scores, font_size=1)
    # save_fig("RF_Gini_importances_all_variables", ext='pdf')
    # plt.show(block=bPlot_blocking)

def show_most_important_attributes(classifier_pipeline, clf_name, attribute_names, topN):
    if clf_name == "log_reg_l1" or clf_name == "elastic_net":
        abs_weights = np.abs(classifier_pipeline[1].coef_)
        abs_weights = np.ravel(abs_weights) # flattern for sorting
        # Sort by absolute weights in decreasing.  Pairing with the corresponding index.
        abs_weights_sorted = sorted(enumerate(abs_weights), key=lambda x:x[1])[::-1]
        topN_attributes =  abs_weights_sorted[:topN]
        print()
        print("Classifier %s top %d important attributes:" % (clf_name, topN))
        print("Name %11s Weight" % " ")
        for i in range(topN):
            print("%-16s %0.6f" % (attribute_names[topN_attributes[i][0]], topN_attributes[i][1]))

        topN_filename = 'p1sc3_topN_'+clf_name+'.txt'
        filepath = topN_filepath(topN_filename)
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filepath, 'w') as out_file:
            for i in range(topN):
                out_file.write("%s\t%0.6f\n" % (attribute_names[topN_attributes[i][0]], topN_attributes[i][1]))

    # Feature importances for SVM with Gaussian RBF kernel is not supported.
    # See https://stats.stackexchange.com/questions/265656/is-there-a-way-to-determine-the-important-features-weight-for-an-svm-that-uses
    if clf_name == "linear_SVM":
        importances = abs(classifier_pipeline[1].coef_[0])
        importances, attribute_names = zip(*sorted(list(zip(importances, attribute_names)), reverse=True))
        print()
        print("Classifier %s top %d important attributes:" % (clf_name, topN))
        print("Name %11s Importance" % " ")
        for i in range(topN):
            print("%-16s %0.6f" % (attribute_names[i], importances[i]))

        topN_filename = 'p1sc3_topN_'+clf_name+'.txt'
        filepath = topN_filepath(topN_filename)
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filepath, 'w') as out_file:
            for i in range(topN):
                out_file.write("%s\t%0.6f\n" % (attribute_names[i], importances[i]))

    if clf_name == "RF":
        # Relative importance of variables
        # Create a data frame with the first column holding the names of the variables and
        # the second column the corresponding importance score.
        importance_scores = classifier_pipeline[1].feature_importances_
        importances = pd.DataFrame({'Name': attribute_names, 'Gini Importance': importance_scores})
        importances = importances.sort_values('Gini Importance', ascending=False)
        print()
        print("Classifier %s top %d important attributes:" % (clf_name, topN))
        print("Name %10s GINI Importance" % " ")
        for i in range(topN):
            print("%-16s %0.6f" % (importances.iloc[i]['Name'], importances.iloc[i]['Gini Importance']))

        topN_filename = 'p1sc3_topN_'+clf_name+'.txt'
        filepath = topN_filepath(topN_filename)
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filepath, 'w') as out_file:
            for i in range(topN):
                out_file.write("%s\t%.6f\n" % (importances.iloc[i]['Name'], importances.iloc[i]['Gini Importance']))

    if clf_name == "xgboost":
        fscore = classifier_pipeline[1].get_booster().get_fscore(fmap='xgb.fmap')
        importances = []
        for ft, score in fscore.items():
            importances.append({'Feature': ft, 'Importance': score})

        importances = pd.DataFrame(importances)
        importances = importances.sort_values(
            by='Importance', ascending=False).reset_index(drop=True)

        # Normalize so that sum of all feature importances is 1
        importances['Importance'] /= importances['Importance'].sum()

        topN_importances = importances.iloc[:topN][:]
        print()
        print("Classifier %s top %d important attributes:" % (clf_name, topN))
        for i in range(topN):
            print("%-16s %0.6f" % (topN_importances.iloc[i][0], topN_importances.iloc[i][1]))
        print()

        topN_filename = 'p1sc3_topN_'+clf_name+'.txt'
        filepath = topN_filepath(topN_filename)
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filepath, 'w') as out_file:
            for i in range(topN):
                out_file.write("%s\t%0.6f\n" % (topN_importances.iloc[i][0], topN_importances.iloc[i][1]))

def tally_up_WHO_grading_vs_cancer_type(data_grading_cancer_type):
    # freq_counts_dict is a dict of dict.  First key is cancer type.
    # Second key is grading, its value is the frequency count.
    freq_counts_dict = {}
    na_status = (data_grading_cancer_type.isnull()).loc[:,'WHO_GRADING']
    for i in range(data_grading_cancer_type.shape[0]):
        if not na_status[i]:
            cancer_type = data_grading_cancer_type.iloc[i]['CANCER_TYPE']
            grading = data_grading_cancer_type.iloc[i]['WHO_GRADING']
            if cancer_type in freq_counts_dict:
                if grading in freq_counts_dict[cancer_type]:
                    freq_counts_dict[cancer_type][grading] += 1
                else:
                    freq_counts_dict[cancer_type][grading] = 1
            else:
                freq_counts_dict[cancer_type] = {grading:1}

    lookup_table = {}
    for cancer_type, value in freq_counts_dict.items():
        max_freq = value[max(value, key=value.get)]
        # There may be more than one grading per cancer type with the same
        # maximum frequency. In this case we include them in and when we do
        # the fillna later on, we will randomly pick one of these grading.

        # Also, there may be cancer types with no WHO grading. In this case
        # the cancer type will not show up in the lookup_table.
        ranking = []
        for k in sorted(value, key=value.get, reverse=True):
            if value[k] == max_freq:
                ranking.append((k, value[k]))

        lookup_table[cancer_type] = ranking

    return lookup_table

def fillna_for_WHO_grading(data_grading_cancer_type):
    lookup_table = tally_up_WHO_grading_vs_cancer_type(data_grading_cancer_type)

    na_status = (data_grading_cancer_type.isnull()).loc[:,'WHO_GRADING']
    for i in range(data_grading_cancer_type.shape[0]):
        if na_status[i]:
            cancer_type = data_grading_cancer_type.iloc[i]['CANCER_TYPE']
            if cancer_type in lookup_table:
                list_len = len(lookup_table[cancer_type])
                if list_len > 1:
                    # More than one candidate, then draw randomly
                    randomly_selected_index = np.random.choice(list_len, 1)[0]
                    assigned_grading = lookup_table[cancer_type][randomly_selected_index][0]
                else:
                    assigned_grading = lookup_table[cancer_type][0][0]

                data_grading_cancer_type.iloc[i]['WHO_GRADING'] = assigned_grading

    # There may be some NaN entries left due to the cancer type has no WHO grading in other samples.
    # Use the SimplyImputer to fill in with the mode of the WHO grading
    imp_mode = SimpleImputer(strategy='most_frequent')
    data_grading_cancer_type[['WHO_GRADING']] = imp_mode.fit_transform(data_grading_cancer_type[['WHO_GRADING']])

    return pd.DataFrame(data_grading_cancer_type)

def model_RNA_DNA_CN_and_phenotype_vs_survival_data(data_rna_dna_CN, data_phenotype, data_survival, bPhenotype):
    # Algorithm:
    # 1. First make sure the entries in data_rna_dna_CN and in data_survival
    #    are aligned.
    # 2. We know that data_rna_dna_CN and data_survival has no NaN entry from
    #    script p1_sc1_inspect_data.py.
    # 3. We normalize the values in data_rna_dna_CN to avoid having any outlier from
    #    skewing the data.
    # 4. Note that data_survival has more deaths than alives (more 1s than 0s).
    #    We should conserve this ratio in the test set.  Hence use StratifiedShuffleSplit.
    # 5. Create a training set and a test set
    # 6. Train and verify with various algorithms and compare the results.
    #    Since the number of attributes of data_rna_dna_CN is huge, we should attempt
    #    using models that have attribute reduction.

    global model_flags
    global topN
    global bBagging
    global bagging_max_sample_ratio
    global bagging_n_estimators
    global bSmote
    global test_set_ratio
    global smote_ratio
    global rnd_seed
    global bPreFilter
    global rna_prefilter_filepath
    global dna_CN_prefilter_filepath

    if bPreFilter:
        rna_prefilter = pd.read_csv(rna_prefilter_filepath, sep='\t', header=None)
        rna_prefilter = list(rna_prefilter.iloc[:,0])
        dna_CN_prefilter = pd.read_csv(dna_CN_prefilter_filepath, sep='\t', header=None)
        dna_CN_prefilter = list(dna_CN_prefilter.iloc[:,0])

        prefilter = rna_prefilter + dna_CN_prefilter
        data_rna_dna_CN = data_rna_dna_CN.loc[:, prefilter]

    numeric_features_names = list(data_rna_dna_CN.columns)
    # Use indices instead of labels because if bagging is enabled, the Bagging Algorithm cannot access
    # the columns based on column names.
    numeric_features_indices = [ i for i in range(data_rna_dna_CN.shape[1])]

    if bPhenotype:
        # Focus on the WHO Grade and Cancer Type.  For now neglect the effects on SEX and RACE.
        data_grading_cancer_type = data_phenotype.loc[:,['WHO_GRADING','CANCER_TYPE']]

        # Cancer type has the following types:
        # ASTROCYTOMA, GBM, UNCLASSIFIED, OLIGODENDROGLIOMA, UNKNOWN, MIXED
        # So far there is no NaN.

        # First, replace NaN (if any) in Cancer Type with 'Unknown'.
        fillna_values = {'CANCER_TYPE':'UNKNOWN'}
        data_grading_cancer_type.fillna(value=fillna_values, inplace=True)

        # The Each NaN value in 'WHO_GRADING" should be replaced by the mode
        # WHO_GRADING type of its associated cancer type.
        data_grading_cancer_type = fillna_for_WHO_grading(data_grading_cancer_type)

        # Reindex data_grading_cancer_type such that it aligns with the index in data_rna_dna_CN
        data_grading_cancer_type = data_grading_cancer_type.reindex(data_rna_dna_CN.index)
        data_input = pd.concat([data_rna_dna_CN, data_grading_cancer_type], axis=1)

        # categorical_features_names for show_most_important_attributes
        categorical_features_names = list(data_grading_cancer_type.columns)
        # Use indices so as to be compatible with Bagging
        categorical_features_indices = [i+data_rna_dna_CN.shape[1] for i in range(data_grading_cancer_type.shape[1])]
    else:
        data_input = data_rna_dna_CN
        categorical_features_names = []
        categorical_features_indices = []

    # Reindex data_survival such that it aligns with the index in data_rna_dna_CN
    # (They should be the same in the raw data but just in case.)
    data_survival = data_survival.reindex(data_rna_dna_CN.index)

    # Swap the data_survival values so that the minority is 1.  This makes the performance
    # of the classifiers more illustrative by the F1 value. We have a class imbalance
    # issue here.
    print("--------------------------------------------------------------------------")
    print("Swap the data_survival values so that the minority is 1 and majority is 0.")
    print("Will flip back this logic in actual model prediction.")
    print("--------------------------------------------------------------------------")
    data_survival["SURVIVAL_STATUS"] = data_survival["SURVIVAL_STATUS"].replace({0:1, 1:0})
    print()

    # Split the data into train and test sets but preserve the alive/death ratio.
    # n_splits=10 means take 10 re-shuffling operations (to make sure there is no
    # order bias.  We pick up the last one.
    print("Split data into training and test sets using StratifiedShuffleSplit. test_set_ratio=%.2f" % test_set_ratio)
    print()
    split = StratifiedShuffleSplit(n_splits=10, test_size=test_set_ratio, random_state=rnd_seed)
    for train_index, test_index in split.split(data_survival, data_survival["SURVIVAL_STATUS"]):
        X_train = data_input.iloc[train_index]
        y_train = data_survival.iloc[train_index]
        X_test = data_input.iloc[test_index]
        y_test = data_survival.iloc[test_index]

    print("Shape of X_train: %s" % str(X_train.shape))
    print("Shape of y_train: %s" % str(y_train.shape))
    print()
    print("Shape of X_test: %s" % str(X_test.shape))
    print("Shape of y_test: %s" % str(y_test.shape))
    print()

    ###DEBUG
    # # Generate test data for model prediction
    # rna_dna_CN_predict = data_rna_dna_CN.iloc[test_index]
    # filepath = data_filepath('rna_dna_CN_predict.tsv')
    # directory = os.path.dirname(filepath)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # rna_dna_CN_predict.to_csv(filepath, sep='\t', index=True)
    # pheno_predict = data_phenotype.iloc[test_index]
    # filepath = data_filepath('pheno_predict.tsv')
    # pheno_predict.to_csv(filepath, sep='\t', index=True)
    ###DEBUG

    print("Original data_survival label distribution (0:deceased, 1:alive):")
    label_dist = np.unique(data_survival, return_counts=True)
    for i in range(len(label_dist[0])):
        print("%d:%d\t%.4f%%" % (label_dist[0][i], label_dist[1][i], label_dist[1][i]/len(data_survival)))
    print()

    print("y_train label distribution (0:deceased, 1:alive):")
    label_dist = np.unique(y_train, return_counts=True)
    for i in range(len(label_dist[0])):
        print("%d :%d\t%.4f%%" % (label_dist[0][i], label_dist[1][i], label_dist[1][i]/len(y_train)))
    print()

    print("y_test label distribution (0:deceased, 1:alive):")
    label_dist = np.unique(y_test, return_counts=True)
    for i in range(len(label_dist[0])):
        print("%d:%d\t%.4f%%" % (label_dist[0][i], label_dist[1][i], label_dist[1][i]/len(y_test)))
    print()

    # Flatten y_train and y_test to a 1-D array for fitting into the
    # classifier models.
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    if bSmote:
        print("--------------------------------------------------------------------------")
        if bPhenotype:
            print("Apply SMOTENC to synthesize more minority data against majority in training set.")
        else:
            print("Apply SMOTE to synthesize more minority data against majority in training set.")
        print("smote_ratio=%.2f" % smote_ratio)
        print("--------------------------------------------------------------------------")

        if bPhenotype:
            cat_features = [X_train.shape[1]-2, X_train.shape[1]-1]
            sm = SMOTENC(sampling_strategy=smote_ratio, categorical_features=cat_features, random_state=rnd_seed)
        else:
            sm = SMOTE(sampling_strategy=smote_ratio, n_jobs=-1, random_state=rnd_seed)
        X_train, y_train = sm.fit_resample(X_train, y_train)

        print("New shape of X_train: %s" % str(X_train.shape))
        print("New shape of y_train: %s" % str(y_train.shape))
        print("SMOTEd y_train label distribution (0:deceased, 1:alive):")
        label_dist = np.unique(y_train, return_counts=True)
        for i in range(len(label_dist[0])):
            print("%d :%d\t%.4f%%" % (label_dist[0][i], label_dist[1][i], label_dist[1][i]/len(y_train)))

    # Standardization processes each attribute such that each attribute's mean is zero
    # and its spread scales in terms of its standard deviation.
    # Since standard deviation is computed among all samples per attribute, we need to
    # "fit" StandardScalar to the training data only, not to all data before the split.
    # After that, we can use the StandardScalar instance to "transform" the training data
    # and the testing data.
    # Default copy=True
    #standard_scaler = StandardScaler()
    #standard_scaler.fit(X_train)

    # Normalization processes each sample independently such that it has unit norm over its
    # attributes. Scikit-learn offers the choice of normalization using l1, l2 or max norm.
    # Default is l2. Default copy=True.

    # Use of standardization and normalization depends on the classifier.
    # Good reference: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py

    # Lasso Regression ("log_reg")
    # Use the "saga" solver because it supports the Lasso "l1" penalty.
    # The "l1" penalty helps to remove some of the irrelevant RNA_DNA_CN attributes
    # is it has a large number of them.  By default, the "C" value in
    # LogisticRegression is 1.0.  The larger C is, the less the regularization
    # strength and vice versa.
    if model_flags & FLAG_LOG_REG:
        log_reg_solver = "liblinear"
        log_reg_C_value = 50.0
        log_reg_max_iter = 500
        print()
        print("**************************************************************************")
        print("Logistic Regression with L1 regularization on RNA_DNA_CN_data vs survival labels:")
        print("solver = %s, C = %.2f, max_ter = %d" % (log_reg_solver, log_reg_C_value, log_reg_max_iter))
        print("Bagging = %s" % ['False', 'True'][bBagging])
        print("**************************************************************************")
        log_reg_clr = LogisticRegression(solver=log_reg_solver, penalty="l1",
                                         C=log_reg_C_value, max_iter=log_reg_max_iter,
                                         random_state=rnd_seed)
        numeric_transformer = Pipeline(steps=[('norm', Normalizer())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        transformers=[('num', numeric_transformer, numeric_features_indices)]
        if bPhenotype:
            transformers.append(('cat', categorical_transformer, categorical_features_indices))
        preprocessor = ColumnTransformer(transformers)
        if bBagging:
            log_reg_pipeline = BaggingClassifier(Pipeline([('preprocessor', preprocessor),
                                                           ('log_reg_l1', log_reg_clr)]),
                                                 n_estimators=bagging_n_estimators,
                                                 max_samples=bagging_max_sample_ratio,
                                                 bootstrap=True, # with replacement
                                                 n_jobs=-1, # spread over CPU cores
                                                 oob_score=True,
                                                 random_state=rnd_seed)
        else:
            log_reg_pipeline = Pipeline([('preprocessor', preprocessor), ('log_reg_l1', log_reg_clr)])

        if not model_flags & FLAG_SOFT_VOTING:
            # Soft voting needs the defintion of log_reg_pipeline but it will do the fit
            # and predictions at its own code location
            log_reg_pipeline.fit(X_train, y_train)
            if bBagging:
                check_classifier_performance(log_reg_pipeline, "log_reg_l1",
                                             X_train,
                                             y_train,
                                             X_test,
                                             y_test,
                                             bagging_oob_score=log_reg_pipeline.oob_score_)
            else:
                check_classifier_performance(log_reg_pipeline, "log_reg_l1",
                                             X_train,
                                             y_train,
                                             X_test,
                                             y_test)
                if bPhenotype:
                    OneHotTransformed_Names = list(log_reg_pipeline[0].transformers_[1][1].named_steps['onehot'].get_feature_names(categorical_features_names))
                    attribute_names = numeric_features_names + OneHotTransformed_Names
                else:
                    attribute_names = numeric_features_names
                show_most_important_attributes(log_reg_pipeline, "log_reg_l1",
                                               attribute_names,
                                               topN)
            # Save model to file
            print()
            print("Save model to file ./data/%s" % 'p1sc3_log_reg_clf.joblib')
            print()
            model_filepath = data_filepath('p1sc3_log_reg_clf.joblib')
            directory = os.path.dirname(model_filepath)
            if not os.path.exists(directory):
                os.makedirs(directory)
            if os.path.isfile(model_filepath):
                os.remove(model_filepath)
            dump(log_reg_pipeline, model_filepath)

    # Elastic Net ("elastic")
    # Setting l1_ratio=0 is equivalent to using penalty='l2', while setting l1_ratio=1 is equivalent
    # to using penalty='l1'. For 0 < l1_ratio <1, the penalty is a combination of L1 and L2.
    if model_flags & FLAG_ELASTIC:
        elasticnet_solver = "saga"
        elasticnet_l1_ratio = 0.75
        elasticnet_C_value = 0.05
        elasticnet_max_iter = 500
        print()
        print("**************************************************************************")
        print("Elastic Net Regression on RNA_DNA_CN_data vs survival labels:")
        print("solver = %s, l1_ratio = %.2f, C = %.2f, max_iter = %d" %
              (elasticnet_solver, elasticnet_l1_ratio, elasticnet_C_value, elasticnet_max_iter))
        print("Bagging = %s" % ['False', 'True'][bBagging])
        print("**************************************************************************")
        elastic_net_clr = LogisticRegression(solver=elasticnet_solver, penalty="elasticnet",
                                             l1_ratio=elasticnet_l1_ratio,
                                             C=elasticnet_C_value,
                                             max_iter=elasticnet_max_iter,
                                             random_state=rnd_seed)
        numeric_transformer = Pipeline(steps=[('norm', Normalizer())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        transformers=[('num', numeric_transformer, numeric_features_indices)]
        if bPhenotype:
            transformers.append(('cat', categorical_transformer, categorical_features_indices))
        preprocessor = ColumnTransformer(transformers)
        if bBagging:
            elastic_net_pipeline = BaggingClassifier(Pipeline([('preprocessor', preprocessor), ('elastic', elastic_net_clr)]),
                                                     n_estimators=bagging_n_estimators,
                                                     max_samples=bagging_max_sample_ratio,
                                                     bootstrap=True, # with replacement
                                                     n_jobs=-1, # spread over CPU cores
                                                     oob_score=True,
                                                     random_state=rnd_seed)
        else:
            elastic_net_pipeline = Pipeline([('preprocessor', preprocessor), ('elastic', elastic_net_clr)])

        if not model_flags & FLAG_SOFT_VOTING:
            # Soft voting needs the defintion of elastic_net_pipeline but it will do the fit
            # and predictions at its own code location
            elastic_net_pipeline.fit(X_train, y_train)
            if bBagging:
                check_classifier_performance(elastic_net_pipeline, "elastic_net",
                                             X_train,
                                             y_train,
                                             X_test,
                                             y_test,
                                             bagging_oob_score=elastic_net_pipeline.oob_score_)
            else:
                check_classifier_performance(elastic_net_pipeline, "elastic_net",
                                             X_train,
                                             y_train,
                                             X_test,
                                             y_test)
                if bPhenotype:
                    OneHotTransformed_Names = list(elastic_net_pipeline[0].transformers_[1][1].named_steps['onehot'].get_feature_names(categorical_features_names))
                    attribute_names = numeric_features_names + OneHotTransformed_Names
                else:
                    attribute_names = numeric_features_names
                show_most_important_attributes(elastic_net_pipeline, "elastic_net",
                                               attribute_names,
                                               topN)
            # Save model to file
            print()
            print("Save model to file ./data/%s" % 'p1sc3_elastic_clf.joblib')
            print()
            model_filepath = data_filepath('p1sc3_elastic_clf.joblib')
            directory = os.path.dirname(model_filepath)
            if not os.path.exists(directory):
                os.makedirs(directory)
            if os.path.isfile(model_filepath):
                os.remove(model_filepath)
            dump(elastic_net_pipeline, model_filepath)

    # SVM
    # SVM with linear kernel ("lin_SVM")
    if model_flags & FLAG_LINEAR_SVM:
        linear_SVM_C_value = 10
        print()
        print("**************************************************************************")
        print("SVM with linear kernel on RNA_DNA_CN_data vs survival labels:")
        print("C = %.2f" % linear_SVM_C_value)
        print("Bagging = %s" % ['False', 'True'][bBagging])
        print("**************************************************************************")
        linear_svm_clr = SVC(kernel="linear", C=linear_SVM_C_value, probability=True, random_state=rnd_seed)
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        transformers=[('num', numeric_transformer, numeric_features_indices)]
        if bPhenotype:
            transformers.append(('cat', categorical_transformer, categorical_features_indices))
        preprocessor = ColumnTransformer(transformers)
        if bBagging:
            linear_svm_pipeline = BaggingClassifier(Pipeline([('preprocessor', preprocessor),
                                                              ('linear_svm', linear_svm_clr)]),
                                                    n_estimators=bagging_n_estimators,
                                                    max_samples=bagging_max_sample_ratio,
                                                    bootstrap=True, # with replacement
                                                    n_jobs=-1, # spread over CPU cores
                                                    oob_score=True,
                                                    random_state=rnd_seed)
        else:
            linear_svm_pipeline = Pipeline([('preprocessor', preprocessor), ('linear_svm', linear_svm_clr)])

        if not model_flags & FLAG_SOFT_VOTING:
            # Soft voting needs the defintion of linear_svm_pipeline but it will do the fit
            # and predictions at its own code location
            linear_svm_pipeline.fit(X_train, y_train)
            if bBagging:
                check_classifier_performance(linear_svm_pipeline, "linear_SVM",
                                             X_train,
                                             y_train,
                                             X_test,
                                             y_test,
                                             bagging_oob_score=linear_svm_pipeline.oob_score_)
            else:
                check_classifier_performance(linear_svm_pipeline, "linear_SVM",
                                             X_train,
                                             y_train,
                                             X_test,
                                             y_test)
                if bPhenotype:
                    OneHotTransformed_Names = list(linear_svm_pipeline[0].transformers_[1][1].named_steps['onehot'].get_feature_names(categorical_features_names))
                    attribute_names = numeric_features_names + OneHotTransformed_Names
                else:
                    attribute_names = numeric_features_names
                show_most_important_attributes(linear_svm_pipeline, "linear_SVM",
                                               attribute_names,
                                               topN)
            # Save model to file
            print()
            print("Save model to file ./data/%s" % 'p1sc3_linSVM_clf.joblib')
            print()
            model_filepath = data_filepath('p1sc3_linSVM_clf.joblib')
            directory = os.path.dirname(model_filepath)
            if not os.path.exists(directory):
                os.makedirs(directory)
            if os.path.isfile(model_filepath):
                os.remove(model_filepath)
            dump(linear_svm_pipeline, model_filepath)

    # SVM with Gaussian RBF kernel ("rbf_SVM")
    if model_flags & FLAG_GAUSSIAN_RBF_SVM:
        rbf_SVM_C_value = 1.0
        rbf_SVM_gamma = "auto"
        print()
        print("**************************************************************************")
        print("SVM with Gaussian RBF kernel on RNA_DNA_CN_data vs survival labels:")
        print("C = %.2f, gamma = %s" % (rbf_SVM_C_value, rbf_SVM_gamma))
        print("Bagging = %s" % ['False', 'True'][bBagging])
        print("**************************************************************************")
        gaussian_svm_clr = SVC(kernel="rbf", gamma=rbf_SVM_gamma, C=rbf_SVM_C_value,
                               probability=True, random_state=rnd_seed)
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        transformers=[('num', numeric_transformer, numeric_features_indices)]
        if bPhenotype:
            transformers.append(('cat', categorical_transformer, categorical_features_indices))
        preprocessor = ColumnTransformer(transformers)
        if bBagging:
            gaussian_svm_pipeline = BaggingClassifier(Pipeline([('preprocessor', preprocessor),
                                                                ('gaussian_svm', gaussian_svm_clr)]),
                                                      n_estimators=bagging_n_estimators,
                                                      max_samples=bagging_max_sample_ratio,
                                                      bootstrap=True, # with replacement
                                                      n_jobs=-1,
                                                      oob_score=True,
                                                      random_state=rnd_seed)
        else:
            gaussian_svm_pipeline = Pipeline([('preprocessor', preprocessor), ('gaussian_svm', gaussian_svm_clr)])

        if not model_flags & FLAG_SOFT_VOTING:
            # Soft voting needs the defintion of gaussian_svm_pipeline but it will do the fit
            # and predictions at its own code location
            gaussian_svm_pipeline.fit(X_train, y_train)
            if bBagging:
                check_classifier_performance(gaussian_svm_pipeline, "Gaussian_RBF_SVM",
                                             X_train,
                                             y_train,
                                             X_test,
                                             y_test,
                                             bagging_oob_score=gaussian_svm_pipeline.oob_score_)
            else:
                check_classifier_performance(gaussian_svm_pipeline, "Gaussian_RBF_SVM",
                                             X_train,
                                             y_train,
                                             X_test,
                                             y_test)
                if bPhenotype:
                    OneHotTransformed_Names = list(gaussian_svm_pipeline[0].transformers_[1][1].named_steps['onehot'].get_feature_names(categorical_features_names))
                    attribute_names = numeric_features_names + OneHotTransformed_Names
                else:
                    attribute_names = numeric_features_names
                show_most_important_attributes(gaussian_svm_pipeline, "Gaussian_RBF_SVM",
                                               attribute_names,
                                               topN)
            # Save model to file
            print()
            print("Save model to file ./data/%s" % 'p1sc3_rbfSVM_clf.joblib')
            print()
            model_filepath = data_filepath('p1sc3_rbfSVM_clf.joblib')
            directory = os.path.dirname(model_filepath)
            if not os.path.exists(directory):
                os.makedirs(directory)
            if os.path.isfile(model_filepath):
                os.remove(model_filepath)
            dump(gaussian_svm_pipeline, model_filepath)

    # Random Forrest ("RF")
    if model_flags & FLAG_RF:
        RF_n_estimators = 1000
        RF_class_weight = 'balanced_subsample'
        print()
        print("**************************************************************************")
        print("Random Forest Classifier on RNA_DNA_CN_data vs survival labels:")
        print("n_estimators = %d, class_weight = %s" % (RF_n_estimators, RF_class_weight))
        print("(No additional bagging on RF since RF itself uses bagging)")
        print("**************************************************************************")
        RF_clr = RandomForestClassifier(n_estimators=RF_n_estimators, n_jobs=-1,
                                        class_weight=RF_class_weight, random_state=rnd_seed)
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        transformers=[('num', numeric_transformer, numeric_features_indices)]
        if bPhenotype:
            transformers.append(('cat', categorical_transformer, categorical_features_indices))
        preprocessor = ColumnTransformer(transformers)

        RF_clr_pipeline = Pipeline([('preprocessor', preprocessor), ('random_forest', RF_clr)])

        if not model_flags & FLAG_SOFT_VOTING:
            # Soft voting needs the defintion of RF_clr_pipeline but it will do the fit
            # and predictions at its own code location
            RF_clr_pipeline.fit(X_train, y_train)
            check_classifier_performance(RF_clr_pipeline, "RF",
                                         X_train,
                                         y_train,
                                         X_test,
                                         y_test)
            if bPhenotype:
                OneHotTransformed_Names = list(RF_clr_pipeline[0].transformers_[1][1].named_steps['onehot'].get_feature_names(categorical_features_names))
                attribute_names = numeric_features_names + OneHotTransformed_Names
            else:
                attribute_names = numeric_features_names
            show_most_important_attributes(RF_clr_pipeline, "RF",
                                           attribute_names,
                                           topN)
            # Save model to file
            print()
            print("Save model to file ./data/%s" % 'p1sc3_rf_clf.joblib')
            print()
            model_filepath = data_filepath('p1sc3_rf_clf.joblib')
            directory = os.path.dirname(model_filepath)
            if not os.path.exists(directory):
                os.makedirs(directory)
            if os.path.isfile(model_filepath):
                os.remove(model_filepath)
            dump(RF_clr_pipeline, model_filepath)

    # XGBoost ("xgboost")
    if model_flags & FLAG_XGBOOST:
        print()
        print("**************************************************************************")
        print("XGBoost Classifier on RNA_DNA_CN_data vs survival labels:")
        print("(No additional bagging on XGBoost since XGBoost itself uses bagging)")
        print("**************************************************************************")
        xgb_clr = xgboost.XGBClassifier(random_state=rnd_seed)
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        transformers=[('num', numeric_transformer, numeric_features_indices)]
        if bPhenotype:
            transformers.append(('cat', categorical_transformer, categorical_features_indices))
        preprocessor = ColumnTransformer(transformers)

        xgb_clr_pipeline = Pipeline([('preprocessor', preprocessor), ('xgboost', xgb_clr)])

        if not model_flags & FLAG_SOFT_VOTING:
            # Soft voting needs the defintion of xgb_clr_pipeline but it will do the fit
            # and predictions at its own code location
            xgb_clr_pipeline.fit(X_train, y_train)
            check_classifier_performance(xgb_clr_pipeline, "xgboost",
                                         X_train,
                                         y_train,
                                         X_test,
                                         y_test)
            if bPhenotype:
                OneHotTransformed_Names = list(xgb_clr_pipeline[0].transformers_[1][1].named_steps['onehot'].get_feature_names(categorical_features_names))
                attribute_names = numeric_features_names + OneHotTransformed_Names
            else:
                attribute_names = numeric_features_names
            create_xgboost_feature_map(attribute_names)
            show_most_important_attributes(xgb_clr_pipeline, "xgboost",
                                           attribute_names,
                                           topN)
            os.remove('xgb.fmap')

            # Save model to file
            print()
            print("Save model to file ./data/%s" % 'p1sc3_xgboost_clf.joblib')
            print()
            model_filepath = data_filepath('p1sc3_xgboost_clf.joblib')
            directory = os.path.dirname(model_filepath)
            if not os.path.exists(directory):
                os.makedirs(directory)
            if os.path.isfile(model_filepath):
                os.remove(model_filepath)
            dump(xgb_clr_pipeline, model_filepath)

    #Ensemble Learning Hard Voting does not work for probabiity-based algorithms
    #such as what we have here. Hence hard voting is skipped.

    #Ensemble Learning: Soft Voting
    if model_flags & FLAG_SOFT_VOTING:
        print()
        print("**************************************************************************")
        print("Soft Voting Classifier on RNA_DNA_CN_data vs survival labels:")
        print("**************************************************************************")
        estimators = []
        if model_flags & FLAG_LOG_REG:
            estimators.append(('log_reg_l1', log_reg_pipeline))
        if model_flags & FLAG_ELASTIC:
            estimators.append(('elastic_net', elastic_net_pipeline))
        if model_flags & FLAG_LINEAR_SVM:
            estimators.append(('linear_SVM', linear_svm_pipeline))
        if model_flags & FLAG_GAUSSIAN_RBF_SVM:
            estimators.append(('Gaussian_RBF_SVM', gaussian_svm_pipeline))
        if model_flags & FLAG_RF:
            estimators.append(('random_forest', RF_clr_pipeline))
        if model_flags & FLAG_XGBOOST:
            estimators.append(('xgboost', xgb_clr_pipeline))

        if len(estimators) > 0:
            # Don't do this if no classifier is specified
            soft_voting_clf = VotingClassifier(estimators, voting='soft', n_jobs=-1)
            soft_voting_clf.fit(X_train, y_train)
            check_classifier_performance(soft_voting_clf, "soft_voting",
                                         X_train,
                                         y_train,
                                         X_test,
                                         y_test)
            # Save model to file
            print()
            print("Save model to file ./data/%s" % 'p1sc3_sv_clf.joblib')
            print()
            model_filepath = data_filepath('p1sc3_sv_clf.joblib')
            directory = os.path.dirname(model_filepath)
            if not os.path.exists(directory):
                os.makedirs(directory)
            if os.path.isfile(model_filepath):
                os.remove(model_filepath)
            dump(soft_voting_clf, model_filepath)

def predict_based_on_model(rna_dna_CN_predict_filepath, phenotype_predict_filepath, trained_model_filepath):
    global bPreFilter

    # clf_pipeline is the trained model
    if os.path.isfile(trained_model_filepath):
        clf_pipeline = load(trained_model_filepath)

    rna_dna_CN_predict = pd.read_csv(rna_dna_CN_predict_filepath, index_col='PATIENTID',
                           sep='\t', skipinitialspace=True)

    if bPreFilter:
        rna_prefilter = pd.read_csv(rna_prefilter_filepath, sep='\t', header=None)
        rna_prefilter = list(rna_prefilter.iloc[:,0])
        dna_CN_prefilter = pd.read_csv(dna_CN_prefilter_filepath, sep='\t', header=None)
        dna_CN_prefilter = list(dna_CN_prefilter.iloc[:,0])

        prefilter = rna_prefilter + dna_CN_prefilter
        rna_dna_CN_predict = rna_dna_CN_predict.loc[:, prefilter]

    phenotype_predict = pd.read_csv(phenotype_predict_filepath, index_col='PATIENTID',
                                 sep='\t', skipinitialspace=True)

    # Focus on the WHO Grade and Cancer Type.  For now neglect the effects on SEX and RACE.
    data_grading_cancer_type = phenotype_predict.loc[:, ['WHO_GRADING', 'CANCER_TYPE']]

    # Cancer type has the following types:
    # ASTROCYTOMA, GBM, UNCLASSIFIED, OLIGODENDROGLIOMA, UNKNOWN, MIXED
    # So far there is no NaN.

    # First, replace NaN (if any) in Cancer Type with 'Unknown'.
    fillna_values = {'CANCER_TYPE': 'UNKNOWN'}
    data_grading_cancer_type.fillna(value=fillna_values, inplace=True)

    # The Each NaN value in 'WHO_GRADING" should be replaced by the mode
    # WHO_GRADING type of its associated cancer type.
    data_grading_cancer_type = fillna_for_WHO_grading(data_grading_cancer_type)

    # Reindex data_grading_cancer_type such that it aligns with the index in rna_dna_CN_predict
    data_grading_cancer_type = data_grading_cancer_type.reindex(rna_dna_CN_predict.index)
    test_data = pd.concat([rna_dna_CN_predict, data_grading_cancer_type], axis=1)

    test_labels_pred = clf_pipeline.predict(test_data)

    # The model was trained with 0 = deceased and 1 = alive.  We flip the logic before
    # reporting
    test_labels_pred = pd.DataFrame(test_labels_pred).replace({0:1, 1:0})
    test_labels_pred.index = test_data.index
    test_labels_pred.index.name = 'PATIENTID'
    test_labels_pred.columns = ['SURVIVAL_STATUS']

    filepath = os.path.join(PROJECT_ROOT_DIR, "outputs", 'subchallenge_3.tsv')
    test_labels_pred.to_csv(filepath, sep='\t', index=True)

    pd.set_option('display.max_rows', None)
    print(test_labels_pred)
    pd.reset_option('display.max_rows')

##############################################################################
# main program
def main(_):
    global rnd_seed
    global model_flags
    global bPlot_blocking
    global bAnalyze_Phenotype
    global bBagging
    global bSmote
    global test_set_ratio
    global smote_ratio
    global default_trained_model_filename
    global bPreFilter
    global default_rna_prefilter_file
    global rna_prefilter_filepath
    global default_dna_CN_prefilter_file
    global dna_CN_prefilter_filepath

    # Print Python version
    print("Python version: {}".format(platform.python_version()))
    print("Scikit-learn version: {}".format(sklearn.__version__))
    print("Numpy version: {}".format(np.__version__))
    print("Pandas version: {}".format(pd.__version__))

    command = [os.path.basename(sys.argv[0])]
    command += sys.argv[1:]
    command_str = " ".join(command)
    print()
    print("Command: \"%s\"" % command_str)

    # Print current date and time
    now = datetime.datetime.now()
    print("Execution date/time: %s" % str(now))
    print()

    # For measuring the run time when running this script non-interactively
    start_time = time.time()

    if FLAGS.model == "log_reg":
        model_flags = FLAG_LOG_REG
    elif FLAGS.model == "elastic":
        model_flags = FLAG_ELASTIC
    elif FLAGS.model == "lin_SVM":
        model_flags = FLAG_LINEAR_SVM
    elif FLAGS.model == "rbf_SVM":
        model_flags = FLAG_GAUSSIAN_RBF_SVM
    elif FLAGS.model == "RF":
        model_flags = FLAG_RF
    elif FLAGS.model == "xgboost":
        model_flags = FLAG_XGBOOST
    elif FLAGS.model == "classifiers_only":
        model_flags = FLAG_ALL_CLASSIFIERS
    elif FLAGS.model == "soft_voting":
        model_flags = (FLAG_LOG_REG + FLAG_LINEAR_SVM + FLAG_RF + FLAG_XGBOOST + FLAG_SOFT_VOTING)
    elif FLAGS.model == "all":
        model_flags = FLAG_ALL

    bPlot_blocking = FLAGS.block
    bAnalyze_Phenotype = bool(FLAGS.phenotype)
    bBagging = bool(FLAGS.bagging)
    bSmote = bool(FLAGS.smote)
    bPredictMode = FLAGS.predict
    bPrefilter = FLAGS.prefilter

    if bPrefilter:
        rna_prefilter_filepath = data_filepath(default_rna_prefilter_file)
        dna_CN_prefilter_filepath = data_filepath(default_dna_CN_prefilter_file)

    if bPredictMode:
        if FLAGS.rna_dna_CN_predict is None:
            print("Must specify RNA DNA CN filename for making a prediction.")
            return

        if FLAGS.pheno_predict is None:
            print("Must specify phenotype filename for making a prediction.")
            return

        rna_dna_CN_predict_filepath = FLAGS.rna_dna_CN_predict.strip('\"')
        phenotype_predict_filepath = FLAGS.pheno_predict.strip('\"')

        if FLAGS.model == "log_reg":
            trained_model_filepath = data_filepath('p1sc3_log_reg_clf.joblib')
        elif FLAGS.model == "elastic":
            trained_model_filepath = data_filepath('p1sc3_elastic_clf.joblib')
        elif FLAGS.model == "lin_SVM":
            trained_model_filepath = data_filepath('p1sc3_linSVM_clf.joblib')
        elif FLAGS.model == "rbf_SVM":
            trained_model_filepath = data_filepath('p1sc3_rbfSVM_clf.joblib')
        elif FLAGS.model == "RF":
            trained_model_filepath = data_filepath('p1sc3_rf_clf.joblib')
        elif FLAGS.model == "xgboost":
            trained_model_filepath = data_filepath('p1sc3_xgboost_clf.joblib')
        elif FLAGS.model == "soft_voting":
            trained_model_filepath = data_filepath('p1sc3_sv_clf.joblib')
        else:
            trained_model_filepath = data_filepath(default_trained_model_filename)

        if not os.path.isfile(trained_model_filepath):
            print("Classifier model %s is missing. Please train model first." % os.path.basename(trained_model_filepath))
            return

        predict_based_on_model(rna_dna_CN_predict_filepath, phenotype_predict_filepath, trained_model_filepath)
        return

    # ******************** The code below is for training a model(s) ***********************

    # Set the seed of the random generator for results to be reproducible
    # May need to skip this later on to test the robustness of the algorithms.
    print("************************************************************************************")
    print("numpy.random rnd_seed=%d test_set_ratio=%.2f SMOTE ratio=%0.2f" %
          (rnd_seed, test_set_ratio, smote_ratio))
    print("************************************************************************************")
    np.random.seed(rnd_seed)

    data_rna_dna_CN, data_phenotype, data_survival = load_input_data()
    if (data_rna_dna_CN is None) or (data_phenotype is None) or (data_survival is None):
    # Quit the program
        return

    # Theoretically for reproducibility, we do not need to set the random_state
    # of each classifier once we have set up np.random.seed().  However, it is
    # noticed that reproducibility cannot be guaranteed unless we set the
    # random_state of each classifier.
    model_RNA_DNA_CN_and_phenotype_vs_survival_data(data_rna_dna_CN, data_phenotype, data_survival, bAnalyze_Phenotype)

    stop_time = time.time()
    print("Run time was %.2fs." % (stop_time - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-block',
        help='Enable a plot to block the running process until the plot is closed.',
        action='store_true'
    )
    parser.add_argument(
        '-phenotype',
        type=int,
        default=0,
        choices=[0, 1],
        help='Specify whether to include the phenotype data in the modeling. (0: Do not include (default); 1: include).'
    )
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        choices=['log_reg', 'elastic', 'lin_SVM', 'rbf_SVM', 'RF', 'xgboost', 'classifiers_only', 'soft_voting', 'all'],
        default='soft_voting',
        help='Specify a specific classifier model. Default "soft_voting".'
    )
    parser.add_argument(
        '-prefilter',
        type=int,
        default=1,
        choices=[0, 1],
        help='Specify to include the attribute prefilters from P1SC1 and P1SC2 (0: No; 1: Yes (default)).'
    )
    parser.add_argument(
        '-bagging',
        type=int,
        default=1,
        choices=[0, 1],
        help='Specify whether bagging should be applied to each classifier (0: No; 1: Yes (default)).'
    )
    parser.add_argument(
        '-smote',
        type=int,
        default=1,
        choices=[0,1],
        help='Specify whether SMOTE should be used to increase minority sample instances (0: No; 1: Yes (default)).'
    )
    parser.add_argument(
        '-predict',
        help='Enable prediction based on a trained model.',
        action='store_true'
    )
    parser.add_argument(
        '-rna_dna_CN_predict',
        type=str,
        help='Specify the input RNA DNA CN filename for generating a prediction.'
    )
    parser.add_argument(
        '-pheno_predict',
        type=str,
        help='Specify the phenotype filename for generating a prediction.'
    )
    #All known agruments go to FLAGS, which is the namespace. The rest go to unparsed.
    FLAGS, unparsed = parser.parse_known_args()
    main(sys.argv)
