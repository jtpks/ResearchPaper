#!/usr/bin/env python

import statistics
import pandas as pd
import multiprocessing
import json
import os
import sys
import argparse
import logging
import time
import datetime
import matplotlib.pyplot as plt
import numpy

from sklearn.neural_network import MLPClassifier
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2
from numpy import where

plot_final = 0
plot_sample_data = 0
sample = 0
debug = 0
normalize = 0
weights = 0

def main():
    try:
        paramFileContaints = open("test.json").read()
        params = json.loads(paramFileContaints)
        files = params["csvs"].split(", ")
        sample = params["sample"]
        plot_sample_data = params["plot_sample_data"]
        plot_final = params["plot_final"]
        debug = params["debug"]
        weights = params["weights"]
        normalize = params["normalize"]
        for model_id in ["ExtraTree", "DecisionTree", "Logistic_L1", "Logistic_L2", "RandomForest", "mlp"]:
            for file in files:
                do_pred(file, params["label"], params["test_size"], model_id)
                if debug == 1:
                    break
            if debug == 1:
                break
    except Exception as e:
        logging.error("ERROR: Correct the error in json file %s, %s", paramFile, str(e))

def clean_data(df):
    #df.drop_duplicates(keep='first', inplace=True)
    return df

def get_df(csv, label):
    df = pd.read_csv(csv) 
    df = clean_data(df)
    col_names = df.columns[1:].values.tolist()
    features = [i for i in col_names if i not in label]
    return df, features

def do_sample(df, features, label):
    # TODO: Evaluate other SMOTE sample algorithms
    counter = Counter(df[label])
    print("Before sample ", counter)
    plot_data(df[features], df[label], label, counter)
    if sample == 1:
        #smote=SMOTE()
        smote=BorderlineSMOTE()
        X, y = smote.fit_resample(df[features], df[label])
    else:
        X = df[features]
        y = df[label]
    counter = Counter(y)
    print("After sample ", counter)
    plot_data(X, y, label, counter)
    return X, y

def plot_data(X, y, label, counter):
    if plot_sample_data == 0:
        return
    for label, _ in counter.items():
        row_ix = where(y == label)[0]
        print(row_ix)
        plt.scatter(X.iloc[row_ix, 0], X.iloc[row_ix, 1], label=str(label))
    plt.legend()
    plt.show()

def get_top_features(X, y):
    return
    #top = len(X.columns)
    top = 3
    best = SelectKBest(chi2, k=top)
    best.fit_transform(X, y)
    print(X.columns)
    print(best.get_params())
    print(best.get_support())
    print(best.get_feature_names_out())

def get_train_test_data(X, y, test_size):
    get_top_features(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return X_train, y_train, X_test, y_test

def print_feature_weights(model, model_id):
    if weights == 0:
        return
    if model_id == "RandomForest":
        importance = model.feature_importances_
        for i,v in enumerate(importance):
	        print('Feature: %0d, Score: %.5f' % (i,v))
    elif model_id == "ExtraTree":
        importance = model.feature_importances_
        for i,v in enumerate(importance):
	        print('Feature: %0d, Score: %.5f' % (i,v))
    elif model_id == "DecisionTree":
        importance = model.feature_importances_
        for i,v in enumerate(importance):
	        print('Feature: %0d, Score: %.5f' % (i,v))
    elif model_id == "Logistic_L2":
        importance = model.coef_[0]
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
    elif model_id == "Logistic_L2_Sag":
        importance = model.coef_[0]
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
    elif model_id == "mlp":
        print("Not implemented yet!")
    else:
        importance = model.coef_[0]
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))

def get_model(X_train, y_train, model_id):
    if model_id == "RandomForest":
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        return model
    elif model_id == "ExtraTree":
        model = ExtraTreesClassifier()
        model.fit(X_train, y_train)
        return model
    elif model_id == "DecisionTree":
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        return model
    elif model_id == "Logistic_L2":
        model = LogisticRegression(solver='liblinear')
        model.fit(X_train, y_train)
        return model
    elif model_id == "Logistic_L2_Sag":
        model = LogisticRegression(solver='sag')
        model.fit(X_train, y_train)
        return model
    elif model_id == "mlp":
        model = MLPClassifier(random_state=1, max_iter=1000)
        model.fit(X_train, y_train)
        return model
    else:
        model = LogisticRegression(solver='liblinear', penalty='l1')
        model.fit(X_train, y_train)
        return model

def dump_reports(y_test, y_pred):
    print("y_test: ", y_test.to_numpy())
    print("y_pred: ", y_pred)
    #print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred, zero_division=0))
    #precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='micro', labels=[0], zero_division=0)
    #print("0- prevision, recall, f1: %.2f %.2f %.2f" % (precision, recall, f1))
    #precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='micro', labels=[1], zero_division=0)
    #print("1- prevision, recall, f1: %.2f %.2f %.2f" % (precision, recall, f1))

def get_feature_to_label_corr(X_test, y_test, y_pred):
    dfx = pd.DataFrame(X_test.to_numpy())
    df = pd.DataFrame(y_test.to_numpy())
    df2 = pd.DataFrame(y_pred)
    R = df.corrwith(df2)[0]
    series = pd.Series(y_pred)
    print("R^2 = Correlation y_test to y_pred %.2f" % (R * R))

    y_test_var = statistics.variance(y_test)
    y_pred_var = statistics.variance(series)

    if (y_test_var != 0):
        print("F-stat: {:.2f}".format(y_pred_var / y_test_var))
    else :
        print("F-stat: NA")
    plot_actual_vs_final(y_test, y_pred, X_test)

def plot_actual_vs_final(y_test, y_pred, X_test):
    if plot_final == 0:
        return
    plt.plot(X_test.iloc[:, 0], y_pred, label = "predicted", marker='o')
    plt.plot(X_test.iloc[:, 0], y_test, label = "actual", marker='x')
    plt.xlabel('Q1')
    plt.ylabel('DND')
    plt.title('Q1 vs DND')
    plt.legend()
    plt.show()

def histogram_intersection(a, b):
    v = numpy.minimum(a, b).sum().round(decimals=1)
    return v

def print_feature_corr(df):
    if (debug):
        print(df.corr(method=histogram_intersection))

def normalize(df):
    num_columns = len(df.columns)
    s0 = df.iloc[:,1:(num_columns-1)]
    df.iloc[:,1:(num_columns-1)] = (s0 - s0.min()) / (s0.max() - s0.min())
    return df

def do_pred(csv, label, test_size, model_id):
    print("Model_id: %s Data file : %s, Label : %s" % (model_id, csv, label))
    df, features = get_df(csv, label)
    df = normalize(df)
    print_feature_corr(df)
    X, y = do_sample(df, features, label)
    X_train, y_train, X_test, y_test = get_train_test_data(X, y, test_size)
    print("Complete Data Num Rows X Features: ", X.shape[0], X.shape[1]-1)
    print("Training Data Num Rows X Features: ", X_train.shape[0], X_train.shape[1]-1)
    print("Test Data Num Rows X Features:     ", X_test.shape[0], X_test.shape[1]-1)

    model = get_model(X_train, y_train, model_id)
    print_feature_weights(model, model_id)
    y_pred = model.predict(X_test)
    dump_reports(y_test, y_pred)
    #get_feature_to_label_corr(X_test, y_test, y_pred)

    #y_pred = model.predict(X)
    #dump_reports(y, y_pred)
    #get_feature_to_label_corr(X, y, y_pred)

    print("=================================================================")


if __name__ == "__main__":
    main()

