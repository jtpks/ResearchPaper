#!/usr/bin/env python

import pandas as pd
import multiprocessing
import json
import os
import sys
import argparse
import logging
import time
import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import BorderlineSMOTE

def main():
    try:
        paramFileContaints = open("test.json").read()
        params = json.loads(paramFileContaints)
        files = params["csvs"].split(", ")
        for model_id in ["Logistic", "RandomForest", "CrossValid"]:
            for file in files:
                do_pred(file, params["label"], params["test_size"], model_id)
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
    smote=BorderlineSMOTE()
    X, y = smote.fit_resample(df[features], df[label])
    return X, y

def get_train_test_data(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return X_train, y_train, X_test, y_test

def get_model(X_train, y_train, model_id):
    if model_id == "RandomForest":
        model = RandomForestClassifier(max_depth=5, n_estimators=100, max_features=1)
        model.fit(X_train, y_train)
        return model
    elif model_id == "Logistic":
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return model
    else:
        base_clf = GaussianNB()
        model = CalibratedClassifierCV(base_estimator=base_clf, cv=3)
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

def dump_charts(y_test, y_pred):
    #TODO: linear model data between test/pred
    #TODO: How to get graphical representation of inputs and linear
    #TODO: Regularization
    
def do_pred(csv, label, test_size, model_id):
    print("Model_id: %s Data file : %s, Label : %s" % (model_id, csv, label))
    df, features = get_df(csv, label)
    X, y = do_sample(df, features, label)
    X_train, y_train, X_test, y_test = get_train_test_data(X, y, test_size)
    print("Complete Data Num Rows X Features: ", X.shape[0], X.shape[1]-1)
    print("Training Data Num Rows X Features: ", X_train.shape[0], X_train.shape[1]-1)
    print("Test Data Num Rows X Features:     ", X_test.shape[0], X_test.shape[1]-1)

    model = get_model(X_train, y_train, model_id)
    y_pred = model.predict(X_test)
    dump_reports(y_test, y_pred)
    dump_charts(y_test, y_pred)

    print("=================================================================")


if __name__ == "__main__":
    main()

