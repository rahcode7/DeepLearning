#!/usr/bin/env python
# coding: utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_colwidth', -1)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "input"]).decode("utf8"))
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score,f1_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


# Create numeric features
def create_numeric_features(df):
    df["num_photos"] = df["photos"].apply(len)
    df["num_features"] = df["features"].apply(len)
    df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
    df["created"] = pd.to_datetime(df["created"])
    df["created_year"] = df["created"].dt.year
    df["created_month"] = df["created"].dt.month
    df["created_day"] = df["created"].dt.day
    return(df)


# All lowercase,Remove Spaces,Concatenate, Remove commas
def clean_features(text):
    text = [x.lower() for x in text]
    text = [x.replace(' ','') for x in text]
    text = ' '.join(text)
    return(text)

# Vectorize text dataset
# vectorizer = CountVectorizer()
# vectorizer.fit(df['features_cleaned'])
# print(vectorizer.vocabulary_)


def label_encode(y):
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(y)
    return(y,list(le.classes_))


from sklearn.feature_extraction.text import TfidfVectorizer
def vectorize(X_train,X_val):
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    #print(vectorizer.get_feature_names())
    return(X_train,X_val)


def prepare_train_test_(X,y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)
    #y_train_encoded = le.fit_transform(y_train)
    #print(pd.Series(y_train_encoded).value_counts())
    #y_val_encoded = le.transform(y_val)
    #print(pd.Series(y_val_encoded).value_counts())
    return(X_train, X_val, y_train, y_val)

from sklearn.naive_bayes import GaussianNB
def classifier_NB(X_train,X_val,y_train):
    model = GaussianNB()
    model.fit(X_train.toarray(),y_train)
    y_pred = model.predict(X_val.toarray())
    #pd.Series(y_pred).value_counts()
    return(y_pred)


import lightgbm as lgb
def classifier_LGBM(X_train,X_val,y_train):
    model = lgb.LGBMClassifier(objective='multiclass', verbose=-1, learning_rate=0.5, max_depth=20, num_leaves=50, n_estimators=120, max_bin=2000,)
    model.fit(X_train.toarray(), y_train, verbose=-1)
    y_pred = model.predict(X_val.toarray())
    y_prob = model.predict_proba(X_val.toarray())
    return(y_pred,y_prob)

def evaluate_metrics(y_val,y_pred,y_prob):
    print("Accuracy:",metrics.accuracy_score(y_val, y_pred))
    print("Log Loss:",log_loss(y_val, y_prob))


if __name__ == "__main__":
    df = pd.read_json(open("input/train.json", "r"))
    test = pd.read_json(open("input/test.json", "r"))


    df = create_numeric_features(df)

    df['features_cleaned'] = df['features'].apply(lambda x : clean_features(x))
    df['features_string']  = df['features_cleaned'].apply(lambda x: [str(i) for i in x.split(' ')])

    #df['features_vectorised'] = df['features_cleaned'].apply(lambda x : list(*vectorizer.transform([x]).toarray()))
    y = df["interest_level"]
    X = df["features_cleaned"]

    y,classes = label_encode(y)
    #print(y)

    X_train, X_val, y_train, y_val = prepare_train_test_(X,y)

    X_train, X_val = vectorize(X_train,X_val)
    print(X_train.shape,X_val.shape)

    y_pred,y_prob = classifier_LGBM(X_train, X_val, y_train)

    evaluate_metrics(y_val,y_pred,y_prob)


    y_prob = pd.DataFrame(y_prob)
    y_prob.columns = classes
    y_prob.to_csv("Predictions_text_classifier.csv",index=None)


    ## Save model predictions on test dataset   
    ## Save model predictions on test.csv dataset 
