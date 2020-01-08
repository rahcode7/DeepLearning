#!/usr/bin/env python
# coding: utf-8

# In[7]:


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
# Save model
import pickle
import joblib

 
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from sklearn import metrics
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score,f1_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

import lightgbm as lgb
import pylev

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

#text1 = ["Dining Room, Pre-War, Laundry in Building, Dishwasher, Hardwood Floors, Dogs Allowed, Cats Allowed"]
#clean_features(text1)


# Check contact shared 
contact_words = ['email','phone','contact','mail','@gmail.com','@yahoo.com']

def contact(str1):
    str1 = str1.lower()
    result = any(substring in str1 for substring in contact_words)
    if(result==True):
        return(1)
    else:
        return(0)
    
# Check highlighters used 
highlight_words = ['..','*','--']

def highlight(str1):
    str1 = str1.lower()
    result = any(substring in str1 for substring in highlight_words)
    if(result==True):
        return(1)
    else:
        return(0)


def feature_engineering(df):
    df['street_display_dist'] = df.apply(lambda x : pylev.levenshtein(x.street_address,x.display_address),axis=1)
    
    df['desc_upper_num'] = df['description'].str.findall(r'[A-Z]').str.len()
    df['desc_uplow_ratio'] = df['description'].str.findall(r'[A-Z]').str.len()/df['description'].str.findall(r'[a-z]').str.len()
    df['desc_uplow_ratio'] = df['desc_uplow_ratio'].replace([np.inf, -np.inf], np.nan)
    df['desc_uplow_ratio'] = df['desc_uplow_ratio'].fillna(0.0)
    
    df['contact_flag'] =  df.apply(lambda x : contact(x.description),axis=1)
    df['highlight_flag'] =  df.apply(lambda x : contact(x.description),axis=1)

    df['highlight_flag'] = df['highlight_flag'].astype('category')
    df['contact_flag'] = df['contact_flag'].astype('category')

    df['weekend'] = df["created"].apply(lambda x : x.weekday())
    df['weekend_flag'] = np.where(df["weekend"]<5,0,1)


    return(df)


def prepare_train_test(X,y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)
    print(X_train.shape,X_val.shape,y_train.shape,y_val.shape)
    return(X_train, X_val, y_train, y_val)

def evaluate_metrics(y_val,y_pred,y_prob):
    #print("Accuracy:",metrics.accuracy_score(y_val, y_pred))
    print("Log Loss:",log_loss(y_val, y_prob))



#from sklearn.model_selection import GridSearchCV
# param_dist = {"max_depth": [25,50, 75],
#               "learning_rate" : [0.01,0.05,0.1],
#               "num_leaves": [300,900,1200],
#               "n_estimators": [200]
#              }


# grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv = 3, scoring="neg_log_loss", verbose=5)
# grid_search.fit(train,y_train)
# grid_search.best_estimator_

if __name__ == "__main__":

    RANDOM_SEED = 123
    params = {
        "boosting":"gbdt",
        "application": "multiclass",
        "max_depth": 50, 
        "learning_rate" : 0.01, 
        "num_leaves": 900,  
        "n_estimators": 300,
        "num_class":3,
        "is_unbalance" :"True",
        "num_threads":4,
        "seed":RANDOM_SEED,
        "metric":"multi_logloss",
        "min_data_per_group":50
         }

    df = pd.read_json(open("input/train.json", "r"))
    df = create_numeric_features(df)
    df = feature_engineering(df)
   
   
    num_feats = ["bedrooms", "latitude", "longitude", "price","num_photos", "num_features", "num_description_words", "created_month",
             "created_day","manager_id","street_display_dist","highlight_flag","contact_flag","desc_uplow_ratio","desc_upper_num"]

    
    X = df[num_feats]

    cate_features_name = ["manager_id","highlight_flag","contact_flag"]
    for c in cate_features_name:
        X[c] = X[c].astype('category')

    print(X.info())
    y = df["interest_level"]

    print(X.info())
    print(y.dtype)

    #X_train, X_val, y_train, y_val = prepare_train_test(X,y)
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(y)
    classes = le.classes_

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33) 


    y_val = pd.DataFrame(y_val) 
    y_val.columns = ["Class"]
    y_val.to_csv("y_val.csv",index=None)


    y_train_encoded = le.fit_transform(y_train)
    print(pd.Series(y_train_encoded).value_counts())


    y_val_encoded = le.transform(y_val)
    print(pd.Series(y_val_encoded).value_counts())


    d_train = lgb.Dataset(X_train,label=y_train_encoded)
    #,categorical_feature=cate_features_name

    lg = lgb.LGBMClassifier(silent=False)

    # ##### LightGBM With categorical features
    model = lgb.train(params, d_train,categorical_feature=cate_features_name)

    y_pred = model.predict(X_val)
    y_pred_proba = model.predict(X_val)
    print(y_pred_proba[:5])
    evaluate_metrics(y_val,y_pred,y_pred_proba)


    # Write Predictions
    pd.DataFrame(y_pred).to_csv('Y_predictions_LGBMcat.csv',index=None)
    y_prob = pd.DataFrame(y_pred_proba)
    y_prob.columns = classes
    y_prob.to_csv("Probabilities_LGBMcat_classifier.csv",index=None)
        
    # Save to file in the current working directory
    final_model = model
    pkl_filename = "/Users/rahulm/Desktop/LEARN/OLX_Shared/restapi/model_rental_lgbmcat.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(final_model, file)
    print("Model Dumped in RestAPI folder")
        
    model_columns = list(X_train.columns)
    joblib.dump(model_columns, '/Users/rahulm/Desktop/LEARN/OLX_Shared/restapi/model_rental_lgbmcat_columns.pkl')
    print("Model Columns Dumped in RestAPI folder")   

    # Write Submission file

    test = pd.read_json(open("input/test.json", "r"))
    print(test.shape)

    sub = pd.DataFrame()
    sub["listing_id"] = test["listing_id"]

    test = create_numeric_features(test)

    test = feature_engineering(test)

    cate_features_name = ["manager_id","highlight_flag","contact_flag"]
    for c in cate_features_name:
        test[c] = test[c].astype('category')

    # num_feats = ["bedrooms", "latitude", "longitude", "price","num_photos", "num_features", "num_description_words", "created_month",
    #             "created_day","street_display_dist","highlight_flag","contact_flag","desc_uplow_ratio","desc_upper_num"]

    test = test[num_feats]

    labels2idx = {label: i for i, label in enumerate(classes)}
    z = model.predict(test)

    
    for label in ["high","medium", "low"]:
        sub[label] = z[:, labels2idx[label]]
    sub.to_csv("submission_test_LGBMCat.csv", index=False)

