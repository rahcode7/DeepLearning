import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_colwidth', -1)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "input"]).decode("utf8"))
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn import metrics
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score,f1_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import pylev

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


def label_encode(y):
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(y)
    return(y,list(le.classes_))


def prepare_train_test(X,y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)
    print(X_train.shape,X_val.shape,y_train.shape,y_val.shape)
    #y_train_encoded = le.fit_transform(y_train)
    #print(pd.Series(y_train_encoded).value_counts())
    #y_val_encoded = le.transform(y_val)
    #print(pd.Series(y_val_encoded).value_counts())
    return(X_train, X_val, y_train, y_val)


import lightgbm as lgb
params = {"boosting":"gbdt",
         "application": "multiclass",
         "max_depth": 50, 
         "learning_rate" : 0.01, 
         "num_leaves": 900,  
         "n_estimators": 300,
         "num_class":3,
         "metric":"multi_logloss",
         "is_unbalance" :"True",
         "min_data_per_group":50
         }
def classifier_LGBM(params,d_train):
    lgb.LGBMClassifier(silent=False)
    model = lgb.train(params,d_train)
    print(model)
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

    df = feature_engineering(df)
   
   
    y = df["interest_level"]
    y,classes = label_encode(y)


    num_feats = ["bedrooms", "latitude", "longitude", "price","num_photos", "num_features", "num_description_words", "created_month",
             "created_day","manager_id","street_display_dist","highlight_flag","contact_flag","desc_uplow_ratio","desc_upper_num"]
    X = df[num_feats]

    cate_features_name = ["manager_id","highlight_flag","contact_flag"]
    for c in cate_features_name:
        X[c] = X[c].astype('category')

    print(X.info())

    X_train, X_val, y_train, y_val = prepare_train_test(X,y)
    d_train = lgb.Dataset(X_train,label=y_train,categorical_feature=cate_features_name)


    y_pred,y_prob = classifier_LGBM(params,d_train)

    evaluate_metrics(y_val,y_pred,y_prob)

    y_prob = pd.DataFrame(y_prob)
    y_prob.columns = classes
    y_prob.to_csv("Predictions_classifier.csv",index=None)




