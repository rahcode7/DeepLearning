from  flask import request, Flask,jsonify
import pickle
import pandas as pd
import joblib
import pylev
import numpy as np


# Load Model Pickle File
def load_model():
    global model
    global model_columns

    model = joblib.load("model_rental_lgbmcat.pkl")
    print("Loading model columns")
    model_columns = joblib.load("model_rental_lgbmcat_columns.pkl")

app = Flask(__name__)
model = None

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

def feature_engg(df):
    print("Feature Engineering started")
    df["num_photos"] = df["photos"].apply(len)
    df["num_features"] = df["features"].apply(len)
    df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
    df["created"] = pd.to_datetime(df["created"])
    df["created_year"] = df["created"].dt.year
    df["created_month"] = df["created"].dt.month
    df["created_day"] = df["created"].dt.day


    df['street_display_dist'] = df.apply(lambda x : pylev.levenshtein(x.street_address,x.display_address),axis=1)
    df['contact_flag'] =  df.apply(lambda x : contact(x.description),axis=1)
    df['highlight_flag'] =  df.apply(lambda x : highlight(x.description),axis=1)
    df['desc_upper_num'] = df['description'].str.findall(r'[A-Z]').str.len()
    df['desc_uplow_ratio'] = df['description'].str.findall(r'[A-Z]').str.len()/df['description'].str.findall(r'[a-z]').str.len()
    df['desc_uplow_ratio'] = df['desc_uplow_ratio'].replace([np.inf, -np.inf], np.nan)
    df['desc_uplow_ratio'] = df['desc_uplow_ratio'].fillna(0.0)


    df['highlight_flag'] = df['highlight_flag'].astype('category')
    df['manager_id'] = df['manager_id'].astype('category')
    df['contact_flag'] = df['contact_flag'].astype('category')
    print("Feature engg done")
    return(df)

# Predict Input data
@app.route("/predict", methods=["POST"])
def predict():
    #data = {"success":False}

    json_ = request.json


    df = pd.DataFrame(json_)
    print("Before subset {}".format(df.columns.shape[0]))
    print(df)

    #Step 1.Feature engineering
    df = feature_engg(df)
    print(df)

    #df.shape
    #
    # # Step 2.Subset input features
    num_feats = ["bedrooms", "latitude", "longitude", "price","num_photos", "num_features", "num_description_words", "created_month",
             "created_day","manager_id","street_display_dist","highlight_flag","contact_flag","desc_uplow_ratio","desc_upper_num"]

    df = df[num_feats]
    print(df.info())
    # #query = df.fillna(0)
    #
    # #query = query.reindex(columns=model_columns, fill_value=0)
    # #print("After Reindexing {}".format(query.columns.shape[0]))
    #
    prediction = model.predict(df)
    print(prediction)

    return({'prediction': str(prediction)})

if __name__  == '__main__':
    print("Loading model")
    load_model()
    app.run(port=5000)
