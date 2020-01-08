import pandas as pd 
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score,f1_score

## Calculate metrics
def evaluate_metrics(y_val,y_pred,y_prob):
    #print("Accuracy:",metrics.accuracy_score(y_val, y_pred))
    print("Log Loss:",log_loss(y_val, y_prob))


## Read all predictions files
model1 = pd.read_csv('Probabilities_text_classifier.csv',header="infer")
model2 = pd.read_csv('Probabilities_LGBMcat_classifier.csv',header="infer")
print("Model1",model1.shape) 
print("Model2",model2.shape) 


y_val = pd.read_csv('Y_predictions_LGBMcat.csv')
print(y_val.shape)

## Generate average results 
avg_model = pd.DataFrame(columns = model1.columns)
avg_model["high"] = (model1["high"]+model2["high"]) / 2
avg_model["medium"] = (model1["medium"]+model2["medium"]) / 2
avg_model["low"] =(model1["low"]+model2["low"]) / 2

evaluate_metrics(y_val,None,avg_model)


