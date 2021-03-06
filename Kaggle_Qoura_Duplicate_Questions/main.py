# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gJY0w43QQA_IH6R15KlzGhg90LC6OY3K
"""

from data_loader.data_loader import data_loader
from models.QuoraModel import QuoraModel
from models.Model_Trainer import Model_Trainer
from models.preprocessing import preprocessing
from keras.models import load_model
from keras import backend as K


from keras import backend as K
# cfg = K.tf.ConfigProto()
# cfg.gpu_options.allow_growth = True
# K.set_session(K.tf.Session(config=cfg))
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

def main():

  # Load training data
  # dr = data_loader()
  # df = dr.get_train_data()
  # data = data_loader()
  # train = data.get_train_data()
  # test =  data.get_test_data()
  #
  # #print("Data loaded",(train.shape,test.shape))
  # print("Data loaded - Training ",train.shape)
  # print("Data loaded - Test ",test.shape)
  #
  #p = preprocessing(train,test)
  #print(p)
  #p.encode()
  # training_X,train_y,val_X_encoded,val_y, test_X_encoded = p.encode()
  #
  #
  # print("Data Preprocessed, Showing Validation dataset",(val_X_encoded.shape,val_y.shape))

  # Define model
  model = QuoraModel()


  print(model)
  print("Model defined")

  # Train the model
  trainer = Model_Trainer()
  trained_model = trainer.train(nn_model)
  print("Model Trained and ready to predict")

  #model.save('../trained_models/model_quora3.h5')

  # Load the trained model
  trained_model = load_model(os.path.join(os.getcwd(),'model/trained_models/model_quora10.h5'))
  val_Ypred = trained_model.predict([val_X_encoded], batch_size=1024, verbose=1)
  test_Ypred = trained_model.predict([val_X_encoded], batch_size=1024, verbose=1)


  # Write output files
  with open(os.path.join(os.getcwd(),'predictions/test_scores.txt', 'w+')) as f:
    for i in range(len(test_Ypred)):
      label = test_Ypred[i][1]
      id = test['id'].iloc[[i]].values[0]
      f.write("%s %s\n"%(id,label))
      i+=1
      print(i)

  with open(os.path.join(os.getcwd(),'predictions/val_scores.txt', 'w+'))  as f:
    for i in range(len(val_Ypred)):
      label = val_Ypred[i][1]
      id = test['id'].iloc[[i]].values[0]
      f.write("%s %s\n"%(id,label))
      i+=1
      print(i)

  if __name__ == '__main__':
    main()
K.clear_session()
