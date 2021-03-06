# -*- coding: utf-8 -*-
"""data_loader.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10jwNbYJKOW-G2IJeFAeT-qxPNhnbNWUQ
"""

import pandas as pd
import os

#data_path = '../datasets'
#data_path = 'drive/My Drive/datasets/iamplus/'
data_path = '/Users/rahulm/Desktop/LEARN/Assignments/Iamplus/project/datasets'

class data_loader():
  # Define datasets
  def __init__(self):
    self.train = pd.read_csv(os.path.join(data_path,'train.csv'),dtype='unicode')
    #self.test  = pd.read_csv(os.path.join(data_path,'test.csv'),dtype='unicode')
    self.test  = pd.read_csv(os.path.join(data_path,'test10000.csv'),dtype='unicode') # Load small if less memory


  def get_train_data(self):
    return(self.train)

  def get_test_data(self):
    return(self.test)

# data = data_loader()
# train = data.get_train_data()
# test =  data.get_test_data()
#print("Data loaded",(train.shape,test.shape))
