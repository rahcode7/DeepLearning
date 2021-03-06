# -*- coding: utf-8 -*-
"""models.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CkdNMKYMfQCsSynfUYcUQBfJrutVb8Wi
"""

#### Import libraries
from keras.layers import Conv1D
from keras.optimizers import Adam
import keras
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer
from keras.layers import Embedding,BatchNormalization,LSTM,Dense,Activation,Dropout
import pandas as pd
from keras.preprocessing.sequence import pad_sequences


### Model Parameters
vocab_size = 10000     # Vocabulary size
input_seqlen = 100     # Length of the input sentence
embedding_size = 300   # Size of each embedded word vector


### Model Class
class QuoraModel():

  def __init__(self):
    self.NN_model()

  def NN_model(self):
    self.model = Sequential()
    self.model.add(Embedding(vocab_size, embedding_size, input_length=input_seqlen))

    self.model.add(Conv1D(32,
                     3,
                     padding='valid',
                     activation='relu',
                     strides=1))

    #self.model.add(Dropout(0.5))
    self.model.add(BatchNormalization())

    self.model.add(LSTM(100))

    self.model.add(Dense(500))
    self.model.add(BatchNormalization())

    self.model.add(Dense(2))
    self.model.add(BatchNormalization())

    self.model.add(Activation('softmax'))

    #Compile Model
    self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

# model = QuoraModel()
# print(model.NN_model)
