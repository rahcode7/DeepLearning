## Deep Learning Models for Obfuscated text classification

### Dataset
The data contain 32513 training text documents 
with each document a sequence of encrypted text only
We have 12 class labels and have to train the model to classify these classes, so therefore its a multiclass classification problem

Dataset URL - https://www.kaggle.com/alaeddineayadi/obfuscated-multiclassification


### Text Preprocessing

The machine learning models understand text in a vector form    
1. First we create a dictionary, Keras has a tokenize class to count the unique words in our vocabulary     
     as it's English in our case it results in vocabulary size of 26  
2. Word Index Look Up using fit_on_texts() on our vocabulary for each document of our training dataset    
3. Numeric Vector : We prepared the training data by calling text_to_sentences method to convert the each doc into a numeric vector for both training and test dataset  
4. Zero Padding : To standardise the size of all inputs,we pad the documents with 0 to make them fix length size of 453 (i.e maximum length among documents)  

Training Output    
1. We have 12 Categorical classes in range of 0 to 11   
   We One-hot encode the classes to make it usable as inputs for neural network  


### Neural Network Architectures

We use the following layers to define our neural network in sequential order

#### 1.Embedding Layer
We define embedding layer, which learns to map word vectors into a lower dimensional vector space
In our case, each character of the each document is represented in 15 dimensional space

#### 2.Batch Normalization Layer
Added between linear and non linear layer and normalizes the input to our activation function

#### 3.Convolutional Layer
We add 1Dimensional convolutional layer to understand temporal representations with kernel size of 5
which kind of understand 5-grams in a sentence

64 filters of size (5,5) scans the inputs with padding of 0 and stride of 1

#### 4.MaxPool Layer
A 1D max-pool layer of size (3,3) which extracts the most significant elements in each convolution 

#### 5.Convolutional Layer  
32 filters of size (5,5) scans the inputs with padding of 0 and stride of 1 to understand the abstract representations

#### 6.MaxPool Layer
A 1D max-pool layer of size (3,3) which extracts the most significant elements in each convolution 

#### 7.DropOut   
Dropout acts a a regularization, because by randomly disabling a fraction of neurons in the layer (set to 50% here) 
it ensure that that model does not overfit. 
This prevents neurons from adapting together and forces them to learn individually useful features.


#### 8.LSTM Layer 
LSTM layer acts as a decode to decode the abstract representations made by the convolutional layers

#### 9.Dense layer  
A fully connected layer, where each neuron will be fully connected to the next layer
Usually in the dimension of power of 2, We use 500 in our network

#### 10.Batch Normalisation 


#### 11.SoftMax Activation Layer
To normalize the results into probability scores, and all values will add up to 1


** We use Cross Entropy as a loss function because each text will output to one value **  
** Also,we use Adam as a regularize because of its computational efficiency and ease of convergence **


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Results 


** We ran the model for 200 Epochs,with batch_size of 128 **  
** After 100 Epoch, the model seems to starts overfitting, coz the Training accuracy keeps on increasing from 80 to 85%, **
** but the test accuracy stops improving at 73% and also the test loss stops decreasing at 0.6 **

** Output File - ytext.txt   
** It contains predictions of novel classes for 3000 test documents

### Accuracy
Accuracy on 20% test set vs Epoch 

![picture](https://bitbucket.org/datageek008/obfuscated_textclassification/downloads/ACCURACY_v14.png)



Loss vs Epoch  
![picture](https://bitbucket.org/datageek008/obfuscated_textclassification/downloads/LOSS_V14.png)

