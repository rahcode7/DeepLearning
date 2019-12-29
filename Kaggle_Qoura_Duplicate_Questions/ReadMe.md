## Problem - Quora Question Pair Duplicates

The task is to identify duplicate question pairs 

Reference - https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs

## Dataset    
    Training Set - 404290
    We split the dataset into 90/10 for training and validation set

    Test Set  - 3563475
    The output prediction file should contain 3563475 entries with probability of duplicates


## Solution - Neural Networks
Because we have sequences of sentences, and we have to predict a binary class of duplicate or not.
We can use neural networks particularly LSTM,CNN to understand the underlying structure of these sequence 
and help predict the dupicates


## Text preprocessing
Text Preprocessing
The neural network models understand text in a vector form 

1. First we create a dictionary, Keras has a tokenize class to count the unique words in our vocabulary  
    Vocab size = 10000 

2. Word Index Look Up using fit_on_texts() on our vocabulary for each document of our training dataset 

3. Numeric Vector : We prepared the training data by calling text_to_sentences method to convert the each doc into a numeric vector for both training and test dataset

4. Zero Padding : To standardise the size of all inputs,we pad the documents with 0 to make them fixed length.
    Max Length of Encoded vector = 100



## Neural Network Architecture
We use the following layers to define our neural network in sequential order

##### 1.Embedding Layer
We define embedding layer, which learns to map word vectors into a lower dimensional vector space
In our case, each character of the each document is represented in 100 dimensional space

##### 2.Convolutional Layer
We add 1Dimensional convolutional layer to understand temporal representations with kernel size of 3
which kind of understand 3-grams in a sentence
32 filters of size (3,3) scans the inputs with padding of 0 and stride of 1

##### 3.MaxPool Layer
A 1D max-pool layer of size (3,3) which extracts the most significant elements in each convolution 

##### 4.Batch Normalization Layer
Added between linear and non linear layer and normalizes the input to our activation function

##### 5.LSTM Layer 
LSTM layer acts as a decoder to decode the abstract representations made by the convolutional layers

##### 6.Dense layer  
A fully connected layer, where each neuron will be fully connected to the next layer
Usually in the dimension of power of 2, We use 500 in our network

##### 7.Batch Normalisation 
##### 8.Dense layer  
##### 9.Batch Normalisation 

##### 10.SoftMax Activation Layer
To normalize the results into probability scores, and all values will add up to 1


** We use Sparse Cross Entropy as a loss function because of only two integer outputs 0 and 1 **  
** Also,we use Adam as a regularizer because of its computational efficiency and ease of convergence **

--------------------------------------------------------------------------------------------------------------------------------------

## Files Structure

```
├── main.py             -  Main function to call other methods 
│
│
├── data_loader      
│   ├── data_loader.py   - Loads the training and test csv datasets from the datasets folder
│   
│
│
├── models                 - this folder contains the models files
│   └── preprocessing.py   - Preprocess the train,test datasets
|   |                        1. Generate training and validation dataset from training data split of 90/10
|   |                        2. Encodes text to numerical vector and adds padding and makes fixed length vectors 
|   |                         of size 100 for each sentence of train,test,val
|   |
|   |--- QuoraModel.py     -  Defines the Neural Network Architecture with 
|   |                          paramters - vocab_size = 10000,sequence length = 100,Embedding size-300
|   |                       -  Compiles the model with cross entropy loss and accuracy as metric│
│   |
|   |---- Model_Trainer.py - Runs the model with checkpoint for 10 epochs and batch size of 512,captures validation loss and training loss and accuracy for both                
│   |-----trained_models    - Contains trained model(with trained weighst and model architecture)  as model_quora10.h5 file, 
                              used for generating predictions on the fly
│
├── datasets                -  Contains train.csv,test.csv
│
│
└── predictions             - Contains predictions files 
                            - 1) validation_scores.txt ( for model training purpose
                            - 2) submission.txt containing 3563475 entries
```


## Rune
( Requires - Keras,Tensorflow,Python-3) 


```
1. Download this repo
2.Download data from https://www.kaggle.com/c/quora-question-pairs and place in datasets folder
3. open terminal and go to this repo and run main.py
    python main.py
```

----------------------------------------------------------------------------------------------------------------------------------------------

## Accuracy & Results 
** We ran the model for 10 Epochs,with batch_size of 512 which results in 78% accuracy in the validation dataset(10% of training dataset)**  

##### Accuracy
![picture](https://bitbucket.org/datageek008/quora_duplicate_questions/downloads/accuracy.png)

##### Loss
![picture](https://bitbucket.org/datageek008/quora_duplicate_questions/downloads/loss.png)


## Output Files

##### 2 Files
    - 1) submission.txt containing 3563475 entries 2) validation_scores.txt (for model training purpose)
    - Each file contains 2 columns 1) The testid 2) probability of being duplicate 



