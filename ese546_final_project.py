# -*- coding: utf-8 -*-
"""ese546_final_project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/anish-neervannan/ese546_final_project/blob/main/ese546_final_project.ipynb

## **ESE 546 Final Project: The Effect of Genetic Algorithms on Long Short Term Memory Networks with Attention Modules**

### **Authors**
- Anish Neervannan (PennKey: anishrn)
- Vinay Senthil (PennKey: vinayksk)
- Bhaskar Abhiraman (PennKey: bhaskara)

### **Import Kaggle Dataset**

1. Use Chrome Extension "[Get cookies.txt](https://chrome.google.com/webstore/detail/get-cookiestxt/bgaddhkoddajcdgocldbbfleckgcbcid?utm_campaign=en&utm_source=en-ha-na-us-bk-ext&utm_medium=ha)" to download cookies from [here](https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) after logging in.
2. Save the file as "cookies.txt" and upload it in the next cell.
3. A version of "cookies.txt" is also in the shared Drive for this project.
"""

from google.colab import files

uploaded = files.upload()

"""This cell imports all the data (492 MB) straight from Kaggle to the Colab workspace."""

!wget -x --load-cookies cookies.txt "https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/download" -O data.zip
!unzip data.zip

"""For example, graph AAPL over time."""

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')

aapl_data = pd.read_csv("Stocks/aapl.us.txt")
print(aapl_data.head())
plt.plot(aapl_data.Open);
plt.xlabel("Days since Sept 7 1984")
plt.ylabel("Opening Price (USD)");
plt.title("AAPL Opening Price Over Time")

"""### **LSTM for Numerical Sequences**

### To do:


*   Plagiarize less - everything is mostly changed except: (1) get_batch to process multiple batches, (2) change LSTM() network architecture (maybe use the one from earlier homework), (3) change eval code, (4) why is test_data not scaled?
*   Validation metrics (MAE per sector)
*   Preprocess all stock data by sector/S&P
*   Implement mini-batching (looking at fixed-length sequences of various stocks)
*   Characterize base LSTM
*   Attention
*   Get S&P 500 info/sector info/stock symbols

LSTM for Numerical Sequences based on:
https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/

"""

# from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocess the data, select the appropriate stocks and convert to numpy
# Rescale training data between -1 and 1
stock = 'aapl'
stock_pd_frame = pd.read_csv("Stocks/{}.us.txt".format(stock))
stock_np = np.asarray(stock_pd_frame.Open)

train_size = 500
test_size = 12

train_data = stock_np[-train_size-test_size:-test_size]
test_data = stock_np[-test_size:]

scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))

# helper method to get a training sequence starting at a particular point in the sequence
def get_batch(input_data, seq_len, i):
    train_seq = input_data[i:i+seq_len]
    train_label = input_data[i+seq_len:i+seq_len+1]
    return (train_seq ,train_label)

# LSTM class module defined
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

epochs = 150
seq_len = 12
print_and_save_interval = 1
num_batches = len(train_data_normalized) - seq_len

model = LSTM().to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


print(model)
#Training Loop

def train(model, epochs):
    #Save losses
    tr_losses = np.zeros(epochs)

    for epoch in range(epochs):
        for j in range(num_batches):
            # Fetch the batched data and push to GPU
            X, y = get_batch(train_data_normalized, seq_len, j)
            X = torch.Tensor(X).to(device)
            y = torch.Tensor(y).to(device)

            # Zero out the gradients and hidden cells
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                            torch.zeros(1, 1, model.hidden_layer_size).to(device))

            # Compute predicted value
            y_pred = model(X)

            # Compute loss and backpropagate
            loss = criterion(y_pred, torch.flatten(y))
            loss.backward()
            optimizer.step()

            # Record average loss over each epoch
            tr_losses[epoch] += loss.item()/num_batches

        # Report loss every few epochs
        if epoch % print_and_save_interval == 0:
            print ('Epoch [{}/{}], Training Loss: {:.4f}'.format(epoch+1, epochs, tr_losses[epoch]))
    return tr_losses
tr_losses = train(model,150)
#TODO: Fix bug where first training loss is underreported...

plt.plot(tr_losses[1:]);
plt.title('Training Loss on AAPL for 150 Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss (L1 Distance)')

num_preds = 12

test_inputs = np.ndarray.flatten(train_data_normalized[-seq_len:]).tolist()

model.eval()

for i in range(num_preds):
    seq = torch.FloatTensor(test_inputs[-seq_len:]).to(device)
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                        torch.zeros(1, 1, model.hidden_layer_size).to(device))
        test_inputs.append(model(seq).item())

actual_predictions = scaler.inverse_transform(np.array(test_inputs[seq_len:] ).reshape(-1, 1))

ts = np.arange(0,train_size + test_size,1)
plt.plot(ts[:train_size],train_data)
plt.plot(ts[-test_size:],actual_predictions)
plt.plot(ts[-test_size:],stock_np[-test_size:])
plt.xlabel('Time (days)')
plt.ylabel('Opening Price')
plt.legend(['Price History', 'Price Prediction (Based on 12 prev days)','Real Price Outcome'])

"""### Genetic Algorithm Approach
Referenced: [Paras Chopra - Towards Data Science](https://towardsdatascience.com/reinforcement-learning-without-gradients-evolving-agents-using-genetic-algorithms-8685817d84f)


"""

import time
import math
import copy
import torch.nn.functional as F

for param in model.parameters():
    try:
        print(param.bias)
    except:
        pass

def random_models(num_models):
    models = []
    for i in range(num_models):
        model = LSTM().to(device)

        #Not using GD for Genetic Algo, turn off grads
        for param in model.parameters():
            model.requites_grad = False
        
        models.append(model)
    
    return models

def eval_model(model):

    epochs = 1 #Epochs without gradients don't really make sense.
    tr_losses = np.zeros(epochs)

    for epoch in range(epochs):
        num_batches_eval = num_batches
        for j in range(num_batches_eval): #was num_batches
            # Fetch the batched data and push to GPU
            X, y = get_batch(train_data_normalized, seq_len, j)
            X = torch.Tensor(X).to(device)
            y = torch.Tensor(y).to(device)

            with torch.no_grad():
                #zero out the hidden state
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                            torch.zeros(1, 1, model.hidden_layer_size).to(device))

                # Compute predicted value
                y_pred = model(X)

                # Compute loss
                loss = criterion(y_pred, torch.flatten(y))

            # Record average loss over each epoch
            tr_losses[epoch] += loss.item()/num_batches_eval
    return tr_losses

def fitness_test(models):
    # Score a bunch of models (golf rules)
    scores = np.zeros(len(models))
    for i,model in enumerate(models):
        scores[i] = eval_model(model).mean()
    return scores

def mutate(model):
    # Modify weights with Gaussian noise - tunable std dev
    child_model = copy.deepcopy(model)

    mutation_power = 0.1 #tune this

    for param in child_model.parameters():
        param.data += torch.empty(param.shape).normal_(mean=0,std=mutation_power).to(device)

    return child_model.to(device)

def make_babies(models, sorted_inds):
    # sorted_inds will be the top group of parents that performed the best
    # !make sure sorted is going the correct direction
    N = len(sorted_inds)

    babies = []
    keep = 1 
    # Make equal amount of new babies which are mutations of the top parents
    for i in range(len(models)-keep):
        mutate_ind = np.random.choice(sorted_inds)
        babies.append(mutate(models[mutate_ind]))
    #Keep the best parent # TODO: implement keep top k
    babies.append(models[sorted_inds[0]])
    best_baby_ind = len(babies)-1 #last bab is superb

    return babies, best_baby_ind

generations = 1000

num_models = 50
num_children = 10

tst_models = random_models(num_models)

min_scores = []
for gen in range(generations):
    scores = fitness_test(tst_models)
    print('Gen {}, Mean score: {:.6f}, Best score: {:.6f}'.format(gen, scores.mean(), scores.min()))
    min_scores.append(scores.min())
    sorted_inds = np.argsort(scores)
    tst_models, best_ind = make_babies(tst_models, sorted_inds)

"""### Loss vs Generation for  GA"""

plt.plot(min_scores);
plt.xlabel('Generations')
plt.ylabel('Training Loss');

"""

*   Loss got down to 0.089752 from 0.6 over the entire AAPL stock.
*   Tweak GA so several mutations of the best cand. forward. Progress will be a lot less incremental 

"""



"""## LSTM With Attention Layer

https://github.com/chrisvdweth/ml-toolkit/blob/master/pytorch/models/text/classifier/rnn.py
https://github.com/keon/seq2seq/blob/master/model.py
https://towardsdatascience.com/time-series-forecasting-with-deep-learning-and-attention-mechanism-2d001fc871fc

"""

# LSTM class module defined
class AttentionLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.encoder = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.encoder(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]