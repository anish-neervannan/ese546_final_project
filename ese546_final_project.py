from nn_architectures import LSTM, BaseLSTM, AttentionLSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = None
attention = False
if (sys.argv[2] == "base"):
    model = LSTM().to(device)
elif (sys.argv[2] == "attention"):
    model = AttentionLSTM().to(device)
    attention = True
else:
    sys.exit("Model flag not set. Please choose from: base, attention.")

print(model)

#! Process genetic algo flag

#! Process data set flag
seq_len = 12
raw_X = np.load(f"train_and_val_{sys.argv[1]}")

X, y = [], []
for i in range(0, raw_X.shape[1] - seq_len):
    X.append(raw_X[:,i:i+12])
    y.append(raw_X[:,i+12])

X = np.asarray(X)
y = np.asarray(y)

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, shuffle=True)

scaler = MinMaxScaler(feature_range=(-1, 1))
for i in len(train_X):
    train_X[i] = scaler.fit_transform(train_X[i])
    train_y[i] = scaler.transform(train_y[i])

for i in len(val_X):
    val_X[i] = scaler.fit_transform(val_X[i])
    val_y[i] = scaler.transform(val_y[i])

train_X_torch = torch.Tensor(train_X)
train_y_torch = torch.Tensor(train_y)
val_X_torch = torch.Tensor(val_X)
val_y_torch = torch.Tensor(val_y)

train_set = TensorDataset(train_X_torch.float(), train_y_torch)
val_set = TensorDataset(val_X_torch.float(), val_y_torch)

epochs = 150
batch_size = 16
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

train_updates, val_updates, train_loss, val_loss = [], [], [], []
updates = 0

for i in range(epochs):
    model.train()
    for j, (batch, target) in enumerate(train_dataloader):
        # Push data to GPU
        X = batch.to(device)
        y = target.to(device)
        total_batches = len(train_dataloader)

        # Zero out the gradients and hidden cells
        optimizer.zero_grad()
        if not attention:
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device), 
                        torch.zeros(1, 1, model.hidden_layer_size).to(device))

        # Compute predicted value
        y_pred = model(y)

        # Compute loss and backpropagate
        loss = criterion(y_pred, torch.flatten(y))
        loss.backward()
        optimizer.step()

        # Record average loss over each epoch
        if i == 0 or (i+1) % 10 == 0:
            train_updates.append(updates)
            train_loss.append(loss.item())
            print('\rEpoch: ', i + 1, 'Train Loss:', train_loss[-1],
                      'Batch', j + 1, 'of', total_batches, ' ' * 10, end='', flush=True)

        updates += 1

    model.eval()
    val_loss = 0.
    for j, (batch, target) in enumerate(val_dataloader):
        X = batch.to(device)
        y = target.to(device)

        # Zero out the gradients and hidden cells
        optimizer.zero_grad()
        if not attention:
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device), 
                        torch.zeros(1, 1, model.hidden_layer_size).to(device))

        # Compute predicted value
        y_pred = model(y)
        loss = criterion(y_pred, torch.flatten(y))

        val_loss += loss.item() / (j + 1)
    print('\nValidation Loss: {}'.format(val_loss))

np.save("results/train_updates.npy", np.asarray(train_updates))
np.save("results/val_updates.npy", np.asarray(val_updates))
np.save("results/train_loss.npy", np.asarray(train_loss))
np.save("results/val_loss.npy", np.asarray(val_loss))







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

def random_models_LSTM(num_models):
    models = []
    for i in range(num_models):
        model = LSTM().to(device)

        #Not using GD for Genetic Algo, turn off grads
        for param in model.parameters():
            param.requires_grad = False
        
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
    mutate_best = 4 
    # Make babies which are mutations of top parents
    for i in range(len(models)-keep-mutate_best):
        mutate_ind = np.random.choice(sorted_inds)
        babies.append(mutate(models[mutate_ind]))
    # Mutate the best parent a few times
    for i in range(mutate_best):
        babies.append(mutate(models[sorted_inds[0]]))
    #Keep the best parent # TODO: implement keep top k
    babies.append(models[sorted_inds[0]])
    best_baby_ind = len(babies)-1 #last bab is superb

    return babies, best_baby_ind

generations = 150

num_models = 50
num_top = 10 #consider the top...
tst_models = random_models_LSTM(num_models)

min_scores = []
for gen in range(generations):
    scores = fitness_test(tst_models)
    print('Gen {}, Mean score: {:.6f}, Best score: {:.6f}'.format(gen, scores.mean(), scores.min()))
    min_scores.append(scores.min())
    sorted_inds = np.argsort(scores)[:num_top]
    tst_models, best_ind = make_babies(tst_models, sorted_inds)

"""### Loss vs Generation for  GA"""

plt.plot(min_scores);
plt.xlabel('Generations')
plt.ylabel('Training Loss');





"""### GA w Attention


"""

def random_models_attn(num_models):
    models = []
    for i in range(num_models):
        model = AttentionLSTM(device=device).to(device)

        #Not using GD for Genetic Algo, turn off grads
        for param in model.parameters():
            param.requires_grad = False
        
        models.append(model)
    
    return models   

generations = 150

num_models = 50
num_top = 10 #consider the top...
tst_models = random_models_attn(num_models)

min_scores = []
for gen in range(generations):
    scores = fitness_test(tst_models)
    print('Gen {}, Mean score: {:.6f}, Best score: {:.6f}'.format(gen, scores.mean(), scores.min()))
    min_scores.append(scores.min())
    sorted_inds = np.argsort(scores)[:num_top]
    tst_models, best_ind = make_babies(tst_models, sorted_inds)

