from nn_architectures import LSTM, BaseLSTM, AttentionLSTM
from genetic_algorithm import random_models, fitness_test, mutate, make_babies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys
import torch
import torch.nn as nn


epochs = 5
batch_size = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = None
attention = False
ga = False
if (sys.argv[1] == "base_ga"):
    ga = True
elif (sys.argv[1] == "attention_ga"):
    ga = True
    attention = True
elif (sys.argv[1] == "base"):
    model = LSTM(batch_size=batch_size).to(device)
elif (sys.argv[1] == "attention"):
    model = AttentionLSTM(device=device).to(device)
    attention = True
else:
    sys.exit("Model flag not set. Please choose from: base, attention, base_ga, attention_ga.")

print(model)

sector = "Information Technology"
seq_len = 90
raw_X = np.load(f"Data/train_and_val_{sector}.npy")

for i in range(len(raw_X)):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    raw_X[i] = np.squeeze(scaler.fit_transform(raw_X[i].reshape(-1, 1)))

X, y = [], []
for i in range(0, raw_X.shape[1] - seq_len):
    X.append(raw_X[:,i:i+seq_len])
    y.append(raw_X[:,i+seq_len])

X = np.asarray(X)
y = np.asarray(y)
X = np.reshape(X, (X.shape[0] * X.shape[1], X.shape[2]))
y = np.reshape(y, (y.shape[0] * y.shape[1], 1))

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, shuffle=True)

train_X_torch = torch.Tensor(train_X)
train_y_torch = torch.Tensor(train_y)
val_X_torch = torch.Tensor(val_X)
val_y_torch = torch.Tensor(val_y)

train_set = TensorDataset(train_X_torch.float(), train_y_torch)
val_set = TensorDataset(val_X_torch.float(), val_y_torch)

criterion = nn.L1Loss()

if not ga:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


def train_epoch(model, epoch, ga, total_epochs=5, prin=True):
    if not ga:
        model.train()
    else:
        model.eval()
    avg_train_loss = 0.
    total_batches = len(train_dataloader)
    train_losses_batch = []
    for j, (batch, target) in enumerate(train_dataloader):
        # Push data to GPU
        bs = batch.shape[0]
        X = torch.reshape(batch, (seq_len, bs)).to(device)
        y = target.to(device)
        
        # Zero out the gradients and hidden cells
        if not ga:
            optimizer.zero_grad()
        if not attention:
            model.hidden_cell = (torch.zeros(1, bs, model.hidden_layer_size).to(device), 
                        torch.zeros(1, bs, model.hidden_layer_size).to(device))

        # Compute predicted value
        y_pred = torch.flatten(model(X))

        # Compute loss and backpropagate
        loss = criterion(y_pred, torch.flatten(y))
        avg_train_loss += loss.item()/len(train_dataloader)

        if not ga:
            loss.backward()
            optimizer.step()

        train_losses_batch.append(loss.item())

        if (j == 0 or (j+1) % 100 == 0) and prin:
            print(f'Epoch: [{epoch+1}/{total_epochs}], Batch [{j+1}/{total_batches}], Train Loss: {loss.item():.4f}')
    return avg_train_loss, train_losses_batch


def val_epoch(model, ga):
    # VALIDATION
    model.eval()
    avg_val_loss = 0.
    for j, (batch, target) in enumerate(val_dataloader):
        # Push data to GPU
        bs = batch.shape[0]
        X = torch.reshape(batch, (seq_len, bs)).to(device)
        y = target.to(device)

        # Zero out the gradients and hidden cells
        if not ga:
            optimizer.zero_grad()
        if not attention:
            model.hidden_cell = (torch.zeros(1, bs, model.hidden_layer_size).to(device), 
                        torch.zeros(1, bs, model.hidden_layer_size).to(device))

        # Compute predicted value
        y_pred = model(X)
        y_pred = torch.flatten(y_pred)

        # Compute loss and backpropagate
        loss = criterion(y_pred, torch.flatten(y))
        avg_val_loss += loss.item()/len(val_dataloader)

    return avg_val_loss


# Runs training loop for base LSTM and LSTM w/ Attention layer
if not ga:

    train_losses_batch, train_loss, val_loss = [], [], []

    for i in range(epochs):
        #TRAINING
        avg_loss_train, train_losses_batch_epoch = train_epoch(model, i, ga, total_epochs=epochs)

        train_loss.append(avg_loss_train)
        train_losses_batch.extend(train_losses_batch_epoch)
        print(f'Epoch: [{i+1}/{epochs}], Average Train Loss: {avg_loss_train:.4f}')

        # VALIDATION
        avg_loss_val = val_epoch(model, ga)
        val_loss.append(avg_loss_val)
        print(f'Epoch: [{i+1}/{epochs}], Validation Loss: {avg_loss_val:.4f}')
        # print('Validation Loss: {}'.format(val_loss))

    folder_name = f"{sys.argv[1]}_bs{batch_size}_sl{seq_len}_ep{epochs}"
    if not os.path.isdir(f"results/{folder_name}"):
        os.mkdir(f"results/{folder_name}")

    plt.plot(np.convolve(train_losses_batch,(1/1000)*np.ones(1000))[1000:-1000])
    plt.savefig(f"results/{folder_name}/train_losses_batch.png")

    np.save(f"results/{folder_name}/train_loss.npy", np.asarray(train_loss))
    np.save(f"results/{folder_name}/val_loss.npy", np.asarray(val_loss))
    np.save(f"results/{folder_name}/train_losses_batch.npy", np.asarray(train_losses_batch))
    torch.save(model.state_dict(), f"results/{folder_name}/model.pt")


#GA Training loop!
else:
    generations = epochs

    num_models = 10 #was 50
    num_top = 4 #kill all but... #was 10
    mutate_best = 4 #how many mutations of the top parent
    lstm_type = 'base'

    if attention:
        tst_models = random_models(num_models, 'attention', device)
    else:
        tst_models = random_models(num_models, lstm_type, device)

    min_tr_losses = []
    mean_tr_losses = []
    val_losses = []
    for gen in range(generations):
        #assess
        scores = fitness_test(tst_models, train_epoch)
        #log metrics
        min_score = scores.min()
        mean_score = scores.mean()
        min_tr_losses.append(min_score)
        mean_tr_losses.append(mean_score)
        #sort by performance
        sorted_inds = np.argsort(scores)[:num_top]
        #validation of best model
        best_val = val_epoch(tst_models[sorted_inds[0]], ga = True)
        val_losses.append(best_val)
        print('Gen [{}/{}], Mean Train Loss: {:.4f}, Best Train Loss: {:.4f}: Best Val Loss: {:.4f}'.format(gen + 1, generations, mean_score, min_score, best_val))
        tst_models, best_ind = make_babies(tst_models, sorted_inds, device, mutate_best)

    folder_name = f"{sys.argv[1]}_bs{batch_size}_sl{seq_len}_ep{epochs}"
    if not os.path.isdir(f"results/{folder_name}"):
        os.mkdir(f"results/{folder_name}")

    #plt.plot(np.convolve(train_losses_batch,(1/1000)*np.ones(1000))[1000:-1000])
    #plt.savefig(f"results/{folder_name}/train_loss_batch_test.png")
    plt.figure()
    plt.plot(min_tr_losses)
    plt.plot(mean_tr_losses)
    plt.plot(val_losses)
    plt.legend(['Min. Train Losses', 'Avg Train Losses', 'Val Loss'])
    plt.title(sys.argv[1])
    plt.ylabel('Loss')
    plt.xlabel('Generations')

    plt.savefig(f"results/{folder_name}/GA_losses.png")

    np.save(f"results/{folder_name}/min_train_loss.npy", np.asarray(min_tr_losses))
    np.save(f"results/{folder_name}/mean_train_loss.npy", np.asarray(mean_tr_losses))
    np.save(f"results/{folder_name}/val_loss.npy", np.asarray(val_losses))
    torch.save(tst_models[best_ind].state_dict(), f"results/{folder_name}/best model.pt")