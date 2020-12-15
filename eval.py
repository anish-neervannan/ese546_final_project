from nn_architectures import LSTM, BaseLSTM, AttentionLSTM
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch

model_folders = []
for root, dirs, files in os.walk("results", topdown=False):
    for name in dirs:
        if name != "eval":
            model_folders.append(name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seq_len = 90
batch_size = 8
sector = "Information Technology"
raw_train_X = np.load(f"data/train_and_val_{sector}.npy")
raw_test_X = np.load(f"data/test_{sector}.npy")

# Create scalers for each stock in the test matrix
scalers = []
for i in range(len(raw_train_X)):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    raw_train_X[i] = np.squeeze(scaler.fit_transform(raw_train_X[i].reshape(-1, 1)))
    scalers.append(scaler)

X, y = [], []
for i in range(0, raw_test_X.shape[1] - seq_len):
    scaled_raw_x = np.zeros_like(raw_test_X[:,i:i+seq_len])
    scaled_raw_y = np.zeros((raw_test_X[:,i+seq_len].shape[0], 1))
    for idx, scaler in enumerate(scalers):
        scaled_raw_x[idx,:] = np.squeeze(scaler.transform(raw_test_X[idx, i:i+seq_len].reshape((-1, 1))))
        scaled_raw_y[idx,:] = np.squeeze(scaler.transform(raw_test_X[idx, i+seq_len].reshape((1, -1))))
    scaled_raw_y = np.squeeze(scaled_raw_y)
    X.append(scaled_raw_x)
    y.append(scaled_raw_y)


appl_data = raw_test_X[7]
appl_orig = appl_data[:90]
appl_gt = appl_data[90:150]
appl_scaler = MinMaxScaler(feature_range=(-1, 1))
appl_data_sc = np.squeeze(appl_scaler.fit_transform(appl_data.copy().reshape(-1, 1)))
appl_X = appl_data_sc[:90]

X = np.asarray(X)
y = np.asarray(y)
X = np.reshape(X, (X.shape[0] * X.shape[1], X.shape[2]))
y = np.reshape(y, (y.shape[0] * y.shape[1], 1))

test_X_torch = torch.Tensor(X)
test_y_torch = torch.Tensor(y)
test_set = TensorDataset(test_X_torch.float(), test_y_torch)
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

criterion = torch.nn.L1Loss()
test_loss = np.zeros((len(model_folders), len(test_dataloader)))
test_directional_acc = np.zeros((len(model_folders), len(test_dataloader)))

predictions = np.zeros((len(model_folders), 60))


for i, model_dir in enumerate(model_folders):
    model = None
    attention = False
    if model_dir.startswith("base"):
        model = LSTM(batch_size=batch_size).to(device)
    elif model_dir.startswith("attention"):
        model = AttentionLSTM(device=device).to(device)
        attention = True
    else:
        sys.exit(f"Unidentified experiment: {model_dir}")

    model.load_state_dict(torch.load(f"results/{model_dir}/model.pt"))
    model.eval()

    with torch.no_grad():
        for j, (batch, target) in enumerate(test_dataloader):
            bs = batch.shape[0]
            re_batch = torch.reshape(batch, (seq_len, bs)).to(device)
            target = target.to(device)

            if not attention:
                model.hidden_cell = (torch.zeros(1, bs, model.hidden_layer_size).to(device), 
                        torch.zeros(1, bs, model.hidden_layer_size).to(device))

            y_pred = model(re_batch)
            y_pred = torch.flatten(y_pred)

            loss = criterion(y_pred, torch.flatten(target))
            test_loss[i, j] = loss
            
            preds_diff = y_pred.cpu() - batch[:,-1]
            gt_diff = np.squeeze(target.cpu()) - batch[:,-1]
            test_directional_acc[i, j] = (np.sign(preds_diff)==np.sign(gt_diff)).sum()

    
    with torch.no_grad():
        for j in range(60):
            appl_X_copy = copy.deepcopy(appl_data_sc[j:j+90])
            re_batch = torch.reshape(torch.Tensor(appl_X_copy), (seq_len, 1)).to(device)

            if not attention:
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device), 
                        torch.zeros(1, 1, model.hidden_layer_size).to(device))

            y_pred = model(re_batch)
            predictions[i, j] = y_pred
            # appl_X_copy = np.roll(appl_X_copy, -1)
            # appl_X_copy[-1] = y_pred

# Plot predictions
predictions = appl_scaler.inverse_transform(predictions)
predictions = np.concatenate((np.ones((4, 1))*appl_data[89], predictions), axis=1)
plt.plot(np.arange(0, 90), appl_data[0:90])
plt.plot(np.arange(90, 150), appl_gt, label="Actual Prices")
plt.plot(np.arange(89, 150), predictions[0], label="Base")
plt.plot(np.arange(89, 150), predictions[1], label="Attention")
plt.plot(np.arange(89, 150), predictions[2], label="Base + GA")
plt.plot(np.arange(89, 150), predictions[3], label="Attention + GA")
plt.xlabel('Time (Days)')
plt.ylabel('Opening Price')
plt.legend()
plt.title('Predicted AAPL Stock Price Comparison')
plt.savefig("results/eval/predictions.png")

print("Average Batch Loss", np.mean(test_loss, axis=1))
print("Average Batch Directional Accuracy", np.sum(test_directional_acc, axis=1) / test_X_torch.shape[0])
    