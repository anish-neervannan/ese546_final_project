import matplotlib.pyplot as plt

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
