import torch
import torch.nn as nn

# LSTM class module defined
class LSTM(nn.Module):
    def __init__(self, batch_size=8, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,batch_size,self.hidden_layer_size),
                            torch.zeros(1,batch_size,self.hidden_layer_size))

    def forward(self, input_seq):
        input_seq = torch.unsqueeze(input_seq, dim=2)
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(lstm_out[-1])
        return predictions


class BaseLSTM(nn.Module):
    def __init__(self, input_size=1, batch_size=8, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        input_seq = torch.unsqueeze(input_seq, dim=2)
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        return lstm_out


class AttentionLSTM(nn.Module):
    def __init__(self, device, input_size=1, batch_size=8, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.device = device
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = batch_size

        self.lstm = BaseLSTM(input_size=input_size, batch_size=batch_size,
                        hidden_layer_size=self.hidden_layer_size, output_size=output_size)
        self.attn_weights = nn.Linear(self.hidden_layer_size, 1)
        self.softmax = nn.Softmax(dim=0)
        self.linear = nn.Linear(self.hidden_layer_size, 1)

    def forward(self, input_seq):
        self.lstm.hidden_cell = (torch.zeros(1, input_seq.shape[1], self.lstm.hidden_layer_size).to(self.device),
                            torch.zeros(1, input_seq.shape[1], self.lstm.hidden_layer_size).to(self.device))
        predictions = self.lstm(input_seq)
        raw_values = torch.zeros(len(input_seq), predictions.shape[1]).to(self.device)
        for i, time_step in enumerate(predictions):
            att_t = torch.squeeze(self.attn_weights(time_step), 1)
            raw_values[i] = att_t
        alphas = self.softmax(raw_values)
        alphas = alphas.unsqueeze(2).repeat(1, 1, self.hidden_layer_size)
        product = alphas * predictions
        final = torch.sum(product, dim=0)
        final = self.linear(final)
        return final