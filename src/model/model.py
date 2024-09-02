import torch
import torch.nn as nn

class MyLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.0):
        super(MyLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, 1, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size // 2, 1, batch_first=True, dropout=0.0)
        self.fc = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)  # Output of LSTM layer 1
        out, _ = self.lstm2(out)  # Output of LSTM layer 2
        out = out[:, -1, :]  # Take the output of the last time step
        out = self.fc(out)  # Pass through the fully connected layer
        return out