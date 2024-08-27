import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        _, (hidden, _) = self.lstm(x)
        # hidden shape: (num_layers, batch_size, hidden_size)

        # Use the output from the last layer
        last_hidden = hidden[-1]
        # last_hidden shape: (batch_size, hidden_size)

        output = self.fc(last_hidden)
        # output shape: (batch_size, 1)

        return output

# Example usage:
# model = LSTMModel(input_size=5, hidden_size=128, num_layers=2)