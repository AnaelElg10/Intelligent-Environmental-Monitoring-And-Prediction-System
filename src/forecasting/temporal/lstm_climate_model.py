"""LSTM Climate Model for time series forecasting."""

import torch
import torch.nn as nn

class LSTMClimateModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.classifier = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        output, _ = self.lstm(x)
        return self.classifier(output[:, -1, :])
