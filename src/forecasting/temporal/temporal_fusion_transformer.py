"""Temporal Fusion Transformer for climate forecasting."""

import torch
import torch.nn as nn

class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_attention_heads, dropout, output_size, prediction_length):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        output, _ = self.lstm(x)
        return self.classifier(output[:, -1, :])
