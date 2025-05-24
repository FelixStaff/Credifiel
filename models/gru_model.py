import torch
import torch.nn as nn

class GRUTimeSeriesModel(nn.Module):
    """
    GRU-based model for time series prediction.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.0):
        super(GRUTimeSeriesModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        # Take the output from the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out
