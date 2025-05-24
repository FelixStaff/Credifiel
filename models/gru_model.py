import torch
import torch.nn as nn

class LinearPaymentPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(-1)


class TransformerPaymentPredictor(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), 
            num_layers
        )
        self.fc = nn.Linear(d_model, 1)

    def forward(self, historial, recibo_actual):
        # historial: [batch, seq_len, input_dim]
        # recibo_actual: [batch, input_dim]
        x = self.embedding(historial)  # [batch, seq_len, d_model]
        x = x.permute(1, 0, 2)  # Transformer expects [seq_len, batch, d_model]
        h = self.transformer(x)
        h_final = h[-1]  # Último token (o usar atención)
        rec_emb = self.embedding(recibo_actual)
        combined = h_final + rec_emb  # simple combinación
        return torch.sigmoid(self.fc(combined)).squeeze(-1)
