import torch
import torch.nn as nn
import torch.optim as optim

class Transaction(nn.Module):
    def __init__(self, input_dim, input_history, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.embedding_history = nn.Linear(input_history, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), 
            num_layers
        )
        self.fc = nn.Linear(d_model, 1)

    def forward(self, historial, recibo_actual):
        # historial: [batch, seq_len, input_dim]
        # recibo_actual: [batch, input_dim]
        x = self.embedding_history(historial)  # [batch, seq_len, d_model]
        x = x.permute(1, 0, 2)  # Transformer expects [seq_len, batch, d_model]
        h = self.transformer(x)
        h_final = h.mean(dim=0)  # Promedio sobre la dimensión de secuencia [batch, d_model]
        rec_emb = self.embedding(recibo_actual)
        combined = h_final + rec_emb  # simple combinación
        return torch.sigmoid(self.fc(combined)).squeeze(-1)
    

if __name__ == "__main__":
    # Test the Transaction model
    input_dim = 10  # Dimensión de entrada
    input_history = 5  # Dimensión de la historia
    model = Transaction(input_dim, input_history)
    seq_len = 5  # Longitud de la secuencia
    # Datos de prueba
    historial = torch.randn(1, seq_len, input_history)  # [batch_size, seq_len, input_history]
    recibo_actual = torch.randn(1, input_dim)  # [batch_size, input_dim]
    
    # Forward pass
    output = model(historial, recibo_actual)
    print(output.shape)  # Debería ser [batch_size]