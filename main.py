import torch
from models.transaction_classifier import Transaction
from pipeline.timeseries_pipeline import pipeline
import random

# Parámetros de ejemplo
input_dim = 8  # Cambia esto según tus datos reales
input_history = 5  # Cambia esto según tus datos reales
batch_size = 10
seq_len = 5

# Instanciar modelo y pipeline
model = Transaction(input_dim, input_history)
p = pipeline(model)

# Crear historial como una lista de vectores de tamaño aleatorio
historial = []
for _ in range(batch_size):
    seq_length = random.randint(1, seq_len)  # longitud aleatoria para cada batch
    historial.append(torch.randn(seq_length, input_history))
# Para compatibilidad con el modelo, convierte la lista a un objeto tipo packed sequence o padéalo si es necesario

# Crear actual como antes
actual = torch.randn(batch_size, input_dim)

# Realizar predicción
for i in range(len(historial)):
    # Meter los valores en el modelo
    output = model(historial[i].unsqueeze(0), actual[i].unsqueeze(0))

    print('Predicción de ejemplo:', output)

# pipeline
pipe = pipeline(model)
histories = [torch.randn(seq_len, input_history) for _ in range(batch_size)]
recibos = [torch.randn(input_dim) for _ in range(batch_size)]
outputs = [torch.randn(1) for _ in range(batch_size)]
trian_loader = list(zip(histories, recibos, outputs))
pipe.train(trian_loader, num_epochs=10)