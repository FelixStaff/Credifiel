import torch
from models.transaction_classifier import Transaction
from pipeline.timeseries_pipeline import pipeline
import random
from readData import PreprocesadorCobros
from readData import convertir_a_tensor

batch_size = 1
# Leemos los datos
print ('Leyendo datos...')
preprocesador = PreprocesadorCobros('Data/ListaCobroDetalle2025.csv')
historial, labels = preprocesador.ejecutar_todo()
print ('Datos leídos')
# Instanciar modelo y pipeline
input_dim = len(historial[0][0])
input_history = len(historial[0][0])
model = Transaction(input_dim, input_history)
pipe = pipeline(model)
print ('Pipeline creado')
# Realizar predicción
for i in range(len(historial)):
    input = torch.tensor(historial[i][:-1])
    actual = torch.tensor(historial[i][-1])
    outp = torch.tensor(labels[i])
    if len(input) != input_history:
        print('Error en la longitud de la entrada')
        break
    # Meter los valores en el modelo
    output = model(input.unsqueeze(0), actual.unsqueeze(0))

    print('Predicción de ejemplo:', output)

# pipeline
train_loader = convertir_a_tensor(historial, labels)
for (historical, actual, label) in train_loader:
    print('Historial:', historical)
    print('Recibo actual:', actual)
    print('Etiqueta:', label)
    break
# Entrenar el modelo

pipe.train(train_loader, num_epochs=10)