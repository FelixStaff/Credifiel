import torch
from models.transaction_classifier import Transaction
from pipeline.timeseries_pipeline import pipeline
import random
from readData import PreprocesadorCobros
from readData import convertir_a_tensor
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

batch_size = 1
# Leemos los datos
print ('Leyendo datos...')
preprocesador = PreprocesadorCobros('Data\TestData.csv')
historial, labels = preprocesador.ejecutar_todo()
print ('Datos leídos')
# Instanciar modelo y pipeline
input_dim = len(historial[0][0])
input_history = len(historial[0][0])
model = Transaction(input_dim, input_history)
# Cargar el modelo preentrenado
model.load_state_dict(torch.load('model_epoch_0_step_2000.pth'))
pipe = pipeline(model)
print ('Pipeline creado')
# Realizar predicción

train_loader = convertir_a_tensor(historial, labels)

accuracy = 0

progress_bar = tqdm(train_loader, desc="Evaluating", unit="batch")
true_labels = []
predicted_labels = []
with torch.no_grad():
    # Desactivar el cálculo de gradientes
    model.eval()
    counter = 0
    # Iterar sobre el conjunto de datos
    for (historical, actual, label) in progress_bar:
        # Convertir a tensores
        historical = historical.unsqueeze(0)
        actual = actual.unsqueeze(0)
        label = label.unsqueeze(0).reshape(-1)

        # Realizar la predicción
        output = model(historical, actual)
        counter += 1
        # Almacenar las etiquetas verdaderas y predicciones
        true_labels.append(label.item())
        predicted_labels.append(output.item())

        # Calcular la precisión
        predicted = (output > 0.5).float()
        accuracy += (predicted == label).float().sum()
        progress_bar.set_postfix(accuracy=accuracy / (counter + 1))


# Transformar las listas a valores enteros en predicciones
predicted_labels = [1 if pred > 0.5 else 0 for pred in predicted_labels]
print(f'Precisión total: {accuracy.item()}')
# Imprimir el reporte de clasificación
print(classification_report(true_labels, predicted_labels, target_names=['No Pagado', 'Pagado']))
# Imprimir la matriz de confusión
from sklearn.metrics import confusion_matrix
print(confusion_matrix(true_labels, predicted_labels))