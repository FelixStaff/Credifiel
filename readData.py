import pandas as pd
import numpy as np
import torch

class PreprocesadorCobros:
    def __init__(self, path_csv):
        self.path_csv = path_csv
        self.df = None
        self.Cobradores = None
        self.historial_por_grupo = None
        self.predicciones = None
        self.top4_bancos = None
        self.bancos = None

    # --- Carga y limpieza de datos ---
    def cargar_datos(self):
        self.df = pd.read_csv(self.path_csv)
        return self

    def ordenar_y_limpiar(self):
        self.df = self.df.sort_values(by=['idCredito'])
        if 'fechaCobroBanco' in self.df.columns:
            self.df = self.df.drop(columns=['fechaCobroBanco'])
        self.df = self.df.fillna(0)
        return self

    # --- Procesamiento de columnas ---
    def agregar_orden(self):
        self.df['orden'] = self.df.groupby('idCredito').cumcount()
        return self

    def procesar_bancos(self):
        self.top4_bancos = self.df['idBanco'].value_counts().nlargest(4).index
        self.bancos = {12: 0, 2: 1, 72: 2, 21: 3}
        self.bancos.update({x: 4 for x in self.df['idBanco'].unique() if x not in self.bancos})
        self.df['idBanco'] = self.df['idBanco'].apply(lambda x: x if x in self.top4_bancos else 1)
        self.df = pd.get_dummies(self.df, columns=['idBanco'], prefix='banco')
        return self

    # --- Agrupación y generación de features ---
    def agrupar_por_credito(self):
        # Quitamos la variable de idListaCobro, consecutivoCobro, idEmisora,IdBanco,IdBancoCliente
        self.df = self.df.drop(columns=['idListaCobro', 'consecutivoCobro', 'idEmisora', 'IdBanco', 'IdBancoCliente'])
        self.Cobradores = self.df.groupby('idCredito')
        # print the columns of the grouped data

        return self

    def crear_historial_por_grupo(self):
        # Quitar la variable de idListaCobro, consecutivoCobro
        #self.historial_por_grupo = [grupo.reset_index().drop("idCredit").values.tolist() for _, grupo in self.Cobradores]
        # Make the same as above but drop the columns idCredito, o mas bien sin el indice
        self.historial_por_grupo = []
        for _, grupo in self.Cobradores:
            # 5
            self.historial_por_grupo.append(grupo.drop(columns=['idCredito']).values.tolist())
            
        
        return self

    # --- Generación de etiquetas ---
    def calcular_predicciones(self):
        self.predicciones = self.Cobradores['idRespuestaBanco'].last().reset_index()
        self.predicciones.rename(columns={'idRespuestaBanco': 'valor_prediccion'}, inplace=True)
        dicccionario = {0: 1, 4: 0, 26: 1}
        self.predicciones['valor_prediccion'] = self.predicciones['valor_prediccion'].replace(dicccionario)
        self.predicciones['valor_prediccion'] = np.where(
            self.predicciones['valor_prediccion'].isin([0, 1]),
            self.predicciones['valor_prediccion'],
            0
        )
        # Make the numpy of the predictions
        self.predicciones = self.predicciones['valor_prediccion'].values.tolist()
        return self

    # --- Ejecutar todo el preprocesamiento ---
    def ejecutar_todo(self):
        self.cargar_datos()
        self.ordenar_y_limpiar()
        self.agregar_orden()
        #self.procesar_bancos()
        self.agrupar_por_credito()
        self.calcular_predicciones()
        self.crear_historial_por_grupo()
        return self.historial_por_grupo, self.predicciones
    

# Pasar los datos a conjunto de tensores
def convertir_a_tensor(historial, labels):
    # Recibimos un conjunto de valores y lo convertimos a tensores
    # Input": [batch_size, *var_lenght, input_dim]
    # Creamos ahora el input variable que es
    # Historial: [batch_size, *var_lenght-1, input_dim]
    # Recibo actual: [batch_size, input_dim]
    historial_tensor = []
    recibo_actual_tensor = []
    labels_tensor = []
    for i in range(len(historial)):
        if len(historial[i]) == 1:
            # Si solo hay un elemento, historial y recibo son iguales
            tensor_val = torch.tensor(historial[i][0], dtype=torch.float32)
            # Modificamos la variable 4 (índice 4) a -1 para el recibo actual
            recibo_actual = tensor_val.clone()
            recibo_actual[4] = -1
            historial_tensor.append(tensor_val.unsqueeze(0))
            recibo_actual_tensor.append(recibo_actual)
        else:
            historial_tensor.append(torch.tensor(historial[i][:-1], dtype=torch.float32))
            recibo_actual = torch.tensor(historial[i][-1], dtype=torch.float32)
            recibo_actual[4] = -1
            recibo_actual_tensor.append(recibo_actual)
        labels_tensor.append(torch.tensor(labels[i], dtype=torch.float32))
    return zip(historial_tensor, recibo_actual_tensor, labels_tensor)

if __name__ == "__main__":
    # Ejemplo de uso
    preprocesador = PreprocesadorCobros('Data\ListaCobroDetalle2025.csv')
    historial, predicciones = preprocesador.ejecutar_todo()
    print(len(historial)) 
    print(len(historial[0]))
    print(len(predicciones))