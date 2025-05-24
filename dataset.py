import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class ProcesadorCobros:
    def __init__(self, path_csv, top_n_bancos=4):
        self.path_csv = path_csv
        self.top_n_bancos = top_n_bancos
        self.dicccionario = {0: 0, 4: 1, 26: 1}
        self.df = None
        self.predicciones = None
        self.historial_por_id = None

    def ejecutar_todo(self):
        self._cargar_y_preprocesar()
        self._crear_orden()
        self._procesar_bancos()
        self._calcular_predicciones()
        self._generar_historial_por_id()
        return self

    def _cargar_y_preprocesar(self):
        self.df = pd.read_csv(self.path_csv)
        self.df = self.df.sort_values(by=['idCredito'])
        if 'fechaCobroBanco' in self.df.columns:
            self.df = self.df.drop(columns=['fechaCobroBanco'])
        self.df = self.df.fillna(0)

    def _crear_orden(self):
        self.df['orden'] = self.df.groupby('idCredito')['consecutivoCobro'].transform(lambda x: x - x.min())

    def _procesar_bancos(self):
        top_bancos = self.df['idBanco'].value_counts().nlargest(self.top_n_bancos).index
        self.df['idBanco'] = self.df['idBanco'].apply(lambda x: x if x in top_bancos else 1)
        self.df = pd.get_dummies(self.df, columns=['idBanco'], prefix='banco')

    def _calcular_predicciones(self):
        cobradores = self.df.groupby('idCredito')
        predicciones = cobradores['idRespuestaBanco'].last().reset_index()
        predicciones.rename(columns={'idRespuestaBanco': 'valor_prediccion'}, inplace=True)
        predicciones['valor_prediccion'] = predicciones['valor_prediccion'].replace(self.dicccionario)
        predicciones['valor_prediccion'] = np.where(
            predicciones['valor_prediccion'].isin([0, 1]),
            predicciones['valor_prediccion'],
            0
        )
        self.predicciones = predicciones

    def _generar_historial_por_id(self):
        self.historial_por_id = {id_credito: grupo.values.tolist() for id_credito, grupo in self.df.groupby('idCredito')}

    def obtener_X_y_splits(self, test_size=0.2, val_size=0.1, random_state=42):
        df_agrupado = self.df.groupby('idCredito').last().reset_index()
        df_agrupado = df_agrupado.merge(self.predicciones, on='idCredito')
        X = df_agrupado.drop(['idCredito', 'valor_prediccion', 'idRespuestaBanco'], axis=1, errors='ignore')
        y = df_agrupado['valor_prediccion']
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        val_relative_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_relative_size, random_state=random_state, stratify=y_trainval
        )
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# Ejemplo de uso:
procesador = ProcesadorCobros('Data/ListaCobroDetalle2025.csv')
procesador.ejecutar_todo()
print(procesador.df.head())
print(procesador.predicciones.head())
# Acceso al historial por idCredito:
# print(procesador.historial_por_id)