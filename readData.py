import pandas as pd
import numpy as np


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
        self.Cobradores = self.df.groupby('idCredito')
        return self

    def crear_historial_por_grupo(self):
        self.historial_por_grupo = [grupo.values.tolist() for _, grupo in self.Cobradores]
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
        return self

    # --- Ejecutar todo el preprocesamiento ---
    def ejecutar_todo(self):
        self.cargar_datos()
        self.ordenar_y_limpiar()
        self.agregar_orden()
        self.procesar_bancos()
        self.agrupar_por_credito()
        self.crear_historial_por_grupo()
        self.calcular_predicciones()
        return self.split_datos()