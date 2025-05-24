import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Tuple

class TransactionTrendModel:
    """
    Modelo simple de regresión lineal para detectar tendencias en series de tiempo de transacciones bancarias.
    """
    def __init__(self):
        self.model = LinearRegression()
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Ajusta el modelo a los datos.
        X: np.ndarray de forma (n_samples, 1) representando el tiempo o índice.
        y: np.ndarray de forma (n_samples,) representando el valor de la transacción.
        """
        self.model.fit(X, y)
        self.fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice la tendencia para los valores dados.
        """
        if not self.fitted:
            raise RuntimeError("El modelo debe ser ajustado antes de predecir.")
        return self.model.predict(X)

    def get_trend_slope(self) -> float:
        """
        Devuelve la pendiente de la tendencia (coeficiente de la regresión lineal).
        """
        if not self.fitted:
            raise RuntimeError("El modelo debe ser ajustado antes de obtener la pendiente.")
        return self.model.coef_[0]
