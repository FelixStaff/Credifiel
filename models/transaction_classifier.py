import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Optional

class TransactionClassifier:
    """
    Modelo de clasificación para transacciones bancarias usando Random Forest.
    """
    def __init__(self, n_estimators: int = 100, random_state: Optional[int] = None):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Ajusta el modelo a los datos de entrenamiento.
        X: np.ndarray de forma (n_samples, n_features) con las características de las transacciones.
        y: np.ndarray de forma (n_samples,) con las etiquetas de clase.
        """
        self.model.fit(X, y)
        self.fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice la clase de las transacciones dadas.
        """
        if not self.fitted:
            raise RuntimeError("El modelo debe ser ajustado antes de predecir.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Devuelve las probabilidades de pertenencia a cada clase.
        """
        if not self.fitted:
            raise RuntimeError("El modelo debe ser ajustado antes de predecir.")
        return self.model.predict_proba(X)
