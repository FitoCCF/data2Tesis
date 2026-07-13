# ============================================================
# src/pipeline/escalado.py — Etapa 3: corrección de asimetría + estandarización
# ============================================================
# Prepara las features del clustering: primero Yeo-Johnson (corrige asimetría),
# luego StandardScaler (media 0, varianza 1). El orden importa y se conserva
# igual en entrenamiento e inferencia.
# ============================================================

import numpy as np                                        # arrays numéricos
import pandas as pd                                        # DataFrames
from sklearn.preprocessing import PowerTransformer, StandardScaler  # transformadores

from .config import FEATS_CLUSTER                          # columnas a escalar


def escalar(df: pd.DataFrame, power: PowerTransformer | None = None,
            scaler: StandardScaler | None = None):
    """Escala las 4 columnas ortogonalizadas para el clustering.

    Parámetros
    ----------
    df : DataFrame que contiene las columnas FEATS_CLUSTER.
    power, scaler : transformadores ya ajustados. Si son None se entrenan
        (modo ENTRENAMIENTO); si se pasan, se aplican (modo INFERENCIA).

    Retorna
    -------
    X_scaled : np.ndarray con las 4 columnas transformadas y estandarizadas.
    power, scaler : los transformadores (nuevos o los recibidos).
    """
    X = df[FEATS_CLUSTER].values                          # matriz de las 4 columnas ortogonalizadas

    if power is None:                                     # --- modo ENTRENAMIENTO (Yeo-Johnson) ---
        power = PowerTransformer(method="yeo-johnson", standardize=False)  # instancia
        power.fit(X)                                      # ajusta la corrección de asimetría
    X_power = power.transform(X)                          # aplica Yeo-Johnson

    if scaler is None:                                    # --- modo ENTRENAMIENTO (StandardScaler) ---
        scaler = StandardScaler()                         # instancia
        scaler.fit(X_power)                               # ajusta media/desviación
    X_scaled = scaler.transform(X_power)                  # aplica estandarización

    return X_scaled, power, scaler                        # devuelve matriz escalada + transformadores
