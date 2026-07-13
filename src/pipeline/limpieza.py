# ============================================================
# src/pipeline/limpieza.py — Etapa 1: limpieza y detección de anomalías
# ============================================================
# Convierte la celda 01_limpieza en funciones reutilizables. Separa el
# "entrenar" (ajustar el IsolationForest) del "aplicar" (usar uno ya
# entrenado), para que entrenamiento e inferencia usen la MISMA lógica.
# ============================================================

import pandas as pd                                       # manejo de DataFrames
from sklearn.ensemble import IsolationForest              # detector de anomalías multivariadas

from .config import CANALES, RANDOM_STATE                 # constantes centralizadas


def _filtrar_invalidos(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica los tres filtros deterministas de calidad (no requieren modelo).

    1. Elimina códigos de error del analizador (-9999).
    2. Elimina lecturas en cero simultáneo (planta/analizador detenido).
    3. Elimina 'sensor congelado' (vector idéntico repetido >= 3 veces).
    """
    df = df[~(df[CANALES] == -9999).any(axis=1)]          # quita filas con algún -9999
    df = df[~(df[CANALES] == 0).all(axis=1)].copy()       # quita filas con los 5 canales en 0

    repeticiones = df.groupby(CANALES)[CANALES[0]].transform("size")  # nº de repeticiones de cada vector
    df = df[repeticiones < 3].copy()                      # conserva vectores que aparecen menos de 3 veces
    return df                                             # devuelve el DataFrame filtrado


def limpiar(df: pd.DataFrame, detector: IsolationForest | None = None):
    """Limpia intensidades crudas y detecta anomalías multivariadas.

    Parámetros
    ----------
    df : DataFrame con los canales de intensidad crudos.
    detector : IsolationForest ya entrenado. Si es None, se entrena uno nuevo
        (modo ENTRENAMIENTO); si se pasa, se aplica el existente (modo INFERENCIA).

    Retorna
    -------
    df_limpio : DataFrame sin filas inválidas ni anómalas.
    detector : el IsolationForest (nuevo o el mismo que se recibió).
    """
    df = _filtrar_invalidos(df)                           # aplica los filtros deterministas

    x = df[CANALES].dropna()                              # matriz sin NaN (IsolationForest no los acepta)

    if detector is None:                                  # --- modo ENTRENAMIENTO ---
        detector = IsolationForest(                       # instancia el detector
            n_estimators=100,                             # 100 árboles de aislamiento
            contamination=0.02,                           # asume ~2% de anomalías
            random_state=RANDOM_STATE,                    # reproducibilidad
        )
        detector.fit(x)                                   # ajusta sobre las intensidades filtradas

    etiquetas = detector.predict(x)                       # -1 = anomalía, 1 = normal
    idx_normales = x.index[etiquetas == 1]                # índices marcados como normales (== 1, no != -1)

    df_limpio = df.loc[idx_normales].copy()               # conserva solo las filas normales
    return df_limpio, detector                            # devuelve datos limpios + el detector
