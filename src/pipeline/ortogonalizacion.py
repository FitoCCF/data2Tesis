# ============================================================
# src/pipeline/ortogonalizacion.py — Etapa 2: corrección por dilución (n6sc)
# ============================================================
# Descuenta de cada canal de metal la componente explicada por n6sc (el sólido
# en la muestra / dilución). El residuo es la señal del metal libre de dilución.
# Separa entrenar (ajustar las regresiones) de aplicar (usarlas congeladas).
# ============================================================

import pandas as pd                                       # DataFrames
from sklearn.linear_model import LinearRegression         # regresión lineal simple

from .config import METALES                               # canales a ortogonalizar


def ortogonalizar(df: pd.DataFrame, regresiones: dict | None = None):
    """Ortogonaliza cada metal respecto a n6sc, conservando n6sc.

    Parámetros
    ----------
    df : DataFrame con las intensidades crudas (incluye n6sc).
    regresiones : dict {metal -> LinearRegression} ya ajustado. Si es None se
        entrena (modo ENTRENAMIENTO); si se pasa, se aplica (modo INFERENCIA).

    Retorna
    -------
    df_ortho : DataFrame con las columnas '<metal>_ortho' agregadas (y n6sc intacto).
    regresiones : dict de modelos de regresión (nuevo o el recibido).
    """
    X = df[["n6sc"]].values                               # variable independiente = sólido/dilución
    df_ortho = df.copy()                                  # copia donde se agregan las columnas _ortho

    if regresiones is None:                               # --- modo ENTRENAMIENTO ---
        regresiones = {}                                  # diccionario a llenar
        for metal in METALES:                             # por cada metal
            if df[metal].isna().all():                    # si la columna está vacía
                continue                                  # la salta
            modelo = LinearRegression().fit(X, df[metal].values)  # ajusta metal ~ n6sc
            regresiones[metal] = modelo                   # guarda el modelo (coeficientes congelados)

    for metal, modelo in regresiones.items():             # --- aplica (entrenamiento e inferencia igual) ---
        residuo = df[metal].values - modelo.predict(X)    # residuo = metal - parte explicada por n6sc
        df_ortho[f"{metal}_ortho"] = residuo              # nueva columna ortogonalizada

    return df_ortho, regresiones                          # devuelve datos + regresiones


def solo_features(df: pd.DataFrame, regresiones: dict) -> pd.DataFrame:
    """Versión ligera para inferencia: aplica regresiones ya entrenadas y
    devuelve solo las columnas ortogonalizadas + n6sc. No entrena nada."""
    X = df[["n6sc"]].values                               # referencia de dilución
    salida = pd.DataFrame(index=df.index)                 # DataFrame de features
    for metal, modelo in regresiones.items():             # por cada metal con modelo
        salida[f"{metal}_ortho"] = df[metal].values - modelo.predict(X)  # residuo
    salida["n6sc"] = df["n6sc"].values                    # conserva n6sc (feature de la regresión)
    return salida                                         # devuelve solo las features
