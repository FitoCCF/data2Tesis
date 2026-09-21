#!/usr/bin/env python3

# ============================================================
# src/pipeline/escalado.py — Etapa 3: corrección de asimetría + estandarización
# ============================================================
# Prepara las features del clustering: primero Yeo-Johnson (corrige asimetría),
# luego StandardScaler (media 0, varianza 1). El orden importa y se conserva
# igual en entrenamiento e inferencia.
#
# Ejecutable de forma independiente:
#   python -m src.pipeline.escalado --input data/processed/intensidad_cobre_24_orthogonalized.csv
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
        scaler.fit(X_power)                                # ajusta media/desviación
    X_scaled = scaler.transform(X_power)                  # aplica estandarización

    return X_scaled, power, scaler                        # devuelve matriz escalada + transformadores


# ============================================================
# CLI — ejecución independiente de la etapa
# ============================================================
def _main():
    import argparse
    import joblib

    from .config import DATA_PROCESSED, MODELS_DIR, ART_POWER, ART_SCALER, FEATS_CLUSTER

    ap = argparse.ArgumentParser(description="Etapa 3: escalado (Yeo-Johnson + StandardScaler)")
    ap.add_argument("--input", default=str(DATA_PROCESSED / "intensidad_cobre_24_orthogonalized.csv"),
                    help="CSV ortogonalizado de entrada (salida de la etapa 2)")
    ap.add_argument("--output", default=str(DATA_PROCESSED / "intensidad_cobre_24_scaled.csv"),
                    help="Ruta del CSV escalado de salida")
    ap.add_argument("--power-in", default=None, help="PowerTransformer .joblib ya entrenado (modo INFERENCIA)")
    ap.add_argument("--scaler-in", default=None, help="StandardScaler .joblib ya entrenado (modo INFERENCIA)")
    ap.add_argument("--power-out", default=str(MODELS_DIR / ART_POWER))
    ap.add_argument("--scaler-out", default=str(MODELS_DIR / ART_SCALER))
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    power = joblib.load(args.power_in) if args.power_in else None
    scaler = joblib.load(args.scaler_in) if args.scaler_in else None

    X_scaled, power, scaler = escalar(df, power, scaler)

    df_scaled = df.copy()
    df_scaled[FEATS_CLUSTER] = X_scaled
    df_scaled.to_csv(args.output, index=False)
    print(f"CSV escalado guardado en: {args.output}")

    if args.power_in is None:
        joblib.dump(power, args.power_out)
        print(f"PowerTransformer guardado en: {args.power_out}")
    if args.scaler_in is None:
        joblib.dump(scaler, args.scaler_out)
        print(f"StandardScaler guardado en: {args.scaler_out}")


if __name__ == "__main__":
    _main()
