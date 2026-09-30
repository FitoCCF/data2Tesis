#!/usr/bin/env python3
# ============================================================
# src/pipeline/escalado.py — Etapa 3: corrección de dilución + estandarización
# ============================================================
# Sobre los 3 log-cocientes de la etapa 2, dos pasos AJUSTADOS SOLO con las
# lecturas válidas del período 'train' y aplicados a todas:
#
#   1. Corrección de dilución: f_dc = f - b_f · dil, con b_f la pendiente de
#      f contra la señal de dilución 'dil' (etapa 2), solo para las features
#      de config.DIL_CORREGIR (hoy lr_fe_cu); las demás b_f = 0. El cociente no cancela
#      del todo el agua de la pulpa (ver config, DIL_HALFLIFE); esto quita la
#      parte que 'dil' logra ver. dil es un proxy débil (corr 0.55 con pSol):
#      la corrección es parcial, no total.
#   2. StandardScaler -> columnas <feature>_z (0 = promedio de train).
#
# Sin Yeo-Johnson (2026-09-30): el log-cociente ya deja las features casi
# simétricas (asimetría 0.02-0.78 en train). Yeo-Johnson solo mejoraba
# lr_zn_cu y con lambda = -6.95, una potencia extrema que amplifica cualquier
# valor de producción fuera del rango de entrenamiento.
#
# El artefacto (models/scaler.joblib) es un dict {"b_dilucion", "scaler"}.
#
# Ejecutable de forma independiente:
#   python -m src.pipeline.escalado                              # entrena y escala
#   python -m src.pipeline.escalado --artifact-in models/scaler.joblib --input nuevo.csv
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import FEATS_CLUSTER, DIL_CORREGIR

FEATS_CORREGIDAS = [f"{c}_dc" for c in FEATS_CLUSTER]      # corregidas por dilución
FEATS_ESCALADAS = [f"{c}_z" for c in FEATS_CLUSTER]        # corregidas y estandarizadas (entrada del GMM)


def escalar(df: pd.DataFrame, artefacto: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Agrega <feature>_dc y <feature>_z.

    artefacto=None -> ENTRENAMIENTO: ajusta b y el scaler con periodo == 'train' & valida.
    Con artefacto  -> INFERENCIA: los aplica tal cual.
    """
    if artefacto is None:
        ajuste = df[(df["periodo"] == "train") & df["valida"]]
        b = {f: float(np.polyfit(ajuste["dil"], ajuste[f], 1)[0]) if f in DIL_CORREGIR else 0.0
             for f in FEATS_CLUSTER}
        corr = pd.DataFrame({f"{f}_dc": ajuste[f] - b[f] * ajuste["dil"] for f in FEATS_CLUSTER})
        artefacto = {"b_dilucion": b, "scaler": StandardScaler().fit(corr[FEATS_CORREGIDAS])}
    df = df.copy()
    for f in FEATS_CLUSTER:
        df[f"{f}_dc"] = df[f] - artefacto["b_dilucion"][f] * df["dil"]
    df[FEATS_ESCALADAS] = artefacto["scaler"].transform(df[FEATS_CORREGIDAS])
    return df, artefacto


def _main():
    import argparse
    import joblib

    from .config import FEATURES, ESCALADO, MODELS_DIR, ART_SCALER

    ap = argparse.ArgumentParser(description="Etapa 3: corrección de dilución + StandardScaler")
    ap.add_argument("--input", default=str(FEATURES))
    ap.add_argument("--output", default=str(ESCALADO))
    ap.add_argument("--artifact-in", default=None, help="Artefacto ya ajustado (modo INFERENCIA)")
    ap.add_argument("--artifact-out", default=str(MODELS_DIR / ART_SCALER))
    args = ap.parse_args()

    df = pd.read_csv(args.input, parse_dates=["ts"])
    artefacto = joblib.load(args.artifact_in) if args.artifact_in else None
    df, artefacto = escalar(df, artefacto)
    df.to_csv(args.output, index=False)
    print(f"{args.input} -> {args.output}")
    print("b_dilucion:", {k: round(v, 4) for k, v in artefacto["b_dilucion"].items()})
    if args.artifact_in is None:
        joblib.dump(artefacto, args.artifact_out)
        print(f"Artefacto guardado en: {args.artifact_out}")


if __name__ == "__main__":
    _main()
