#!/usr/bin/env python3
# ============================================================
# CELDA 1c — LIMPIEZA GENERAL (tabla maestra)
# ============================================================
# Sobre courier_unificado.csv (etapa 1b) marca congeladas, anomalías
# (IsolationForest sobre fracciones de cierre, ajustado SOLO hasta
# FECHA_CORTE_TRAIN) y la validez final de cada lectura. No borra filas.
# La lógica vive en src/pipeline/limpieza.py.
#
#   courier_unificado.csv -> courier_limpio.csv  + models/anomaly_detector.joblib
#
#   pixi run python notebooks/01c_limpieza.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # independiente del CWD

import joblib
import pandas as pd

from src.pipeline.config import UNIFICADO, LIMPIO, MODELS_DIR, ART_ANOMALIAS, FECHA_CORTE_TRAIN
from src.pipeline.limpieza import limpiar

df = pd.read_csv(UNIFICADO, parse_dates=['ts'])
df, detector, resumen = limpiar(df, corte=FECHA_CORTE_TRAIN)
df.to_csv(LIMPIO, index=False)
joblib.dump(detector, MODELS_DIR / ART_ANOMALIAS)
print(f"{UNIFICADO.name} -> {LIMPIO.name}  (corte train: {FECHA_CORTE_TRAIN})")
for k, v in resumen.items():
    print(f"  {k}: {v}")
