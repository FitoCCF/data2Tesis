#!/usr/bin/env python3
# ============================================================
# CELDA 3 — CORRECCIÓN DE DILUCIÓN + ESCALADO (ajustados solo con train)
# ============================================================
# A los 3 log-cocientes de la etapa 2 les resta la parte explicada por la
# señal de dilución 'dil' (-> <feature>_dc) y los estandariza (-> <feature>_z).
# Ambos pasos se ajustan con periodo == 'train' & valida y se aplican a todo.
# Detalle y cifras en la cabecera de src/pipeline/escalado.py.
#
#   courier_features.csv -> courier_escalado.csv + models/scaler.joblib
#
#   pixi run python notebooks/03_escalado.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # independiente del CWD

import joblib
import pandas as pd

from src.pipeline.config import FEATURES, ESCALADO, MODELS_DIR, ART_SCALER
from src.pipeline.escalado import escalar, FEATS_ESCALADAS

df, artefacto = escalar(pd.read_csv(FEATURES, parse_dates=['ts']))
df.to_csv(ESCALADO, index=False)
joblib.dump(artefacto, MODELS_DIR / ART_SCALER)
print(f"{FEATURES.name} -> {ESCALADO.name}")
print("b_dilucion:", {k: round(v, 4) for k, v in artefacto['b_dilucion'].items()})
print("media y desviación por período (lecturas válidas); train debe dar 0 y 1:")
print(df.loc[df['valida']].groupby('periodo')[FEATS_ESCALADAS].agg(['mean', 'std']).round(3).to_string())
