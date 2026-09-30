#!/usr/bin/env python3
# ============================================================
# CELDA 2 — FEATURES ROBUSTAS A LA DERIVA
# ============================================================
# Reemplaza a la ortogonalización contra n6sc. Agrega a la tabla maestra:
#   clustering  lr_fe_cu, lr_zn_cu, lr_mo_cu   log-cocientes contra Cu
#   regresión   n1fe_f..n4mo_f, logSumI        fracciones de cierre + magnitud
# Transformación fija fila a fila: no ajusta nada, no guarda artefacto.
# La lógica vive en src/pipeline/features.py.
#
#   courier_limpio.csv -> courier_features.csv
#
#   pixi run python notebooks/02_features.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # independiente del CWD

import pandas as pd

from src.pipeline.config import LIMPIO, FEATURES, FEATS_CLUSTER, FEATS_REGRESION
from src.pipeline.features import agregar_features

df = agregar_features(pd.read_csv(LIMPIO, parse_dates=['ts']))
df.to_csv(FEATURES, index=False)
feats = FEATS_CLUSTER + FEATS_REGRESION
print(f"{LIMPIO.name} -> {FEATURES.name}  ({len(df)} filas)")
print(f"  features con NaN: {int(df[feats].isna().any(axis=1).sum())}")
print(df.loc[df['valida'], feats].describe().round(4).to_string())
