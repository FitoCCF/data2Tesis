#!/usr/bin/env python3
# ============================================================
# CELDA 6 — ESTIMACIÓN DE LEYES POR LECTURA
# ============================================================
# Aplica el bundle de la etapa 5 a todas las lecturas del courier. La lógica
# vive en src/pipeline/inferencia.py.
#
#   courier_clusterizado.csv + models/modelos_locales_por_cluster.joblib
#       -> courier_estimado.csv (+ pFe_est, pCu_est, pMo_est, pZn_est)
#
#   pixi run python notebooks/06_inferencia_hibrida.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # independiente del CWD

import joblib
import pandas as pd

from src.pipeline.config import CLUSTERIZADO, ESTIMADO, MODELS_DIR, ART_REGRESION, TARGETS
from src.pipeline.inferencia import estimar

df = pd.read_csv(CLUSTERIZADO, parse_dates=['ts'])
bundle = joblib.load(MODELS_DIR / ART_REGRESION)
print("Ruteo:", {f"{l}/{g}": d for (l, g), d in bundle['tabla_ruteo'].items() if d == 'local'} or "todo global")

out = estimar(df, bundle)
out.to_csv(ESTIMADO, index=False)
print(f"{len(out)} lecturas estimadas -> {ESTIMADO}")

# --- Cordura: rango de la estimación vs. la ley de laboratorio, por periodo ---
est = [f"{t}_est" for t in TARGETS]
v = out[out['valida']]
print("\n== Estimación en lecturas válidas (percentiles 1/50/99) por periodo ==")
print(v.groupby('periodo')[est].quantile([.01, .5, .99]).round(3).to_string())
print("\n== Ley de laboratorio (BD, train) para comparar rangos ==")
print(out.loc[out['valida_ley'], TARGETS].quantile([.01, .5, .99]).round(3).to_string())
print("\nNaN en estimaciones:", out[est].isna().sum().to_dict())
