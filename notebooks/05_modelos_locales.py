#!/usr/bin/env python3
# ============================================================
# CELDA 5 — MODELOS POR GRUPO DE MINERAL + TABLA DE RUTEO
# ============================================================
# Un modelo global por ley y uno local por (ley, grupo) cuando el grupo tiene
# N >= N_MIN_LOCAL. Validación walk-forward con purga y embargo (split_purgado),
# folds compartidos; global y local comparados sobre los mismos puntos; el local
# se rutea solo si supera al global Y al control aleatorio (percentil
# PCTL_CONTROL de la mejora con grupos al azar del mismo tamaño).
# La lógica vive en src/pipeline/modelos.py.
#
#   courier_clusterizado.csv -> models/modelos_locales_por_cluster.joblib
#
#   pixi run python notebooks/05_modelos_locales.py
#   pixi run python notebooks/05_modelos_locales.py --n-control 20   (rápido, solo para probar)
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # independiente del CWD

import argparse

import joblib
import pandas as pd

from src.pipeline.config import CLUSTERIZADO, MODELS_DIR, ART_REGRESION, N_CONTROL, PCTL_CONTROL
from src.pipeline.modelos import entrenar_modelos_locales, evaluar_test, conjunto_supervisado

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument('--n-control', type=int, default=N_CONTROL)
args, _ = _ap.parse_known_args()

df = pd.read_csv(CLUSTERIZADO, parse_dates=['ts'])

# --- 1. Folds (una sola vez, sobre train con ley válida) ---
bundle = entrenar_modelos_locales(df, args.n_control)
s = conjunto_supervisado(df)
print(f"== Folds walk-forward sobre {len(s)} muestras de train ==")
for k, (tr, te) in enumerate(bundle['folds']):
    print(f"  fold {k}: train {len(tr):3d} hasta {s.ts.iloc[tr].max():%Y-%m-%d}  |  "
          f"prueba {len(te):3d}  {s.ts.iloc[te].min():%Y-%m-%d} -> {s.ts.iloc[te].max():%Y-%m-%d}  |  "
          f"purgadas {int(te.min()) - len(tr)}")

# --- 2. Validación por celda y ruteo ---
v = bundle['validacion'].copy()
v['ruteo'] = [bundle['tabla_ruteo'].get((l, g), '') for l, g in zip(v['ley'], v['grupo'])]
print(f"\n== Validación fuera de fold (control: {args.n_control} particiones, percentil {PCTL_CONTROL}) ==")
print(v.round(3).to_string(index=False))

# --- 3. Chequeo en test (jun-ago, pocas muestras de la BD) ---
print("\n== Test (muestras de laboratorio de jun-ago; control de cordura, no evaluación) ==")
print(evaluar_test(bundle, df).round(3).to_string(index=False))

# --- 4. Guardar ---
MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(bundle, MODELS_DIR / ART_REGRESION)
print(f"\nBundle guardado en: {MODELS_DIR / ART_REGRESION}")
