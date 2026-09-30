#!/usr/bin/env python3
# ============================================================
# CELDA 4 — GRUPOS DE MINERAL (GMM)
# ============================================================
# 1. Diagnóstico del número de grupos sobre train (silhouette + estabilidad).
# 2. GMM con K_GRUPOS, ajustado SOLO con train & valida, aplicado a todo.
# 3. Caracterización de cada grupo contra el laboratorio (BD y compósito).
# 4. Producción (septiembre): a qué grupo pertenece cada lectura, por semana.
# La lógica vive en src/pipeline/clustering.py.
#
#   courier_escalado.csv -> courier_clusterizado.csv + models/gmm_model.joblib
#
#   pixi run python notebooks/04_clusterizado.py
#   pixi run python notebooks/04_clusterizado.py --sin-diagnostico
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # independiente del CWD

import argparse

import joblib
import pandas as pd

from src.pipeline.config import ESCALADO, CLUSTERIZADO, MODELS_DIR, ART_GMM, LIMPIO_COMPOSITO, K_GRUPOS
from src.pipeline.clustering import diagnostico_k, entrenar, asignar, caracterizar

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument('--sin-diagnostico', action='store_true')
_ap.add_argument('--k', type=int, default=K_GRUPOS)
args, _ = _ap.parse_known_args()

df = pd.read_csv(ESCALADO, parse_dates=['ts'])
comp = pd.read_csv(LIMPIO_COMPOSITO, parse_dates=['ts'])

if not args.sin_diagnostico:
    print("== Diagnóstico del número de grupos (train) ==")
    print(diagnostico_k(df).to_string(index=False))

gmm = entrenar(df, args.k)
df = asignar(df, gmm)
df.to_csv(CLUSTERIZADO, index=False)
joblib.dump(gmm, MODELS_DIR / ART_GMM)
print(f"\n== k={args.k}: grupos {gmm.nombres_} ==")
print(caracterizar(df, comp).to_string())

v = df[df['valida'] & (df['fuente'] != 'bd')]
racha = v['grupo_id'].ne(v['grupo_id'].shift()).cumsum()
print(f"\nPermanencia en un grupo (lecturas seguidas): mediana {v.groupby(racha).size().median():.0f}, "
      f"p90 {v.groupby(racha).size().quantile(.9):.0f}  (1 lectura ≈ 21 min)")

p = v[v['periodo'] == 'produccion'].copy()
p['semana'] = p['ts'].dt.to_period('W-SUN').dt.start_time.dt.date
c = comp[comp['ts'] >= p['ts'].min()].copy()
c['semana'] = c['ts'].dt.to_period('W-SUN').dt.start_time.dt.date
print("\n== Producción: % de lecturas por grupo y semana, junto al compósito de laboratorio ==")
print(pd.concat([pd.crosstab(p['semana'], p['grupo'], normalize='index').mul(100).round(0),
                 c.groupby('semana')[['cu', 'fe', 'mo', 'ins']].mean().round(2).add_prefix('comp_')],
                axis=1).to_string())
