#!/usr/bin/env python3
# ============================================================
# CELDA 9 — MODELO vs MUESTRAS PUNTUALES DE LABORATORIO (BD), SIN Y CON KALMAN
# ============================================================
# Dos comparaciones contra las muestras de la BD (las 4 leyes, Zn incluido):
#
#   A. Historia 2023 -> may-2026: predicciones fuera de fold del walk-forward
#      de la etapa 5 (modelo global). Es la única referencia sin fuga para el
#      periodo anterior al PI continuo (2025-07). Sin Kalman: el corrector
#      necesita compósitos alineados con lecturas del PI.
#   B. Test jun-ago 2026: la estimación de la lectura exacta de cada muestra
#      (etapa 6) contra su ley; Fe/Cu/Mo también corregidos con el Kalman de
#      la etapa 8 (mismo q_por_dia, sesgo de los compósitos ya cerrados).
#
#   models/modelos_locales_por_cluster.joblib (oof) + courier_estimado.csv
#   + data/final/eval_composito*.csv (etapa 8)
#       -> data/final/eval_bd_metricas.csv
#       -> reports/fig_09_historia_walkforward.png, reports/fig_09_bd_test.png
#
#   pixi run python notebooks/09_evaluacion_vs_lab_real.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # independiente del CWD

import joblib
import pandas as pd

from src.pipeline.config import ESTIMADO, MODELS_DIR, ART_REGRESION, DATA_FINAL, REPORTS_DIR, TARGETS
from src.pipeline.modelos import conjunto_supervisado
from src.evaluacion.evaluar import LEYES_COMPOSITO, metricas, kalman_en_instantes
from src.evaluacion.figuras import figura_tendencia

bundle = joblib.load(MODELS_DIR / ART_REGRESION)
est = pd.read_csv(ESTIMADO, parse_dates=['ts'])
A = pd.read_csv(DATA_FINAL / 'eval_composito.csv', parse_dates=['ts'])          # etapa 8
M8 = pd.read_csv(DATA_FINAL / 'eval_composito_metricas.csv')
q = M8.dropna(subset=['q_por_dia']).groupby('ley')['q_por_dia'].first().to_dict()  # q elegido en train

filas = []

# --- A. Historia walk-forward (fuera de fold) ---
H = bundle['oof'].dropna(subset=[f'{t}_oof' for t in TARGETS], how='all').copy()
for t in TARGETS:
    H[f'{t}_est'] = H[f'{t}_oof']
    filas.append({'comparacion': 'historia walk-forward', 'ley': t, 'variante': 'modelo',
                  **metricas(H[t], H[f'{t}_est'])})
print(f"== A. Historia walk-forward: {H.ts.min():%Y-%m-%d} -> {H.ts.max():%Y-%m-%d} ==")

# --- B. Test: estimación en la lectura exacta de cada muestra ---
T = conjunto_supervisado(est, 'test')
for t in TARGETS:
    filas.append({'comparacion': 'test (BD)', 'ley': t, 'variante': 'modelo',
                  **metricas(T[t], T[f'{t}_est'])})
    if t in LEYES_COMPOSITO:
        T[f'{t}_kal'] = kalman_en_instantes(A, t, q[t], T)
        filas.append({'comparacion': 'test (BD)', 'ley': t, 'variante': 'modelo + Kalman',
                      'q_por_dia': q[t], **metricas(T[t], T[f'{t}_kal'])})
print(f"== B. Test BD: {len(T)} muestras, {T.ts.min():%Y-%m-%d} -> {T.ts.max():%Y-%m-%d} ==")

M = pd.DataFrame(filas)
pd.set_option('display.width', 200)
print("\n== Métricas vs muestras de laboratorio (est - lab para el sesgo) ==")
print(M.round(3).to_string(index=False))
M.to_csv(DATA_FINAL / 'eval_bd_metricas.csv', index=False)


def _txt(comparacion):
    """ley -> {'modelo': métricas, 'kalman': métricas} para el recuadro de la figura."""
    d = {t: {} for t in TARGETS}
    for _, r in M[M.comparacion == comparacion].iterrows():
        d[r['ley']]['kalman' if r['variante'] == 'modelo + Kalman' else 'modelo'] = r.to_dict()
    return d


ruta = REPORTS_DIR / 'fig_09_historia_walkforward.png'
figura_tendencia(H, TARGETS, "Estimación del modelo vs muestras de laboratorio químico (BD) — "
                 "walk-forward 2023–2026, fuera de fold", ruta, _txt('historia walk-forward'),
                 con_kalman=False, marcador=True, max_hueco="21D")
print(f"\nFigura: {ruta}")
ruta = REPORTS_DIR / 'fig_09_bd_test.png'
figura_tendencia(T, TARGETS, "Estimación del modelo vs muestras de laboratorio químico (BD) — test jun–ago 2026",
                 ruta, _txt('test (BD)'), marcador=True, max_hueco="10D")
print(f"Figura: {ruta}")
