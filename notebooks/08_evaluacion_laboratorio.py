#!/usr/bin/env python3
# ============================================================
# CELDA 8 — EVALUACIÓN CONTRA LABORATORIO DE ALTA FRECUENCIA
# ============================================================
# Evaluación DEFENDIBLE del pipeline completo sobre datos fuera de muestra:
# corre la inferencia sobre las intensidades del período de test, promedia las
# estimaciones en ventanas de 12 h (trailing) y las compara contra las leyes
# reales de laboratorio (que llegan cada 12 h).
#
# A diferencia de V1: ya NO usa exec(open('06_inferencia_hibrida.py').read())
# para "importar" la clase (eso rompía si el CWD no era exactamente
# notebooks/, y triplicaba la lógica). EstimadorHibrido se importa normal
# desde src.pipeline.inferencia.
#
# REQUIERE (no vienen en el repo, hay que generarlos/exportarlos primero):
#   - data/processed/intensidades_cobre_test.csv (o el split que uses como test)
#   - data/processed/AssayLab_combined_limpio.csv (ver scripts/assay_lab_test.py)
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # independiente del CWD

import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

from src.pipeline.config import CANALES, DATA_PROCESSED
from src.pipeline.inferencia import EstimadorHibrido

VENTANA = '12h'                                         # ventana de agregación = frecuencia del lab

# --- 1. Correr el modelo sobre cada lectura de intensidad del test ---
est = EstimadorHibrido()                                # carga todos los modelos congelados
inten = pd.read_csv(DATA_PROCESSED / 'intensidades_cobre_test.csv')  # intensidades nuevas de campo
inten['ts'] = pd.to_datetime(inten['date'] + ' ' + inten['time'])  # timestamp por lectura

filas = []                                              # estimaciones por lectura
for _, f in inten.iterrows():
    r = est.predecir({c: f[c] for c in CANALES})        # estima leyes de esta lectura
    if r['leyes']:                                      # si pasó la compuerta de calidad
        filas.append({'ts': f['ts'], **r['leyes']})     # guarda ts + leyes estimadas
pred = pd.DataFrame(filas).set_index('ts').sort_index() # serie temporal de estimaciones

# --- 2. Cargar las leyes de laboratorio del mismo período ---
lab = pd.read_csv(DATA_PROCESSED / 'AssayLab_combined_limpio.csv', parse_dates=['fecha_hora'])
lab = lab[(lab[['cu', 'fe', 'mo']] > 0).all(axis=1)]    # descarta filas con 0 (dato inválido)
lab = lab[(lab['fecha_hora'] >= pred.index.min()) &
          (lab['fecha_hora'] <= pred.index.max())].copy()  # recorta al período de test

# --- 3. Alinear: para cada punto de lab, promediar el modelo en las 12 h previas ---
comparacion = []
for _, L in lab.iterrows():
    t = L['fecha_hora']                                 # instante del ensayo de lab
    w = pred[(pred.index > t - pd.Timedelta(VENTANA)) & (pred.index <= t)]  # ventana previa
    if len(w) == 0:                                     # sin intensidades en esa ventana
        continue
    comparacion.append({
        'fecha_hora': t, 'n_lecturas': len(w),          # nº de intensidades promediadas
        'pCu_est': w['pCu'].mean(), 'cu_lab': L['cu'],   # Cu estimado promedio vs lab
        'pFe_est': w['pFe'].mean(), 'fe_lab': L['fe'],   # Fe estimado promedio vs lab
        'pMo_est': w['pMo'].mean(), 'mo_lab': L['mo'],   # Mo estimado promedio vs lab
    })
M = pd.DataFrame(comparacion)                            # tabla comparativa
print(f'Puntos de laboratorio comparados: {len(M)}')
print(f'Lecturas de intensidad promediadas por ventana (media): {M["n_lecturas"].mean():.1f}')

# --- 4. Métricas por elemento ---
print('\n=== Desempeño del pipeline vs laboratorio (fuera de muestra) ===')
print(f'{"elem":5s} {"R2":>8s} {"MAE":>8s} {"correlación":>13s}')
for est_c, lab_c, nombre in [('pCu_est', 'cu_lab', 'Cu'),
                             ('pFe_est', 'fe_lab', 'Fe'),
                             ('pMo_est', 'mo_lab', 'Mo')]:
    r2 = r2_score(M[lab_c], M[est_c])                    # R2
    mae = mean_absolute_error(M[lab_c], M[est_c])        # error absoluto medio
    corr = M[est_c].corr(M[lab_c])                       # correlación lineal
    print(f'{nombre:5s} {r2:8.3f} {mae:8.3f} {corr:13.3f}')

# --- 5. Guardar la tabla de comparación (para graficar / auditar) ---
output_path = DATA_PROCESSED / 'comparacion_modelo_vs_lab.csv'
M.to_csv(output_path, index=False)
print(f'\nComparación guardada en: {output_path}')
