# ============================================================
# CELDA 8 — EVALUACIÓN CONTRA LABORATORIO DE ALTA FRECUENCIA
# ============================================================
# Evaluación DEFENDIBLE del pipeline completo sobre datos fuera de muestra:
# corre la inferencia (celda 6) sobre las intensidades del período de test,
# promedia las estimaciones en ventanas de 12 h y las compara contra las leyes
# reales de laboratorio (que llegan cada 12 h). Este período (2026-06-27 en
# adelante) es POSTERIOR a todo el entrenamiento -> sin fuga de datos.
#
# El laboratorio trae cu, fe, mo (no zinc), así que se validan pFe, pCu, pMo.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

# Reutiliza la clase de inferencia de la celda 6 (debe estar en el mismo proyecto)
from importlib import import_module
import sys
sys.argv = ['x']
exec(open('06_inferencia_hibrida.py').read().split('if __name__')[0])  # carga EstimadorHibrido

VENTANA = '12h'                                         # ventana de agregación = frecuencia del lab
CANALES = ['n1fe', 'n2cu', 'n3zn', 'n4mo', 'n6sc']

# --- 1. Correr el modelo sobre cada lectura de intensidad del test ---
est = EstimadorHibrido()                                # carga todos los modelos congelados
inten = pd.read_csv('data/processed/intensidades_test_cobre.csv')  # intensidades nuevas de campo
inten['ts'] = pd.to_datetime(inten['date'] + ' ' + inten['time'])  # timestamp por lectura

filas = []                                              # estimaciones por lectura
for _, f in inten.iterrows():
    r = est.predecir({c: f[c] for c in CANALES})        # estima leyes de esta lectura
    if r['leyes']:                                      # si pasó la compuerta de calidad
        filas.append({'ts': f['ts'], **r['leyes']})     # guarda ts + leyes estimadas
pred = pd.DataFrame(filas).set_index('ts').sort_index() # serie temporal de estimaciones

# --- 2. Cargar las leyes de laboratorio del mismo período ---
lab = pd.read_csv('AssayLab_combined_limpio.csv', parse_dates=['fecha_hora'])  # ajusta la ruta si hace falta
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
M.to_csv('data/processed/comparacion_modelo_vs_lab.csv', index=False)
print('\nComparación guardada en: data/processed/comparacion_modelo_vs_lab.csv')
