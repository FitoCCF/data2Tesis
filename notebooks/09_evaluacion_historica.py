# ============================================================
# CELDA 9 — EVALUACIÓN HISTÓRICA (cualquier período, sin fuga)
# ============================================================
# Generaliza la evaluación de la celda 8 a un período arbitrario (p. ej. todo
# 2025) usando las intensidades del stream + las leyes de laboratorio de alta
# frecuencia. Maneja la FUGA DE DATOS: excluye las ventanas de 12 h que
# contengan muestras de calibración que la regresión ya vio en entrenamiento.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import sys
sys.argv = ['x']
exec(open('06_inferencia_hibrida.py').read().split('if __name__')[0])  # carga EstimadorHibrido

# --- Parámetros del período a evaluar ---
FECHA_INI = '2025-01-01'
FECHA_FIN = '2026-01-01'
VENTANA = '12h'
CANALES = ['n1fe', 'n2cu', 'n3zn', 'n4mo', 'n6sc']

est = EstimadorHibrido()                                # modelos congelados

# --- 1. Intensidades del período ---
inten = pd.read_csv('data/processed/intensidades_cobre.csv', parse_dates=['date'])
inten['ts'] = pd.to_datetime(inten['date'].astype(str) + ' ' + inten['time'])
inten = inten[(inten['ts'] >= FECHA_INI) & (inten['ts'] < FECHA_FIN)].copy()

# --- 2. Marcar muestras de calibración (las que la regresión ya vio = fuga) ---
cal = pd.read_csv('data/processed/intensidad_cobre_24_completo_filtrado.csv')
cal_keys = set(map(tuple, cal[CANALES].round(3).values))
inten['es_calib'] = [tuple(np.round(r, 3)) in cal_keys for r in inten[CANALES].values]
print(f'Intensidades en período: {len(inten)}  | de calibración (fuga): {inten["es_calib"].sum()}')

# --- 3. Inferencia sobre cada lectura ---
filas = []
for _, f in inten.iterrows():
    r = est.predecir({c: f[c] for c in CANALES})
    if r['leyes']:
        filas.append({'ts': f['ts'], 'es_calib': f['es_calib'], **r['leyes']})
pred = pd.DataFrame(filas).set_index('ts').sort_index()

# --- 4. Leyes de laboratorio del período ---
lab = pd.read_csv('AssayLab_combined_limpio.csv', parse_dates=['fecha_hora'])
lab = lab[(lab[['cu', 'fe', 'mo']] > 0).all(axis=1)]    # descarta ceros inválidos
lab = lab[(lab['fecha_hora'] >= FECHA_INI) & (lab['fecha_hora'] < FECHA_FIN)]

# --- 5. Alinear en ventanas de 12 h y marcar ventanas con fuga ---
rows = []
for _, L in lab.iterrows():
    t = L['fecha_hora']
    w = pred[(pred.index > t - pd.Timedelta(VENTANA)) & (pred.index <= t)]
    if len(w) == 0:
        continue
    rows.append({'fecha_hora': t, 'n': len(w), 'tiene_calib': bool(w['es_calib'].any()),
                 'pCu': w['pCu'].mean(), 'cu': L['cu'],
                 'pFe': w['pFe'].mean(), 'fe': L['fe'],
                 'pMo': w['pMo'].mean(), 'mo': L['mo']})
M = pd.DataFrame(rows)
print(f'Ventanas: {len(M)}  | con fuga (excluidas del cálculo limpio): {M["tiene_calib"].sum()}')

# --- 6. Métricas: sobre las ventanas LIMPIAS (defendibles) ---
limpio = M[~M['tiene_calib']]
print(f'\n=== Desempeño 2025 sobre ventanas limpias (n={len(limpio)}) ===')
print(f'{"elem":5s} {"R2":>8s} {"MAE":>8s} {"corr":>8s}')
for e, l, n in [('pCu', 'cu', 'Cu'), ('pFe', 'fe', 'Fe'), ('pMo', 'mo', 'Mo')]:
    print(f'{n:5s} {r2_score(limpio[l], limpio[e]):8.3f} '
          f'{mean_absolute_error(limpio[l], limpio[e]):8.3f} {limpio[e].corr(limpio[l]):8.3f}')

M.to_csv('data/processed/comparacion_2025_vs_lab.csv', index=False)
print('\nComparación guardada en: data/processed/comparacion_2025_vs_lab.csv')
