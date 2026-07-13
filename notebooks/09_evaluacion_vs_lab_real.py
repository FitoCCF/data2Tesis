# ============================================================
# CELDA 9 — EVALUACIÓN vs LABORATORIO QUÍMICO REAL
# ============================================================
# Evalúa el pipeline contra el reporte de laboratorio químico REAL
# (assay_lab_raw.csv), no contra las ecuaciones genéricas del analizador.
# Alinea por ventanas de 12 h CENTRADAS en el timestamp del ensayo (el lag
# óptimo se determinó por escaneo de correlación). Excluye ventanas con
# muestras de calibración (fuga). Válido para cualquier período.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import sys
sys.argv = ['x']
exec(open('06_inferencia_hibrida.py').read().split('if __name__')[0])  # EstimadorHibrido

FECHA_INI, FECHA_FIN = '2025-01-01', '2026-01-01'
CANALES = ['n1fe', 'n2cu', 'n3zn', 'n4mo', 'n6sc']
MESES = {'ene':'01','feb':'02','mar':'03','abr':'04','may':'05','jun':'06',
         'jul':'07','ago':'08','sep':'09','oct':'10','nov':'11','dic':'12'}


def parse_fecha_es(s):
    """Convierte '01-ene-24 06:00:00' a Timestamp."""
    d, t = s.split(' ')
    dd, mm, yy = d.split('-')
    return pd.Timestamp(f'20{yy}-{MESES[mm]}-{dd} {t}')


# --- 1. Cargar laboratorio REAL y quedarse con assays distintos (cada 12 h) ---
lab = pd.read_csv('assay_lab_raw.csv')
lab['ts'] = lab['date_time'].apply(parse_fecha_es)
lab = lab.drop_duplicates(subset=['Cu', 'Mo', 'Fe'], keep='first').sort_values('ts')  # 1 valor por 12 h
lab = lab[(lab['ts'] >= FECHA_INI) & (lab['ts'] < FECHA_FIN)]

# --- 2. Intensidades del período + marca de fuga (muestras de calibración) ---
inten = pd.read_csv('data/processed/intensidades_cobre.csv', parse_dates=['date'])
inten['ts'] = pd.to_datetime(inten['date'].astype(str) + ' ' + inten['time'])
inten = inten[(inten['ts'] >= FECHA_INI) & (inten['ts'] < FECHA_FIN)].copy()
cal = pd.read_csv('data/processed/intensidad_cobre_24_completo_filtrado.csv')
cal_keys = set(map(tuple, cal[CANALES].round(3).values))
inten['es_calib'] = [tuple(np.round(r, 3)) in cal_keys for r in inten[CANALES].values]

# --- 3. Inferencia sobre cada lectura ---
est = EstimadorHibrido()
filas = []
for _, f in inten.iterrows():
    r = est.predecir({c: f[c] for c in CANALES})
    if r['leyes']:
        filas.append({'ts': f['ts'], 'es_calib': f['es_calib'], **r['leyes']})
pred = pd.DataFrame(filas).set_index('ts').sort_index()
pred = pred[~pred['es_calib']]                          # excluir lecturas de calibración (fuga)

# --- 4. Alinear en ventanas de 12 h CENTRADAS en el timestamp del lab ---
rows = []
for _, L in lab.iterrows():
    c = L['ts']                                         # centro de la ventana (lag 0)
    w = pred[(pred.index > c - pd.Timedelta('6h')) & (pred.index <= c + pd.Timedelta('6h'))]
    if len(w) == 0:
        continue
    rows.append({'fecha': c,
                 'pCu': w['pCu'].mean(), 'Cu': L['Cu'],
                 'pFe': w['pFe'].mean(), 'Fe': L['Fe'],
                 'pMo': w['pMo'].mean(), 'Mo': L['Mo']})
M = pd.DataFrame(rows)
print(f'Ventanas comparadas (sin fuga): {len(M)}')

# --- 5. Métricas contra laboratorio real ---
print(f'\n=== Pipeline vs LABORATORIO REAL — {FECHA_INI[:4]} ===')
print(f'{"elem":5s} {"R2":>8s} {"MAE":>8s} {"corr":>8s}')
for e, l, n in [('pCu', 'Cu', 'Cu'), ('pFe', 'Fe', 'Fe'), ('pMo', 'Mo', 'Mo')]:
    print(f'{n:5s} {r2_score(M[l], M[e]):8.3f} {mean_absolute_error(M[l], M[e]):8.3f} {M[e].corr(M[l]):8.3f}')

M.to_csv('data/processed/comparacion_vs_lab_real.csv', index=False)
print('\nGuardado en: data/processed/comparacion_vs_lab_real.csv')
