#!/usr/bin/env python3
# ============================================================
# CELDA 9b — EVALUACIÓN vs LABORATORIO QUÍMICO REAL
# ============================================================
# Evalúa el pipeline contra el reporte de laboratorio químico REAL
# (assay_lab_raw.csv), no contra las ecuaciones genéricas del analizador.
# Alinea por ventanas de 12 h CENTRADAS en el timestamp del ensayo. Excluye
# ventanas con muestras de calibración (fuga).
#
# A diferencia de V1: toda la lógica (parseo de fecha en español, ventana
# centrada, exclusión de fuga, métricas) vive en src/evaluacion/evaluar.py -
# esta celda solo la invoca. Ese mismo módulo tiene su propio CLI:
#   python -m src.evaluacion.evaluar --desde 2025-01-01 --hasta 2026-01-01
#
# REQUIERE (no viene en el repo, hay que exportarlo desde la BD/analizador):
#   - data/raw/assay_lab_raw.csv
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # independiente del CWD

import pandas as pd

from src.pipeline.config import DATA_PROCESSED, DATA_RAW
from src.evaluacion.evaluar import cargar_lab_real, evaluar_periodo, reportar_metricas

FECHA_INI, FECHA_FIN = '2025-01-01', '2026-01-01'

inten = pd.read_csv(DATA_PROCESSED / 'intensidades_cobre.csv')
lab = cargar_lab_real(DATA_RAW / 'assay_lab_raw.csv')
cal = pd.read_csv(DATA_PROCESSED / 'intensidad_cobre_24_completo_filtrado.csv')

M = evaluar_periodo(inten, lab, cal, FECHA_INI, FECHA_FIN)
print(f'Ventanas comparadas (sin fuga): {len(M)}')

print(f'\n=== Pipeline vs LABORATORIO REAL — {FECHA_INI[:4]} ===')
reportar_metricas(M)

output_path = DATA_PROCESSED / 'comparacion_vs_lab_real.csv'
M.to_csv(output_path, index=False)
print(f'\nGuardado en: {output_path}')
