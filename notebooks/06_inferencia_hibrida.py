#!/usr/bin/env python3
# ============================================================
# CELDA 6 — INFERENCIA HÍBRIDA (estimación de leyes en producción)
# ============================================================
# Ya NO reimplementa EstimadorHibrido (a diferencia de V1): lo importa de
# src.pipeline.inferencia, la MISMA clase que usa src/evaluacion/evaluar.py y
# el CLI `python -m src.pipeline.inferencia`.
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # independiente del CWD

import pandas as pd

from src.pipeline.config import CANALES, DATA_PROCESSED
from src.pipeline.inferencia import EstimadorHibrido

est = EstimadorHibrido()  # carga todos los modelos (etapas 1-5)

print('Tabla de ruteo cargada (local vs global):')
for cl in sorted({cl for (_, cl) in est.tabla_ruteo}):
    print(f'  cluster {cl}:', {ley: est.tabla_ruteo[(ley, cl)] for ley in est.targets})

print('\nSimulación de lecturas de campo:')
stream = pd.read_csv(DATA_PROCESSED / 'intensidades_cobre.csv').head(500)
mostradas = 0
for _, fila in stream.iterrows():
    ints = {c: fila[c] for c in CANALES}
    res = est.predecir(ints)
    if res['leyes'] and not res['alertas'] and mostradas < 4:
        le = res['leyes']
        print(f"\n  Lectura {ints}")
        print(f"  -> cluster {res['cluster']}  |  ruteo {res['ruteo']}")
        print(f"  -> pFe={le['pFe']:.2f}  pCu={le['pCu']:.2f}  pMo={le['pMo']:.2f}  pZn={le['pZn']:.3f}")
        mostradas += 1
    if mostradas >= 4:
        break
