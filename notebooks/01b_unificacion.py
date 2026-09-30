#!/usr/bin/env python3
# ============================================================
# CELDA 1b — UNIFICACIÓN (tabla maestra del courier)
# ============================================================
# Une la BD limpia (muestreo con leyes) y las lecturas del PI (etapa 1a) en
# una sola tabla, una fila por lectura real del courier, con columna 'fuente'
# (pi / ambos / bd). Las leyes se pegan por calce de VALOR de los 4 metales,
# no por hora. La lógica vive en src/pipeline/unificacion.py.
#
#   courier_bd_limpio.csv + courier_pi_lecturas.csv -> courier_unificado.csv
#
#   pixi run python notebooks/01b_unificacion.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # independiente del CWD

import pandas as pd

from src.pipeline.config import LIMPIO_COURIER_BD, LIMPIO_COURIER_PI, UNIFICADO
from src.pipeline.unificacion import unificar

bd = pd.read_csv(LIMPIO_COURIER_BD, parse_dates=['ts'])
pi = pd.read_csv(LIMPIO_COURIER_PI, parse_dates=['ts'])
u, resumen = unificar(bd, pi)
u.to_csv(UNIFICADO, index=False)
print(f"{LIMPIO_COURIER_BD.name} + {LIMPIO_COURIER_PI.name} -> {UNIFICADO.name}")
for k, v in resumen.items():
    print(f"  {k}: {v}")
