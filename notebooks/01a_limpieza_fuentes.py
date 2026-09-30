#!/usr/bin/env python3
# ============================================================
# CELDA 1a — LIMPIEZA POR FUENTE
# ============================================================
# Limpia cada crudo de la etapa 0 con sus propias reglas deterministas, antes
# de unificarlos (1b). La lógica vive en src/pipeline/limpieza_fuentes.py.
#
#   courier_pi.csv   -> courier_pi_lecturas.csv  una fila por lectura real
#   courier_bd.csv   -> courier_bd_limpio.csv    leyes en 0 -> NaN, duplicados unidos
#   composito_pi.csv -> composito_limpio.csv
#
# No borra filas por un campo malo: marca con banderas (sostenida,
# n6sc_desfasado, leyes_sospechosas) y la decisión queda para después.
#
#   pixi run python notebooks/01a_limpieza_fuentes.py
#   pixi run python notebooks/01a_limpieza_fuentes.py --fuentes pi
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # independiente del CWD

import argparse

import pandas as pd

from src.pipeline.config import (RAW_COURIER_PI, RAW_COURIER_BD, RAW_COMPOSITO,
                                 LIMPIO_COURIER_PI, LIMPIO_COURIER_BD, LIMPIO_COMPOSITO)
from src.pipeline.limpieza_fuentes import limpiar_pi, limpiar_bd, limpiar_composito

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument('--fuentes', nargs='+', choices=['bd', 'pi', 'composito'], default=['bd', 'pi', 'composito'])
args, _ = _ap.parse_known_args()

tareas = {'bd': (limpiar_bd, RAW_COURIER_BD, LIMPIO_COURIER_BD, None),
          'pi': (limpiar_pi, RAW_COURIER_PI, LIMPIO_COURIER_PI, ['ts']),
          'composito': (limpiar_composito, RAW_COMPOSITO, LIMPIO_COMPOSITO, ['ts'])}

for nombre in args.fuentes:
    fn, entrada, salida, fechas = tareas[nombre]
    df, resumen = fn(pd.read_csv(entrada, parse_dates=fechas))
    df.to_csv(salida, index=False)
    print(f"[{nombre}] {entrada.name} -> {salida.name}")
    for k, v in resumen.items():
        print(f"  {k}: {v}")
