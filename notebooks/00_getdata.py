#!/usr/bin/env python3
# ============================================================
# CELDA 0 — ADQUISICIÓN (tres fuentes, un crudo por fuente)
# ============================================================
#   bd         A  data/raw/courier_bd.csv     muestreo + leyes de laboratorio
#                 (Postgres works4cdp_assay; intensidades y pFe..pSol en la
#                 misma fila; ~cada 3 días desde 2022-12)
#   pi         B  data/raw/courier_pi.csv     operación continua del courier
#                 (PI recorded desde 2025-07-13, sin leyes; lectura real ~cada 21 min,
#                 re-archivada por el PI cada 100 s)
#   composito  C  data/raw/composito_pi.csv   compósito de lab de 12 h (PI)
#
# Cada fuente se extrae por separado y falla por separado: si la pasarela PI
# no responde, la BD igual se guarda. Los crudos no se modifican después;
# limpieza (1a por fuente) y unificación (1b) son de la etapa 1.
#
# Todos los crudos salen con el tiempo en HORA LOCAL naive (America/Lima):
# la BD ya lo guarda así (date, time) y el PI se convierte desde UTC.
#
# Se extrae todo hasta hoy en los mismos crudos. Train (hasta
# FECHA_CORTE_TRAIN), test y producción (desde FECHA_INICIO_PRODUCCION) los
# separa la etapa 1c; ninguna etapa ajusta nada fuera de 'train'.
#
# Requisitos de conexión (dependen de la máquina):
#   BD  DB_PORT=5432 si el contenedor no publica en 5433 (default del código)
#   PI  PI_TOKEN=... (o ~/.pi_token); host/puerto en config (PI_HOST, PI_PUERTO)
#
#   DB_PORT=5432 PI_TOKEN=... pixi run python notebooks/00_getdata.py
#   pixi run python notebooks/00_getdata.py --fuentes bd
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # independiente del CWD

import argparse
import time

import pandas as pd

from src.pipeline.config import (SAMPLE_ID_COURIER, COLS_BD, FECHA_INICIO_COURIER,
                                 RAW_COURIER_BD, RAW_COURIER_PI, RAW_COMPOSITO)
from src.acquisition.from_db import extraer, resumen_leyes
from src.acquisition.from_pi import extraer_courier, extraer_composito, FECHA_INICIO_COMPOSITO

FECHA_DESDE = None          # None = inicio de la historia de cada fuente
FECHA_HASTA = None          # None = hasta hoy. La separación train/test/producción NO se hace
                            # al extraer: la marca la etapa 1c (columna 'periodo', config.FECHA_CORTE_TRAIN
                            # y FECHA_INICIO_PRODUCCION), y todo lo que se ajusta usa solo 'train'.

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument('--fuentes', nargs='+', choices=['bd', 'pi', 'composito'], default=['bd', 'pi', 'composito'])
_ap.add_argument('--desde', default=FECHA_DESDE, help="'YYYY-MM-DD' (inclusive), opcional")
_ap.add_argument('--hasta', default=FECHA_HASTA, help="'YYYY-MM-DD' (inclusive)")
_ap.add_argument('--sample-id', type=int, default=SAMPLE_ID_COURIER)
args, _ = _ap.parse_known_args()

# El fin del PI es exclusivo: se pide hasta el día siguiente a --hasta para incluirlo entero
fin_pi = None                                           # None = ahora
if args.hasta:
    fin_pi = (pd.Timestamp(args.hasta) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')


def _bd():
    df = extraer(args.sample_id, desde=args.desde, hasta=args.hasta)
    df.to_csv(RAW_COURIER_BD, index=False)
    print(f"  filas: {len(df)}  | con ≥1 ley: {int(df[COLS_BD['leyes']].notna().any(axis=1).sum())}")
    print(resumen_leyes(df).to_string())
    return RAW_COURIER_BD


def _pi():
    df = extraer_courier(args.desde or FECHA_INICIO_COURIER, fin_pi)
    df.to_csv(RAW_COURIER_PI, index=False)
    print(f"  filas: {len(df)}  | con los 5 canales: {int(df.drop(columns='ts').notna().all(axis=1).sum())}"
          f"  | rango {df['ts'].min()} .. {df['ts'].max()}")
    return RAW_COURIER_PI


def _composito():
    df = extraer_composito(args.desde or FECHA_INICIO_COMPOSITO, fin_pi)
    df.to_csv(RAW_COMPOSITO, index=False)
    print(f"  filas: {len(df)}  | rango {df['ts'].min()} .. {df['ts'].max()}")
    return RAW_COMPOSITO


fallos = []
for nombre, fn in {'bd': _bd, 'pi': _pi, 'composito': _composito}.items():
    if nombre not in args.fuentes:
        continue
    print(f"[{nombre}] extrayendo (desde={args.desde or 'inicio'} hasta={args.hasta or 'hoy'}) ...")
    t0 = time.time()
    try:
        ruta = fn()
        print(f"  guardado en {ruta}  ({time.time() - t0:.0f} s)")
    except Exception as e:                                # una fuente caída no tumba a las demás
        fallos.append(nombre)
        print(f"  FALLÓ: {type(e).__name__}: {str(e)[:300]}")

if fallos:
    sys.exit(f"Fuentes que fallaron: {fallos}")
