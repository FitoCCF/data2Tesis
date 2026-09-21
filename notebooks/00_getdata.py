#!/usr/bin/env python3
# ============================================================
# CELDA 0 — ADQUISICIÓN (extracción desde la BD Postgres)
# ============================================================
# Reemplaza notebooks/00_getdata.ipynb de V1: delega en
# src.acquisition.from_db en vez de instanciar el Extractor a mano, y ya no
# tiene el puerto hardcodeado (V1 tenía 5433 aquí y 5432 en scripts/assay_int.py
# -> inconsistente; ahora ambos usan src.database.connection.DB_CONFIG_DEFAULT).
#
# FECHA_HASTA por defecto = fin de agosto: en esta fase solo se quiere data
# hasta agosto para calibrar/entrenar el pipeline. Septiembre en adelante se
# reserva como datos de PRODUCCIÓN (se extraen y procesan aparte, para scoring
# en línea, no para entrenamiento) -> por eso NO se incluyen aquí por defecto.
# Ajusta/quita FECHA_HASTA cuando ese corte ya no aplique.
#
# --desde/--hasta se pueden sobreescribir por línea de comandos sin romper el
# uso como celda de notebook (los argumentos desconocidos del kernel de
# Jupyter se ignoran vía parse_known_args):
#   pixi run notebooks/00_getdata.py --hasta 2026-08-31
#   pixi run notebooks/00_getdata.py --desde 2026-09-01 --hasta 2026-09-21 --out data/raw/produccion_sept.csv
#
# Equivalente por CLI (mismos parámetros, sin pasar por el wrapper de notebook):
#   python -m src.acquisition.from_db --sample-id 24 --tabla intensidades \
#       --desde ... --hasta ... --out data/raw/intensidad_cobre_db.csv
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # independiente del CWD

import argparse

from src.pipeline.config import DATA_RAW
from src.acquisition.from_db import extraer_intensidades

SAMPLE_ID = 24              # Concentrado final de courier cobre
FECHA_DESDE = None          # None = sin límite inferior (todo el histórico disponible en la BD)
FECHA_HASTA = '2026-08-31'  # solo hasta agosto; septiembre en adelante = datos de producción

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument('--sample-id', type=int, default=SAMPLE_ID)
_ap.add_argument('--desde', default=FECHA_DESDE, help="'YYYY-MM-DD' (inclusive), opcional")
_ap.add_argument('--hasta', default=FECHA_HASTA, help="'YYYY-MM-DD' (inclusive), opcional")
_ap.add_argument('--out', default=str(DATA_RAW / 'intensidad_cobre_db.csv'))
args, _ = _ap.parse_known_args()

df_columnas = extraer_intensidades(args.sample_id, desde=args.desde, hasta=args.hasta)
df_columnas.to_csv(args.out, index=False)
print(f"Filas extraídas: {len(df_columnas)}  (desde={args.desde or 'inicio'} hasta={args.hasta or 'hoy'})")
print(f"Guardado en: {args.out}")
