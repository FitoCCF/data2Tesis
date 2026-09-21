#!/usr/bin/env python3
# ============================================================
# CELDA 0b — ADQUISICIÓN (extracción desde PI OSIsoft vía gateway, WSL)
# ============================================================
# Igual que notebooks/00_getdata.py, pero para la fuente de ALTA FRECUENCIA:
# delega en src.acquisition.from_pi (PiGateway -> gateway HTTP en Windows,
# data4cdpv1_local), con los 5 tags del courier ya incorporados por defecto
# (_296290_ConcFinal_Canal{Cu,Fe,Mo,Sc,Zn}_ABB -> cu/fe/mo/sc/zn). Corre desde
# WSL, sin AF SDK ni pythonnet.
#
# FECHA_HASTA por defecto = fin de agosto, mismo criterio que 00_getdata.py:
# en esta fase solo se quiere data hasta agosto para calibrar/entrenar;
# septiembre en adelante se reserva como datos de PRODUCCIÓN.
# FECHA_DESDE por defecto = 2025-07-13 15:00:00: no hay historia de estos tags
# en PI antes de esa fecha (no cambiar salvo que sepas lo que haces).
#
# Requiere el gateway accesible (arrancado en la máquina Windows). Si el
# autodescubrimiento no lo encuentra, exportar PI_GATEWAY_HOST antes de correr.
#
# --desde/--hasta/--out se pueden sobreescribir por línea de comandos sin
# romper el uso como celda de notebook (los argumentos propios del kernel de
# Jupyter se ignoran vía parse_known_args):
#   pixi run notebooks/00_getdata_pi.py --hasta 2026-08-31
#   pixi run notebooks/00_getdata_pi.py --desde 2026-09-01 --hasta 2026-09-21 --out data/raw/produccion_sept_pi.csv
#
# Equivalente por CLI (mismos parámetros, sin pasar por el wrapper de notebook):
#   python -m src.acquisition.from_pi --hasta 2026-08-31 --out data/raw/Intensidades_pi.csv
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # independiente del CWD

import argparse

from src.pipeline.config import DATA_RAW
from src.acquisition.from_pi import extraer_courier, FECHA_INICIO_COURIER

FECHA_DESDE = FECHA_INICIO_COURIER  # 2025-07-13 15:00:00 -- no cambiar salvo que sepas lo que haces
FECHA_HASTA = '2026-08-31'          # solo hasta agosto; septiembre en adelante = datos de producción

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument('--desde', default=FECHA_DESDE)
_ap.add_argument('--hasta', default=FECHA_HASTA)
_ap.add_argument('--intervalo', default='15m')
_ap.add_argument('--metodo', choices=['interpolated', 'recorded'], default='interpolated')
_ap.add_argument('--host', default=None, help="IP del host Windows con el gateway (si no, autodetecta)")
_ap.add_argument('--out', default=str(DATA_RAW / 'Intensidades_pi.csv'))
args, _ = _ap.parse_known_args()

ancho = extraer_courier(inicio=args.desde, fin=args.hasta, host=args.host,
                        intervalo=args.intervalo, metodo=args.metodo)
ancho.to_csv(args.out)
print(f"Filas: {len(ancho)}  |  columnas: {list(ancho.columns)}  (desde={args.desde} hasta={args.hasta})")
print(f"Guardado en: {args.out}")
print(f"Listo para: python scripts/merge_cobre_data.py --cobre-24 {args.out}")
