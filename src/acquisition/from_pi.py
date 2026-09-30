# ============================================================
# src/acquisition/from_pi.py — Adquisición desde PI OSIsoft (vía pasarela PiGateway)
# ============================================================
# Dos fuentes, cada una a su propio crudo en data/raw/ (etapa 0):
#
#   B  courier_pi.csv     operación continua del courier: dato ARCHIVADO
#                         (recorded), 5 canales, desde 2025-07-13.
#   C  composito_pi.csv   compósito de laboratorio de 12 h (fe/cu/ins/mo).
#
# Ambas con recorded(), nunca interpolated(): interpolar a una grilla inventa
# valores entre lecturas reales y borra la distinción entre un canal estable
# y uno congelado. Se verificó (2026-09-30) que 108 de 126 muestras de la BD
# aparecen con valores IDÉNTICOS en el recorded del PI; en una grilla
# interpolada de 15 min esa correspondencia se pierde.
#
# El crudo NO se limpia aquí: el courier sostiene cada lectura del
# concentrado ~21 min y el PI la vuelve a archivar cada 100 s (ExcMax=100).
# Colapsar esas repeticiones es de la etapa 1a (limpieza_fuentes.limpiar_pi).
#
# Cliente: src/acquisition/pi_client.py, copia tal cual de
# data4cdpv1_local/scripts/pi_client.py (fuente canónica; si cambia allá,
# se vuelve a copiar). Host/puerto en config (PI_HOST, PI_PUERTO), o por
# PI_GATEWAY_HOST / PI_GATEWAY_PORT. Token por PI_TOKEN o ~/.pi_token:
# nunca en el código, el repo está en GitHub.
#
# Ejecutable de forma independiente:
#   PI_TOKEN=... python -m src.acquisition.from_pi courier --hasta 2026-08-31
#   PI_TOKEN=... python -m src.acquisition.from_pi composito --desde 2022-01-01 --hasta 2026-08-31
# ============================================================

import argparse
import os
from pathlib import Path

import pandas as pd

from .pi_client import PiGateway
from ..pipeline.config import (PI_HOST, PI_PUERTO, TAGS_COURIER, FECHA_INICIO_COURIER,
                               TAGS_COMPOSITO, RAW_COURIER_PI, RAW_COMPOSITO)

FECHA_INICIO_COMPOSITO = "2022-01-01"                     # antes de la historia del compósito: trae todo


def cargar_token() -> str:
    """Token de la pasarela: PI_TOKEN o, si no, ~/.pi_token. Nunca se imprime."""
    tok = os.environ.get("PI_TOKEN", "").strip()
    if tok:
        return tok
    archivo = Path.home() / ".pi_token"
    return archivo.read_text(encoding="utf-8").strip() if archivo.exists() else ""


def cliente() -> PiGateway:
    """PiGateway con host/puerto de config (sobreescribibles por entorno) y el token."""
    return PiGateway(host=os.environ.get("PI_GATEWAY_HOST", PI_HOST),
                     puerto=int(os.environ.get("PI_GATEWAY_PORT", PI_PUERTO)),
                     token=cargar_token())


def _a_ancho(largo: pd.DataFrame, nombres: dict) -> pd.DataFrame:
    """Largo (tag, t, value, good) -> ancho con columna 'ts' naive en hora local.

    - Los valores con good=False (estado de error del PI) pasan a NaN: su
      'value' no es numérico. No se descarta la fila.
    - Cada tag se archiva con su propio timestamp (con o sin milisegundos);
      se redondea a 1 s para que los canales de un mismo ciclo del courier
      queden en la misma fila. Si dos eventos de un tag caen en el mismo
      segundo, queda el último.
    """
    largo = largo.copy()
    largo["value"] = pd.to_numeric(largo["value"], errors="coerce").where(largo["good"].astype(bool))
    # to_datetime explícito: si algún bloque del troceado vino vacío, el concat
    # deja 't' como object aunque los demás bloques traigan datetime con zona
    largo["ts"] = (pd.to_datetime(largo["t"], utc=True).dt.tz_convert("America/Lima")
                   .dt.tz_localize(None).dt.round("1s"))
    ancho = largo.pivot_table(index="ts", columns="tag", values="value", aggfunc="last", dropna=False)
    ancho = ancho.rename(columns=nombres)
    ancho = ancho[[c for c in nombres.values() if c in ancho.columns]]
    ancho.columns.name = None
    return ancho.sort_index().reset_index()


def extraer_courier(desde: str = FECHA_INICIO_COURIER, hasta: str | None = None,
                    chunk_dias: float = 7) -> pd.DataFrame:
    """Fuente B: dato archivado de los 5 canales del courier.
    Columnas: ts (naive, hora local) + n1fe/n2cu/n3zn/n4mo/n6sc."""
    hasta = hasta or pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    largo = cliente().recorded(list(TAGS_COURIER), desde, hasta, chunk_dias=chunk_dias)
    return _a_ancho(largo, TAGS_COURIER)


def extraer_composito(desde: str = FECHA_INICIO_COMPOSITO, hasta: str | None = None,
                      chunk_dias: float = 90) -> pd.DataFrame:
    """Fuente C: compósito de 12 h. Columnas: ts (naive, hora local) + fe/cu/ins/mo.
    Los ceros del laboratorio se dejan: son de la limpieza."""
    hasta = hasta or pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    largo = cliente().recorded(list(TAGS_COMPOSITO), desde, hasta, chunk_dias=chunk_dias)
    return _a_ancho(largo, TAGS_COMPOSITO)


def _main():
    ap = argparse.ArgumentParser(description="Adquisición desde PI OSIsoft (courier o compósito)")
    ap.add_argument("fuente", choices=["courier", "composito"])
    ap.add_argument("--desde", default=None, help="Inicio 'YYYY-MM-DD[ HH:MM:SS]' (default: inicio de la historia)")
    ap.add_argument("--hasta", default=None, help="Fin 'YYYY-MM-DD[ HH:MM:SS]' (default: ahora)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.fuente == "courier":
        df = extraer_courier(args.desde or FECHA_INICIO_COURIER, args.hasta)
        out = args.out or RAW_COURIER_PI
    else:
        df = extraer_composito(args.desde or FECHA_INICIO_COMPOSITO, args.hasta)
        out = args.out or RAW_COMPOSITO

    df.to_csv(out, index=False)
    print(f"Filas: {len(df)}  (rango {df['ts'].min()} .. {df['ts'].max()})")
    print(f"Guardado en: {out}")


if __name__ == "__main__":
    _main()
