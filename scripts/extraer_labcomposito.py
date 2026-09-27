#!/usr/bin/env python3
# ============================================================
# scripts/extraer_labcomposito.py — Extrae el compósito de laboratorio del PI
# ============================================================
# Extrae los tags del laboratorio químico del compósito de 12 h a
# data/raw/LABCOMPOSITO.csv, usando la pasarela PiGateway de
# data4cdpv1_local/scripts/pi_client.py.
#
# Usa recorded() y NO interpolated(): los ensayos de laboratorio son eventos
# discretos cada 12 h. Interpolar inventaría valores entre turnos y arruinaría
# cualquier validación posterior.
#
# TOKEN — la pasarela exige X-PI-Token. Dos formas de aportarlo:
#   1. Variable de entorno:   PI_TOKEN="..." pixi run python scripts/extraer_labcomposito.py
#   2. Archivo privado:       echo "..." > ~/.pi_token && chmod 600 ~/.pi_token
# El token nunca se imprime ni se guarda en el CSV.
#
# Uso:
#   pixi run python scripts/extraer_labcomposito.py --desde 2022-01-01
#   pixi run python scripts/extraer_labcomposito.py --solo-verificar   # no extrae
# ============================================================

import argparse                                           # argumentos de linea de comandos
import os                                                 # entorno y rutas
import sys                                                # path de modulos
from datetime import date                                 # fecha de fin por defecto
from pathlib import Path                                  # rutas multiplataforma

# La pasarela vive en otro proyecto; se agrega su ruta sin copiarla aqui
RUTA_PI_CLIENT = os.environ.get(
    "PI_CLIENT_DIR", "/home/daigo/data4cdpv1_local/scripts")
sys.path.insert(0, RUTA_PI_CLIENT)                        # para poder importar pi_client

import pandas as pd                                       # DataFrames
from pi_client import obtener_cliente, PiError            # cliente de la pasarela

# --- Tags del compósito de 12 h (turno dia 07:30 / noche 19:30) ---
TAGS = ["7100AIP101MAN", "7100AIP102MAN",                 # muestras cortadas antes del
        "7100AIP103MAN", "7100AIP104MAN"]                 # multiplexor, antes de los rayos X

DESTINO = Path(__file__).resolve().parents[1] / "data" / "raw" / "LABCOMPOSITO.csv"


def cargar_token() -> str:
    """Obtiene el token de la pasarela sin exponerlo en la linea de comandos."""
    tok = os.environ.get("PI_TOKEN", "").strip()          # 1) variable de entorno
    if tok:
        return tok
    archivo = Path.home() / ".pi_token"                   # 2) archivo privado
    if archivo.exists():
        return archivo.read_text(encoding="utf-8").strip()
    return ""                                             # sin token: la pasarela dara 401


def main():
    ap = argparse.ArgumentParser(description="Extrae el compósito de laboratorio del PI")
    ap.add_argument("--desde", default="2022-01-01", help="inicio, 'YYYY-MM-DD' o sintaxis PI ('*-3y')")
    # OJO: el troceado del cliente hace pd.Timestamp(fin), que no entiende la
    # sintaxis relativa del PI ('*', '*-3y'). El fin debe ser una fecha concreta.
    ap.add_argument("--hasta", default=date.today().isoformat(),
                    help="fin, 'YYYY-MM-DD' (hoy por defecto; no acepta '*')")
    ap.add_argument("--salida", default=str(DESTINO), help="ruta del CSV de salida")
    ap.add_argument("--solo-verificar", action="store_true",
                    help="comprueba conexión y metadata de los tags, sin extraer")
    ap.add_argument("--chunk-dias", type=float, default=90,
                    help="tamaño del troceado temporal (evita timeouts del servidor)")
    ap.add_argument("--host", default=os.environ.get("PI_GATEWAY_HOST", "127.0.0.1"),
                    help="host de la pasarela PiGateway")
    ap.add_argument("--puerto", type=int, default=int(os.environ.get("PI_GATEWAY_PORT", 5173)),
                    help="puerto de la pasarela (5173 en esta instalación, no el 5000 por defecto)")
    args = ap.parse_args()

    token = cargar_token()                                # token, si lo hay
    if not token:
        print("AVISO: no se encontró PI_TOKEN (ni variable de entorno ni ~/.pi_token).")
        print("       Si la pasarela exige token, la petición será rechazada con HTTP 401.\n")

    cli = obtener_cliente(host=args.host, puerto=args.puerto,  # host/puerto explícitos
                          token=token)                         # token desde entorno o ~/.pi_token

    # --- 1. Salud de la pasarela ---
    try:
        salud = cli.health()                              # confirma que responde
        print(f"Pasarela  : {cli}")
        print(f"Health    : {salud}\n")
    except PiError as e:
        print("FALLO DE CONEXION:\n", str(e)[:1200])
        return 1

    # --- 2. Metadata de los tags (confirma QUE mide cada uno antes de extraer) ---
    try:
        meta = cli.attributes(TAGS)                       # descriptor, unidades, tipo
        cols = [c for c in ("tag", "Name", "Descriptor", "EngineeringUnits",
                            "PointType", "Zero", "Span") if c in meta.columns]
        print("METADATA DE LOS TAGS")
        print(meta[cols].to_string(index=False) if cols else meta.to_string(index=False))
        print()
    except PiError as e:
        print("No se pudo leer la metadata:", str(e)[:400], "\n")

    if args.solo_verificar:                               # modo comprobacion
        print("--solo-verificar: no se extrajo nada.")
        return 0

    # --- 3. Extraccion del dato ARCHIVADO (no interpolado) ---
    print(f"Extrayendo recorded() de {args.desde} a {args.hasta} ...")
    df = cli.recorded(TAGS, args.desde, args.hasta, chunk_dias=args.chunk_dias)
    print(f"Eventos recibidos: {len(df)}")
    if df.empty:
        print("Sin datos en el rango pedido.")
        return 1

    # --- 4. Pivote a formato ancho: una fila por timestamp, una columna por tag ---
    ancho = cli.to_wide(df)                               # tag -> columna
    ancho = ancho.sort_index()                            # orden cronologico

    destino = Path(args.salida)                           # ruta de salida
    destino.parent.mkdir(parents=True, exist_ok=True)     # crea data/raw si falta
    ancho.to_csv(destino)                                 # guarda con el indice de tiempo

    print(f"\nFilas   : {len(ancho)}")
    print(f"Rango   : {ancho.index.min()}  ->  {ancho.index.max()}")
    print(f"Columnas: {list(ancho.columns)}")
    print("\nCobertura por tag (valores no nulos):")
    print((ancho.notna().sum()).to_string())
    print(f"\nGuardado en: {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
