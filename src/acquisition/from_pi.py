# ============================================================
# src/acquisition/from_pi.py — Adquisición: intensidades del courier desde PI OSIsoft (vía gateway WSL)
# ============================================================
# Envoltorio sobre pi_client.PiGateway (copiado tal cual desde
# data4cdpv1_local/scripts/pi_client.py, autocontenido: solo requiere
# `requests` + `pandas`). PiGateway habla HTTP/JSON con una pasarela (gw4Pi.exe)
# que corre en Windows y expone el PI Data Archive, sin necesitar AF SDK ni
# pythonnet en este entorno Linux/WSL.
#
# ESTA es la ruta de extracción vigente para los canales del courier: se
# extrae DESDE WSL, a través del gateway de data4cdpv1_local, usando los
# mismos tags que ya se conocían por el script AF SDK original (ver también
# from_pi_afsdk.py, que documenta la vía directa por AF SDK en Windows como
# alternativa/referencia -misma fuente PI, mismos tags, transporte distinto-):
#
#   _296290_ConcFinal_CanalCu_ABB -> n2cu
#   _296290_ConcFinal_CanalFe_ABB -> n1fe
#   _296290_ConcFinal_CanalMo_ABB -> n4mo
#   _296290_ConcFinal_CanalSc_ABB -> n6sc
#   _296290_ConcFinal_CanalZn_ABB -> n3zn
#
# El servidor PI real (tpi.southernperu.com.pe) lo resuelve el gateway del
# lado Windows; el cliente de este módulo no necesita saberlo.
#
# FECHA_INICIO_COURIER no se cambia salvo que se sepa lo que se hace: no hay
# historia de estos tags antes de esa fecha.
#
# La salida (índice = timestamp, columnas n1fe/n2cu/n3zn/n4mo/n6sc) ya está en
# el formato que espera directamente la etapa 1 (src.pipeline.limpieza,
# CANALES) para scorear producción sin pasar por merge_cobre_data.py. También
# sigue funcionando como --cobre-24 de scripts/merge_cobre_data.py: su
# `mapping_24` (que espera cu/fe/zn/mo/sc) simplemente no encuentra nada que
# renombrar y sigue de largo, porque las columnas ya vienen con el nombre final.
#
# Ejecutable de forma independiente (desde WSL):
#   export PI_GATEWAY_HOST=<ip_windows>   # si el autodescubrimiento no lo encuentra
#   python -m src.acquisition.from_pi --out data/raw/Intensidades_nuevo.csv
#   python -m src.acquisition.from_pi --hasta 2026-09-21 --out data/raw/Intensidades_sept.csv
#
# Para OTRAS variables de proceso (p.ej. espesadores/relaves, tags distintos
# a los del courier), usar --tags explícito:
#   python -m src.acquisition.from_pi --tags _294100_LIT_1011_ABB \
#       --desde 2026-08-01 --hasta 2026-09-21 --out data/raw/pi_espesadores.csv
# ============================================================

import argparse

import pandas as pd

from .pi_client import PiGateway
from ..pipeline.config import DATA_RAW

# --- Fecha desde la que existen datos para los tags del courier en PI: NO cambiar ---
FECHA_INICIO_COURIER = "2025-07-13 15:00:00"

# Tag PI -> nombre de columna FINAL del pipeline (src.pipeline.config.CANALES),
# no el nombre corto crudo -- así la salida sirve directo para la etapa 1 sin
# pasar por merge_cobre_data.py, y ese script sigue funcionando igual (ver nota
# arriba).
TAGS_COURIER = {
    "_296290_ConcFinal_CanalCu_ABB": "n2cu",
    "_296290_ConcFinal_CanalFe_ABB": "n1fe",
    "_296290_ConcFinal_CanalMo_ABB": "n4mo",
    "_296290_ConcFinal_CanalSc_ABB": "n6sc",
    "_296290_ConcFinal_CanalZn_ABB": "n3zn",
}


def extraer_courier(inicio: str = FECHA_INICIO_COURIER, fin: str | None = None,
                    host: str | None = None, intervalo: str = "15m",
                    metodo: str = "interpolated") -> pd.DataFrame:
    """Extrae los 5 tags del courier vía el gateway y devuelve un DataFrame
    ancho (índice = timestamp naive, columnas n1fe/n2cu/n3zn/n4mo/n6sc -- ya
    con los nombres finales del pipeline), listo tanto para alimentar
    directamente la etapa 1 (producción) como para scripts/merge_cobre_data.py
    --cobre-24.

    metodo='interpolated' reproduce el mismo criterio que el script AF SDK
    original (InterpolatedValues cada `intervalo`); 'recorded' trae el dato
    crudo archivado (sin interpolar) si se prefiere para auditoría.
    """
    fin = fin or "*"  # '*' = ahora, mismo significado que en pi_client
    pi = PiGateway(host=host)
    tags = list(TAGS_COURIER.keys())

    if metodo == "interpolated":
        df_largo = pi.interpolated(tags, inicio, fin, intervalo=intervalo)
    else:
        df_largo = pi.recorded(tags, inicio, fin)

    ancho = pi.to_wide(df_largo)                     # pivote sin relleno, columnas = tags completos
    ancho = ancho.rename(columns=TAGS_COURIER)        # tags -> n1fe/n2cu/n3zn/n4mo/n6sc
    ancho = ancho.ffill().bfill()                     # mismo criterio de relleno que el script AF SDK original
    if ancho.index.tz is not None:
        ancho.index = ancho.index.tz_localize(None)   # timestamp naive, igual que Intensidades_*.csv existentes
    ancho.index.name = None                           # PiGateway.to_wide() nombra el índice 't'; sin nombre =
                                                        # columna en blanco al hacer to_csv() (mismo formato que
                                                        # el script AF SDK original), que merge_cobre_data.py
                                                        # reconoce como 'Unnamed: 0' al releerlo
    return ancho


def extraer_generico(tags: list[str], inicio: str, fin: str, host: str | None = None) -> pd.DataFrame:
    """Dato crudo archivado en PI para tags arbitrarios (p.ej. espesadores/relaves),
    en formato ancho (pivote sin relleno). No aplica al courier -> usar extraer_courier()."""
    pi = PiGateway(host=host)
    df_largo = pi.recorded(tags, inicio, fin)
    return pi.to_wide(df_largo)


def _main():
    ap = argparse.ArgumentParser(
        description="Adquisición: extrae intensidades del courier (o tags arbitrarios) desde PI OSIsoft vía gateway")
    ap.add_argument("--tags", nargs="+", default=None,
                    help="Tags PI arbitrarios (modo genérico). Si se omite, extrae los 5 tags del courier.")
    ap.add_argument("--desde", default=FECHA_INICIO_COURIER,
                    help=f"Inicio. Default: {FECHA_INICIO_COURIER} (inicio de historia de los tags del courier)")
    ap.add_argument("--hasta", default=None, help="Fin, p.ej. '2026-09-21' o '*' (default: ahora)")
    ap.add_argument("--intervalo", default="15m", help="Solo modo courier: intervalo de interpolación")
    ap.add_argument("--metodo", choices=["interpolated", "recorded"], default="interpolated",
                    help="Solo modo courier")
    ap.add_argument("--host", default=None, help="IP del host Windows con la pasarela (si no, autodetecta)")
    ap.add_argument("--out", default=str(DATA_RAW / "Intensidades_pi.csv"))
    args = ap.parse_args()

    if args.tags:
        ancho = extraer_generico(args.tags, args.desde, args.hasta or "*", host=args.host)
    else:
        ancho = extraer_courier(args.desde, args.hasta, host=args.host,
                                intervalo=args.intervalo, metodo=args.metodo)

    ancho.to_csv(args.out)
    print(f"Filas: {len(ancho)}  |  columnas: {list(ancho.columns)}")
    print(f"Guardado en: {args.out}")
    if not args.tags:
        print("Listo para: python scripts/merge_cobre_data.py --cobre-24", args.out)


if __name__ == "__main__":
    _main()
