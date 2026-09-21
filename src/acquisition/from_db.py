# ============================================================
# src/acquisition/from_db.py — Adquisición: intensidades desde Postgres
# ============================================================
# Fuente REAL de las intensidades del courier de cobre (n1fe, n2cu, n3zn, n4mo,
# n6sc): reemplaza a notebooks/00_getdata.ipynb de V1. El analizador expone una
# API HTTP local (CLB) que otro proyecto (api2db.py en data4cdpv1_local) vuelca
# a la tabla Postgres `works4cdp_assay`; este módulo solo lee esa tabla.
#
# Ejecutable de forma independiente:
#   python -m src.acquisition.from_db --sample-id 24 --hasta 2026-08-31 \
#       --out data/raw/intensidad_cobre_db.csv
# ============================================================

import argparse

from ..database import Extractor, DB_CONFIG_DEFAULT
from ..pipeline.config import DATA_RAW


def extraer_intensidades(sample_id: int, desde: str | None = None, hasta: str | None = None,
                         db_config: dict | None = None):
    """Extrae intensidades crudas de la BD, opcionalmente acotadas por fecha."""
    extractor = Extractor(table_name="works4cdp_assay", **(db_config or DB_CONFIG_DEFAULT))
    return extractor.get_intensity(sample_id, desde=desde, hasta=hasta)


def extraer_assays(sample_id: int, desde: str | None = None, hasta: str | None = None,
                   db_config: dict | None = None):
    """Extrae ensayos de laboratorio (leyes) de la BD, opcionalmente acotados por fecha."""
    extractor = Extractor(table_name="works4cdp_assay", **(db_config or DB_CONFIG_DEFAULT))
    return extractor.get_assays(sample_id, desde=desde, hasta=hasta)


def _main():
    ap = argparse.ArgumentParser(description="Adquisición: extrae intensidades (o ensayos) de la BD Postgres")
    ap.add_argument("--sample-id", type=int, default=24, help="sample_id del courier (24 = concentrado final cobre)")
    ap.add_argument("--tabla", default="assays", choices=["assays", "intensidades"],
                    help="'intensidades' trae solo canales; 'assays' trae canales + leyes de laboratorio")
    ap.add_argument("--desde", default=None, help="Fecha mínima 'YYYY-MM-DD' (inclusive), opcional")
    ap.add_argument("--hasta", default=None, help="Fecha máxima 'YYYY-MM-DD' (inclusive), opcional")
    ap.add_argument("--out", default=str(DATA_RAW / "intensidad_cobre_db.csv"))
    args = ap.parse_args()

    if args.tabla == "intensidades":
        df = extraer_intensidades(args.sample_id, args.desde, args.hasta)
    else:
        df = extraer_assays(args.sample_id, args.desde, args.hasta)

    df.to_csv(args.out, index=False)
    print(f"Filas extraídas: {len(df)} (rango date: {df['date'].min()} .. {df['date'].max()})")
    print(f"Guardado en: {args.out}")


if __name__ == "__main__":
    _main()
