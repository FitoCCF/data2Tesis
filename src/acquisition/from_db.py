# ============================================================
# src/acquisition/from_db.py — Adquisición desde Postgres (baja frecuencia)
# ============================================================
# Fuente de las intensidades del courier Y de las leyes de laboratorio
# (pFe..pSol): el analizador expone una API HTTP local (CLB) que otro proyecto
# (api2db.py en data4cdpv1_local) vuelca a la tabla works4cdp_assay. Este
# módulo solo lee esa tabla, vía src.database.ExtractorBD.
#
# Por defecto trae intensidades + leyes en un solo CSV (misma fila en la BD).
# Qué columnas hay en cada grupo se define en config.COLS_BD.
#
# Ejecutable de forma independiente:
#   python -m src.acquisition.from_db --hasta 2026-08-31            # -> data/raw/courier_bd.csv
#   python -m src.acquisition.from_db --grupos leyes --out data/raw/leyes.csv
# ============================================================

import argparse

import pandas as pd

from ..database import ExtractorBD
from ..pipeline.config import COLS_BD, SAMPLE_ID_COURIER, RAW_COURIER_BD


def extraer(sample_id: int = SAMPLE_ID_COURIER, grupos=("intensidades", "leyes"),
            desde: str | None = None, hasta: str | None = None,
            db_config: dict | None = None) -> pd.DataFrame:
    """Llaves (date, time, instance) + los grupos de columnas pedidos, de un sample_id."""
    return ExtractorBD(db_config).extraer(sample_id, grupos=grupos, desde=desde, hasta=hasta)


def resumen_leyes(df: pd.DataFrame) -> pd.DataFrame:
    """Por cada ley presente: cuántas filas traen dato y cuántas vienen en cero.
    Los ceros se reportan, no se tratan: decidir si son centinela es de la limpieza."""
    leyes = [c for c in COLS_BD["leyes"] if c in df.columns]
    return pd.DataFrame({"con_dato": df[leyes].notna().sum(),
                         "en_cero": (df[leyes] == 0).sum()})


def _main():
    ap = argparse.ArgumentParser(description="Adquisición desde la BD Postgres (works4cdp_assay)")
    ap.add_argument("--sample-id", type=int, default=SAMPLE_ID_COURIER,
                    help=f"sample_id a extraer ({SAMPLE_ID_COURIER} = Concentrado Colectivo)")
    ap.add_argument("--grupos", nargs="+", default=["intensidades", "leyes"], choices=list(COLS_BD),
                    help="Grupos de columnas a extraer (definidos en config.COLS_BD)")
    ap.add_argument("--desde", default=None, help="Fecha mínima 'YYYY-MM-DD' (inclusive), opcional")
    ap.add_argument("--hasta", default=None, help="Fecha máxima 'YYYY-MM-DD' (inclusive), opcional")
    ap.add_argument("--out", default=str(RAW_COURIER_BD))
    args = ap.parse_args()

    df = extraer(args.sample_id, grupos=args.grupos, desde=args.desde, hasta=args.hasta)
    df.to_csv(args.out, index=False)
    print(f"Filas extraídas: {len(df)} (rango date: {df['date'].min()} .. {df['date'].max()})")
    if "leyes" in args.grupos:
        print(resumen_leyes(df).to_string())
    print(f"Guardado en: {args.out}")


if __name__ == "__main__":
    _main()
