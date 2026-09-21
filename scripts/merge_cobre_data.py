#!/usr/bin/env python3
# ============================================================
# scripts/merge_cobre_data.py — Combina extracciones crudas + train/test split
# ============================================================
# --cobre es BD Postgres (baja frecuencia, ver src/acquisition/from_db.py) y
# --cobre-24 es PI OSIsoft vía AF SDK (alta frecuencia, cada 15 min desde
# 2025-07-13, ver src/acquisition/from_pi_afsdk.py). Al combinar ambas fuentes
# el dataset final queda con más datos que cualquiera de las dos por separado.
#
# Ejecutable de forma independiente:
#   python scripts/merge_cobre_data.py
#   python scripts/merge_cobre_data.py --cobre otra_extraccion.csv --cobre-24 otra.csv
#   python scripts/merge_cobre_data.py --train-hasta 2026-08-31   # split por FECHA en vez de 70/30
# ============================================================
import argparse
import pandas as pd
import os


def parse_datetime_column(df):
    if 'date' in df.columns and 'time' in df.columns:
        return df

    # 'Unnamed: 0' = índice sin nombre (CSV del script AF SDK original); 't' =
    # nombre que le da PiGateway.to_wide() (src/acquisition/from_pi.py) al
    # pivotar -- ambos son la misma columna de timestamp, solo cambia el header.
    for col in ('Unnamed: 0', 't'):
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors='coerce')
            if dt.notna().any():
                df['date'] = dt.dt.strftime('%Y-%m-%d')
                df['time'] = dt.dt.strftime('%H:%M:%S')
                return df

    raise ValueError("No se pudo obtener date/time de df_cobre_24")


def main():
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))

    ap = argparse.ArgumentParser(description="Combina dos extracciones crudas de intensidades y hace train/test split")
    ap.add_argument('--cobre', default=os.path.join(raw_dir, 'intensidad_cobre_db.csv'),
                    help="CSV de la BD (baja frecuencia), columnas date/time/instance/n1fe..n7ech7 "
                         "-- ver src/acquisition/from_db.py")
    ap.add_argument('--cobre-24', default=os.path.join(raw_dir, 'Intensidades_20260717_1715.csv'),
                    help="CSV de PI OSIsoft (alta frecuencia, cada 15 min), columnas cu/fe/zn/mo/sc "
                         "(se renombran a n2cu/n1fe/n3zn/n4mo/n6sc) -- ver src/acquisition/from_pi_afsdk.py")
    ap.add_argument('--out', default=os.path.join(processed_dir, 'intensidades_cobre.csv'))
    ap.add_argument('--out-train', default=os.path.join(processed_dir, 'intensidades_cobre_train.csv'))
    ap.add_argument('--out-test', default=os.path.join(processed_dir, 'intensidades_cobre_test.csv'))
    ap.add_argument('--train-hasta', default=None,
                    help="Fecha 'YYYY-MM-DD' (inclusive): si se da, el split es por FECHA "
                         "(train = fecha <= esta, test = el resto) en vez del 70/30 por posición")
    args = ap.parse_args()

    print(f"Reading '{args.cobre}'...")
    df_cobre = pd.read_csv(args.cobre)
    print(f"Original shape: {df_cobre.shape}")
    print("Columns:", list(df_cobre.columns))

    print(f"\nReading '{args.cobre_24}'...")
    df_cobre_24 = pd.read_csv(args.cobre_24)
    print(f"Original shape: {df_cobre_24.shape}")
    print("Columns:", list(df_cobre_24.columns))

    df_cobre_24 = parse_datetime_column(df_cobre_24)

    mapping_24 = {
        'fe': 'n1fe',
        'cu': 'n2cu',
        'zn': 'n3zn',
        'mo': 'n4mo',
        'sc': 'n6sc'
    }
    df_cobre_24 = df_cobre_24.rename(columns=mapping_24)

    expected_columns = [
        'date', 'time', 'instance',
        'n1fe', 'n2cu', 'n3zn', 'n4mo', 'n5ech5', 'n6sc', 'n7ech7'
    ]

    df_cobre = df_cobre.reindex(columns=expected_columns)
    df_cobre_24 = df_cobre_24.reindex(columns=expected_columns)

    print("\nMerging datasets...")
    df_merged = pd.concat([df_cobre, df_cobre_24], ignore_index=True)

    df_merged['datetime'] = pd.to_datetime(
        df_merged['date'].astype(str) + ' ' + df_merged['time'].astype(str),
        errors='coerce'
    )
    df_merged = df_merged.sort_values(by='datetime').reset_index(drop=True)

    os.makedirs(processed_dir, exist_ok=True)
    df_merged.to_csv(args.out, index=False)

    if args.train_hasta:
        es_train = df_merged['datetime'] <= pd.Timestamp(args.train_hasta) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df_train = df_merged[es_train].drop(columns=['datetime'])
        df_test = df_merged[~es_train].drop(columns=['datetime'])
    else:
        split_index = int(len(df_merged) * 0.7)
        df_train = df_merged.iloc[:split_index].drop(columns=['datetime'])
        df_test = df_merged.iloc[split_index:].drop(columns=['datetime'])

    df_train.to_csv(args.out_train, index=False)
    df_test.to_csv(args.out_test, index=False)

    print("Saved:", args.out)
    print(f"Saved train split ({len(df_train)} filas):", args.out_train)
    print(f"Saved test split ({len(df_test)} filas):", args.out_test)


if __name__ == '__main__':
    main()
