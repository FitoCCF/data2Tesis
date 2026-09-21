# ============================================================
# scripts/split_date_time.py — Separa una columna 'date' datetime en date+time
# ============================================================
# NOTA vs V1: el script original tenía la ruta hardcodeada a
# '/home/fito/Proyects/data2Tesis/...' (usuario/máquina distintos a los
# actuales) -> se reemplaza por un argumento --file explícito.
#
# Ejecutable de forma independiente:
#   python scripts/split_date_time.py --file data/raw/intensidades_cobre.csv
# ============================================================
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Separa 'date' (YYYY-MM-DD HH:MM:SS) en columnas date + time")
    ap.add_argument('--file', required=True, help="CSV a modificar in-place")
    args = ap.parse_args()

    print(f"Reading {args.file}...")
    df = pd.read_csv(args.file)

    print("\nOriginal columns:", list(df.columns))
    print("Original first 5 rows:")
    print(df.head())

    # Split 'date' column by the space
    # The format is 'YYYY-MM-DD HH:MM:SS'
    # So split(' ', n=1, expand=True) will give date in [0] and time in [1]
    split_df = df['date'].astype(str).str.split(' ', n=1, expand=True)

    # Insert 'time' column right after 'date' (which is at index 0)
    df.insert(1, 'time', split_df[1])
    df['date'] = split_df[0]

    print("\nModified columns:", list(df.columns))
    print("Modified first 5 rows:")
    print(df.head())

    # Save the modified dataframe back to the same file
    df.to_csv(args.file, index=False)
    print("\nFile saved successfully.")


if __name__ == '__main__':
    main()
