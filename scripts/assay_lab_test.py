# ============================================================
# scripts/assay_lab_test.py — Combina y limpia exportaciones de AssayLab
# ============================================================
# Genera data/processed/AssayLab_combined_limpio.csv, que consume
# src/evaluacion/evaluar.py (vía --lab) y scripts/validaciones_estadisticas.py.
#
# NOTA: requiere que existan los dos CSV crudos de AssayLab (--file1/--file2).
# En V1 (data2Tesis) esos dos archivos NO estaban presentes en data/raw/, por
# lo que este script -y todo lo que depende de su salida- no podía correr;
# hay que exportarlos de nuevo desde el analizador/BD antes de usar esto.
#
# Ejecutable de forma independiente:
#   python scripts/assay_lab_test.py --file1 data/raw/AssayLab_A.csv --file2 data/raw/AssayLab_B.csv
# ============================================================
import argparse
import os
import pandas as pd


def main():
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))

    ap = argparse.ArgumentParser(description="Combina y limpia dos exportaciones de AssayLab")
    ap.add_argument('--file1', required=True)
    ap.add_argument('--file2', required=True)
    ap.add_argument('--out', default=os.path.join(processed_dir, 'AssayLab_combined_limpio.csv'))
    args = ap.parse_args()

    # 1. Cargar los datos
    df1 = pd.read_csv(args.file1)
    df2 = pd.read_csv(args.file2)

    # 2. Unir y normalizar la columna de tiempo
    df_combined = pd.concat([df1, df2], ignore_index=True)
    df_combined.rename(columns={'Unnamed: 0': 'fecha_hora'}, inplace=True)
    df_combined['fecha_hora'] = pd.to_datetime(df_combined['fecha_hora'])

    # 3. Ordenar cronológicamente (crítico antes de comparar filas consecutivas)
    df_combined.sort_values(by='fecha_hora', inplace=True)

    # 4. Eliminar el traslape estricto de fecha/hora (1 fila)
    df_combined.drop_duplicates(subset=['fecha_hora'], keep='first', inplace=True)

    # 5. ELIMINAR ELEMENTOS REPETIDOS EN LOS VALORES (Leyes)
    # Filtra conservando solo las filas donde al menos un valor de 'cu', 'fe' o 'mo' sea distinto a la fila anterior
    cols_leyes = ['cu', 'fe', 'mo']
    df_final = df_combined.loc[(df_combined[cols_leyes] != df_combined[cols_leyes].shift(1)).any(axis=1)]

    # 6. Guardar el archivo limpio
    df_final.to_csv(args.out, index=False)
    print(f"Filas finales: {len(df_final)} -> {args.out}")


if __name__ == '__main__':
    main()
