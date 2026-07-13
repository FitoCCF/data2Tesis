import pandas as pd

file1 = '../data/raw/AssayLab_2420260704_1906.csv'
file2 = '../data/raw/AssayLab_20260704_1900.csv'
output_file = '../data/processed/AssayLab_combined_limpio.csv'

# 1. Cargar los datos
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

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
df_final.to_csv(output_file, index=False)