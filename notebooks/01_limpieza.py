# ============================================================
# CELDA 1 — LIMPIEZA
# ============================================================

import pandas as pd                                  # pandas: manejo de tablas (DataFrames)
import joblib                                         # joblib: guardar/cargar modelos entrenados
import os                                             # os: crear carpetas en el sistema de archivos
from sklearn.ensemble import IsolationForest          # IsolationForest: detector de anomalías multivariadas

# --- Carga de datos ---
file_path = 'data/processed/intensidades_cobre.csv'   # ruta del CSV de intensidades crudas
xx = pd.read_csv(file_path)                            # lee el CSV a un DataFrame llamado xx

# Columnas espectrales que vamos a limpiar (n6sc = sólido en la muestra)
cols = ['n1fe', 'n2cu', 'n3zn', 'n4mo', 'n6sc']       # lista de canales de intensidad

# --- FIX 3: eliminar códigos de error del analizador (-9999) de forma EXPLÍCITA ---
xx = xx[~(xx[cols] == -9999).any(axis=1)]             # conserva filas SIN ningún -9999 en los canales
#   (xx[cols] == -9999)   -> matriz booleana: True donde hay error
#   .any(axis=1)          -> True si la fila tiene AL MENOS un -9999
#   ~(...)                -> niega: True para filas SIN error
#   xx[...]               -> se queda solo con esas filas

# --- Eliminar filas de planta/analizador detenido (todos los canales en 0) ---
df_intensidad = xx[~(xx[cols] == 0).all(axis=1)].copy()  # quita filas donde TODOS los canales son 0
#   .all(axis=1)          -> True solo si los 5 canales son 0 en esa fila
#   ~(...)                -> conserva las filas que NO son todo-ceros
#   .copy()               -> FIX menor: copia real para evitar SettingWithCopyWarning al asignar después

# --- FIX 1: eliminar "sensor congelado" (vector idéntico repetido) ---
repeticiones = df_intensidad.groupby(cols)[cols[0]].transform('size')  # cuántas veces se repite cada vector exacto
df_intensidad = df_intensidad[repeticiones < 3].copy()                 # conserva solo vectores que aparecen < 3 veces
#   groupby(cols)                 -> agrupa filas con los 5 canales idénticos
#   [cols[0]].transform('size')   -> asigna a cada fila el tamaño de su grupo (nº de repeticiones)
#   repeticiones < 3              -> True si ese vector se repite menos de 3 veces (lectura real, no atascada)
#   IsolationForest NO detecta esto porque son puntos densos, no aislados -> hay que filtrarlo aparte

# --- Preparar matriz para IsolationForest ---
feature = ['n1fe', 'n2cu', 'n3zn', 'n4mo', 'n6sc']    # features que entran al detector de anomalías
x = df_intensidad[feature].dropna()                   # matriz sin filas con NaN (IsolationForest no acepta NaN)

# --- Entrenar el detector de anomalías multivariadas ---
model = IsolationForest(                              # instancia el modelo
    n_estimators=100,                                 # 100 árboles de aislamiento
    contamination=0.02,                               # asume ~2% de anomalías esperadas
    random_state=42                                   # semilla fija -> resultados reproducibles
)
model.fit(x)                                          # ajusta el modelo sobre las intensidades limpias

# --- Aplicar el detector y guardar resultados ---
df_intensidad.loc[x.index, 'anomaly'] = model.predict(x)                  # -1 = anomalía, 1 = normal (solo en filas de x)
df_intensidad.loc[x.index, 'anomaly_score'] = model.decision_function(x)  # score continuo (más bajo = más anómalo)

# --- FIX 2: seleccionar filas normales con == 1 (NO con != -1) ---
normal_indices = df_intensidad[df_intensidad['anomaly'] == 1].index       # índices marcados explícitamente como normales
#   Antes usabas 'anomaly' != -1, pero las filas con NaN quedan con anomaly = NaN,
#   y NaN != -1 da True -> las filas con NaN SOBREVIVÍAN. Con == 1 solo pasan las validadas como normales.

# --- Reconstruir el DataFrame limpio sin columnas auxiliares ---
original_columns = [col for col in df_intensidad.columns                  # todas las columnas...
                    if col not in ['anomaly', 'anomaly_score']]           # ...menos las dos auxiliares del detector
df_clean = df_intensidad.loc[normal_indices, original_columns].copy()     # subconjunto final: filas normales, columnas originales

# --- Guardar el CSV limpio ---
output_path = 'data/processed/intensidad_cobre_24_clean.csv'  # ruta de salida
df_clean.to_csv(output_path, index=False)                     # escribe el CSV sin el índice de pandas

# --- Guardar el modelo detector de anomalías ---
directorio_modelos = 'models'                                 # carpeta destino de modelos
os.makedirs(directorio_modelos, exist_ok=True)                # la crea si no existe (no falla si ya existe)
ruta_modelo_anomalias = 'models/anomaly_detector.joblib'      # ruta del archivo del modelo
joblib.dump(model, ruta_modelo_anomalias)                     # serializa el IsolationForest a disco

print(f'Filas finales tras limpieza: {len(df_clean)}')        # reporta cuántas filas quedaron
