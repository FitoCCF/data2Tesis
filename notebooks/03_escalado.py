# ============================================================
# CELDA 3 — ESCALADO (corrección de asimetría + estandarización)
# ============================================================

import pandas as pd                                  # manejo de DataFrames
import joblib                                          # guardar transformadores
import os                                              # carpetas del sistema
from sklearn.preprocessing import PowerTransformer, StandardScaler  # transformadores de escala

# --- 1. Cargar el dataset ortogonalizado (salida de la Celda 2) ---
input_path = 'data/processed/intensidad_cobre_24_orthogonalized.csv'  # ruta del CSV ortogonalizado
df = pd.read_csv(input_path)                                          # lo carga

# --- Features que entran al clustering (las ortogonalizadas) ---
features_to_scale = ['n1fe_ortho', 'n2cu_ortho', 'n3zn_ortho', 'n4mo_ortho']  # columnas a escalar

df_scaled = df.copy()                                 # copia donde guardaremos las versiones escaladas

# --- 2. FIX: corregir asimetría ANTES de estandarizar (Yeo-Johnson) ---
power = PowerTransformer(method='yeo-johnson', standardize=False)     # instancia el transformador de potencia
#   Yeo-Johnson acerca cada variable a una forma gaussiana; admite valores negativos
#   (los residuos ortogonalizados pueden ser negativos). standardize=False -> solo corrige forma, no escala aún.
X_power = power.fit_transform(df[features_to_scale])                  # ajusta y transforma las 4 columnas

# --- 3. Estandarizar (media 0, desviación 1) ---
scaler = StandardScaler()                             # instancia el estandarizador
X_scaled = scaler.fit_transform(X_power)              # ajusta y transforma sobre los datos ya des-sesgados
#   KMeans/GMM usan distancia euclidiana: sin estandarizar, la variable de mayor rango domina el clustering.

df_scaled[features_to_scale] = X_scaled               # reemplaza las columnas por su versión escalada

# --- 4. Guardar el dataset escalado ---
output_path = 'data/processed/intensidad_cobre_24_scaled.csv'  # ruta de salida
df_scaled.to_csv(output_path, index=False)                     # escribe el CSV

# --- Guardar AMBOS transformadores (orden importa en tiempo real) ---
directorio_modelos = 'models'                         # carpeta de modelos
os.makedirs(directorio_modelos, exist_ok=True)        # la crea si no existe

joblib.dump(power, 'models/power_transformer.joblib')  # guarda el Yeo-Johnson
joblib.dump(scaler, 'models/scaler.joblib')            # guarda el StandardScaler
#   En producción se aplican EN ESTE ORDEN: primero power_transformer, luego scaler.
