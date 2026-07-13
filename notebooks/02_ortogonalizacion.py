# ============================================================
# CELDA 2 — ORTOGONALIZACIÓN (corrección por dilución vía n6sc)
# ============================================================

import pandas as pd                                  # pandas: manejo de DataFrames
from sklearn.linear_model import LinearRegression     # regresión lineal simple (para ortogonalizar)
import joblib                                          # guardar los modelos de regresión
import os                                              # crear carpetas si hace falta

# --- 1. Cargar el dataset limpio (salida de la Celda 1) ---
file_path = 'data/processed/intensidad_cobre_24_clean.csv'  # ruta del CSV ya limpio
df = pd.read_csv(file_path)                                  # lo carga a un DataFrame

# --- Definir variable de referencia y metales a corregir ---
X = df[['n6sc']].values                               # X = sólido en la muestra (proxy de dilución), como matriz 2D
metals = ['n1fe', 'n2cu', 'n3zn', 'n4mo']             # canales de metal que se ortogonalizan contra n6sc

# --- 2. Preparar estructuras de salida ---
df_ortho = df.copy()                                  # copia del DataFrame donde se agregarán las columnas _ortho
processed_metals = []                                 # lista de metales efectivamente procesados
modelos_ortogonalizacion = {}                         # diccionario: metal -> modelo de regresión ajustado

# --- Ortogonalización por regresión residual, canal por canal ---
for metal in metals:                                  # itera sobre cada metal
    if df[metal].isna().all():                        # si la columna está 100% vacía (todo NaN)...
        print(f"Columna '{metal}' omitida: está completamente vacía (NaN).")  # avisa
        continue                                       # ...la salta y pasa al siguiente metal

    y = df[metal].values                              # y = intensidad cruda del metal (variable dependiente)

    model = LinearRegression()                        # instancia la regresión lineal
    model.fit(X, y)                                   # ajusta:  metal ≈ a·n6sc + b  (parte explicada por la dilución)

    modelos_ortogonalizacion[metal] = model           # guarda el modelo (coeficientes) para reusarlo en tiempo real

    residuals = y - model.predict(X)                  # residuo = metal - parte explicada por n6sc
    #   El residuo es la señal del metal LIBRE del efecto de dilución/sólido (centrada en 0)

    df_ortho[f'{metal}_ortho'] = residuals            # nueva columna con la intensidad ortogonalizada
    processed_metals.append(metal)                    # registra que este metal sí se procesó

# --- 3. Eliminar las columnas crudas ya reemplazadas por su versión _ortho ---
df_ortho = df_ortho.drop(columns=processed_metals)    # quita n1fe, n2cu, n3zn, n4mo originales
#   NOTA: n6sc se CONSERVA en df_ortho. Para el CLUSTERING de tipo de mineral está bien
#   dejar n6sc fuera de las features (se agrupa por mineralogía, no por densidad de pulpa).
#   PERO si más adelante estimas leyes (pCu/pFe/pMo), ahí NO descartes n6sc: aporta info de la ley.

# --- 4. Guardar el dataset ortogonalizado ---
output_path = 'data/processed/intensidad_cobre_24_orthogonalized.csv'  # ruta de salida
df_ortho.to_csv(output_path, index=False)                              # escribe el CSV

# --- Guardar los modelos de regresión de la ortogonalización ---
directorio_modelos = 'models'                                     # carpeta de modelos
os.makedirs(directorio_modelos, exist_ok=True)                    # la crea si no existe
ruta_archivo_regresiones = 'models/orthogonalization_regressions.joblib'  # ruta del archivo
joblib.dump(modelos_ortogonalizacion, ruta_archivo_regresiones)   # guarda el diccionario de regresiones
#   Guardar estos modelos es CLAVE: en tiempo real debes aplicar EXACTAMENTE los mismos
#   coeficientes (no re-ajustar), igual que un scaler congelado.
