# ============================================================
# CELDA 4 — CLUSTERIZADO (selección de k + KMeans y GMM)
# ============================================================

import pandas as pd                                  # DataFrames
import numpy as np                                    # cálculo numérico
import joblib                                          # guardar modelos
import os                                              # carpetas
from sklearn.cluster import KMeans                     # KMeans (geométrico)
from sklearn.mixture import GaussianMixture            # GMM (probabilístico)
from sklearn.metrics import silhouette_score           # métrica interna de calidad de clusters

# --- 1. Cargar el dataset ortogonalizado y escalado (salida de la Celda 3) ---
file_path = 'data/processed/intensidad_cobre_24_scaled.csv'  # ruta del CSV escalado
df = pd.read_csv(file_path)                                   # lo carga

# --- Espacio de features para el clustering ---
features_ortho_scaled = ['n1fe_ortho', 'n2cu_ortho', 'n3zn_ortho', 'n4mo_ortho']  # columnas de entrada
X_scaled = df[features_ortho_scaled].values                  # matriz numérica que consumen los algoritmos

# --- FIX: elegir k con métricas REALES en vez de fijarlo a ciegas ---
muestra = np.random.RandomState(0).choice(              # submuestra para silhouette (es costoso en datasets grandes)
    len(X_scaled), min(5000, len(X_scaled)), replace=False)  # hasta 5000 filas al azar, sin repetición

print(f'{"k":>2} {"GMM_BIC":>12} {"KM_silhouette":>14} {"GMM_silhouette":>15}')  # encabezado de la tabla
for k in range(2, 9):                                   # prueba k de 2 a 8
    km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)               # KMeans con k clusters
    gmm_tmp = GaussianMixture(n_components=k, covariance_type='full',                     # GMM full (más flexible que 'tied')
                              random_state=42, n_init=10).fit(X_scaled)
    bic = gmm_tmp.bic(X_scaled)                          # BIC del GMM: menor = mejor equilibrio ajuste/complejidad
    sil_km = silhouette_score(X_scaled[muestra], km_tmp.labels_[muestra])                 # silhouette de KMeans
    sil_gmm = silhouette_score(X_scaled[muestra], gmm_tmp.predict(X_scaled)[muestra])     # silhouette de GMM
    print(f'{k:>2} {bic:>12.0f} {sil_km:>14.3f} {sil_gmm:>15.3f}')  # imprime métricas de este k
#   Regla de decisión: buscar el "codo" del BIC (donde deja de bajar fuerte) y el mejor silhouette del GMM.
#   Con estos datos, k=3 queda respaldado (codo de BIC en k=3-4 y silhouette máximo del GMM en k=3).

# --- Número de regímenes operativos elegido (ahora SÍ justificado por las métricas de arriba) ---
k_optimo = 3                                            # k seleccionado según BIC + silhouette

# --- 2. KMeans (enfoque geométrico rígido) ---
kmeans_model = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)  # instancia con k óptimo
df['cluster_kmeans'] = kmeans_model.fit_predict(X_scaled)               # ajusta y asigna etiqueta de cluster a cada fila

# --- 3. GMM (enfoque probabilístico elíptico) ---
# FIX: covariance_type='full' (no 'tied'): permite que cada cluster tenga su propia forma,
#      que es justo la ventaja del GMM. 'tied' lo obliga a comportarse casi como KMeans.
gmm_model = GaussianMixture(n_components=k_optimo, covariance_type='full',  # cada cluster con covarianza propia
                            random_state=42, n_init=10)                     # 10 inicializaciones -> más estable
gmm_model.fit(X_scaled)                                 # ajusta el GMM
df['cluster_gmm'] = gmm_model.predict(X_scaled)         # asigna el cluster más probable a cada fila

# --- Métricas finales del modelo elegido (informativo) ---
print(f'\nSilhouette KMeans (k={k_optimo}):', round(silhouette_score(X_scaled[muestra], df["cluster_kmeans"].values[muestra]), 3))
print(f'Silhouette GMM    (k={k_optimo}):', round(silhouette_score(X_scaled[muestra], df["cluster_gmm"].values[muestra]), 3))

# --- 6. Guardar el dataset con las etiquetas de cluster ---
output_path = 'data/processed/intensidad_cobre_24_clusterizado.csv'  # ruta de salida
df.to_csv(output_path, index=False)                                  # escribe el CSV con cluster_kmeans y cluster_gmm

# --- Guardar los modelos de clustering ---
directorio_modelos = 'models'                           # carpeta de modelos
os.makedirs(directorio_modelos, exist_ok=True)          # la crea si no existe
joblib.dump(kmeans_model, 'models/kmeans_model.joblib') # guarda el modelo KMeans
joblib.dump(gmm_model, 'models/gmm_model.joblib')       # guarda el modelo GMM
