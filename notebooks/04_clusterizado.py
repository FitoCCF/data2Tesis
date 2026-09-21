#!/usr/bin/env python3
# ============================================================
# CELDA 4 — CLUSTERIZADO (selección de k + KMeans y GMM)
# ============================================================
# Delega en src.pipeline.clustering (diagnostico_k + entrenar_clustering).
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # independiente del CWD

import pandas as pd
import joblib

from src.pipeline.config import DATA_PROCESSED, MODELS_DIR, ART_KMEANS, ART_GMM, FEATS_CLUSTER
from src.pipeline.clustering import diagnostico_k, entrenar_clustering

# --- 1. Cargar el dataset ortogonalizado y escalado (salida de la Celda 3) ---
file_path = DATA_PROCESSED / 'intensidad_cobre_24_scaled.csv'
df = pd.read_csv(file_path)
X_scaled = df[FEATS_CLUSTER].values

# --- Elegir k con métricas reales (BIC + silhouette) ---
diagnostico_k(X_scaled)
# Regla de decisión: buscar el "codo" del BIC y el máximo silhouette del GMM.
# Con estos datos, k=3 queda respaldado (codo de BIC en k=3-4, silhouette máximo del GMM en k=3).
k_optimo = 3

# --- Entrenar KMeans + GMM con k óptimo (etapa 4) ---
kmeans_model, gmm_model, labels_kmeans, labels_gmm = entrenar_clustering(X_scaled, k=k_optimo)
df['cluster_kmeans'] = labels_kmeans
df['cluster_gmm'] = labels_gmm

# --- Guardar el dataset con las etiquetas de cluster ---
output_path = DATA_PROCESSED / 'intensidad_cobre_24_clusterizado.csv'
df.to_csv(output_path, index=False)

# --- Guardar los modelos de clustering ---
MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(kmeans_model, MODELS_DIR / ART_KMEANS)
joblib.dump(gmm_model, MODELS_DIR / ART_GMM)

print(f'Clusterizado listo (k={k_optimo}):', output_path)
