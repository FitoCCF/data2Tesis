#!/usr/bin/env python3

# ============================================================
# src/pipeline/clustering.py — Etapa 4: segmentación en regímenes de mineral
# ============================================================
# Entrena KMeans y GMM (k=3) sobre las features escaladas, y expone una función
# de diagnóstico para elegir k por BIC y silhouette. En inferencia se aplica el
# modelo ya entrenado.
#
# Ejecutable de forma independiente:
#   python -m src.pipeline.clustering --input data/processed/intensidad_cobre_24_scaled.csv
#   python -m src.pipeline.clustering --input ... --diagnostico   (solo imprime tabla de k, no entrena)
# ============================================================

import numpy as np                                        # arrays numéricos
import pandas as pd                                        # DataFrames
from sklearn.cluster import KMeans                         # KMeans (geométrico)
from sklearn.mixture import GaussianMixture                # GMM (probabilístico)
from sklearn.metrics import silhouette_score               # métrica interna de clustering

from .config import RANDOM_STATE, FEATS_CLUSTER            # semilla global + columnas de features


def diagnostico_k(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 8):
    """Calcula BIC (GMM) y silhouette (KMeans y GMM) para elegir k.

    No entrena el modelo final; solo imprime la tabla para justificar la
    elección de k (codo del BIC + máximo silhouette).
    """
    muestra = np.random.RandomState(0).choice(            # submuestra para silhouette (costoso)
        len(X_scaled), min(5000, len(X_scaled)), replace=False)
    print(f'{"k":>2} {"GMM_BIC":>12} {"KM_sil":>10} {"GMM_sil":>10}')  # encabezado
    for k in range(k_min, k_max + 1):                     # recorre k candidatos
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_scaled)  # KMeans
        gmm = GaussianMixture(n_components=k, covariance_type="full",                  # GMM full
                              random_state=RANDOM_STATE, n_init=10).fit(X_scaled)
        bic = gmm.bic(X_scaled)                           # BIC del GMM (menor = mejor)
        sk = silhouette_score(X_scaled[muestra], km.labels_[muestra])                  # silhouette KMeans
        sg = silhouette_score(X_scaled[muestra], gmm.predict(X_scaled)[muestra])       # silhouette GMM
        print(f"{k:>2} {bic:>12.0f} {sk:>10.3f} {sg:>10.3f}")  # fila de la tabla


def entrenar_clustering(X_scaled: np.ndarray, k: int = 3):
    """Entrena KMeans y GMM con k fijo y devuelve ambos modelos + etiquetas.

    Retorna
    -------
    kmeans, gmm : modelos entrenados.
    labels_kmeans, labels_gmm : etiquetas de cluster por fila.
    """
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)  # instancia KMeans
    labels_kmeans = kmeans.fit_predict(X_scaled)          # ajusta y asigna cluster

    gmm = GaussianMixture(n_components=k, covariance_type="full",        # instancia GMM (covarianza propia)
                          random_state=RANDOM_STATE, n_init=10)
    gmm.fit(X_scaled)                                     # ajusta el GMM
    labels_gmm = gmm.predict(X_scaled)                    # asigna el cluster más probable

    return kmeans, gmm, labels_kmeans, labels_gmm         # devuelve modelos + etiquetas


def asignar_cluster(X_scaled: np.ndarray, modelo) -> np.ndarray:
    """Aplica un modelo de clustering ya entrenado (KMeans o GMM) a datos nuevos."""
    return modelo.predict(X_scaled)                       # etiqueta de cluster por fila


# ============================================================
# CLI — ejecución independiente de la etapa
# ============================================================
def _main():
    import argparse
    import joblib

    from .config import DATA_PROCESSED, MODELS_DIR, ART_KMEANS, ART_GMM

    ap = argparse.ArgumentParser(description="Etapa 4: clustering (KMeans + GMM)")
    ap.add_argument("--input", default=str(DATA_PROCESSED / "intensidad_cobre_24_scaled.csv"),
                    help="CSV escalado de entrada (salida de la etapa 3)")
    ap.add_argument("--output", default=str(DATA_PROCESSED / "intensidad_cobre_24_clusterizado.csv"),
                    help="Ruta del CSV clusterizado de salida")
    ap.add_argument("--k", type=int, default=3, help="Número de clusters (regímenes de mineral)")
    ap.add_argument("--diagnostico", action="store_true",
                    help="Solo imprime la tabla BIC/silhouette para k=2..8 y termina, sin entrenar ni guardar")
    ap.add_argument("--kmeans-in", default=None, help="Modelo KMeans .joblib ya entrenado (modo INFERENCIA)")
    ap.add_argument("--gmm-in", default=None, help="Modelo GMM .joblib ya entrenado (modo INFERENCIA)")
    ap.add_argument("--kmeans-out", default=str(MODELS_DIR / ART_KMEANS))
    ap.add_argument("--gmm-out", default=str(MODELS_DIR / ART_GMM))
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    X_scaled = df[FEATS_CLUSTER].values

    if args.diagnostico:
        diagnostico_k(X_scaled)
        return

    if args.kmeans_in and args.gmm_in:                     # --- modo INFERENCIA: aplica ambos modelos ---
        kmeans = joblib.load(args.kmeans_in)
        gmm = joblib.load(args.gmm_in)
        df["cluster_kmeans"] = asignar_cluster(X_scaled, kmeans)
        df["cluster_gmm"] = asignar_cluster(X_scaled, gmm)
    else:                                                    # --- modo ENTRENAMIENTO ---
        kmeans, gmm, labels_kmeans, labels_gmm = entrenar_clustering(X_scaled, k=args.k)
        df["cluster_kmeans"] = labels_kmeans
        df["cluster_gmm"] = labels_gmm
        joblib.dump(kmeans, args.kmeans_out)
        joblib.dump(gmm, args.gmm_out)
        print(f"Modelo KMeans guardado en: {args.kmeans_out}")
        print(f"Modelo GMM guardado en: {args.gmm_out}")

    _reportar_metricas(X_scaled, df["cluster_kmeans"].values, df["cluster_gmm"].values)

    df.to_csv(args.output, index=False)
    print(f"CSV clusterizado guardado en: {args.output}")


def _reportar_metricas(X_scaled: np.ndarray, labels_kmeans: np.ndarray, labels_gmm: np.ndarray):
    """Silhouette (submuestra, igual criterio que diagnostico_k) + distribución
    de filas por cluster + "firma química" (centroide en unidades escaladas)
    de cada cluster. Sirve tanto en modo entrenamiento como en modo INFERENCIA
    (p.ej. al aplicar los modelos ya calibrados sobre datos nuevos de
    producción), para comparar si el régimen se mantiene estable.

    La firma está en las mismas unidades escaladas (StandardScaler ya
    congelado en la etapa 3) en train e inferencia -> 0 siempre significa
    "promedio del período de calibración (train)", así que el mismo número de
    cluster es directamente comparable entre corridas (mismo modelo
    congelado, no se reajusta) y sirve para identificar qué tipo de mineral
    representa cada cluster (p.ej. alto n2cu_ortho = más cobre relativo)."""
    muestra = np.random.RandomState(0).choice(len(X_scaled), min(5000, len(X_scaled)), replace=False)
    sil_km = silhouette_score(X_scaled[muestra], labels_kmeans[muestra])
    sil_gmm = silhouette_score(X_scaled[muestra], labels_gmm[muestra])
    print(f"\nSilhouette KMeans: {sil_km:.3f}")
    print(f"Silhouette GMM   : {sil_gmm:.3f}")

    print("\nDistribución de filas por cluster:")
    n = len(labels_gmm)
    for cl in sorted(set(labels_gmm)):
        cnt = int((labels_gmm == cl).sum())
        print(f"  cluster_gmm={cl}: {cnt:>7d}  ({100 * cnt / n:5.1f}%)")

    print("\nFirma química por cluster (media de features escaladas; "
          "0 = promedio del período de calibración/train, +/- = por encima/debajo):")
    header = f'{"cluster":>7s}' + "".join(f"{c:>14s}" for c in FEATS_CLUSTER)
    print(header)
    for cl in sorted(set(labels_gmm)):
        medias = X_scaled[labels_gmm == cl].mean(axis=0)
        fila = f"{cl:>7d}" + "".join(f"{m:>14.3f}" for m in medias)
        print(fila)


if __name__ == "__main__":
    _main()
