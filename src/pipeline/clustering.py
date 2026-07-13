# ============================================================
# src/pipeline/clustering.py — Etapa 4: segmentación en regímenes de mineral
# ============================================================
# Entrena KMeans y GMM (k=3) sobre las features escaladas, y expone una función
# de diagnóstico para elegir k por BIC y silhouette. En inferencia se aplica el
# modelo ya entrenado.
# ============================================================

import numpy as np                                        # arrays numéricos
from sklearn.cluster import KMeans                         # KMeans (geométrico)
from sklearn.mixture import GaussianMixture                # GMM (probabilístico)
from sklearn.metrics import silhouette_score               # métrica interna de clustering

from .config import RANDOM_STATE                           # semilla global


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
