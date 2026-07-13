# ============================================================
# src/pipeline/modelos.py — Etapa 5: modelos locales + tabla de ruteo
# ============================================================
# Entrena un modelo de regresión por cada (ley, cluster) y uno global de
# respaldo. Construye la tabla de ruteo (usa local solo si supera al global en
# validación cruzada). Recomputa ortho y cluster con los modelos congelados para
# que entrenamiento e inferencia sean idénticos.
# ============================================================

import numpy as np                                        # arrays numéricos
import pandas as pd                                        # DataFrames
from sklearn.linear_model import Ridge                     # regresión regularizada
from sklearn.svm import SVR                                # regresión de vectores soporte
from sklearn.cross_decomposition import PLSRegression      # mínimos cuadrados parciales
from sklearn.preprocessing import StandardScaler           # estandarización interna
from sklearn.pipeline import Pipeline                      # encadena escala + modelo
from sklearn.model_selection import KFold, cross_val_predict  # validación cruzada
from sklearn.metrics import r2_score                       # métrica R2

from .config import (METALES, FEATS_CLUSTER, FEATS_REGRESION,   # constantes
                     TARGETS, MODELO_POR_TARGET, RANDOM_STATE)


def _construir_modelo(nombre: str) -> Pipeline:
    """Devuelve un pipeline (estandarizar -> modelo) según la familia elegida."""
    if nombre == "SVR":                                   # cobre
        modelo = SVR(kernel="rbf", C=10, epsilon=0.3)     # SVR con kernel radial
    elif nombre == "Ridge":                               # hierro / molibdeno
        modelo = Ridge(alpha=1.0)                         # Ridge con regularización moderada
    else:                                                 # zinc
        modelo = PLSRegression(n_components=2)            # PLS con 2 componentes
    return Pipeline([("scaler", StandardScaler()), ("modelo", modelo)])  # escala + modelo


def _recomputar_features(df: pd.DataFrame, regresiones_ortho: dict,
                         power, scaler_cluster, gmm) -> pd.DataFrame:
    """Recalcula ortho y cluster con los modelos CONGELADOS (consistencia
    entrenamiento/inferencia). Agrega columnas '<metal>_ortho' y 'cluster'."""
    X_ref = df[["n6sc"]].values                           # referencia de dilución
    for metal in METALES:                                 # por cada metal
        df[f"{metal}_ortho"] = df[metal].values - regresiones_ortho[metal].predict(X_ref)  # residuo
    X_cluster = scaler_cluster.transform(power.transform(df[FEATS_CLUSTER].values))  # ortho->power->escala
    df["cluster"] = gmm.predict(X_cluster)                # cluster consistente con inferencia
    return df                                             # DataFrame con ortho + cluster


def entrenar_modelos_locales(df: pd.DataFrame, regresiones_ortho: dict,
                             power, scaler_cluster, gmm) -> dict:
    """Entrena modelos locales y globales, construye la tabla de ruteo y
    devuelve el bundle completo listo para serializar.

    Parámetros
    ----------
    df : dataset de calibración (intensidades crudas + leyes de laboratorio).
    regresiones_ortho, power, scaler_cluster, gmm : artefactos congelados de
        las etapas 2-4 (para recomputar ortho y cluster de forma consistente).

    Retorna
    -------
    bundle : dict con modelos_locales, modelos_globales, tabla_ruteo, etc.
    """
    df = _recomputar_features(df.copy(), regresiones_ortho, power, scaler_cluster, gmm)  # features consistentes
    df = df.dropna(subset=TARGETS).reset_index(drop=True) # quita filas sin ley medida

    modelos_locales, modelos_globales = {}, {}            # diccionarios de modelos
    r2_global_por_ley, r2_local_por_celda = {}, {}        # trazabilidad de R2

    for tgt in TARGETS:                                   # por cada ley
        familia = MODELO_POR_TARGET[tgt]                  # familia de modelo para esa ley
        y = df[tgt].values                                # vector de la ley

        # --- Modelo GLOBAL: validación cruzada + ajuste final ---
        pred_g = cross_val_predict(_construir_modelo(familia), df[FEATS_REGRESION].values, y,
                                   cv=KFold(5, shuffle=True, random_state=RANDOM_STATE))  # CV global
        r2_global_por_ley[tgt] = r2_score(y, pred_g)      # R2 global
        modelos_globales[tgt] = _construir_modelo(familia).fit(df[FEATS_REGRESION].values, y)  # ajuste final

        # --- Modelos LOCALES: uno por cluster ---
        for cl in sorted(df["cluster"].unique()):         # por cada cluster
            idx = df["cluster"] == cl                     # máscara de ese cluster
            Xc = df.loc[idx, FEATS_REGRESION].values      # features del cluster
            yc = df.loc[idx, tgt].values                  # ley del cluster
            k = min(5, int(idx.sum()))                    # folds (no más que muestras)
            pred_l = cross_val_predict(_construir_modelo(familia), Xc, yc,
                                       cv=KFold(k, shuffle=True, random_state=RANDOM_STATE))  # CV local
            r2_local_por_celda[(tgt, cl)] = r2_score(yc, pred_l)  # R2 local
            modelos_locales[(tgt, cl)] = _construir_modelo(familia).fit(Xc, yc)  # ajuste local final

    # --- Tabla de ruteo: local solo si supera al global ---
    tabla_ruteo = {}                                      # (ley, cluster) -> 'local' | 'global'
    for tgt in TARGETS:                                   # por cada ley
        for cl in sorted(df["cluster"].unique()):         # por cada cluster
            usa_local = r2_local_por_celda[(tgt, cl)] > r2_global_por_ley[tgt]  # compara R2
            tabla_ruteo[(tgt, cl)] = "local" if usa_local else "global"        # decisión congelada

    return {                                              # bundle a serializar
        "modelos_locales": modelos_locales,               # (ley, cluster) -> pipeline
        "modelos_globales": modelos_globales,             # ley -> pipeline
        "features": FEATS_REGRESION,                      # orden de features esperado
        "targets": TARGETS,                               # leyes estimadas
        "cluster_col": "cluster",                         # columna de cluster usada
        "tabla_ruteo": tabla_ruteo,                       # decisiones local/global
        "r2_global_por_ley": r2_global_por_ley,           # trazabilidad
        "r2_local_por_celda": r2_local_por_celda,         # trazabilidad
    }
