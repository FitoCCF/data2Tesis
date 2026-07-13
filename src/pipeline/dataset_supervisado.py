# ============================================================
# src/pipeline/dataset_supervisado.py — Puente intensidades ↔ leyes de lab
# ============================================================
# Reemplaza el script assay_int.py. Une las intensidades clusterizadas con las
# leyes de laboratorio para construir el dataset de entrenamiento supervisado.
#
# CORRECCIÓN CLAVE vs el original: arrastra las INTENSIDADES CRUDAS (n1fe..n4mo,
# n6sc), no solo las _ortho. La etapa 5 (modelos) necesita las crudas para
# recomputar el ortho de forma consistente con la inferencia.
# ============================================================

import pandas as pd                                       # DataFrames


def _limpiar_hora(valor) -> str:
    """Normaliza la hora: string, sin espacios, sin fracción de segundos."""
    s = str(valor).strip()                                # a string y sin espacios
    if "." in s:                                          # si trae fracción de segundos
        s = s.split(".")[0]                               # conserva solo hasta el segundo entero
    return s                                              # hora normalizada


def construir_dataset_supervisado(df_assays: pd.DataFrame,
                                  df_clusterizado: pd.DataFrame):
    """Une ensayos de laboratorio con intensidades clusterizadas.

    Parámetros
    ----------
    df_assays : DataFrame con date, time y las leyes (pFe, pCu, pZn, pMo, ...).
    df_clusterizado : salida de la etapa 4 (intensidades crudas + _ortho + cluster).

    Retorna
    -------
    df_completo : merge completo por (date, time).
    df_filtrado : solo filas con al menos una ley de laboratorio no nula.
    """
    df_cluster = df_clusterizado.dropna(subset=["instance"]).copy()  # descarta filas sin instancia

    # Normaliza date/time en ambos lados para que el merge coincida
    df_assays = df_assays.copy()                          # no mutar el original
    df_assays["date"] = df_assays["date"].astype(str).str.strip()   # date a string limpio
    df_assays["time"] = df_assays["time"].apply(_limpiar_hora)      # time normalizado

    # CORRECCIÓN: incluir intensidades CRUDAS + n6sc, no solo las _ortho
    cols = ["date", "time", "instance",
            "n1fe", "n2cu", "n3zn", "n4mo", "n6sc",       # crudas (necesarias para la etapa 5)
            "n1fe_ortho", "n2cu_ortho", "n3zn_ortho", "n4mo_ortho",  # ortogonalizadas
            "cluster_kmeans", "cluster_gmm"]              # etiquetas de cluster
    cols = [c for c in cols if c in df_cluster.columns]   # solo las que existan
    df_cluster = df_cluster[cols].copy()                  # subconjunto de columnas
    df_cluster["date"] = df_cluster["date"].astype(str).str.strip()  # date limpio
    df_cluster["time"] = df_cluster["time"].apply(_limpiar_hora)     # time normalizado

    df_completo = pd.merge(df_assays, df_cluster,         # fusiona por date+time
                           on=["date", "time"], how="inner")

    filtro = ["pFe", "pCu", "pZn", "pMo", "pIns", "pSol"] # columnas de ley/laboratorio
    filtro = [c for c in filtro if c in df_completo.columns]  # solo las que existan
    df_filtrado = df_completo.dropna(subset=filtro, how="all")  # filas con al menos una ley

    return df_completo, df_filtrado                       # devuelve ambos datasets
