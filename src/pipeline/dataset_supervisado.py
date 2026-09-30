# ============================================================
# src/pipeline/dataset_supervisado.py — Puente intensidades <-> leyes de lab
# ============================================================
# Reemplaza al script assay_int.py de V1. Une las intensidades clusterizadas con
# las leyes de laboratorio para construir el dataset de entrenamiento supervisado.
#
# Arrastra las INTENSIDADES CRUDAS (n1fe..n4mo, n6sc), no solo las _ortho: la
# etapa 5 (modelos) necesita las crudas para recomputar el ortho de forma
# consistente con la inferencia.
#
# Ejecutable de forma independiente (los ensayos de lab pueden venir de un CSV
# ya exportado, o pedirse en vivo a la BD):
#   python -m src.pipeline.dataset_supervisado \
#       --clusterizado data/processed/intensidad_cobre_24_clusterizado.csv \
#       --assays-csv data/raw/assays.csv
#   python -m src.pipeline.dataset_supervisado --from-db --sample-id 24
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

    Fusiona por 'instance' cuando está disponible en ambos lados (id de fila,
    no depende de cómo vengan formateadas date/time -- es lo que ya trae
    from_db.extraer() de la BD). Si df_assays no tiene 'instance' (p.ej. un CSV de
    ensayos exportado a mano sin esa columna), cae a (date, time) normalizados.

    Parámetros
    ----------
    df_assays : DataFrame con date, time, (idealmente) instance y las leyes
        (pFe, pCu, pZn, pMo, ...).
    df_clusterizado : salida de la etapa 4 (intensidades crudas + _ortho + cluster).

    Retorna
    -------
    df_completo : merge completo.
    df_filtrado : solo filas con al menos una ley de laboratorio no nula.
    """
    df_cluster = df_clusterizado.dropna(subset=["instance"]).copy()  # descarta filas sin instancia
    df_assays = df_assays.copy()                          # no mutar el original

    por_instance = "instance" in df_assays.columns        # llave robusta si está disponible
    if por_instance:
        df_assays = df_assays.dropna(subset=["instance"])
        df_assays["instance"] = df_assays["instance"].astype("int64")
        df_cluster["instance"] = df_cluster["instance"].astype("int64")
        llave = ["instance"]
    else:                                                  # --- fallback: (date, time) normalizados ---
        df_assays["date"] = df_assays["date"].astype(str).str.strip()
        df_assays["time"] = df_assays["time"].apply(_limpiar_hora)
        df_cluster["date"] = df_cluster["date"].astype(str).str.strip()
        df_cluster["time"] = df_cluster["time"].apply(_limpiar_hora)
        llave = ["date", "time"]

    # Incluye intensidades CRUDAS + n6sc, no solo las _ortho (necesarias para la etapa 5).
    # OJO: df_assays (works4cdp_assay) puede traer n1fe..n6sc junto con las leyes -> si
    # también las tomamos de df_cluster, pandas las duplica como n6sc_x/n6sc_y (la columna
    # 'n6sc' deja de existir). Por eso solo se toman de df_cluster las que NO estén ya en df_assays.
    #
    # NO se arrastran n1fe_ortho..n4mo_ortho: son la salida de la etapa 3 (YA escaladas,
    # pese al nombre "_ortho" heredado de la etapa 2) y ningún consumidor las usa -- la
    # etapa 5 (modelos.py: _recomputar_features) siempre recalcula ortho desde las crudas
    # con el artefacto congelado, nunca lee estas columnas. Se detectó en sesión que
    # dejarlas invita a reaplicar power+scaler sobre un valor ya escalado (doble escalado
    # -> colapso del GMM a un solo cluster). Ver docs/bitacora_analisis.md.
    crudas = ["n1fe", "n2cu", "n3zn", "n4mo", "n6sc"]
    crudas_de_cluster = [c for c in crudas if c not in df_assays.columns]
    cols = llave + crudas_de_cluster + [
        "cluster_kmeans", "cluster_gmm",                        # etiquetas de cluster (no features)
    ]
    cols = [c for c in dict.fromkeys(cols) if c in df_cluster.columns]  # solo las que existan, sin duplicar
    df_cluster = df_cluster[cols].copy()                  # subconjunto de columnas

    df_completo = pd.merge(df_assays, df_cluster, on=llave, how="inner")  # fusiona por la llave elegida

    filtro = ["pFe", "pCu", "pZn", "pMo", "pIns", "pSol"] # columnas de ley/laboratorio
    filtro = [c for c in filtro if c in df_completo.columns]  # solo las que existan
    df_filtrado = df_completo.dropna(subset=filtro, how="all")  # filas con al menos una ley

    return df_completo, df_filtrado                       # devuelve ambos datasets


# ============================================================
# CLI — ejecución independiente de la etapa
# ============================================================
def _main():
    import argparse

    from .config import DATA_PROCESSED, SAMPLE_ID_COURIER

    ap = argparse.ArgumentParser(description="Fusión: intensidades clusterizadas + leyes de laboratorio")
    ap.add_argument("--clusterizado", default=str(DATA_PROCESSED / "intensidad_cobre_24_clusterizado.csv"),
                    help="CSV clusterizado de entrada (salida de la etapa 4)")
    ap.add_argument("--assays-csv", default=None,
                    help="CSV ya exportado con los ensayos de laboratorio (date, time, pFe, pCu, pZn, pMo, ...)")
    ap.add_argument("--from-db", action="store_true",
                    help="En vez de --assays-csv, pide los ensayos en vivo a la BD (usa src.acquisition.from_db)")
    ap.add_argument("--sample-id", type=int, default=SAMPLE_ID_COURIER,
                    help="sample_id a extraer de la BD cuando se usa --from-db")
    ap.add_argument("--output-completo", default=str(DATA_PROCESSED / "intensidad_cobre_24_completo.csv"))
    ap.add_argument("--output-filtrado", default=str(DATA_PROCESSED / "intensidad_cobre_24_completo_filtrado.csv"))
    args = ap.parse_args()

    if not args.assays_csv and not args.from_db:
        ap.error("indica --assays-csv <archivo> o --from-db")

    df_cluster = pd.read_csv(args.clusterizado)

    if args.from_db:
        from ..acquisition.from_db import extraer
        df_assays = extraer(args.sample_id, grupos=("leyes",))
    else:
        df_assays = pd.read_csv(args.assays_csv)

    df_completo, df_filtrado = construir_dataset_supervisado(df_assays, df_cluster)

    df_completo.to_csv(args.output_completo, index=False)
    df_filtrado.to_csv(args.output_filtrado, index=False)
    print(f"Filas fusionadas (completo): {len(df_completo)} -> {args.output_completo}")
    print(f"Filas con al menos una ley (filtrado): {len(df_filtrado)} -> {args.output_filtrado}")


if __name__ == "__main__":
    _main()
