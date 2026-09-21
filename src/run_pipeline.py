# ============================================================
# src/run_pipeline.py — Orquestador: corre las etapas encadenadas
# ============================================================
# Cada etapa (src/pipeline/*.py) es ejecutable de forma independiente vía
# `python -m src.pipeline.<etapa>`. Este script las encadena en un solo
# comando, usando las mismas rutas por defecto que cada CLI individual, y
# permite detenerse en cualquier etapa con --hasta-etapa (p.ej. para dejar el
# pipeline "en línea solo hasta clustering", sin entrenar la etapa 5).
#
# --input debe ser el split TRAIN de scripts/merge_cobre_data.py
# (intensidades_cobre_train.csv), NO intensidades_cobre.csv (el archivo
# completo sin dividir) ni el _test.csv (subconjunto de VALIDACIÓN del mismo
# período histórico, no se re-entrena con él).
#
# Los datos de PRODUCCIÓN (p.ej. septiembre en adelante) no son el _test.csv:
# son un extracto aparte que nunca pasa por aquí -> se scorean después con
# `python -m src.pipeline.inferencia` (EstimadorCluster/EstimadorHibrido),
# usando los artefactos que este orquestador deja en models/. Ver "Flujo
# train/test vs. producción" en el README.
#
# Uso:
#   python -m src.run_pipeline --input data/processed/intensidades_cobre_train.csv
#   python -m src.run_pipeline --input ... --hasta-etapa clustering
#   python -m src.run_pipeline --input ... --assays-csv data/raw/assays.csv   # incluye fusión + etapa 5
# ============================================================

import argparse
import joblib
import pandas as pd

from .pipeline.config import (DATA_PROCESSED, MODELS_DIR,
                              ART_ANOMALIAS, ART_ORTHO, ART_POWER, ART_SCALER,
                              ART_KMEANS, ART_GMM, ART_REGRESION, FEATS_CLUSTER)
from .pipeline.limpieza import limpiar
from .pipeline.ortogonalizacion import ortogonalizar
from .pipeline.escalado import escalar
from .pipeline.clustering import entrenar_clustering
from .pipeline.dataset_supervisado import construir_dataset_supervisado
from .pipeline.modelos import entrenar_modelos_locales

ETAPAS = ["limpieza", "ortogonalizacion", "escalado", "clustering", "fusion", "modelos"]


def main():
    ap = argparse.ArgumentParser(description="Corre el pipeline completo (o hasta cierta etapa) en un solo comando")
    ap.add_argument("--input", required=True,
                    help="CSV de intensidades crudas (entrada de la etapa 1). Debe ser el split TRAIN "
                         "de merge_cobre_data.py -- no el archivo completo ni el split TEST")
    ap.add_argument("--hasta-etapa", choices=ETAPAS, default="clustering",
                    help="Última etapa a ejecutar (default: clustering, no requiere leyes de lab)")
    ap.add_argument("--assays-csv", default=None,
                    help="Requerido si --hasta-etapa es 'fusion' o 'modelos': CSV con leyes de laboratorio")
    ap.add_argument("--k", type=int, default=3, help="Número de clusters para la etapa 4")
    args = ap.parse_args()

    if args.hasta_etapa in ("fusion", "modelos") and not args.assays_csv:
        ap.error("--hasta-etapa fusion|modelos requiere --assays-csv")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # --- Etapa 1: limpieza ---
    print("== Etapa 1: limpieza ==")
    df = pd.read_csv(args.input)
    df_clean, detector = limpiar(df)
    df_clean.to_csv(DATA_PROCESSED / "intensidad_cobre_24_clean.csv", index=False)
    joblib.dump(detector, MODELS_DIR / ART_ANOMALIAS)
    print(f"  filas: {len(df_clean)}")
    if args.hasta_etapa == "limpieza":
        return

    # --- Etapa 2: ortogonalización ---
    print("== Etapa 2: ortogonalización ==")
    df_ortho, regresiones = ortogonalizar(df_clean)
    df_ortho.to_csv(DATA_PROCESSED / "intensidad_cobre_24_orthogonalized.csv", index=False)
    joblib.dump(regresiones, MODELS_DIR / ART_ORTHO)
    if args.hasta_etapa == "ortogonalizacion":
        return

    # --- Etapa 3: escalado ---
    print("== Etapa 3: escalado ==")
    X_scaled, power, scaler = escalar(df_ortho)
    df_scaled = df_ortho.copy()
    df_scaled[FEATS_CLUSTER] = X_scaled
    df_scaled.to_csv(DATA_PROCESSED / "intensidad_cobre_24_scaled.csv", index=False)
    joblib.dump(power, MODELS_DIR / ART_POWER)
    joblib.dump(scaler, MODELS_DIR / ART_SCALER)
    if args.hasta_etapa == "escalado":
        return

    # --- Etapa 4: clustering ---
    print("== Etapa 4: clustering ==")
    kmeans, gmm, labels_kmeans, labels_gmm = entrenar_clustering(X_scaled, k=args.k)
    df_cluster = df_scaled.copy()
    df_cluster["cluster_kmeans"] = labels_kmeans
    df_cluster["cluster_gmm"] = labels_gmm
    df_cluster.to_csv(DATA_PROCESSED / "intensidad_cobre_24_clusterizado.csv", index=False)
    joblib.dump(kmeans, MODELS_DIR / ART_KMEANS)
    joblib.dump(gmm, MODELS_DIR / ART_GMM)
    print(f"  k={args.k}")
    if args.hasta_etapa == "clustering":
        return

    # --- Fusión con leyes de laboratorio ---
    print("== Fusión con leyes de laboratorio ==")
    df_assays = pd.read_csv(args.assays_csv)
    df_completo, df_filtrado = construir_dataset_supervisado(df_assays, df_cluster)
    df_completo.to_csv(DATA_PROCESSED / "intensidad_cobre_24_completo.csv", index=False)
    df_filtrado.to_csv(DATA_PROCESSED / "intensidad_cobre_24_completo_filtrado.csv", index=False)
    print(f"  filas con >=1 ley: {len(df_filtrado)}")
    if args.hasta_etapa == "fusion":
        return

    # --- Etapa 5: modelos locales ---
    print("== Etapa 5: modelos locales ==")
    bundle = entrenar_modelos_locales(df_filtrado, regresiones, power, scaler, gmm)
    joblib.dump(bundle, MODELS_DIR / ART_REGRESION)
    for tgt, r2 in bundle["r2_global_por_ley"].items():
        print(f"  {tgt}: R2 global={r2:.3f}")

    print("\nPipeline completo. Artefactos en:", MODELS_DIR)


if __name__ == "__main__":
    main()
