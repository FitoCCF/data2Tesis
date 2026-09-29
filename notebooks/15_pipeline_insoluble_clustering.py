#!/usr/bin/env python3
# ============================================================
# CELDA 15 — Pipeline con insoluble en clustering (etapa 0.5 + 3 + 4)
# ============================================================
# Implementa el diseño acordado:
#
#   LABCOMPOSITO.csv (12h, ins)  ---+
#                                    +--> alineación (0.5) --> ins_t (15min)
#   intensidad_cobre_24_orthogonalized.csv (15min, ya limpio+ortogonalizado,
#   etapas 1-2 SIN CAMBIOS) ---------+
#                                          |
#                                          v
#   Etapa 3 (extendida)  Yeo-Johnson + StandardScaler sobre
#                        [n1fe_ortho..n4mo_ortho] + [ins_t]
#                                          |
#                                          v
#   Etapa 4              GMM/KMeans k=3 (el k que ya usa el proyecto)
#
# ins_t = último insoluble de laboratorio CONOCIDO (merge_asof backward) --
# igual que ins_lag en el experimento de ruteo: nunca mira el compósito
# futuro.
#
# CORRECCIÓN IMPORTANTE (encontrada en esta sesión): las columnas
# n1fe_ortho..n4mo_ortho de intensidad_cobre_24_completo_filtrado.csv NO son
# los valores ortogonalizados crudos -- están YA escalados (mismo orden de
# magnitud que intensidad_cobre_24_scaled.csv, media~0 std~1), pese al
# nombre. Verificado: aplicar power+scaler+gmm oficiales sobre esas columnas
# tal cual da 100% de las 314 filas en un solo cluster (colapso); usando en
# cambio los ortho CRUDOS del mismo timestamp desde
# intensidad_cobre_24_orthogonalized.csv (existen para las 314/314 filas), el
# modelo oficial reproduce el cluster_gmm ya guardado con 100% de acuerdo.
# Este script usa siempre los crudos del stream, nunca las columnas _ortho
# del archivo supervisado.
#
# Comparación con datos de laboratorio: eta^2 = SS_entre_grupos / SS_total
# por ley, sobre las 314 muestras con ley real -- misma métrica de
# "dispersión entre/dentro" de la bitácora (2.6).
#
# Uso:
#   pixi run python notebooks/15_pipeline_insoluble_clustering.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler

from src.pipeline import config as CFG
from src.pipeline.ruteo_insoluble import cargar_composito_crudo
from src.pipeline.clustering import diagnostico_k, entrenar_clustering


def construir_ts(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))


def escalar_cols(frame: pd.DataFrame, cols: list[str]):
    p = PowerTransformer(method="yeo-johnson", standardize=False).fit(frame[cols].values)
    Xp = p.transform(frame[cols].values)
    s = StandardScaler().fit(Xp)
    return s.transform(Xp), p, s


def eta2(valores: pd.Series, etiquetas: pd.Series) -> float:
    d = pd.DataFrame({"v": valores, "g": etiquetas}).dropna()
    media = d["v"].mean()
    ss_total = ((d["v"] - media) ** 2).sum()
    if ss_total == 0:
        return float("nan")
    ss_entre = d.groupby("g")["v"].apply(lambda s: len(s) * (s.mean() - media) ** 2).sum()
    return float(ss_entre / ss_total)


def _main():
    print("=" * 62)
    print("PASO 1 — Etapa 0.5: alinear insoluble sobre el stream ya limpio+ortogonalizado")
    print("=" * 62)
    stream = pd.read_csv(CFG.DATA_PROCESSED / "intensidad_cobre_24_orthogonalized.csv")
    stream["ts"] = construir_ts(stream)
    stream = stream.sort_values("ts").reset_index(drop=True)

    lab = cargar_composito_crudo()
    stream = pd.merge_asof(stream, lab[["ts", "ins"]].rename(columns={"ins": "ins_t"}),
                           on="ts", direction="backward")
    n_total = len(stream)
    stream = stream.dropna(subset=["ins_t"]).reset_index(drop=True)
    print(f"Stream orthogonalizado: {n_total} filas -> {len(stream)} con ins_t conocido")

    print("\n" + "=" * 62)
    print("PASO 2 — Etapa 3: escalado baseline (4 feats) vs aumentado (+ins_t) vs control")
    print("=" * 62)
    feats_base = CFG.FEATS_CLUSTER
    feats_aug = CFG.FEATS_CLUSTER + ["ins_t"]

    X_base, p_base, s_base = escalar_cols(stream, feats_base)
    X_aug, p_aug, s_aug = escalar_cols(stream, feats_aug)

    rng = np.random.default_rng(CFG.RANDOM_STATE)
    stream_ctrl = stream.copy()
    stream_ctrl["ins_t"] = rng.permutation(stream_ctrl["ins_t"].values)
    X_ctrl, p_ctrl, s_ctrl = escalar_cols(stream_ctrl, feats_aug)

    print("\n--- Diagnóstico k: BASELINE (solo Concentrado Colectivo) ---")
    diagnostico_k(X_base)
    print("\n--- Diagnóstico k: AUMENTADO (+ins_t real) ---")
    diagnostico_k(X_aug)
    print("\n--- Diagnóstico k: CONTROL (+ins_t barajado en el entrenamiento) ---")
    diagnostico_k(X_ctrl)

    print("\n" + "=" * 62)
    print("PASO 3 — Etapa 4: entrenar k=3")
    print("=" * 62)
    _, gmm_base, _, _ = entrenar_clustering(X_base, k=3)
    _, gmm_aug, _, _ = entrenar_clustering(X_aug, k=3)
    _, gmm_ctrl, _, _ = entrenar_clustering(X_ctrl, k=3)

    print("\n" + "=" * 62)
    print("PASO 4 — Comparación contra datos REALES de laboratorio (314 muestras puntuales)")
    print("=" * 62)
    sup = pd.read_csv(CFG.DATA_PROCESSED / "intensidad_cobre_24_completo_filtrado.csv")
    sup["ts"] = construir_ts(sup)

    # CLAVE: se descartan las columnas _ortho de sup (mal etiquetadas, ya
    # escaladas) y se toman los ortho CRUDOS del stream por timestamp.
    leyes_cols = [c for c in ["pFe", "pCu", "pMo", "pZn", "pIns", "pSol", "cluster_gmm"] if c in sup.columns]
    sup = sup[["ts"] + leyes_cols].merge(stream[["ts", "ins_t"] + feats_base], on="ts", how="inner")
    print(f"Muestras supervisadas emparejadas con ortho crudo + ins_t: {len(sup)} / 314")

    Xb_sup = s_base.transform(p_base.transform(sup[feats_base].values))
    Xa_sup = s_aug.transform(p_aug.transform(sup[feats_aug].values))
    Xc_sup = s_ctrl.transform(p_ctrl.transform(sup[feats_aug].values))

    sup["cluster_oficial"] = sup["cluster_gmm"]
    sup["cluster_base_refit"] = gmm_base.predict(Xb_sup)
    sup["cluster_aug"] = gmm_aug.predict(Xa_sup)
    sup["cluster_ctrl"] = gmm_ctrl.predict(Xc_sup)

    # sanity check: el refit del baseline debe reproducir (casi) el oficial
    acuerdo = (sup["cluster_base_refit"] == sup["cluster_oficial"]).mean()
    print(f"Acuerdo cluster_base_refit vs cluster_oficial (sanity check): {acuerdo:.1%}")

    print("\neta^2 por ley (SS_entre_clusters / SS_total; 0=no separa, 1=separación perfecta):\n")
    print(f"{'ley':6s}{'oficial(prod)':>14s}{'base(refit)':>14s}{'+ins_t real':>14s}{'+ins_t control':>16s}")
    for ley in ["pFe", "pCu", "pMo", "pZn", "pIns", "pSol"]:
        if ley not in sup.columns:
            continue
        e_of = eta2(sup[ley], sup["cluster_oficial"])
        e_br = eta2(sup[ley], sup["cluster_base_refit"])
        e_ag = eta2(sup[ley], sup["cluster_aug"])
        e_ct = eta2(sup[ley], sup["cluster_ctrl"])
        print(f"{ley:6s}{e_of:14.3f}{e_br:14.3f}{e_ag:14.3f}{e_ct:16.3f}")

    print("\nDistribución de filas por cluster (n={}):".format(len(sup)))
    for col in ["cluster_oficial", "cluster_base_refit", "cluster_aug", "cluster_ctrl"]:
        conteo = sup[col].value_counts().sort_index()
        print(f"  {col:20s}: " + ", ".join(f"{k}={v}" for k, v in conteo.items()))


if __name__ == "__main__":
    _main()
