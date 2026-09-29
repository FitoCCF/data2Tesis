#!/usr/bin/env python3
# ============================================================
# CELDA 16 — Etapa 5 completa: modelos locales, ruteo oficial vs ruteo por
# clustering-con-insoluble. Compara R2 contra ley REAL de laboratorio.
# ============================================================
# Cierra el experimento de las celdas 13-15: aunque el clustering con ins_t
# separa peor la química (eta^2 más bajo, celda 15), la pregunta final que
# importa para la tesis es si el R2 de los MODELOS LOCALES contra el
# laboratorio cambia. Se corre la etapa 5 real (misma validación que
# src.pipeline.modelos: KFold(5, shuffle=True) por celda, igual que el
# pipeline en producción) con dos rutas de cluster distintas:
#
#   oficial : power+scaler+gmm ya congelados (4 features, sin insoluble)
#   aug     : power+scaler+gmm reajustados con ins_t (5 features)
#
# Mismo df, mismas leyes, misma familia de modelo por ley (config.py) y misma
# validación cruzada -- la ÚNICA diferencia es qué cluster usa cada celda
# para rutear. Así el R2 medido aísla el efecto del insoluble en el ruteo.
#
# Uso:
#   pixi run python notebooks/16_modelos_locales_con_insoluble.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.preprocessing import PowerTransformer, StandardScaler

from src.pipeline import config as CFG
from src.pipeline.modelos import _construir_modelo
from src.pipeline.clustering import entrenar_clustering
from src.pipeline.ruteo_insoluble import cargar_composito_crudo


def construir_ts(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))


def entrenar_y_medir(df: pd.DataFrame, col_cluster: str) -> tuple[dict, dict]:
    """Copia el núcleo de modelos.entrenar_modelos_locales, parametrizado por
    la columna de cluster a usar (para poder comparar dos rutas de ruteo)."""
    r2_global, r2_local = {}, {}
    for tgt in CFG.TARGETS:
        familia = CFG.MODELO_POR_TARGET[tgt]
        sub = df.dropna(subset=[tgt])
        y = sub[tgt].values

        pred_g = cross_val_predict(_construir_modelo(familia), sub[CFG.FEATS_REGRESION].values, y,
                                   cv=KFold(5, shuffle=True, random_state=CFG.RANDOM_STATE))
        r2_global[tgt] = r2_score(y, pred_g)

        for cl in sorted(sub[col_cluster].unique()):
            idx = sub[col_cluster] == cl
            Xc, yc = sub.loc[idx, CFG.FEATS_REGRESION].values, sub.loc[idx, tgt].values
            k = min(5, int(idx.sum()))
            if k < 2:
                continue
            pred_l = cross_val_predict(_construir_modelo(familia), Xc, yc,
                                       cv=KFold(k, shuffle=True, random_state=CFG.RANDOM_STATE))
            r2_local[(tgt, cl)] = (r2_score(yc, pred_l), int(idx.sum()))
    return r2_global, r2_local


def r2_local_ponderado(r2_local: dict, tgt: str) -> float:
    """Promedio de R2 local ponderado por tamaño de celda (una sola cifra por ley)."""
    celdas = [(r2, n) for (t, cl), (r2, n) in r2_local.items() if t == tgt]
    if not celdas:
        return float("nan")
    total = sum(n for _, n in celdas)
    return sum(r2 * n for r2, n in celdas) / total


def _main():
    print("=" * 62)
    print("PASO 1 — Recomputar ortho (etapa 2, SIN CAMBIOS) para regresión")
    print("=" * 62)
    df = pd.read_csv(CFG.DATA_PROCESSED / "intensidad_cobre_24_completo_filtrado.csv")
    regresiones_ortho = joblib.load(CFG.MODELS_DIR / CFG.ART_ORTHO)
    X_ref = df[["n6sc"]].values
    for metal in CFG.METALES:
        df[f"{metal}_ortho"] = df[metal].values - regresiones_ortho[metal].predict(X_ref)
    print(f"Filas: {len(df)}  |  features de regresión: {CFG.FEATS_REGRESION}")

    print("\n" + "=" * 62)
    print("PASO 2 — Cluster OFICIAL (4 features, artefactos ya congelados)")
    print("=" * 62)
    power = joblib.load(CFG.MODELS_DIR / CFG.ART_POWER)
    scaler = joblib.load(CFG.MODELS_DIR / CFG.ART_SCALER)
    gmm_of = joblib.load(CFG.MODELS_DIR / CFG.ART_GMM)
    X_of = scaler.transform(power.transform(df[CFG.FEATS_CLUSTER].values))
    df["cluster_oficial"] = gmm_of.predict(X_of)
    acuerdo = (df["cluster_oficial"] == df["cluster_gmm"]).mean()
    print(f"Acuerdo con cluster_gmm ya guardado (sanity check): {acuerdo:.1%}")
    print("Distribución:", df["cluster_oficial"].value_counts().sort_index().to_dict())

    print("\n" + "=" * 62)
    print("PASO 3 — Cluster AUMENTADO (+ins_t, reajustado desde el stream completo)")
    print("=" * 62)
    stream = pd.read_csv(CFG.DATA_PROCESSED / "intensidad_cobre_24_orthogonalized.csv")
    stream["ts"] = construir_ts(stream)
    lab = cargar_composito_crudo()
    stream = pd.merge_asof(stream.sort_values("ts"),
                           lab[["ts", "ins"]].rename(columns={"ins": "ins_t"}),
                           on="ts", direction="backward").dropna(subset=["ins_t"])

    feats_aug = CFG.FEATS_CLUSTER + ["ins_t"]
    p_aug = PowerTransformer(method="yeo-johnson", standardize=False).fit(stream[feats_aug].values)
    Xp_aug = p_aug.transform(stream[feats_aug].values)
    s_aug = StandardScaler().fit(Xp_aug)
    X_aug_stream = s_aug.transform(Xp_aug)
    _, gmm_aug, _, _ = entrenar_clustering(X_aug_stream, k=3)

    df["ts"] = construir_ts(df)
    df = pd.merge_asof(df.sort_values("ts"),
                       lab[["ts", "ins"]].rename(columns={"ins": "ins_t"}),
                       on="ts", direction="backward")
    n_antes = len(df)
    df = df.dropna(subset=["ins_t"]).reset_index(drop=True)
    print(f"Filas con ins_t conocido: {n_antes} -> {len(df)}")

    X_aug_sup = s_aug.transform(p_aug.transform(df[feats_aug].values))
    df["cluster_aug"] = gmm_aug.predict(X_aug_sup)
    print("Distribución:", df["cluster_aug"].value_counts().sort_index().to_dict())

    print("\n" + "=" * 62)
    print("PASO 4 — Etapa 5: entrenar modelos locales con cada ruteo (misma CV)")
    print("=" * 62)
    r2g_of, r2l_of = entrenar_y_medir(df, "cluster_oficial")
    r2g_aug, r2l_aug = entrenar_y_medir(df, "cluster_aug")

    print(f"\n{'ley':6s}{'R2 global':>11s}{'R2 local(oficial)':>19s}{'R2 local(+ins_t)':>18s}{'diferencia':>12s}")
    for tgt in CFG.TARGETS:
        rg = r2g_of[tgt]                       # el global no depende del cluster -> igual en ambos
        rl_of = r2_local_ponderado(r2l_of, tgt)
        rl_aug = r2_local_ponderado(r2l_aug, tgt)
        print(f"{tgt:6s}{rg:11.3f}{rl_of:19.3f}{rl_aug:18.3f}{rl_aug - rl_of:+12.3f}")

    print("\nDetalle por celda (ley, cluster) -> R2 (n):")
    print("\n-- ruteo OFICIAL --")
    for (tgt, cl), (r2, n) in sorted(r2l_of.items()):
        print(f"  {tgt} / cluster {cl}: R2={r2:.3f}  (n={n})")
    print("\n-- ruteo +ins_t --")
    for (tgt, cl), (r2, n) in sorted(r2l_aug.items()):
        print(f"  {tgt} / cluster {cl}: R2={r2:.3f}  (n={n})")


if __name__ == "__main__":
    _main()
