#!/usr/bin/env python3
# ============================================================
# CELDA 25 — ¿El orden escalado/ortogonalización cambia Cu/Mo reales?
# ============================================================
# Completa el experimento de la celda 24: entrena un GMM nuevo con el orden
# invertido (variante B) y mide el R2 local de Cu/Mo contra ley real,
# comparado con el cluster oficial (variante A, orden actual) -- mismo arnés
# de entrenar_y_medir() que ya uso en la celda 16.
#
# Uso:
#   pixi run python notebooks/25_impacto_orden_en_cu_mo.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer, StandardScaler

from src.pipeline import config as CFG
from src.pipeline.ortogonalizacion import ortogonalizar
from src.pipeline.clustering import entrenar_clustering

mod16 = __import__("16_modelos_locales_con_insoluble")


def _main():
    print("=" * 62)
    print("PASO 1 — Entrenar clustering variante B (escalar -> ortogonalizar) en el stream completo")
    print("=" * 62)
    clean = pd.read_csv(CFG.DATA_PROCESSED / "intensidad_cobre_24_clean.csv")

    p_pre = PowerTransformer(method="yeo-johnson", standardize=False).fit(clean[CFG.METALES].values)
    Xp_pre = p_pre.transform(clean[CFG.METALES].values)
    s_pre = StandardScaler().fit(Xp_pre)
    metales_escalados = s_pre.transform(Xp_pre)

    df_pre = clean.copy()
    df_pre[CFG.METALES] = metales_escalados
    df_ortho_b, reg_ortho_b = ortogonalizar(df_pre)

    s_post = StandardScaler().fit(df_ortho_b[CFG.FEATS_CLUSTER].values)
    X_b = s_post.transform(df_ortho_b[CFG.FEATS_CLUSTER].values)
    _, gmm_b, _, _ = entrenar_clustering(X_b, k=3)
    print("GMM variante B entrenado.")

    print("\n" + "=" * 62)
    print("PASO 2 — Cluster oficial (A) vs variante B, sobre las 314 muestras con ley real")
    print("=" * 62)
    sup = pd.read_csv(CFG.DATA_PROCESSED / "intensidad_cobre_24_completo_filtrado.csv")
    regresiones_ortho = joblib.load(CFG.MODELS_DIR / CFG.ART_ORTHO)
    X_ref = sup[["n6sc"]].values
    for metal in CFG.METALES:
        sup[f"{metal}_ortho"] = sup[metal].values - regresiones_ortho[metal].predict(X_ref)

    power_a = joblib.load(CFG.MODELS_DIR / CFG.ART_POWER)
    scaler_a = joblib.load(CFG.MODELS_DIR / CFG.ART_SCALER)
    gmm_a = joblib.load(CFG.MODELS_DIR / CFG.ART_GMM)
    sup["cluster_a"] = gmm_a.predict(scaler_a.transform(power_a.transform(sup[CFG.FEATS_CLUSTER].values)))

    # variante B sobre las 314: mismas transformaciones ajustadas en el PASO 1
    metales_sup_b = s_pre.transform(p_pre.transform(sup[CFG.METALES].values))
    sup_b = sup.copy()
    sup_b[CFG.METALES] = metales_sup_b
    sup_b_ortho, _ = ortogonalizar(sup_b, reg_ortho_b)
    X_sup_b = s_post.transform(sup_b_ortho[CFG.FEATS_CLUSTER].values)
    sup["cluster_b"] = gmm_b.predict(X_sup_b)

    print("Distribución cluster_a:", sup["cluster_a"].value_counts().sort_index().to_dict())
    print("Distribución cluster_b:", sup["cluster_b"].value_counts().sort_index().to_dict())

    print("\n" + "=" * 62)
    print("PASO 3 — R2 local de Cu/Mo/Fe/Zn con cada clustering (misma CV)")
    print("=" * 62)
    r2g_a, r2l_a = mod16.entrenar_y_medir(sup, "cluster_a")
    r2g_b, r2l_b = mod16.entrenar_y_medir(sup, "cluster_b")

    print(f"{'ley':6s}{'R2 local (A, actual)':>22s}{'R2 local (B, invertido)':>25s}{'diferencia':>12s}")
    for tgt in CFG.TARGETS:
        rl_a = mod16.r2_local_ponderado(r2l_a, tgt)
        rl_b = mod16.r2_local_ponderado(r2l_b, tgt)
        print(f"{tgt:6s}{rl_a:22.3f}{rl_b:25.3f}{rl_b - rl_a:+12.3f}")


if __name__ == "__main__":
    _main()
