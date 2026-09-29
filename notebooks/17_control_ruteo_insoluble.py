#!/usr/bin/env python3
# ============================================================
# CELDA 17 — Control negativo de la celda 16: cluster barajado (mismos
# tamaños de grupo que cluster_aug, sin información real).
# ============================================================
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import PowerTransformer, StandardScaler

from src.pipeline import config as CFG
from src.pipeline.clustering import entrenar_clustering
from src.pipeline.ruteo_insoluble import cargar_composito_crudo

sys.path.insert(0, os.path.dirname(__file__))
mod16 = __import__("16_modelos_locales_con_insoluble")


def _main():
    df = pd.read_csv(CFG.DATA_PROCESSED / "intensidad_cobre_24_completo_filtrado.csv")
    regresiones_ortho = joblib.load(CFG.MODELS_DIR / CFG.ART_ORTHO)
    X_ref = df[["n6sc"]].values
    for metal in CFG.METALES:
        df[f"{metal}_ortho"] = df[metal].values - regresiones_ortho[metal].predict(X_ref)

    stream = pd.read_csv(CFG.DATA_PROCESSED / "intensidad_cobre_24_orthogonalized.csv")
    stream["ts"] = mod16.construir_ts(stream)
    lab = cargar_composito_crudo()
    stream = pd.merge_asof(stream.sort_values("ts"),
                           lab[["ts", "ins"]].rename(columns={"ins": "ins_t"}),
                           on="ts", direction="backward").dropna(subset=["ins_t"])

    feats_aug = CFG.FEATS_CLUSTER + ["ins_t"]
    p_aug = PowerTransformer(method="yeo-johnson", standardize=False).fit(stream[feats_aug].values)
    Xp_aug = p_aug.transform(stream[feats_aug].values)
    s_aug = StandardScaler().fit(Xp_aug)
    _, gmm_aug, _, _ = entrenar_clustering(s_aug.transform(Xp_aug), k=3)

    df["ts"] = mod16.construir_ts(df)
    df = pd.merge_asof(df.sort_values("ts"),
                       lab[["ts", "ins"]].rename(columns={"ins": "ins_t"}),
                       on="ts", direction="backward").dropna(subset=["ins_t"]).reset_index(drop=True)
    X_aug_sup = s_aug.transform(p_aug.transform(df[feats_aug].values))
    df["cluster_aug"] = gmm_aug.predict(X_aug_sup)

    rng = np.random.default_rng(CFG.RANDOM_STATE)
    df["cluster_ctrl"] = rng.permutation(df["cluster_aug"].values)   # mismos tamaños, sin info real

    r2g_of, r2l_of = mod16.entrenar_y_medir(df, "cluster_aug")   # reusa cluster_aug real ya calculado arriba
    r2g_ctrl, r2l_ctrl = mod16.entrenar_y_medir(df, "cluster_ctrl")

    print(f"{'ley':6s}{'R2 local(+ins_t real)':>24s}{'R2 local(control barajado)':>28s}{'diferencia real-control':>26s}")
    for tgt in CFG.TARGETS:
        rl_aug = mod16.r2_local_ponderado(r2l_of, tgt)
        rl_ctrl = mod16.r2_local_ponderado(r2l_ctrl, tgt)
        print(f"{tgt:6s}{rl_aug:24.3f}{rl_ctrl:28.3f}{rl_aug - rl_ctrl:+26.3f}")

    print("\n-- control barajado, detalle por celda --")
    for (tgt, cl), (r2, n) in sorted(r2l_ctrl.items()):
        print(f"  {tgt} / cluster {cl}: R2={r2:.3f}  (n={n})")


if __name__ == "__main__":
    _main()
