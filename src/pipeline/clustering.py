#!/usr/bin/env python3
# ============================================================
# src/pipeline/clustering.py — Etapa 4: grupos de mineral (GMM)
# ============================================================
# GMM de covarianza completa sobre los 3 log-cocientes escalados (etapa 3),
# AJUSTADO SOLO con las lecturas válidas del período 'train' y aplicado a
# todas (train, test, producción).
#
# Número de grupos (K_GRUPOS en config): lo decide diagnostico_k(), no se fija
# a priori. Medido el 2026-09-30 sobre ~18,6k lecturas de train, con la
# corrección de dilución de la etapa 3 aplicada a lr_fe_cu:
#
#     k   silhouette   estabilidad (ARI entre mitades)
#     2      0.290          0.958
#     3      0.230          0.661   <- menos estable: el k=3 histórico no es el mejor
#     4      0.249          0.856   <- máximo local de ambas métricas para k >= 3
#     5+    <=0.227         0.43-0.62
#
# (Sin la corrección de dilución: k=4 daba silhouette 0.239 / ARI 0.833 y k=3
# 0.232 / 0.382; la corrección refuerza la partición en 4.)
#
# El BIC baja monótono con k (con ~18k puntos siempre lo hace): no sirve para
# elegir. Silhouette < 0.3 en todo k: la composición es un continuo y los
# grupos son una partición de ese continuo, no nubes separadas.
#
# Cada grupo se NOMBRA por su centroide: los elementos cuyo log-cociente
# contra Cu queda > UMBRAL_NOMBRE desviaciones sobre el promedio de train
# (ordenados de mayor a menor); si ninguno, el grupo es relativamente rico en
# Cu. El número interno del GMM es arbitrario; el nombre no.
#
# Ejecutable de forma independiente:
#   python -m src.pipeline.clustering --diagnostico     # tabla de k, no entrena
#   python -m src.pipeline.clustering                   # entrena con K_GRUPOS y asigna
#   python -m src.pipeline.clustering --gmm-in models/gmm_model.joblib --input nuevo.csv
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

from .config import RANDOM_STATE, K_GRUPOS
from .escalado import FEATS_ESCALADAS

ELEMENTO_DE_FEATURE = {"lr_fe_cu_z": "Fe", "lr_zn_cu_z": "Zn", "lr_mo_cu_z": "Mo"}
UMBRAL_NOMBRE = 0.3


def _train(df: pd.DataFrame) -> np.ndarray:
    return df.loc[(df["periodo"] == "train") & df["valida"], FEATS_ESCALADAS].to_numpy()


def diagnostico_k(df: pd.DataFrame, k_min: int = 2, k_max: int = 8, n_mitades: int = 3) -> pd.DataFrame:
    """BIC, silhouette (submuestra de 5000) y estabilidad por k, sobre train.

    Estabilidad = ARI medio entre dos GMM ajustados en mitades disjuntas
    aleatorias de train y comparados sobre todo train. Cerca de 1: la
    partición no depende de qué lecturas se usaron; bajo: es un artefacto.
    """
    X = _train(df)
    muestra = np.random.RandomState(0).choice(len(X), min(5000, len(X)), replace=False)
    filas = []
    for k in range(k_min, k_max + 1):
        g = GaussianMixture(k, covariance_type="full", random_state=RANDOM_STATE, n_init=5).fit(X)
        lab = g.predict(X)
        aris = []
        for s in range(n_mitades):
            idx = np.random.RandomState(s).permutation(len(X))
            a, b = idx[: len(X) // 2], idx[len(X) // 2:]
            la = GaussianMixture(k, covariance_type="full", random_state=s, n_init=3).fit(X[a]).predict(X)
            lb = GaussianMixture(k, covariance_type="full", random_state=s + 10, n_init=3).fit(X[b]).predict(X)
            aris.append(adjusted_rand_score(la, lb))
        filas.append({"k": k, "bic": round(g.bic(X)), "silhouette": round(silhouette_score(X[muestra], lab[muestra]), 3),
                      "estabilidad_ari": round(float(np.mean(aris)), 3),
                      "grupo_min_pct": round(100 * np.bincount(lab, minlength=k).min() / len(lab), 1)})
    return pd.DataFrame(filas)


def nombrar_grupos(gmm: GaussianMixture) -> dict[int, str]:
    """{id del GMM -> nombre} según los elementos que el centroide sobre-representa."""
    nombres = {}
    for i, centro in enumerate(gmm.means_):
        altos = sorted(((z, ELEMENTO_DE_FEATURE[f]) for f, z in zip(FEATS_ESCALADAS, centro) if z > UMBRAL_NOMBRE),
                       reverse=True)
        nombres[i] = "+".join(e for _, e in altos) if altos else "Cu"
    if len(set(nombres.values())) < len(nombres):          # dos grupos con el mismo nombre: se numeran
        vistos = {}
        for i, n in nombres.items():
            vistos[n] = vistos.get(n, 0) + 1
            nombres[i] = f"{n}-{vistos[n]}" if list(nombres.values()).count(n) > 1 else n
    return nombres


def entrenar(df: pd.DataFrame, k: int = K_GRUPOS) -> GaussianMixture:
    """GMM full con k grupos, ajustado con periodo == 'train' & valida."""
    gmm = GaussianMixture(k, covariance_type="full", random_state=RANDOM_STATE, n_init=10).fit(_train(df))
    gmm.nombres_ = nombrar_grupos(gmm)                     # quedan dentro del artefacto
    return gmm


def asignar(df: pd.DataFrame, gmm: GaussianMixture) -> pd.DataFrame:
    """Agrega grupo_id, grupo (nombre) y prob_grupo (probabilidad posterior del asignado)."""
    df = df.copy()
    post = gmm.predict_proba(df[FEATS_ESCALADAS].to_numpy())
    df["grupo_id"] = post.argmax(axis=1)
    df["grupo"] = df["grupo_id"].map(gmm.nombres_)
    df["prob_grupo"] = post.max(axis=1).round(3)
    return df


def caracterizar(df: pd.DataFrame, composito: pd.DataFrame | None = None,
                 min_lecturas: int = 10, min_mayoria: float = 0.7) -> pd.DataFrame:
    """Qué es cada grupo, medido contra el laboratorio.

    - Leyes de la BD (muestras valida_ley) promediadas por grupo.
    - Compósito de 12 h: se le asigna el grupo mayoritario de las lecturas
      válidas de las 12 h previas, solo si ese grupo es >= min_mayoria de
      >= min_lecturas lecturas (ventanas mezcladas no se usan).
    - Reparto de lecturas válidas por período (% de cada período).
    """
    leyes = (df[df["valida_ley"]].groupby("grupo")[["pCu", "pFe", "pMo", "pZn", "pIns"]].mean().round(2)
             .join(df[df["valida_ley"]].groupby("grupo").size().rename("n_muestras_bd")))
    partes = [leyes]
    if composito is not None:
        lect = df[df["valida"] & (df["fuente"] != "bd")].set_index("ts").sort_index()["grupo"]
        filas = []
        for t, r in composito[composito["ts"] >= lect.index.min()].set_index("ts").iterrows():
            w = lect.loc[t - pd.Timedelta("12h"):t]
            if len(w) >= min_lecturas and w.value_counts(normalize=True).iloc[0] >= min_mayoria:
                filas.append({"grupo": w.mode()[0], "cu": r["cu"], "fe": r["fe"], "mo": r["mo"], "ins": r["ins"]})
        c = pd.DataFrame(filas)
        if len(c):
            partes.append(c.groupby("grupo").agg(n_compositos=("cu", "size"), comp_cu=("cu", "mean"),
                                                 comp_fe=("fe", "mean"), comp_mo=("mo", "mean"),
                                                 comp_ins=("ins", "mean")).round(2))
    reparto = pd.crosstab(df.loc[df["valida"], "grupo"], df.loc[df["valida"], "periodo"],
                          normalize="columns").mul(100).round(1).add_prefix("pct_")
    partes.append(reparto)
    return pd.concat(partes, axis=1)


def _main():
    import argparse
    import joblib

    from .config import ESCALADO, CLUSTERIZADO, MODELS_DIR, ART_GMM, LIMPIO_COMPOSITO

    ap = argparse.ArgumentParser(description="Etapa 4: grupos de mineral (GMM)")
    ap.add_argument("--input", default=str(ESCALADO))
    ap.add_argument("--output", default=str(CLUSTERIZADO))
    ap.add_argument("--k", type=int, default=K_GRUPOS)
    ap.add_argument("--diagnostico", action="store_true", help="Solo imprime la tabla de k y termina")
    ap.add_argument("--gmm-in", default=None, help="GMM ya entrenado (modo INFERENCIA)")
    ap.add_argument("--gmm-out", default=str(MODELS_DIR / ART_GMM))
    args = ap.parse_args()

    df = pd.read_csv(args.input, parse_dates=["ts"])
    if args.diagnostico:
        print(diagnostico_k(df).to_string(index=False))
        return
    gmm = joblib.load(args.gmm_in) if args.gmm_in else entrenar(df, args.k)
    df = asignar(df, gmm)
    df.to_csv(args.output, index=False)
    comp = pd.read_csv(LIMPIO_COMPOSITO, parse_dates=["ts"])
    print(caracterizar(df, comp).to_string())
    if args.gmm_in is None:
        joblib.dump(gmm, args.gmm_out)
        print(f"GMM guardado en: {args.gmm_out}")


if __name__ == "__main__":
    _main()
