#!/usr/bin/env python3
# ============================================================
# scripts/prueba_grupos_psol.py — ¿un clustering menos sensible al agua?
# ============================================================
# Compara variantes de features del GMM de la etapa 4 con tres jueces:
#   - control de dilución (clustering.control_psol): eta² del pSol de
#     laboratorio a química fija contra su control circular -- se quiere BAJO;
#   - química de la BD: eta² de pFe/pCu/pMo por grupo -- se quiere ALTO;
#   - compósito de 12 h (validación externa, ventana centrada): eta² de
#     cu/fe/mo por grupo mayoritario, solo compósitos posteriores al corte
#     de train (fuera de muestra) -- se quiere ALTO;
#   - estabilidad: ARI entre GMM ajustados en mitades disjuntas de train.
# Todo GMM se ajusta solo con train & valida, como la etapa 4.
#
#   pixi run python scripts/prueba_grupos_psol.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from src.pipeline.config import CLUSTERIZADO, LIMPIO_COMPOSITO, RANDOM_STATE, FECHA_CORTE_TRAIN, VENTANA_COMPOSITO_H
from src.pipeline.clustering import control_psol, _eta2

MEDIA = pd.Timedelta(hours=VENTANA_COMPOSITO_H / 2)
VARIANTES = {                                  # nombre -> peso de cada feature (0 = fuera)
    "actual (Fe, Zn, Mo)":    {"lr_fe_cu_z": 1.0, "lr_zn_cu_z": 1.0, "lr_mo_cu_z": 1.0},
    "Fe a la mitad":          {"lr_fe_cu_z": 0.5, "lr_zn_cu_z": 1.0, "lr_mo_cu_z": 1.0},
    "sin Fe (Zn, Mo)":        {"lr_zn_cu_z": 1.0, "lr_mo_cu_z": 1.0},
    "solo Mo":                {"lr_mo_cu_z": 1.0},
    "solo Zn (invariante)":   {"lr_zn_cu_z": 1.0},
}


def _matriz(df, pesos):
    return df[list(pesos)].to_numpy() * np.array(list(pesos.values()))


def _estabilidad(X, k, n=3):
    aris = []
    for s in range(n):
        idx = np.random.RandomState(s).permutation(len(X))
        a, b = idx[: len(X) // 2], idx[len(X) // 2:]
        la = GaussianMixture(k, covariance_type="full", random_state=s, n_init=3).fit(X[a]).predict(X)
        lb = GaussianMixture(k, covariance_type="full", random_state=s + 10, n_init=3).fit(X[b]).predict(X)
        aris.append(adjusted_rand_score(la, lb))
    return float(np.mean(aris))


def _composito(df, comp, col, min_lecturas=10, min_mayoria=0.7):
    lect = df[df["valida"] & (df["fuente"] != "bd")].set_index("ts").sort_index()[col]
    filas = []
    for t, r in comp[comp["ts"] > pd.Timestamp(FECHA_CORTE_TRAIN)].set_index("ts").iterrows():
        w = lect.loc[t - MEDIA: t + MEDIA]
        if len(w) >= min_lecturas and w.value_counts(normalize=True).iloc[0] >= min_mayoria:
            filas.append({"g": w.mode()[0], "cu": r["cu"], "fe": r["fe"], "mo": r["mo"]})
    c = pd.DataFrame(filas).dropna()
    return len(c), {e: _eta2(c[e].to_numpy(), c["g"].to_numpy()) for e in ("cu", "fe", "mo")}


def evaluar(df, comp, pesos, k):
    tr = (df["periodo"] == "train") & df["valida"]
    X = _matriz(df, pesos)
    gmm = GaussianMixture(k, covariance_type="full", random_state=RANDOM_STATE, n_init=5).fit(X[tr.to_numpy()])
    d = df.assign(g=gmm.predict(X))
    c = control_psol(d, "g")
    n_comp, eta_comp = _composito(d, comp, "g")
    return {"eta2_psol": c["eta2_psol"], "p95_ctrl": c["p95_control"], "pasa": "sí" if c["pasa"] else "NO",
            **{f"bd_{e[1:]}": v for e, v in c["eta2_quimica"].items()},
            **{f"comp_{e}": v for e, v in eta_comp.items()}, "n_comp": n_comp,
            "estab": _estabilidad(X[tr.to_numpy()], k)}


def _main():
    df = pd.read_csv(CLUSTERIZADO, parse_dates=["ts"])
    comp = pd.read_csv(LIMPIO_COMPOSITO, parse_dates=["ts"])
    filas = []
    for nombre, pesos in VARIANTES.items():
        for k in (2, 3, 4):
            filas.append({"variante": nombre, "k": k, **evaluar(df, comp, pesos, k)})
            print(f"  {nombre} k={k} listo", flush=True)
    T = pd.DataFrame(filas)
    pd.set_option("display.width", 200)
    print("\neta²: fracción de varianza explicada por el grupo. pSol: bajo = mejor; bd_*/comp_*: alto = mejor.")
    print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    _main()
