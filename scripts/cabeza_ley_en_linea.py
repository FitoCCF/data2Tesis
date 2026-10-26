#!/usr/bin/env python3
# ============================================================
# scripts/cabeza_ley_en_linea.py — ley de CABEZA (Cu, Fe, Mo) en línea desde el courier
# ============================================================
# Pendiente 3 (idea b) de §20-21, rama AlimConc. Mismo esquema que el
# concentrado, pero con la cabeza (rebose de hidrociclones, RebHidroL2):
#
#   lecturas por evento (etapa 1a) -> log-cocientes + logSumI (etapa 2)
#   -> media en la ventana CENTRADA de cada compósito de cabeza (7100AIP011-014MAN)
#   -> Ridge por ley, ajustado SOLO en train -> se aplica a cada lectura
#      (lineal: la media de las estimaciones = la estimación de la media)
#   -> Kalman "escala balde" (q elegido en train), como la etapa 8.
#
# Global contra LOCAL por tipo de cabeza (tipos_mineral.py, k=3 sobre el
# compósito de laboratorio). En línea el tipo no se conoce hasta que el lab
# publica, así que se PREDICE desde las intensidades (regresión logística
# ajustada en train) y los locales se entrenan y rutean con el tipo predicho.
# Se reporta también el tipo real del lab ("oráculo") como techo.
# Un local se adopta solo si su mejora en test supera el p95 de 200
# particiones aleatorias del mismo tamaño Y MEJORA_MIN (regla de la etapa 5).
#
# Juez adicional: PERSISTENCIA (el último compósito publicado). La ley de
# cabeza está muy autocorrelacionada; un modelo que no le gane no sirve en línea.
#
# Corte propio: las intensidades de cabeza empiezan el 2026-03-01.
#   train: 2026-03-01 -> FIN_TRAIN_CABEZA      test: después
#
# Salidas: data/final/cabeza_estimado.csv (por lectura),
#          data/final/eval_cabeza.csv (por compósito), eval_cabeza_metricas.csv
#
#   pixi run python scripts/cabeza_ley_en_linea.py [--linea L2]
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.pipeline.config import DATA_FINAL, RAW_CABEZA, RANDOM_STATE
from src.pipeline import tipos_mineral as tm
from src.evaluacion.evaluar import kalman_serie, metricas, Q_GRID
from scripts.cabeza_intensidades import lecturas, por_ventana, FEATS, MEDIA

LEYES = ["cu", "fe", "mo"]
FIN_TRAIN_CABEZA = "2026-07-31"        # ~5 meses de train, ~3 de test
N_CONTROL, MEJORA_MIN, MIN_N_LOCAL = 200, 0.01, 30


def _ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))


def tabla_compositos(linea):
    """Una fila por compósito de cabeza: features medias de su ventana, leyes y tipo del lab."""
    d = tm.cargar("cabeza")
    sc, gmm, nombres = tm.ajustar(d, "cabeza", tm.K_POR_STREAM["cabeza"])   # tipos: protocolo de §17
    d = tm.asignar(d, "cabeza", sc, gmm, nombres).set_index("ts")
    lect = lecturas(linea)
    V = por_ventana(lect, d.index[d.index >= lect.index.min()])
    A = V.join(d[LEYES + ["tipo_cabeza"]], how="inner").dropna(subset=FEATS + LEYES).reset_index()
    A["periodo"] = np.where(A["ts"] <= pd.Timestamp(FIN_TRAIN_CABEZA) + pd.Timedelta(days=1), "train", "test")
    A["turno"] = np.where(A["ts"].dt.hour < 12, "dia", "noche")
    return A, lect


def ajustar_global(A, tr):
    return {l: _ridge().fit(A.loc[tr, FEATS], A.loc[tr, l]) for l in LEYES}


def ajustar_locales(A, tr, col):
    """{(ley, tipo) -> Ridge}; tipo con < MIN_N_LOCAL compósitos en train -> sin local."""
    M = {}
    for t, g in A[tr].groupby(col):
        if len(g) >= MIN_N_LOCAL:
            for l in LEYES:
                M[(l, t)] = _ridge().fit(g[FEATS], g[l])
    return M


def predecir_locales(A, M, G, col):
    out = {l: G[l].predict(A[FEATS]) for l in LEYES}
    for (l, t), m in M.items():
        s = (A[col] == t).to_numpy()
        if s.any():
            out[l][s] = m.predict(A.loc[s, FEATS])
    return out


def r2(y, p):
    return metricas(y, p).get("R2", np.nan)


def comparar_locales(A, G, col, nombre):
    """R² test del local vs global, y control: misma partición barajada en train y test."""
    tr, te = A["periodo"] == "train", A["periodo"] == "test"
    pg = {l: G[l].predict(A.loc[te, FEATS]) for l in LEYES}
    pl = predecir_locales(A.loc[te], ajustar_locales(A, tr, col), G, col)
    rng = np.random.RandomState(RANDOM_STATE)
    ctrl = {l: [] for l in LEYES}
    for _ in range(N_CONTROL):
        B = A.copy()
        B.loc[tr, col] = rng.permutation(B.loc[tr, col].to_numpy())
        B.loc[te, col] = rng.permutation(B.loc[te, col].to_numpy())
        pc = predecir_locales(B.loc[te], ajustar_locales(B, tr, col), G, col)
        for l in LEYES:
            ctrl[l].append(r2(A.loc[te, l], pc[l]) - r2(A.loc[te, l], pg[l]))
    filas = []
    for l in LEYES:
        rg, rl = r2(A.loc[te, l], pg[l]), r2(A.loc[te, l], pl[l])
        p95 = float(np.quantile(ctrl[l], 0.95))
        filas.append({"tipo": nombre, "ley": l, "R2_global": rg, "R2_local": rl, "mejora": rl - rg, "p95_control": p95,
                      "veredicto": "LOCAL" if rl - rg > p95 and rl - rg >= MEJORA_MIN else "global"})
    return pd.DataFrame(filas)


def _main():
    ap = argparse.ArgumentParser(description="Ley de cabeza en línea: global vs local por tipo de cabeza")
    ap.add_argument("--linea", default="L2")
    args = ap.parse_args()
    pd.set_option("display.width", 200)

    A, lect = tabla_compositos(args.linea)
    tr, te = A["periodo"] == "train", A["periodo"] == "test"
    print(f"Cabeza {args.linea}: {len(A)} compósitos con ventana (train {tr.sum()}, test {te.sum()}), "
          f"{A.ts.min():%Y-%m-%d} -> {A.ts.max():%Y-%m-%d}")
    print("\nTipos de cabeza del lab (tipos_mineral k=3), medianas por tipo:")
    print(A.groupby("tipo_cabeza")[LEYES].median().join(A.groupby(["tipo_cabeza", "periodo"]).size().unstack())
          .round(4).to_string())

    # --- tipo de cabeza PREDICHO desde las intensidades (lo único disponible en línea) ---
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(A.loc[tr, FEATS], A.loc[tr, "tipo_cabeza"])
    A["tipo_pred"] = clf.predict(A[FEATS])
    acierto = (A.loc[te, "tipo_pred"] == A.loc[te, "tipo_cabeza"]).mean()
    base = A.loc[te, "tipo_cabeza"].value_counts(normalize=True).iloc[0]
    print(f"\nTipo predicho desde intensidades: acierto en test {acierto:.2f} (clase mayoritaria {base:.2f})")

    # --- global vs local ---
    G = ajustar_global(A, tr)
    print(f"\n== Local por tipo vs global (test; control {N_CONTROL} particiones barajadas, p95) ==")
    T = pd.concat([comparar_locales(A, G, "tipo_pred", "predicho"),
                   comparar_locales(A, G, "tipo_cabeza", "lab (oráculo)")])
    print(T.round(3).to_string(index=False))

    # --- estimación final: global (o local si alguno pasó con el tipo predicho) + Kalman ---
    usar_local = set(T[(T.tipo == "predicho") & (T.veredicto == "LOCAL")].ley)
    M = ajustar_locales(A, tr, "tipo_pred")
    pl = predecir_locales(A, M, G, "tipo_pred")
    filas = []
    for l in LEYES:
        A[f"{l}_est"] = pl[l] if l in usar_local else G[l].predict(A[FEATS])
        scores = {q: r2(A.loc[tr, l], kalman_serie(A, l, q)[0][tr]) for q in Q_GRID}
        q = max(scores, key=scores.get)
        A[f"{l}_kal"] = kalman_serie(A, l, q)[0]
        A[f"{l}_pers"] = A[l].shift(1)                     # último compósito publicado
        for var, col in (("modelo", f"{l}_est"), (f"modelo + Kalman (q={q})", f"{l}_kal"),
                         ("persistencia", f"{l}_pers")):
            filas.append({"ley": l, "variante": var, **metricas(A.loc[te, l], A.loc[te, col])})
    R = pd.DataFrame(filas)
    print(f"\n== Cabeza {args.linea} vs compósito de cabeza, TEST ({FIN_TRAIN_CABEZA} ->) ==")
    print(R.round(4).to_string(index=False))

    # --- por lectura (lo que vería el monitor) ---
    E = lect[FEATS].copy()
    for l in LEYES:
        E[f"{l}_est"] = G[l].predict(E[FEATS])             # global por lectura (los locales necesitan tipo por ventana)
    E["periodo"] = np.where(E.index <= pd.Timestamp(FIN_TRAIN_CABEZA) + pd.Timedelta(days=1), "train", "test")
    E.reset_index().to_csv(DATA_FINAL / f"cabeza_estimado_{args.linea}.csv", index=False)
    A.to_csv(DATA_FINAL / f"eval_cabeza_{args.linea}.csv", index=False)
    pd.concat([T.assign(tabla="local_vs_global"), R.assign(tabla="test")]).to_csv(
        DATA_FINAL / f"eval_cabeza_{args.linea}_metricas.csv", index=False)
    print(f"\nPor lectura: {len(E)} -> data/final/cabeza_estimado_{args.linea}.csv")
    print("Últimos compósitos (estimado / Kalman / lab):")
    print(A[["ts"] + [c for l in LEYES for c in (f"{l}_est", f"{l}_kal", l)]].tail(6).round(4).to_string(index=False))


if __name__ == "__main__":
    _main()
