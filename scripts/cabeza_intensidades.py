#!/usr/bin/env python3
# ============================================================
# scripts/cabeza_intensidades.py — intensidades del courier en la CABEZA (rebose HC L1/L2)
# ============================================================
# Abre el hilo §16.5(1) de la bitácora: las intensidades de cabeza
# (_296290_RebHidroL1/L2_Canal*, desde 2026-03-01) son señal aguas arriba y EN
# TIEMPO REAL, a diferencia del compósito de cabeza (llega con el del concentrado).
#
#   A. ¿El courier lee la cabeza? intensidades (ventana centrada 12 h) ->
#      compósito de cabeza del laboratorio (7100AIP011-014MAN).
#   B. ¿Mejora la estimación del concentrado? compósito del concentrado ~
#      estimación actual (<ley>_est de la etapa 8) [+ cabeza], Ridge con CV
#      por bloques de tiempo y purga, y control: las features de cabeza corridas
#      >= 14 días en el tiempo (conserva su forma, rompe el calce). Aporta si la
#      mejora supera el p95 del control Y el piso MEJORA_MIN (§16.4).
#
# Crudos que lee (etapa 0 los deja en data/raw; los extrae este mismo script con
# --extraer, PI_GATEWAY_HOST/PORT por entorno):
#   courier_cabeza_L1_pi.csv, courier_cabeza_L2_pi.csv, composito_cabeza_pi.csv
#
#   pixi run python scripts/cabeza_intensidades.py [--extraer]
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.pipeline.config import DATA_RAW, DATA_FINAL, RAW_CABEZA, TAGS_CABEZA, VENTANA_COMPOSITO_H
from src.pipeline.limpieza_fuentes import lecturas_por_evento

CANALES = {"Fe": "n1fe", "Cu": "n2cu", "Zn": "n3zn", "Mo": "n4mo", "Sc": "n6sc"}
FEATS = ["lr_fe_cu", "lr_zn_cu", "lr_mo_cu", "logSumI"]
MEDIA = pd.Timedelta(hours=VENTANA_COMPOSITO_H / 2)
MIN_LECTURAS = 10
DESDE = "2026-03-01"
N_FOLDS, PURGA = 6, pd.Timedelta(days=2)
SEP_CONTROL = 28                     # compósitos (= 14 días) de corrimiento mínimo del control
N_CONTROL = 60
MEJORA_MIN = 0.01


def raw_linea(L):
    return DATA_RAW / f"courier_cabeza_{L}_pi.csv"


def extraer():
    from src.acquisition.from_pi import cliente, _a_ancho, extraer_stream
    hasta = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    for L in ("L1", "L2"):
        tags = {f"_296290_RebHidro{L}_Canal{k}_ABB": v for k, v in CANALES.items()}
        _a_ancho(cliente().recorded(list(tags), f"{DESDE} 00:00:00", hasta, chunk_dias=7), tags).to_csv(
            raw_linea(L), index=False)
    extraer_stream(TAGS_CABEZA, "2022-01-01 00:00:00").to_csv(RAW_CABEZA, index=False)


def lecturas(L):
    """Una fila por lectura real (etapa 1a), con log-cocientes; fuera lo atípico
    (> 5 MAD robustas en algún log-cociente: lecturas de arranque o atoro)."""
    d, _ = lecturas_por_evento(pd.read_csv(raw_linea(L), parse_dates=["ts"]))
    d = d[d[list(CANALES.values())].notna().all(axis=1)].copy()
    d["lr_fe_cu"] = np.log(d.n1fe / d.n2cu)
    d["lr_zn_cu"] = np.log(d.n3zn / d.n2cu)
    d["lr_mo_cu"] = np.log(d.n4mo / d.n2cu)
    d["logSumI"] = np.log(d[["n1fe", "n2cu", "n3zn", "n4mo"]].sum(axis=1))
    z = (d[FEATS] - d[FEATS].median()) / (1.4826 * (d[FEATS] - d[FEATS].median()).abs().median())
    return d[(z.abs() <= 5).all(axis=1)].set_index("ts").sort_index()


def por_ventana(lect, marcas):
    filas = []
    for t in marcas:
        w = lect.loc[t - MEDIA: t + MEDIA, FEATS]
        if len(w) >= MIN_LECTURAS:
            filas.append({"ts": t, "n": len(w), **w.mean().to_dict()})
    return pd.DataFrame(filas).set_index("ts")


def _folds(ts):
    """Bloques contiguos de tiempo; del train se purgan PURGA a cada lado del test."""
    bordes = np.array_split(np.arange(len(ts)), N_FOLDS)
    for b in bordes:
        t0, t1 = ts[b[0]], ts[b[-1]]
        tr = np.where((ts < t0 - PURGA) | (ts > t1 + PURGA))[0]
        yield tr, b


def r2_oof(X, y, ts):
    p = np.full(len(y), np.nan)
    for tr, te in _folds(ts):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(X[tr], y[tr])
        p[te] = m.predict(X[te])
    return r2_score(y, p)


def parte_a(V, cab):
    print("\n== A. ¿El courier lee la cabeza? intensidades L2 (ventana centrada) vs compósito de cabeza ==")
    d = V.join(cab, how="inner").dropna()
    ts = d.index.to_numpy()
    filas = []
    for ley in ("cu", "fe", "zn", "mo"):
        rhos = {f: stats.spearmanr(d[f], d[ley])[0] for f in FEATS}
        filas.append({"ley": ley, **{f"rho_{f}": round(v, 2) for f, v in rhos.items()},
                      "R2_oof_ridge": round(r2_oof(d[FEATS].to_numpy(), d[ley].to_numpy(), ts), 3)})
    print(f"n={len(d)} compósitos de cabeza ({d.index.min():%Y-%m-%d} -> {d.index.max():%Y-%m-%d})")
    print(pd.DataFrame(filas).to_string(index=False))


def parte_b(V, ev, nombre):
    print(f"\n== B. ¿Mejora el compósito del CONCENTRADO? base = <ley>_est de la etapa 8; + cabeza {nombre} ==")
    d = ev.join(V[FEATS].add_prefix("cab_"), how="inner").dropna(subset=["pCu", "pFe", "pMo"])
    ts = d.index.to_numpy()
    cab = [f"cab_{f}" for f in FEATS]
    filas = []
    for ley in ("pCu", "pFe", "pMo"):
        dd = d.dropna(subset=[f"{ley}_est"])
        y, tts = dd[ley].to_numpy(), dd.index.to_numpy()
        base = r2_oof(dd[[f"{ley}_est"]].to_numpy(), y, tts)
        trat = r2_oof(dd[[f"{ley}_est"] + cab].to_numpy(), y, tts)
        n = len(dd)
        corrimientos = np.linspace(SEP_CONTROL, n - SEP_CONTROL, N_CONTROL).astype(int)
        ctrl = [r2_oof(np.column_stack([dd[f"{ley}_est"].to_numpy(), np.roll(dd[cab].to_numpy(), s, axis=0)]),
                       y, tts) - base for s in corrimientos]
        p95 = float(np.quantile(ctrl, 0.95))
        mejora = trat - base
        filas.append({"ley": ley, "n": n, "R2_base": round(base, 3), "R2_+cabeza": round(trat, 3),
                      "mejora": round(mejora, 3), "p95_control": round(p95, 3),
                      "veredicto": "APORTA" if mejora > p95 and mejora >= MEJORA_MIN else "no aporta"})
    print(pd.DataFrame(filas).to_string(index=False))


def _main():
    ap = argparse.ArgumentParser(description="Intensidades de cabeza: lectura y aporte al concentrado")
    ap.add_argument("--extraer", action="store_true", help="re-extrae los crudos del PI")
    args = ap.parse_args()
    if args.extraer:
        extraer()

    cab = pd.read_csv(RAW_CABEZA, parse_dates=["ts"]).set_index("ts")
    ev = pd.read_csv(DATA_FINAL / "eval_composito.csv", parse_dates=["ts"]).set_index("ts")
    marcas = cab.index[cab.index >= DESDE]
    for L in ("L1", "L2"):
        lect = lecturas(L)
        print(f"{L}: {len(lect)} lecturas válidas, {lect.index.min():%Y-%m-%d} -> {lect.index.max():%Y-%m-%d}, "
              f"intervalo mediano {lect.index.to_series().diff().median()}")
    V2 = por_ventana(lecturas("L2"), marcas)
    parte_a(V2, cab)
    parte_b(V2, ev, "L2 (mar-oct)")
    V1 = por_ventana(lecturas("L1"), marcas)
    if len(V1) > 60:
        parte_b(V1, ev, "L1 (ago-oct)")


if __name__ == "__main__":
    _main()
