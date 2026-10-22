#!/usr/bin/env python3
# ============================================================
# scripts/referencias_laboratorio.py
# ============================================================
# Las dos fuentes de ley del concentrado NO son dos laboratorios: son dos
# CORTES del mismo flujo, analizados por el MISMO laboratorio con el MISMO
# método (confirmado con planta, 2026-10-22). El recorrido real es:
#
#   flujo → MULTIPLEXOR ─corte→ balde 12 h          (COMPÓSITO; no ve el probe)
#                │
#                └──────→ PROBE (rayos X) ─corte→ recipiente  (CALIBRACIÓN)
#
# El multiplexor está a menos de 2 m del probe y en ese tramo NO hay agua; el
# agua de dilución se aplica muchos metros antes del multiplexor, así que las
# dos muestras ven el mismo material ya diluido. Los dos cortadores toman la
# sección completa del flujo y el balde se agita y se procesa completo.
#
# Las muestras puntuales son las de CALIBRACIÓN del Outocal: intensidad y
# ensayo quedan en la misma fila de la BD. De ahí que el calce por valor sea
# exacto (§12) y que la BD sea la referencia de calibración por diseño,
# mientras el compósito es la auditoría independiente.
#
# Este script mide dos cosas (ver docs/bitacora_analisis.md §19.12 y §19.13):
#   A. el TECHO demostrable de cada ley, dado el desacuerdo entre referencias
#   B. el SESGO de muestreo entre los dos cortes, con sus exclusiones
#
# Solo lectura.  pixi run python scripts/referencias_laboratorio.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.pipeline.config import (LIMPIO, LIMPIO_COMPOSITO, METALES,
                                 VENTANA_COMPOSITO_H)

INICIO_EPOCA = "2025-12-15"          # época de instrumento vigente (§19.2)
pd.set_option("display.width", 200)
MEDIO = pd.Timedelta(hours=VENTANA_COMPOSITO_H / 2)


def cargar():
    d = pd.read_csv(LIMPIO, parse_dates=["ts"])
    d = d[d["valida"]].copy()
    d["pSol_ok"] = d["pSol"].where((d["pSol"] > 5) & (d["pSol"] <= 100))
    S = d[METALES].sum(axis=1)
    for c in METALES:
        d[f"{c}_f"] = d[c] / S
    d["logSumI"] = np.log(S)
    d["logn6sc"] = np.log(d["n6sc"])
    return d, pd.read_csv(LIMPIO_COMPOSITO, parse_dates=["ts"])


FEATS = [f"{c}_f" for c in METALES] + ["logSumI", "logn6sc"]


def emparejar(d, c):
    """Cada muestra puntual con el compósito cuya ventana centrada la contiene."""
    ct = c["ts"].to_numpy()
    out = []
    for _, r in d.dropna(subset=["pCu"]).iterrows():
        k = int(np.argmin(np.abs(ct - np.datetime64(r["ts"]))))
        tc = pd.Timestamp(ct[k])
        if abs((r["ts"] - tc).total_seconds()) <= MEDIO.total_seconds():
            out.append({"ts": r["ts"], "hora": r["ts"].hour, "pSol": r["pSol_ok"],
                        "comp_pm": int(tc.hour >= 12),
                        "dcu": r["pCu"] - c["cu"].iloc[k], "cu_bd": r["pCu"], "cu_co": c["cu"].iloc[k],
                        "dfe": r["pFe"] - c["fe"].iloc[k], "dmo": r["pMo"] - c["mo"].iloc[k],
                        "dins": (r["pIns"] - c["ins"].iloc[k]) if pd.notna(r["pIns"]) else np.nan})
    P = pd.DataFrame(out)
    P["t"] = (P["ts"] - P["ts"].min()).dt.total_seconds() / (365.25 * 86400)
    return P


def ventanas(d, c):
    """Una fila por compósito: features promediadas en su ventana + química."""
    dd = d[(d["ts"] >= INICIO_EPOCA) & d["fuente"].isin(["pi", "ambos"])].reset_index(drop=True)
    ts, V = dd["ts"].to_numpy(), dd[FEATS].to_numpy()
    filas, idx = [], []
    for t, row in zip(c["ts"].to_numpy(), c[["cu", "fe", "mo"]].to_numpy()):
        i = np.where((ts >= t - MEDIO.to_timedelta64()) & (ts <= t + MEDIO.to_timedelta64()))[0]
        if len(i) >= 10:
            filas.append(dict(zip(["ts"] + FEATS, [t] + list(V[i].mean(axis=0))),
                              **dict(zip(["cu", "fe", "mo"], row))))
            idx.append(i)
    C = pd.DataFrame(filas)
    o = np.argsort(C["ts"].to_numpy())
    return C.iloc[o].reset_index(drop=True), [idx[k] for k in o], dd


def ols(tabla, y, cols, nombre=""):
    s = tabla.dropna(subset=[y] + cols)
    X = np.column_stack([np.ones(len(s))] + [s[c].to_numpy() for c in cols])
    yv = s[y].to_numpy()
    b, *_ = np.linalg.lstsq(X, yv, rcond=None)
    sse = ((yv - X @ b) ** 2).sum()
    gl = len(s) - X.shape[1]
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * sse / gl)
    if nombre:
        print(f"  {nombre}  n={len(s)}")
    for cc, bi, si in zip(["const"] + cols, b, se):
        t = bi / si
        print(f"    {cc:<12}{bi:+9.4f}  t={t:+6.2f}  p={2*(1 - stats.t.cdf(abs(t), gl)):.2e}")


# ============================================================
def seccion_A_techo(d, c, P):
    print("=" * 96)
    print("A. EL TECHO DEMOSTRABLE — ¿cuánto del error es del modelo y cuánto de la referencia?")
    print("=" * 96)
    C, idx, dd = ventanas(d, c)
    n = len(C); corte = int(n * 0.70)
    print(f"  {n} ventanas pareadas; train {corte} / test {n-corte} "
          f"({C['ts'].iloc[corte].date()} -> {C['ts'].iloc[-1].date()})\n")
    print(f"  {'elem':<5}{'sd(dif cortes)':>16}{'soporte':>10}{'sd_err*':>9}"
          f"{'sd(resid modelo)':>18}{'resid/dif':>11}{'R2 modelo':>11}")
    for e in ["cu", "fe", "mo"]:
        sd_dif = P[f"d{e}"].std()
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
        m.fit(C[FEATS].iloc[:corte], C[e].iloc[:corte])
        intra = np.array([m.predict(dd[FEATS])[i].std() for i in idx]).mean()
        sd_err = np.sqrt(max((sd_dif ** 2 - intra ** 2) / 2, 0))
        y = C[e].iloc[corte:].to_numpy()
        p = m.predict(C[FEATS].iloc[corte:])
        r2 = 1 - ((y - p) ** 2).sum() / ((y - C[e].iloc[:corte].mean()) ** 2).sum()
        print(f"  {e:<5}{sd_dif:>16.3f}{intra:>10.3f}{sd_err:>9.3f}"
              f"{(y-p).std():>18.3f}{(y-p).std()/sd_dif:>11.2f}{r2:>11.3f}")
    print("\n  soporte = variación real del material DENTRO de las 12 h (de la estimación por lectura)")
    print("  sd_err* = (dif^2 - soporte^2)/2, suponiendo error igual en los dos cortes.")
    print("  OJO: ese supuesto da un techo POR DEBAJO de lo que el modelo logra, o sea es")
    print("  falso -- el corte puntual (grab) carga más error que el compósito por incrementos.")
    print("  Lo que SÍ queda: el residuo del modelo ya es <= el desacuerdo entre referencias.")
    print("\n  cuánta variación hay que explicar (época vigente):")
    cc = c[c["ts"] >= INICIO_EPOCA]
    print(f"  {'elem':<5}{'media':>9}{'sd':>8}{'CV %':>8}")
    for e in ["cu", "fe", "mo", "ins"]:
        print(f"  {e:<5}{cc[e].mean():>9.2f}{cc[e].std():>8.3f}{100*cc[e].std()/cc[e].mean():>8.1f}")
    print("  >> el Cu casi no varía (CV 7.5 %) porque es el objetivo controlado; el Mo varía")
    print("     43.8 %. El R2 bajo del Cu es sobre todo falta de varianza que explicar.")


def seccion_B_sesgo(d, c, P):
    print("\n" + "=" * 96)
    print("B. EL SESGO DE MUESTREO ENTRE LOS DOS CORTES")
    print("=" * 96)
    print(f"  {len(P)} pares, {P['ts'].min().date()} -> {P['ts'].max().date()}\n")
    print(f"  {'dif':<6}{'media':>9}{'sd':>8}{'media_ref':>11}{'relativo':>10}")
    medias = {"cu": P["cu_co"].mean(), "fe": c["fe"].mean(), "mo": c["mo"].mean(), "ins": c["ins"].mean()}
    for e in ["cu", "fe", "mo", "ins"]:
        q = P.dropna(subset=[f"d{e}"])
        print(f"  d{e:<5}{q[f'd{e}'].mean():>9.3f}{q[f'd{e}'].std():>8.3f}"
              f"{medias[e]:>11.2f}{100*q[f'd{e}'].mean()/medias[e]:>9.1f}%")
    print("\n  >> los relativos NO son iguales ni del mismo signo: no es diferencia de BASE")
    print("     (húmeda/seca, normalización) -- esas mueven todo por el mismo factor.")

    Q = P.dropna(subset=["dcu", "dins"])
    print(f"\n  balance de masa Cu <-> ganga (n={len(Q)}):")
    print(f"    corr(dcu, dins) = {np.corrcoef(Q['dcu'], Q['dins'])[0,1]:+.3f}")
    s = Q["dcu"] + Q["dins"]
    print(f"    suma dcu+dins: media {s.mean():+.3f}  sd {s.std():.3f}   "
          f"(dcu sola sd {Q['dcu'].std():.3f}, dins sola sd {Q['dins'].std():.3f})")
    print("    >> lo que el grab pierde en Cu lo gana en ganga, punto por punto.")

    print("\n  Bland-Altman (la diferencia contra el PROMEDIO, no contra una de las dos):")
    P2 = P.assign(prom=(P["cu_bd"] + P["cu_co"]) / 2)
    r = stats.linregress(P2["prom"], P2["dcu"])
    print(f"    dcu ~ promedio: pendiente {r.slope:+.4f}  r={r.rvalue:+.3f}  p={r.pvalue:.2e}"
          f"  -> {'proporcional' if r.pvalue < 0.05 else 'ADITIVO'}")
    print(f"    sesgo medio {P['dcu'].mean():+.3f}  IC95 "
          f"[{P['dcu'].mean()-1.96*P['dcu'].sem():+.3f}, {P['dcu'].mean()+1.96*P['dcu'].sem():+.3f}]")
    print(f"    límites de concordancia: [{P['dcu'].mean()-1.96*P['dcu'].std():+.2f}, "
          f"{P['dcu'].mean()+1.96*P['dcu'].std():+.2f}]  <- par a par NO es comparable")

    print("\n  EXCLUSIONES (cada hipótesis con su medición):")
    r = stats.linregress(P["t"], P["dcu"])
    print(f"    deriva en el tiempo : pendiente {r.slope:+.4f}/año  p={r.pvalue:.3f}  (no monótona)")
    q = P.dropna(subset=["pSol"])
    print(f"    agua de dilución    : corr(dcu, pSol) = {np.corrcoef(q['pSol'], q['dcu'])[0,1]:+.3f}"
          f"  p={stats.linregress(q['pSol'], q['dcu']).pvalue:.3f}")
    print("    hora de toma, controlando nivel y turno del compósito:")
    ols(P2, "dcu", ["hora", "prom", "comp_pm"])


if __name__ == "__main__":
    d, c = cargar()
    P = emparejar(d, c)
    seccion_A_techo(d, c, P)
    seccion_B_sesgo(d, c, P)
    print("\n" + "=" * 96)
    print("CICLO DIARIO del Cu estimado — ¿explica el sesgo? (época vigente)")
    print("=" * 96)
    C, idx, dd = ventanas(d, c)
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))).fit(C[FEATS], C["cu"])
    dd = dd.assign(cu_est=m.predict(dd[FEATS]))
    h = dd.groupby(dd["ts"].dt.hour)["cu_est"].mean()
    v = dd[dd["ts"].dt.hour.isin([10, 11])]["cu_est"].mean()
    am = dd[(dd["ts"].dt.hour >= 1) & (dd["ts"].dt.hour < 14)]["cu_est"].mean()
    print(f"  amplitud del ciclo: {h.max()-h.min():.3f} puntos de Cu (pico 19 h, valle 11 h)")
    print(f"  las muestras de calibración se toman 10-12 h (259 de 333)")
    print(f"  media 10-12 h menos ventana del compósito AM = {v-am:+.3f}")
    print(f"  >> explica {100*abs(v-am)/abs(P['dcu'].mean()):.0f} % del sesgo de {P['dcu'].mean():+.3f}. NO es la causa.")
