#!/usr/bin/env python3
# ============================================================
# scripts/estructura_quimica_composito.py
# ============================================================
# Cómo se mueven entre sí las leyes del Concentrado Colectivo, medido en la
# QUÍMICA del laboratorio (compósito de 12 h, 4 años), sin instrumento en medio.
# Es la base de la arquitectura de estimación (bitácora §19.14):
#
#   A. Mo no arrastra al Cu, pero sí al Fe: −0.56 de Fe por punto de Mo.
#   B. Esa relación Fe↔Mo NO es aritmética del cierre de masa: si lo fuera,
#      a Cu e insoluble fijos el coeficiente sería −1, y la suma de los cuatro
#      sería constante. No es ni lo uno ni lo otro.
#   C. Cu, Fe e insoluble se intercambian (balance de masa con la ganga).
#   D. Por qué el Zn salió de TARGETS: su varianza la cargan unos pocos picos.
#
# Todo va también centrado por mes, para que la deriva lenta del insoluble
# (2022 → 2026) no fabrique correlaciones.
#
# Solo lectura.  pixi run python scripts/estructura_quimica_composito.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import itertools
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.pipeline.config import LIMPIO_COMPOSITO, LIMPIO_COURIER_BD

LEYES = ["cu", "fe", "mo", "ins"]                         # columnas del compósito limpio


def parcial(df: pd.DataFrame, x: str, y: str, ctrl: list) -> float:
    """Correlación de x con y después de quitarles a ambos lo que explica ctrl."""
    A = np.c_[np.ones(len(df)), df[ctrl].to_numpy()]
    res = lambda v: df[v].to_numpy() - A @ np.linalg.lstsq(A, df[v].to_numpy(), rcond=None)[0]
    return float(np.corrcoef(res(x), res(y))[0, 1])


def main() -> None:
    d = pd.read_csv(LIMPIO_COMPOSITO, parse_dates=["ts"]).dropna(subset=LEYES)
    d["mes"] = d["ts"].dt.to_period("M").astype(str)
    print(f"Compósito: n = {len(d)}, {d.ts.min().date()} -> {d.ts.max().date()}\n")

    print("== CV (%) ==")
    print((d[LEYES].std() / d[LEYES].mean() * 100).round(1).to_string(), "\n")

    centrado = d.groupby("mes")[LEYES].transform(lambda s: s - s.mean())
    print("== Pearson crudo ==")
    print(d[LEYES].corr().round(3).to_string(), "\n")
    print("== Pearson centrado por mes ==")
    print(centrado.corr().round(3).to_string(), "\n")

    print("== Parciales (controlando las otras dos leyes) ==")
    for x, y in itertools.combinations(LEYES, 2):
        ctrl = [c for c in LEYES if c not in (x, y)]
        print(f"  {x:>3s} ~ {y:<3s} | {ctrl}: {parcial(d, x, y, ctrl):+.3f}")

    print("\n== B. ¿Fe↔Mo es cierre de masa? (cierre puro -> coef mo = -1) ==")
    for f in ["fe ~ mo + cu + ins", "fe ~ mo + cu + ins + C(mes)", "fe ~ mo + C(mes)"]:
        m = smf.ols(f, d).fit()
        lo, hi = m.conf_int().loc["mo"]
        print(f"  {f:30s} coef mo = {m.params['mo']:+.3f}  IC95 [{lo:+.3f}, {hi:+.3f}]")
    d["suma"] = d[LEYES].sum(axis=1)
    m = smf.ols("suma ~ mo + C(mes)", d).fit()
    print(f"  suma cu+fe+mo+ins = {d.suma.mean():.2f} ± {d.suma.std():.2f}; "
          f"suma ~ mo + C(mes): coef {m.params['mo']:+.3f}, p={m.pvalues['mo']:.2g}")

    print("\n== D. Zn en la BD (el compósito no lo reporta) ==")
    z = pd.read_csv(LIMPIO_COURIER_BD)["pZn"].dropna()
    cola = z[z > z.quantile(0.95)]
    ss = lambda v: ((v - z.mean()) ** 2).sum()
    print(f"  n={len(z)}  mediana {z.median():.3f}  CV {z.std() / z.mean() * 100:.1f} %  max {z.max():.2f}")
    print(f"  el 5 % superior ({len(cola)} muestras) aporta el {ss(cola) / ss(z) * 100:.1f} % de la suma de cuadrados")


if __name__ == "__main__":
    main()
