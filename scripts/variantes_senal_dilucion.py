#!/usr/bin/env python3
# ============================================================
# scripts/variantes_senal_dilucion.py
# ============================================================
# ¿Conviene cambiar `dil = logSumI - EWMA(90 d)` de la etapa 2?
#
# §19.9 lo había anotado como defecto: la EWMA de 90 días atraviesa el escalón
# de recalibración de nov-2025 y durante ~3 meses absorbe un cambio de
# instrumento como si fuera dilución (media mensual de `dil` de -0.077 en
# dic-ene, sobre sd 0.095). El diagnóstico del mecanismo es correcto.
#
# Medida la CONSECUENCIA, la conclusión se invierte: toda variante que quita el
# escalón empeora la corrección. Ver §19.10. El motivo es que el escalón de
# logSumI fue -7.4 %, en la misma dirección en que venía cayendo el pSol y en
# los mismos meses, así que quitarlo le resta a `dil` algo que correlaciona con
# la dilución real.
#
# Métrica del objetivo (la de §10.7): efecto de pSol sobre `lr_fe_cu` a química
# fija, p10->p90 en unidades de sd del feature. Sin corregir: +0.896 sd.
#
# Solo lectura.  pixi run python scripts/variantes_senal_dilucion.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from src.pipeline.config import LIMPIO, METALES, DIL_HALFLIFE, DIL_CORREGIR

CORTE = pd.Timestamp("2025-12-10")       # escalón de la recalibración (§19.2)
LEYES = ["pFe", "pCu", "pMo", "pIns"]
pd.set_option("display.width", 210)

df = pd.read_csv(LIMPIO, parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)
df["logSumI"] = np.log(df[METALES].sum(axis=1).replace(0, np.nan))
df["lr_fe_cu"] = np.log(df["n1fe"] / df["n2cu"])
df["pSol_ok"] = df["pSol"].where((df["pSol"] > 5) & (df["pSol"] <= 100))

v = df["valida"] & df["logSumI"].notna() & df["fuente"].isin(["pi", "ambos"])
d = df[v].set_index("ts")["logSumI"].resample("D").median().dropna()
SALTO = (d.loc[CORTE:CORTE + pd.Timedelta("45D")].median()
         - d.loc[CORTE - pd.Timedelta("45D"):CORTE - pd.Timedelta("1D")].median())
print(f"escalón de logSumI el {CORTE.date()}: {SALTO:+.4f} "
      f"({100*(np.exp(SALTO)-1):+.1f} % de intensidad total)")
print(f"actual en config: DIL_HALFLIFE={DIL_HALFLIFE}, DIL_CORREGIR={DIL_CORREGIR}\n")


def efecto_psol(feat):
    """Efecto de pSol sobre el feature a química fija, en sd del feature."""
    s = df.dropna(subset=["pSol_ok", "lr_fe_cu"] + LEYES).copy()
    s["y"] = feat.loc[s.index]
    X = np.column_stack([np.ones(len(s))] + [s[c].to_numpy() for c in LEYES]
                        + [s["pSol_ok"].to_numpy()])
    b, *_ = np.linalg.lstsq(X, s["y"], rcond=None)
    p10, p90 = np.percentile(s["pSol_ok"], [10, 90])
    return b[-1] * (p90 - p10) / s["y"].std()


def dil(halflife=DIL_HALFLIFE, quitar_escalon=False, por_epoca=False):
    L = df["logSumI"] - (SALTO * (df["ts"] >= CORTE) if quitar_escalon else 0)
    out = pd.Series(0.0, index=df.index)
    tramos = ([(df["ts"] < CORTE), (df["ts"] >= CORTE)] if por_epoca
              else [pd.Series(True, index=df.index)])
    for m in tramos:
        ok = m & df["valida"] & L.notna()
        e = L[ok].ewm(halflife=pd.Timedelta(halflife), times=df.loc[ok, "ts"]).mean()
        out[m] = (L[m] - e.shift(1).reindex(df.index[m]).ffill()).fillna(0.0)
    return out


tr = (df["periodo"] == "train") & df["valida"]
vv = df[df["valida"]]
print(f"{'variante':<36}{'corr(dil,pSol)':>16}{'b':>9}{'efecto resid.':>15}"
      f"{'dic-ene':>10}{'may-jul':>10}")
print(f"{'SIN corregir':<36}{'-':>16}{'-':>9}{efecto_psol(df['lr_fe_cu']):>15.3f}"
      f"{'-':>10}{'-':>10}")
VARIANTES = [("EWMA 90D global (ACTUAL)", dict()),
             ("EWMA 90D por época", dict(por_epoca=True)),
             ("EWMA 90D, escalón re-nivelado", dict(quitar_escalon=True)),
             ("EWMA 90D, re-nivelado + por época", dict(quitar_escalon=True, por_epoca=True)),
             ("EWMA 30D global", dict(halflife="30D")),
             ("EWMA 14D global", dict(halflife="14D")),
             ("EWMA 7D global", dict(halflife="7D")),
             ("EWMA 180D global", dict(halflife="180D")),
             ("EWMA 180D, escalón re-nivelado", dict(halflife="180D", quitar_escalon=True)),
             ("EWMA 365D, escalón re-nivelado", dict(halflife="365D", quitar_escalon=True))]
for nombre, kw in VARIANTES:
    D = dil(**kw)
    s = df.dropna(subset=["pSol_ok"])
    b = float(np.polyfit(D[tr], df.loc[tr, "lr_fe_cu"], 1)[0])
    m1 = D[vv.index][(vv["ts"] >= "2025-12-01") & (vv["ts"] < "2026-02-01")].mean()
    m2 = D[vv.index][(vv["ts"] >= "2026-05-01") & (vv["ts"] < "2026-08-01")].mean()
    print(f"{nombre:<36}{np.corrcoef(D.loc[s.index], s['pSol_ok'])[0,1]:>16.3f}{b:>9.4f}"
          f"{efecto_psol(df['lr_fe_cu'] - b * D):>15.3f}{m1:>10.4f}{m2:>10.4f}")

print("\n>> NO CAMBIAR `dil`. La variante actual está en el óptimo (la de 180D empata en la")
print("   tercera cifra). Quitar el escalón degrada el proxy, porque el escalón empujó")
print("   logSumI en la MISMA dirección que la caída del pSol y en los mismos meses.")
print("   Riesgo registrado: si un mantenimiento futuro empuja las intensidades en la")
print("   dirección CONTRARIA a la del sólido, `dil` corromperá la corrección. La")
print("   salvaguarda no es cambiar la fórmula: es el monitor de cocientes y volver a")
print("   ajustar `b` después de cada mantenimiento.")
