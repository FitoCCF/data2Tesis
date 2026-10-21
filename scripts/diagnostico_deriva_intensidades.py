#!/usr/bin/env python3
# ============================================================
# scripts/diagnostico_deriva_intensidades.py
# ============================================================
# Descompone la deriva de la magnitud de las intensidades del concentrado.
# La bitácora §2.3 la tenía registrada como "22-25 % entre trimestres" y la
# atribuía a decaimiento de fuente. Son TRES causas independientes, y este
# script las separa. Ver docs/bitacora_analisis.md §19.
#
#   1. Escalón por canal de nov-2025 a dic-2025 (mantenimiento confirmado con
#      planta; después se rehicieron las ecuaciones del Outocal con una
#      campaña de muestreo nueva).
#   2. Dilución de la muestra: pSol cae ~2 puntos/año y domina la magnitud.
#   3. Decaimiento de fuente: -2.7 %/año, lo que queda tras controlar 1 y 2.
#
# Más la invarianza de los log-cocientes a la dilución, que depende de la
# ENERGÍA de los canales y no de ninguna corrección.
#
# Solo lectura: no escribe en data/.
#   pixi run python scripts/diagnostico_deriva_intensidades.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src.pipeline.config import LIMPIO, LIMPIO_COMPOSITO, CANALES, METALES

CORTE = pd.Timestamp("2025-12-10")        # escalón limpio de n6sc (mediana diaria)
ENERGIA_KA = {"n1fe": 6.40, "n2cu": 8.05, "n3zn": 8.64, "n4mo": 17.48}   # keV
pd.set_option("display.width", 210)


def cargar():
    df = pd.read_csv(LIMPIO, parse_dates=["ts"])
    df = df[df["valida"]].sort_values("ts").reset_index(drop=True)
    # pSol: limpieza de §2.10 (hay un 295 %, un 0 y 4 bajo 5 %)
    df["pSol_ok"] = df["pSol"].where((df["pSol"] > 5) & (df["pSol"] <= 100))
    df["logSumI"] = np.log(df[METALES].sum(axis=1))
    for a, i in [("fe", 1), ("zn", 3), ("mo", 4)]:
        df[f"lr_{a}_cu"] = np.log(df[f"n{i}{a}"] / df["n2cu"])
    return df


def ols(tabla, y, cols, nombre):
    """OLS con t y p, sin dependencias extra. Devuelve (coef, R2)."""
    s = tabla.dropna(subset=[y] + cols)
    X = np.column_stack([np.ones(len(s))] + [s[c].to_numpy() for c in cols])
    yv = s[y].to_numpy()
    b, *_ = np.linalg.lstsq(X, yv, rcond=None)
    sse = ((yv - X @ b) ** 2).sum()
    gl = len(s) - X.shape[1]
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * sse / gl)
    r2 = 1 - sse / ((yv - yv.mean()) ** 2).sum()
    print(f"  {nombre}  n={len(s)}  R2={r2:.3f}")
    for c, bi, si in zip(["const"] + cols, b, se):
        t = bi / si
        print(f"    {c:10s} {bi:+9.4f}  t={t:+6.2f}  p={2 * (1 - stats.t.cdf(abs(t), gl)):.2e}")
    return dict(zip(["const"] + cols, b)), r2


def seccion_1_escalon(df):
    print("=" * 100)
    print("1. EL ESCALÓN DE NOV-DIC 2025 — por canal, no común")
    print("=" * 100)
    pi = df[df["fuente"].isin(["pi", "ambos"])]
    d = pi.set_index("ts")[CANALES].resample("D").median().dropna()
    pre = d.loc[CORTE - pd.Timedelta("45D"):CORTE - pd.Timedelta("1D")].median()
    pos = d.loc[CORTE:CORTE + pd.Timedelta("45D")].median()
    raz = pos / pre
    print(pd.DataFrame({"antes": pre.round(1), "despues": pos.round(1), "razon": raz.round(4),
                        "cambio_%": ((raz - 1) * 100).round(1)}).to_string())
    print(f"\n  Si fuera dilución (ganancia COMÚN) las 5 razones serían iguales.")
    print(f"  Rango {raz.min():.3f} a {raz.max():.3f}; dispersión sd/media = {raz.std()/raz.mean():.3f}")
    print("\n  cocientes entre canales (invariantes a una ganancia común):")
    for num in ["n1fe", "n3zn", "n4mo", "n6sc"]:
        a, b = pre[num] / pre["n2cu"], pos[num] / pos["n2cu"]
        print(f"    {num}/n2cu: {a:.5f} -> {b:.5f}   ({(b/a - 1)*100:+.1f} %)")
    print("\n  mediana SEMANAL alrededor del escalón (el de n6sc es de un día):")
    print(pi.set_index("ts")[CANALES].resample("W").median()
            .loc["2025-11-23":"2026-01-11"].to_string(float_format=lambda v: f"{v:9.1f}"))


def seccion_2_psol(df):
    print("\n" + "=" * 100)
    print("2. DILUCIÓN DE LA MUESTRA — pSol cae, y no lo explica la química")
    print("=" * 100)
    s = df.dropna(subset=["pSol_ok"])
    print("  %sólido medido por año (laboratorio, independiente del courier):")
    print("   " + " | ".join(f"{a} {v:.2f}(n{int(n)})" for a, (n, v) in
                             s.groupby(s["ts"].dt.year)["pSol_ok"].agg(["count", "mean"]).iterrows()))
    t = s.groupby(s["ts"].dt.to_period("Q"))["pSol_ok"].agg(["count", "mean"])
    t = t[t["count"] >= 4]
    print("\n  por trimestre (máximo en 2023H2; la caída arranca en 2024, NO es monótona):")
    print("   " + " | ".join(f"{str(i)} {v:.1f}" for i, (n, v) in t.iterrows()))
    s = s.assign(t=(s["ts"] - s["ts"].min()).dt.total_seconds() / (365.25 * 86400))
    print("\n  pSol ~ tiempo:")
    ols(s, "pSol_ok", ["t"], "")
    print("\n  pSol ~ química del sólido + tiempo  (si 't' sobrevive, hay dilución de muestreo):")
    ols(s, "pSol_ok", ["pFe", "pCu", "pMo", "pIns", "t"], "")
    print("\n  >> la caída NO se atenúa al descontar química: es dilución, no mineral.")
    print("     Hipótesis de planta: líneas de agua agregadas contra espuma y atoro.")
    print("     No verificable de forma independiente: NO existe tag de densidad en la")
    print("     línea de concentrado final (confirmado con planta).")


def seccion_3_tres_causas(df):
    print("\n" + "=" * 100)
    print("3. LAS TRES CAUSAS, SEPARADAS (solo período PRE-recalibración)")
    print("=" * 100)
    s = df[df["ts"] < "2025-11-01"].dropna(subset=["pSol_ok", "pFe", "pCu", "pMo", "pIns", "logSumI"])
    s = s.assign(t=(s["ts"] - s["ts"].min()).dt.total_seconds() / (365.25 * 86400))
    print("  logSumI ~ tiempo:")
    ols(s, "logSumI", ["t"], "")
    print("\n  logSumI ~ química + tiempo:")
    ols(s, "logSumI", ["pFe", "pCu", "pMo", "pIns", "t"], "")
    print("\n  logSumI ~ química + pSol + tiempo:")
    c, _ = ols(s, "logSumI", ["pFe", "pCu", "pMo", "pIns", "pSol_ok", "t"], "")
    print(f"\n  >> pSol domina (R2 0.22 -> 0.70). Queda {100*c['t']:+.1f} %/año que sobrevive")
    print("     controlando química Y sólido: ese es el decaimiento de fuente.")


def seccion_4_invarianza(df):
    print("\n" + "=" * 100)
    print("4. INVARIANZA A LA DILUCIÓN — es física del canal, no una corrección")
    print("=" * 100)
    print("  El agua absorbe los rayos X según la energía (~E^-3). Un cociente entre")
    print("  canales de energía PARECIDA es invariante a la dilución; uno entre")
    print("  energías distintas no puede serlo, y el signo lo fija cuál se absorbe más.")
    print(f"\n  Energía Ka: " + "  ".join(f"{k} {v}" for k, v in ENERGIA_KA.items()) + " keV\n")
    s = df.dropna(subset=["pSol_ok"]).copy()
    s["mes"] = s["ts"].dt.to_period("M")
    print(f"  {'feature':<12}{'|dE| vs Cu':>11}{'corr(.,pSol)':>14}{'centrado mes':>14}")
    for f, dE in [("logSumI", None), ("lr_zn_cu", 0.59), ("lr_fe_cu", 1.65), ("lr_mo_cu", 9.43)]:
        q = s.dropna(subset=[f])
        qc = q.assign(a=q[f] - q.groupby("mes")[f].transform("mean"),
                      b=q["pSol_ok"] - q.groupby("mes")["pSol_ok"].transform("mean"))
        e = f"{dE:11.2f}" if dE is not None else f"{'-':>11}"
        print(f"  {f:<12}{e}{np.corrcoef(q[f], q['pSol_ok'])[0,1]:14.3f}"
              f"{np.corrcoef(qc['a'], qc['b'])[0,1]:14.3f}")
    print("\n  >> lr_zn_cu (Zn 8.64 ~ Cu 8.05) es el invariante; §10.7 lo midió en t=-1.3")
    print("     controlando química. lr_fe_cu y lr_mo_cu no pueden serlo: es física.")


def seccion_5_dos_niveles(df):
    print("\n" + "=" * 100)
    print("5. DOS NIVELES DE DILUCIÓN DENTRO DE CADA ÉPOCA")
    print("=" * 100)
    print("  El k=2 sobre los 15 meses es la recalibración (notebook 28). Quitándola y")
    print("  agrupando DENTRO de cada época aparecen dos niveles de sólido, y sobreviven")
    print("  al centrado por mes, que es el control con el que cayeron cuatro hallazgos.")
    for nom, m in [("pre  (2025-07-13..2025-12-09)", (df["ts"] >= "2025-07-13") & (df["ts"] < CORTE)),
                   ("post (2025-12-10..)", df["ts"] >= CORTE)]:
        sub = df[m & df["fuente"].isin(["pi", "ambos"])].copy()
        if len(sub) < 200:
            continue
        sub["gs"] = KMeans(2, n_init=10, random_state=42).fit_predict(
            StandardScaler().fit_transform(sub[CANALES]))
        q = sub.dropna(subset=["pSol_ok"]).copy()
        q["mes"] = q["ts"].dt.to_period("M")
        q["pSol_c"] = q["pSol_ok"] - q.groupby("mes")["pSol_ok"].transform("mean")
        g = [v["pSol_c"].to_numpy() for _, v in q.groupby("gs") if len(v) >= 5]
        print(f"\n  {nom}  n={len(sub)}")
        print(f"    pSol crudo por subgrupo    {q.groupby('gs')['pSol_ok'].mean().round(2).to_dict()}"
              f"  n={q.groupby('gs').size().to_dict()}")
        if len(g) >= 2:
            print(f"    pSol CENTRADO POR MES      {q.groupby('gs')['pSol_c'].mean().round(2).to_dict()}"
                  f"  p={stats.mannwhitneyu(*g).pvalue:.4f}")
        qc = q.assign(a=q["logSumI"] - q.groupby("mes")["logSumI"].transform("mean"),
                      b=q["pSol_c"])
        print(f"    corr(logSumI, pSol)        crudo {np.corrcoef(q['logSumI'], q['pSol_ok'])[0,1]:+.3f}"
              f"   centrado por mes {np.corrcoef(qc['a'], qc['b'])[0,1]:+.3f}")
    print("\n  >> dentro de época el proxy es MEJOR que el +0.659 global de §2.4:")
    print("     lo degradaba mezclar épocas de calibración.")


if __name__ == "__main__":
    df = cargar()
    print(f"lecturas válidas {len(df)}  |  con pSol {int(df['pSol_ok'].notna().sum())}\n")
    seccion_1_escalon(df)
    seccion_2_psol(df)
    seccion_3_tres_causas(df)
    seccion_4_invarianza(df)
    seccion_5_dos_niveles(df)
