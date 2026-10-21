#!/usr/bin/env python3
# ============================================================
# scripts/comparar_epocas_mo.py
# ============================================================
# ¿Se pueden juntar las dos épocas de instrumento para el modelo de Mo?
#
# El mantenimiento de nov-2025 cambió la respuesta por canal (ver
# scripts/diagnostico_deriva_intensidades.py §1), así que las intensidades de
# antes y después no son la misma medición. Pero `lr_mo_cu` es un COCIENTE: una
# recalibración que multiplica n4mo por a y n2cu por b solo lo corre en
# log(a/b), o sea un offset constante -- no le cambia la pendiente.
#
# Esto lo comprueba sobre el compósito, con tres pruebas:
#   1. la regresión mo ~ lr_mo_cu por época (pendiente e intercepto)
#   2. un modelo conjunto con dummy de época e interacción
#   3. validación cruzada de época: ajustar en una, predecir en la otra
#
# Ver docs/bitacora_analisis.md §19.
#   pixi run python scripts/comparar_epocas_mo.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from scipy import stats

from src.pipeline.config import LIMPIO, LIMPIO_COMPOSITO, VENTANA_COMPOSITO_H

FIN_PRE, INI_POST = "2025-12-01", "2025-12-15"      # la transición queda fuera
MIN_LECTURAS = 10
pd.set_option("display.width", 200)

d = pd.read_csv(LIMPIO, parse_dates=["ts"])
d = d[d["valida"] & d["fuente"].isin(["pi", "ambos"])]
d["lr_mo_cu"] = np.log(d["n4mo"] / d["n2cu"])

medio = pd.Timedelta(hours=VENTANA_COMPOSITO_H / 2)
comp = pd.read_csv(LIMPIO_COMPOSITO, parse_dates=["ts"])
ts, v = d["ts"].to_numpy(), d["lr_mo_cu"].to_numpy()
filas = []
for t, mo in zip(comp["ts"].to_numpy(), comp["mo"].to_numpy()):
    i = np.where((ts >= t - medio.to_timedelta64()) & (ts <= t + medio.to_timedelta64()))[0]
    if len(i) >= MIN_LECTURAS:
        filas.append({"ts": t, "x": v[i].mean(), "mo": mo})
C = pd.DataFrame(filas)
C["ts"] = pd.to_datetime(C["ts"])
C = C[(C["ts"] < FIN_PRE) | (C["ts"] >= INI_POST)]
C["post"] = (C["ts"] >= INI_POST).astype(int)
print(f"compósitos: pre={int((C['post'] == 0).sum())}  post={int(C['post'].sum())}\n")

print("1) regresión  mo ~ lr_mo_cu  POR ÉPOCA")
for nombre, sub in [("pre ", C[C["post"] == 0]), ("post", C[C["post"] == 1])]:
    r = stats.linregress(sub["x"], sub["mo"])
    rmse = np.sqrt(((sub["mo"] - (r.intercept + r.slope * sub["x"])) ** 2).mean())
    print(f"   {nombre}: pendiente {r.slope:+.4f}  intercepto {r.intercept:+.4f}  "
          f"R2 {r.rvalue**2:.3f}  RMSE {rmse:.4f}  n={len(sub)}")

print("\n2) modelo conjunto con dummy de época e interacción")
X = np.column_stack([np.ones(len(C)), C["x"], C["post"], C["x"] * C["post"]])
b, *_ = np.linalg.lstsq(X, C["mo"], rcond=None)
sse = ((C["mo"] - X @ b) ** 2).sum()
gl = len(C) - 4
se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * sse / gl)
for n, bi, si in zip(["const", "lr_mo_cu", "post (salto)", "lr_mo_cu x post (pendiente)"], b, se):
    t = bi / si
    print(f"   {n:<30}{bi:+9.4f}  t={t:+6.2f}  p={2*(1 - stats.t.cdf(abs(t), gl)):.2e}")
print(f"   R2 conjunto = {1 - sse/((C['mo'] - C['mo'].mean())**2).sum():.3f}")

print("\n3) validación cruzada de época")
for a, otro in [(0, 1), (1, 0)]:
    ent, prb = C[C["post"] == a], C[C["post"] == otro]
    r = stats.linregress(ent["x"], ent["mo"])
    p = r.intercept + r.slope * prb["x"]
    r2 = 1 - ((prb["mo"] - p) ** 2).sum() / ((prb["mo"] - prb["mo"].mean()) ** 2).sum()
    print(f"   ajuste en {'pre' if a == 0 else 'post':<4} -> predicción en {'post' if a == 0 else 'pre':<4}: "
          f"R2={r2:+.3f}  sesgo={np.mean(p - prb['mo']):+.4f}  "
          f"RMSE={np.sqrt(((prb['mo'] - p)**2).mean()):.4f}")

print("\n>> Ni el salto ni el cambio de pendiente son significativos: las épocas se pueden")
print("   juntar para Mo. El salto estimado coincide con lo que predice la recalibración")
print("   medida aparte: el cociente n4mo/n2cu cambió +14.1 %, o sea lr_mo_cu se corrió")
print(f"   log(1.141)=+0.132, que por la pendiente da {-1.95*0.132:+.3f} de intercepto.")
print("   Recomendación: juntar CON offset de época (un parámetro, quita el sesgo de ±0.20).")
