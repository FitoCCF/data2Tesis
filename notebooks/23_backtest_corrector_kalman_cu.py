#!/usr/bin/env python3
# ============================================================
# CELDA 23 — Corrector Kalman de sesgo aplicado a Cu (nunca probado)
# ============================================================
# Cu usa el modelo original (etapa 5) sin ningun corrector de sesgo posterior
# -- el "cambio 3" (CorrectorSesgo/Kalman) solo se aplico a Fe. La bitacora ya
# midio que el sesgo de Cu tambien deambula por trimestre, incluso en la
# calibracion de fabrica (2.3: sesgo_cu 0.462 -> 1.109 -> 0.055 -> 0.018).
# Aqui se agrega, sin retocar el modelo de Cu ni recalibrarlo (eso ya se probo
# y empeora, ver "Los cuatro cambios implementados"), solo el mismo tipo de
# correccion de sesgo posterior que ya funciono en Fe.
#
# Protocolo identico al de backtest() en calibracion_composito.py, aplicado a
# mano sobre las 228 ventanas ya construidas por construir_evaluacion(): para
# cada ventana, se corrige con el estado ANTERIOR del corrector, y solo
# DESPUES se revela el valor real y se actualiza -- sin fuga.
#
# Uso:
#   pixi run python notebooks/23_backtest_corrector_kalman_cu.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

from src.pipeline import config as CFG
from src.pipeline.corrector_kalman import CorrectorKalman

mod11 = __import__("11_figuras_resultados")

COLOR_ACTUAL = "#8a8a85"
COLOR_KAL = "#eb6834"   # el mismo naranja de Cu en el resto del proyecto
COLOR_LAB = "#0b0b0b"
SUPERFICIE = "#fcfcfb"
REJILLA = "#e6e5e1"
TINTA, TINTA2 = "#0b0b0b", "#52514e"


def metricas(y, p):
    return {"R2": r2_score(y, p), "MAE": mean_absolute_error(y, p),
            "corr": float(np.corrcoef(y, p)[0, 1])}


def corregir_serie(E, col_pred, col_real, q_por_dia):
    """Aplica CorrectorKalman sobre una serie ya construida (predicción
    cruda + real + turno + ts), sin fuga: corrige con el estado anterior,
    revela y actualiza recién después."""
    corrector = CorrectorKalman(q_por_dia=q_por_dia)
    correg = np.empty(len(E))
    for i, r in enumerate(E.itertuples()):
        pred = getattr(r, col_pred)
        real = getattr(r, col_real)
        correg[i] = corrector.corregir("x", pred, r.turno)
        corrector.actualizar("x", pred, real, r.turno, ts=r.ts)
    return correg


def _main():
    print("=" * 62)
    print("PASO 1 — Evaluación unificada (igual que la celda 11, sin tocar Fe)")
    print("=" * 62)
    E = mod11.construir_evaluacion(str(CFG.DATA_PROCESSED / "intensidades_cobre.csv"))
    E = E.sort_values("ts").reset_index(drop=True)
    print(f"n = {len(E)} ventanas  ({E.ts.min():%Y-%m-%d} a {E.ts.max():%Y-%m-%d})")

    base = metricas(E.cu.values, E.pCu.values)
    print(f"\nCu SIN corrector (actual): corr={base['corr']:.3f}  R2={base['R2']:.3f}  MAE={base['MAE']:.3f}")

    print("\n" + "=" * 62)
    print("PASO 2 — Barrido de q_por_dia")
    print("=" * 62)
    candidatos = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    resultados = {}
    for q in candidatos:
        corr_serie = corregir_serie(E, "pCu", "cu", q)
        m = metricas(E.cu.values, corr_serie)
        resultados[q] = m
        print(f"  q_por_dia={q:<7} corr={m['corr']:.3f}  R2={m['R2']:.3f}  MAE={m['MAE']:.3f}")

    mejor_q = max(resultados, key=lambda q: resultados[q]["R2"])
    print(f"\nMejor q_por_dia = {mejor_q}  (R2={resultados[mejor_q]['R2']:.3f}, corr={resultados[mejor_q]['corr']:.3f})")

    print("\n" + "=" * 62)
    print("PASO 3 — Comparación final")
    print("=" * 62)
    E["pCu_kalman"] = corregir_serie(E, "pCu", "cu", mejor_q)
    kal = metricas(E.cu.values, E.pCu_kalman.values)

    print(f"{'':22s}{'corr':>8s}{'R2':>8s}{'MAE':>8s}")
    print(f"{'sin corrector (actual)':22s}{base['corr']:8.3f}{base['R2']:8.3f}{base['MAE']:8.3f}")
    print(f"{'kalman (nuevo)':22s}{kal['corr']:8.3f}{kal['R2']:8.3f}{kal['MAE']:8.3f}")
    print(f"{'diferencia':22s}{kal['corr']-base['corr']:+8.3f}{kal['R2']-base['R2']:+8.3f}{kal['MAE']-base['MAE']:+8.3f}")

    print("\n" + "=" * 62)
    print("PASO 4 — Figura: tendencia comparativa contra el laboratorio")
    print("=" * 62)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), facecolor=SUPERFICIE,
                             sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})

    ax = axes[0]
    ax.set_facecolor(SUPERFICIE)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(REJILLA)
    ax.tick_params(colors=TINTA2, labelsize=8.5, length=3)
    ax.grid(True, color=REJILLA, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.plot(E.ts, E.cu, "-", color=COLOR_LAB, linewidth=1.6, alpha=0.9,
           zorder=2, label="laboratorio (compósito 12h)")
    ax.plot(E.ts, E.pCu, "-", color=COLOR_ACTUAL, linewidth=1.6,
           zorder=3, label=f"sin corrector, actual (corr={base['corr']:.3f})")
    ax.plot(E.ts, E.pCu_kalman, "-", color=COLOR_KAL, linewidth=1.8,
           zorder=4, label=f"Kalman q={mejor_q} (corr={kal['corr']:.3f})")
    ax.set_ylabel("pCu  [%]", fontsize=9.5, color=TINTA2)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False, labelcolor=TINTA2, ncol=3)
    ax.set_title("pCu: modelo original sin corrector vs. con corrector Kalman, contra laboratorio",
                fontsize=13, color=TINTA, loc="left", pad=10)

    ax2 = axes[1]
    ax2.set_facecolor(SUPERFICIE)
    for lado in ("top", "right"):
        ax2.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax2.spines[lado].set_color(REJILLA)
    ax2.tick_params(colors=TINTA2, labelsize=8.5, length=3)
    ax2.grid(True, color=REJILLA, linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    ax2.axhline(0, color=REJILLA, linewidth=1.2, zorder=1)
    ax2.plot(E.ts, E.pCu - E.cu, "-", color=COLOR_ACTUAL, linewidth=1.3,
            alpha=0.85, zorder=2, label="error, sin corrector")
    ax2.plot(E.ts, E.pCu_kalman - E.cu, "-", color=COLOR_KAL, linewidth=1.3,
            zorder=3, label="error, Kalman")
    ax2.set_ylabel("error  [%]", fontsize=9.5, color=TINTA2)
    ax2.set_xlabel("fecha", fontsize=9.5, color=TINTA2)
    ax2.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=TINTA2, ncol=2)
    for etiqueta in ax2.get_xticklabels():
        etiqueta.set_rotation(30); etiqueta.set_ha("right")

    fig.tight_layout()
    CFG.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = CFG.REPORTS_DIR / "fig_tendencia_kalman_cu_vs_lab"
    for ext in ("png", "pdf"):
        fig.savefig(f"{ruta}.{ext}", dpi=200, facecolor=SUPERFICIE, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura guardada en: {ruta}.png / .pdf")


if __name__ == "__main__":
    _main()
