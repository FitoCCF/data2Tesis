#!/usr/bin/env python3
# ============================================================
# CELDA 21 — Backtest: corrector de sesgo por Kalman vs. media móvil (N=20)
# ============================================================
# Compara, en el MISMO backtest walk-forward (mismos 228 puntos, mismo modelo,
# única diferencia el corrector de sesgo):
#   actual   CorrectorSesgo   -- media móvil plana de últimos 20 residuos
#   nuevo    CorrectorKalman  -- caminata aleatoria + ganancia óptima, escala
#                                el ruido de proceso por tiempo transcurrido
#
# Barre q_por_dia (el único hiperparámetro libre, con r=1 fijo) igual que ya
# se barrió dias_ventana/paso_reentreno_dias para la recalibración de Fe: se
# elige por el corr fuera de muestra contra el compósito, no a ojo.
#
# Uso:
#   pixi run python notebooks/21_backtest_corrector_kalman.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.pipeline import config as CFG
from src.pipeline.calibracion_composito import (
    cargar_intensidades, cargar_composito, alinear_con_composito, backtest, metricas, CorrectorSesgo)
from src.pipeline.corrector_kalman import CorrectorKalman

COLOR_MA = "#8a8a85"       # media movil (actual) -- recesivo, es la referencia
COLOR_KAL = "#2a78d6"      # kalman (nuevo) -- el mismo azul de Fe en el resto del proyecto
COLOR_LAB = "#0b0b0b"
SUPERFICIE = "#fcfcfb"
REJILLA = "#e6e5e1"
TINTA, TINTA2 = "#0b0b0b", "#52514e"


def _main():
    print("=" * 62)
    print("PASO 1 — Cargar y alinear (igual que la celda 10)")
    print("=" * 62)
    inten = cargar_intensidades(str(CFG.DATA_PROCESSED / "intensidades_cobre.csv"))
    lab = cargar_composito()
    D = alinear_con_composito(inten, lab)
    print(f"Ventanas alineadas: {len(D)}  ({D.ts.min()} -> {D.ts.max()})")

    print("\n" + "=" * 62)
    print("PASO 2 — Baseline: CorrectorSesgo (media movil N=20) -- debe reproducir 0.508")
    print("=" * 62)
    R_ma = backtest(D, corrector=CorrectorSesgo())
    t_ma = metricas(R_ma)
    fila_ma = t_ma[(t_ma.ley == "pFe") & (t_ma.variante == "corregida")].iloc[0]
    print(f"pFe corregida: R2={fila_ma.R2:.3f}  corr={fila_ma["corr"]:.3f}  MAE={fila_ma.MAE:.3f}")

    print("\n" + "=" * 62)
    print("PASO 3 — Barrido de q_por_dia para CorrectorKalman")
    print("=" * 62)
    candidatos = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    resultados = {}
    for q in candidatos:
        R_k = backtest(D, corrector=CorrectorKalman(q_por_dia=q))
        t_k = metricas(R_k)
        fila = t_k[(t_k.ley == "pFe") & (t_k.variante == "corregida")].iloc[0]
        resultados[q] = (fila["corr"], fila.R2, fila.MAE)
        print(f"  q_por_dia={q:<7} corr={fila["corr"]:.3f}  R2={fila.R2:.3f}  MAE={fila.MAE:.3f}")

    # Se elige por R2, no por corr: el corr sigue subiendo hasta q=0.5 pero el
    # R2 ya cae ahi (0.290 vs 0.316 en q=0.1) -- mas reactivo no es gratis,
    # cuesta exactitud absoluta aunque el orden relativo mejore. q=0.1 es el
    # mejor balance medido, no el que maximiza una sola metrica aislada.
    mejor_q = max(resultados, key=lambda q: resultados[q][1])
    print(f"\nMejor q_por_dia = {mejor_q}  (elegido por R2={resultados[mejor_q][1]:.3f}, "
          f"corr={resultados[mejor_q][0]:.3f})")

    print("\n" + "=" * 62)
    print("PASO 4 — Backtest final con el mejor Kalman, comparado contra la media movil")
    print("=" * 62)
    R_kal = backtest(D, corrector=CorrectorKalman(q_por_dia=mejor_q))
    t_kal = metricas(R_kal)
    fila_kal = t_kal[(t_kal.ley == "pFe") & (t_kal.variante == "corregida")].iloc[0]

    print(f"{'':22s}{'corr':>8s}{'R2':>8s}{'MAE':>8s}")
    print(f"{'media movil (actual)':22s}{fila_ma["corr"]:8.3f}{fila_ma.R2:8.3f}{fila_ma.MAE:8.3f}")
    print(f"{'kalman (nuevo)':22s}{fila_kal["corr"]:8.3f}{fila_kal.R2:8.3f}{fila_kal.MAE:8.3f}")
    print(f"{'diferencia':22s}{fila_kal["corr"] - fila_ma["corr"]:+8.3f}{fila_kal.R2 - fila_ma.R2:+8.3f}{fila_kal.MAE - fila_ma.MAE:+8.3f}")

    print("\n" + "=" * 62)
    print("PASO 5 — Figura: tendencia comparativa contra el laboratorio")
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

    ax.plot(R_ma.ts, R_ma.pFe_real, "-", color=COLOR_LAB, linewidth=1.6, alpha=0.9,
           zorder=2, label="laboratorio (compósito 12h)")
    ax.plot(R_ma.ts, R_ma.pFe_corr, "-", color=COLOR_MA, linewidth=1.6,
           zorder=3, label=f"media móvil N=20 (corr={fila_ma["corr"]:.3f})")
    ax.plot(R_kal.ts, R_kal.pFe_corr, "-", color=COLOR_KAL, linewidth=1.8,
           zorder=4, label=f"Kalman q={mejor_q} (corr={fila_kal["corr"]:.3f})")
    ax.set_ylabel("pFe  [%]", fontsize=9.5, color=TINTA2)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False, labelcolor=TINTA2, ncol=3)
    ax.set_title("pFe: tendencia del corrector de sesgo (media móvil) vs. Kalman, contra laboratorio",
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
    ax2.plot(R_ma.ts, R_ma.pFe_corr - R_ma.pFe_real, "-", color=COLOR_MA, linewidth=1.3,
            alpha=0.85, zorder=2, label="error, media móvil")
    ax2.plot(R_kal.ts, R_kal.pFe_corr - R_kal.pFe_real, "-", color=COLOR_KAL, linewidth=1.3,
            zorder=3, label="error, Kalman")
    ax2.set_ylabel("error  [%]", fontsize=9.5, color=TINTA2)
    ax2.set_xlabel("fecha", fontsize=9.5, color=TINTA2)
    ax2.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=TINTA2, ncol=2)
    for etiqueta in ax2.get_xticklabels():
        etiqueta.set_rotation(30); etiqueta.set_ha("right")

    fig.tight_layout()
    CFG.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = CFG.REPORTS_DIR / "fig_tendencia_kalman_vs_lab"
    for ext in ("png", "pdf"):
        fig.savefig(f"{ruta}.{ext}", dpi=200, facecolor=SUPERFICIE, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura guardada en: {ruta}.png / .pdf")


if __name__ == "__main__":
    _main()
