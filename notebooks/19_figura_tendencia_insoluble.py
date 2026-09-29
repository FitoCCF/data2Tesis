#!/usr/bin/env python3
# ============================================================
# CELDA 19 — Gráfica de tendencia: oficial -> +insoluble -> control
# ============================================================
# Slope chart: una línea por elemento, conectando los tres R2 de la celda 18.
# Hace visible de un vistazo hacia dónde se mueve cada elemento al agregar
# insoluble, y si el control (sin información real) lo supera igual.
# Reusa las funciones de la celda 18 -- ningún número hardcodeado.
#
# Uso:
#   pixi run python notebooks/19_figura_tendencia_insoluble.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.pipeline import config as CFG

mod18 = __import__("18_figura_comparacion_insoluble")

COLOR = mod18.COLOR
TINTA, TINTA2 = mod18.TINTA, mod18.TINTA2
GRIS, SUPERFICIE, REJILLA = mod18.GRIS, mod18.SUPERFICIE, mod18.REJILLA
NOMBRES = {"pFe": "Fe", "pCu": "Cu", "pMo": "Mo", "pZn": "Zn"}


def _main():
    df = mod18._preparar_datos()
    _, r2l_of = mod18.mod16.entrenar_y_medir(df, "cluster_oficial")
    _, r2l_aug = mod18.mod16.entrenar_y_medir(df, "cluster_aug")
    _, r2l_ctrl = mod18.mod16.entrenar_y_medir(df, "cluster_ctrl")

    etapas = ["oficial", "+ insoluble", "control (azar)"]
    x = [0, 1, 2]

    print("Datos de la tendencia:")
    series = {}
    for tgt in CFG.TARGETS:
        vals = [mod18.mod16.r2_local_ponderado(r2l, tgt) for r2l in (r2l_of, r2l_aug, r2l_ctrl)]
        series[tgt] = vals
        print(f"  {tgt}: {[round(v, 3) for v in vals]}")

    fig, ax = plt.subplots(figsize=(7.6, 5.6), facecolor=SUPERFICIE)
    ax.set_facecolor(SUPERFICIE)
    ax.grid(True, axis="y", color=REJILLA, linewidth=0.8, zorder=0)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(REJILLA)
    ax.tick_params(colors=TINTA2, labelsize=9.5, length=3)

    for tgt in CFG.TARGETS:
        vals = series[tgt]
        color = COLOR[tgt]
        sube_al_final = vals[2] >= vals[1]   # el control termina igual o por encima del insoluble real
        ax.plot(x, vals, "-o", color=color, linewidth=2.2, markersize=6.5,
                markeredgecolor=SUPERFICIE, markeredgewidth=1.3, zorder=3,
                alpha=1.0 if sube_al_final else 0.55)
        ax.text(x[0] - 0.07, vals[0], f"{NOMBRES[tgt]}", ha="right", va="center",
               fontsize=10.5, color=color, fontweight="bold")
        ax.text(x[-1] + 0.07, vals[-1], f"{vals[-1]:.2f}", ha="left", va="center",
               fontsize=9, color=TINTA2)
        offsets = {"pMo": -0.022, "pFe": -0.022, "pCu": 0.018, "pZn": 0.018}
        va = "top" if offsets[tgt] < 0 else "bottom"
        ax.text(x[1], vals[1] + offsets[tgt], f"{vals[1]:.2f}",
               ha="center", va=va, fontsize=8, color=color)

    ax.set_xlim(-0.42, 2.28)
    ax.set_xticks(x)
    ax.set_xticklabels(etapas, fontsize=10.5, color=TINTA)
    ax.set_ylabel("R² local ponderado", fontsize=10, color=TINTA2)
    ax.set_title("El insoluble no cambia la tendencia: el control la sigue igual o la supera",
                fontsize=13, color=TINTA, loc="left", pad=14)
    fig.text(0.01, -0.02,
             "Cada línea es un elemento (Fe, Cu, Mo, Zn). Si el insoluble aportara información real, el tramo\n"
             "final (+insoluble -> control) debería bajar; en 3 de 4 elementos sigue subiendo.",
             ha="left", fontsize=8, color=TINTA2)

    CFG.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = CFG.REPORTS_DIR / "fig_tendencia_insoluble"
    for ext in ("png", "pdf"):
        fig.savefig(f"{ruta}.{ext}", dpi=200, facecolor=SUPERFICIE, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigura guardada en: {ruta}.png / .pdf")


if __name__ == "__main__":
    _main()
