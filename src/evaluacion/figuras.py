# ============================================================
# src/evaluacion/figuras.py — Tendencia comparativa: modelo vs laboratorio
# ============================================================
# Un panel por ley: laboratorio (referencia, tinta oscura), estimación del
# modelo (azul) y estimación corregida por Kalman (naranja). Periodo de
# entrenamiento sombreado y SIN métricas; las métricas del recuadro son solo
# fuera de muestra. Paleta validada con el validador de dataviz (azul/naranja:
# CVD ΔE 24.7, normal 33.6, contraste >= 3:1).
# ============================================================

import matplotlib
matplotlib.use("Agg")                                      # sin pantalla
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

SUPERFICIE, TINTA, TINTA_2, REJILLA = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOR_LAB, COLOR_MODELO, COLOR_KALMAN = "#0b0b0b", "#2a78d6", "#eb6834"
SOMBRA_TRAIN = "#f0efec"
NOMBRE = {"pFe": "Fe", "pCu": "Cu", "pMo": "Mo", "pZn": "Zn"}


def _fmt(m: dict) -> str:
    if "corr" not in m:
        return f"n={m.get('n', 0)}"
    return f"corr {m['corr']:.3f} · R² {m['R2']:.3f} · MAE {m['MAE']:.3f} · sesgo {m['sesgo']:+.3f}"


def _cortar_huecos(A: pd.DataFrame, max_hueco) -> pd.DataFrame:
    """Inserta una fila NaN donde dos puntos seguidos distan más de max_hueco,
    para que la línea no una tramos sin datos."""
    A = A.sort_values("ts").reset_index(drop=True)
    salto = A["ts"].diff() > pd.Timedelta(max_hueco)
    if not salto.any():
        return A
    cortes = pd.DataFrame({"ts": A.loc[salto, "ts"] - pd.Timedelta(seconds=1)})
    return pd.concat([A, cortes]).sort_values("ts").reset_index(drop=True)


def figura_tendencia(A: pd.DataFrame, leyes: list, titulo: str, ruta_png,
                     metricas_txt: dict, con_kalman: bool = True,
                     inicio_test=None, inicio_prod=None, marcador: bool = False, desde=None,
                     max_hueco="4D"):
    """A: ts, <ley> (lab), <ley>_est, opcional <ley>_kal.
    metricas_txt: ley -> {'modelo': dict, 'kalman': dict} (fuera de muestra).
    desde: recorta el eje x (las métricas no cambian: vienen calculadas)."""
    if desde is not None:
        A = A[A["ts"] >= pd.Timestamp(desde)]
    A = _cortar_huecos(A, max_hueco)                      # sin líneas sobre huecos de datos
    fig, axes = plt.subplots(len(leyes), 1, figsize=(12, 3.1 * len(leyes)), sharex=True,
                             facecolor=SUPERFICIE)
    axes = [axes] if len(leyes) == 1 else axes
    estilo = dict(marker="o", markersize=3.5, linewidth=0.9) if marcador else dict(linewidth=1.2)

    for ax, ley in zip(axes, leyes):
        ax.set_facecolor(SUPERFICIE)
        ax.grid(True, color=REJILLA, linewidth=0.8, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(REJILLA)
        ax.tick_params(colors=TINTA_2, labelsize=9)

        if inicio_test is not None and A["ts"].min() < inicio_test:   # train sombreado
            ax.axvspan(A["ts"].min(), inicio_test, color=SOMBRA_TRAIN, zorder=1, linewidth=0)
            ax.text(A["ts"].min(), 0.98, "  entrenamiento (dentro de muestra)", transform=ax.get_xaxis_transform(),
                    va="top", ha="left", fontsize=8, color=TINTA_2)
        if inicio_prod is not None:
            ax.axvline(inicio_prod, color=TINTA_2, linewidth=0.8, linestyle="--", zorder=1)
            ax.text(inicio_prod, 0.98, " producción", transform=ax.get_xaxis_transform(),
                    va="top", ha="left", fontsize=8, color=TINTA_2)

        ax.plot(A["ts"], A[ley], color=COLOR_LAB, zorder=4, label="laboratorio químico", **estilo)
        ax.plot(A["ts"], A[f"{ley}_est"], color=COLOR_MODELO, zorder=3, label="modelo", **estilo)
        if con_kalman and f"{ley}_kal" in A:
            ax.plot(A["ts"], A[f"{ley}_kal"], color=COLOR_KALMAN, zorder=3,
                    label="modelo + Kalman", **estilo)

        ax.set_ylabel(f"% {NOMBRE[ley]}", color=TINTA, fontsize=10)
        m = metricas_txt.get(ley, {})
        lineas = [f"modelo:            {_fmt(m['modelo'])}"] if "modelo" in m else []
        if con_kalman and "kalman" in m:
            lineas.append(f"modelo + Kalman: {_fmt(m['kalman'])}")
        if lineas:                                        # métricas encima del panel, fuera de los datos
            ax.text(1.0, 1.02, "\n".join(lineas), transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8, color=TINTA, family="monospace")

    meses = max(1, round((A["ts"].max() - A["ts"].min()).days / 30 / 12))   # ~12 ticks como máximo
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=meses))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.5)
    fig.suptitle(titulo, x=0.01, y=0.995, ha="left", va="top", color=TINTA, fontsize=12)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.965), ncol=3, frameon=False,
               fontsize=9, labelcolor=TINTA)
    fig.savefig(ruta_png, dpi=150, facecolor=SUPERFICIE)
    fig.savefig(str(ruta_png).replace(".png", ".pdf"), facecolor=SUPERFICIE)
    plt.close(fig)
