#!/usr/bin/env python3
# ============================================================
# scripts/tendencia_en_linea.py — tendencia de lo estimado en línea vs el compósito
# ============================================================
# Lee lo que deja scripts/en_linea.py y dibuja, para Cu, Fe y Mo, los últimos
# N días:
#   - modelo por lectura (bloques de 30 min, azul fino): lo que el courier
#     estima en tiempo real, en la escala del modelo;
#   - modelo del compósito (azul grueso): ese mismo modelo promediado sobre
#     la ventana centrada de 12 h, antes del Kalman;
#   - estimado del compósito (naranja): promedio de la ventana centrada de
#     12 h llevado a escala balde con el Kalman -- la cifra que se reporta;
#     tramo horizontal sobre las 12 h que cubre; hueco si aún no está publicado;
#   - laboratorio (tinta oscura): el compósito cuando se publica (~20 h después).
# Cada tramo lleva su valor escrito al centro.
#
# Salida: reports/tendencia_en_linea.png y reports/tendencia_en_linea.html (la
# página se recarga sola cada minuto: abrirla una vez en el navegador basta).
# en_linea.py la regenera al final de cada ciclo; también se puede correr sola:
#
#   pixi run python scripts/tendencia_en_linea.py              # últimos 7 días
#   pixi run python scripts/tendencia_en_linea.py --dias 30
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import html
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pipeline.config import ESTIMADO, DATA_FINAL, REPORTS_DIR, VENTANA_COMPOSITO_H, BLOQUE_RESAMPLE
from src.evaluacion.evaluar import mascara_lecturas
from src.evaluacion.descripcion import modelo_concentrado
from src.evaluacion.figuras import SUPERFICIE, TINTA, TINTA_2, REJILLA, COLOR_LAB, COLOR_MODELO, COLOR_KALMAN

EN_LINEA = DATA_FINAL / "en_linea.csv"
PNG = REPORTS_DIR / "tendencia_en_linea.png"
HTML = REPORTS_DIR / "tendencia_en_linea.html"
LEYES = [("pCu", "Cu", 2), ("pFe", "Fe", 2), ("pMo", "Mo", 2)]      # columna, elemento, decimales
MEDIA = pd.Timedelta(hours=VENTANA_COMPOSITO_H / 2)


def _lecturas(desde: pd.Timestamp) -> pd.DataFrame:
    """Estimación por lectura válida, en bloques de 30 min (como la evaluación)."""
    est = pd.read_csv(ESTIMADO, parse_dates=["ts"])
    est = est[mascara_lecturas(est) & (est["ts"] >= desde - MEDIA)]
    cols = [f"{l}_est" for l, _, _ in LEYES]
    b = est.set_index("ts")[cols].resample(BLOQUE_RESAMPLE).mean()
    return b.reset_index()                                 # los bloques sin lectura quedan NaN: la línea se corta


@dataclass
class Perfil:
    """Qué se dibuja: el concentrado (por defecto) o la cabeza (scripts/en_linea_cabeza.py)."""
    titulo: str
    en_linea: Path
    png: Path
    html: Path
    leyes: list                                            # (columna, elemento, decimales)
    lecturas: Callable                                     # desde -> bloques de 30 min con <columna>_est
    monitor: str = "en_linea.py"
    modelo: str = ""                                      # vacío = el GMM del concentrado


CONCENTRADO = Perfil("Concentrado colectivo", EN_LINEA, PNG, HTML, LEYES, _lecturas)


def _metricas(C: pd.DataFrame, ley: str, dec: int = 2) -> str:
    d = C.dropna(subset=[f"{ley}_kal", ley])
    if len(d) < 2:
        return f"n={len(d)} comparados"
    e = d[f"{ley}_kal"] - d[ley]
    return f"n={len(d)} · MAE {e.abs().mean():.{dec}f} · sesgo {e.mean():+.{dec}f}"


def _valores(ax, C: pd.DataFrame, ley: str, dpi: float, dec: int = 2, sep_pt: float = 8.5) -> None:
    """Escribe el valor de modelo, Kalman y lab al centro de cada tramo. Si dos
    quedan a menos de sep_pt puntos en vertical, los separa hacia arriba (el
    más bajo se queda pegado a su tramo); el texto va en tinta, no en el color
    de la serie: el tramo de al lado ya dice cuál es."""
    a_pt = 72 / dpi
    for f in C.itertuples(index=False):
        vals = [(v, suf) for suf in ("_est", "_kal", "") if not np.isnan(v := getattr(f, f"{ley}{suf}"))]
        vals.sort()
        tope = -np.inf
        for v, suf in vals:
            y_pt = ax.transData.transform((0, v))[1] * a_pt   # altura del tramo, en puntos
            y_txt = max(y_pt + 1.5, tope + sep_pt)
            tope = y_txt
            ax.annotate(f"{v:.{dec}f}", xy=(f.ts, v), xytext=(0, y_txt - y_pt), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5, zorder=5,
                        bbox=dict(boxstyle="square,pad=0.1", facecolor=SUPERFICIE, edgecolor="none", alpha=0.85),
                        color=TINTA if suf != "_est" else TINTA_2,
                        fontweight="bold" if suf == "_kal" else "normal")


def figura(C: pd.DataFrame, L: pd.DataFrame, desde: pd.Timestamp, ultima: pd.Timestamp,
           p: Perfil = CONCENTRADO) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.3), sharex=True, facecolor=SUPERFICIE)
    ahora = pd.Timestamp.now()
    for ax, (ley, el, dec) in zip(axes, p.leyes):
        ax.set_facecolor(SUPERFICIE)
        ax.grid(True, color=REJILLA, linewidth=0.8, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(REJILLA)
        ax.tick_params(colors=TINTA_2, labelsize=9)

        ax.plot(L["ts"], L[f"{ley}_est"], color=COLOR_MODELO, linewidth=0.8, alpha=0.35, zorder=2,
                label="modelo por lectura (30 min)")
        series = [("_est", COLOR_MODELO, 1.6, "modelo, promedio 12 h (sin Kalman)"),
                  ("_kal", COLOR_KALMAN, 2.2, "estimado del compósito (Kalman)"),
                  ("", COLOR_LAB, 2.2, "laboratorio (compósito 12 h)")]
        etiquetas = {suf: nom for suf, _, _, nom in series}   # la leyenda lleva cada serie una sola vez
        for f in C.itertuples(index=False):              # un tramo por compósito, sobre sus 12 h
            x = [f.ts - MEDIA, f.ts + MEDIA]
            for suf, color, ancho, _ in series:
                v = getattr(f, f"{ley}{suf}")
                if np.isnan(v):
                    continue
                punteado = suf and not f.publicado        # estimación de un compósito aún sin publicar
                ax.plot(x, [v, v], color=color, linewidth=ancho, solid_capstyle="round", zorder=3 if suf else 4,
                        linestyle=(0, (2, 1.5)) if punteado else "-", label=etiquetas[suf])
                etiquetas[suf] = None
        ax.margins(y=0.18)                                # aire para los valores sobre los tramos
        ax.axvline(ahora, color=TINTA_2, linewidth=0.8, linestyle=":", zorder=1)
        ax.set_ylabel(f"% {el}", color=TINTA, fontsize=10)
        ax.text(1.0, 1.02, _metricas(C, ley, dec), transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5, color=TINTA, family="monospace")

    axes[0].text(ahora, 0.98, " ahora", transform=axes[0].get_xaxis_transform(), va="top", fontsize=8,
                 color=TINTA_2)
    axes[-1].set_xlim(desde, ahora + pd.Timedelta(hours=8))
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%m\n%H:%M"))
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.2)
    for ax, (ley, _, dec) in zip(axes, p.leyes):          # después del layout: ya se conoce la escala en píxeles
        _valores(ax, C, ley, fig.dpi, dec)
    fig.suptitle(f"{p.titulo} en línea — última lectura del courier {ultima:%d-%m %H:%M}  "
                 f"(actualizado {datetime.now():%d-%m %H:%M})", x=0.01, y=0.995, ha="left", va="top",
                 color=TINTA, fontsize=12)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.965), ncol=3, frameon=False,
               fontsize=9, labelcolor=TINTA)
    fig.text(0.99, 0.965, "punteado = compósito aún sin publicar", ha="right", va="top",
             fontsize=8, color=TINTA_2)
    fig.savefig(p.png, dpi=120, facecolor=SUPERFICIE)
    plt.close(fig)


def pagina(C: pd.DataFrame, ultima: pd.Timestamp, p: Perfil = CONCENTRADO) -> None:
    """HTML que se recarga solo cada 60 s: la figura y la tabla de los últimos compósitos."""
    filas = []
    for f in C.tail(10).iloc[::-1].itertuples(index=False):
        estado = "lab" if f.publicado else ("ventana cerrada" if f.ventana_completa else "en curso")
        celdas = [f"{f.ts:%d-%m %H:%M}", f"{f.n_bloques}/{2 * VENTANA_COMPOSITO_H + 1}", estado]
        for ley, _, dec in p.leyes:
            mod, kal, lab = (getattr(f, f"{ley}{s}") for s in ("_est", "_kal", ""))
            celdas.append("—" if np.isnan(mod) else f"<span class='d'>{mod:.{dec}f}</span>")
            celdas.append("—" if np.isnan(kal) else f"<b>{kal:.{dec}f}</b>")
            if np.isnan(lab):
                celdas.append("—")
            else:
                dif = "" if np.isnan(kal) else f" <span class='d'>({kal - lab:+.{dec}f})</span>"
                celdas.append(f"{lab:.{dec}f}{dif}")
        filas.append("<tr>" + "".join(f"<td>{c}</td>" for c in celdas) + "</tr>")
    horas = (pd.Timestamp.now() - ultima) / pd.Timedelta(hours=1)
    aviso = (f"<p class='aviso'>⚠ El courier no entrega lecturas hace {horas:.1f} h</p>" if horas > 1 else "")
    p.html.write_text(f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta http-equiv="refresh" content="60">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{p.titulo} en línea</title>
<style>
  body {{ background:{SUPERFICIE}; color:{TINTA}; font:14px system-ui, sans-serif; margin:16px; }}
  img {{ max-width:100%; height:auto; border:1px solid {REJILLA}; }}
  table {{ border-collapse:collapse; margin-top:12px; font-variant-numeric:tabular-nums; }}
  th, td {{ padding:4px 10px; border-bottom:1px solid {REJILLA}; text-align:right; }}
  th {{ color:{TINTA_2}; font-weight:600; }} th.g {{ text-align:center; }}
  .d {{ color:{TINTA_2}; }} .aviso {{ font-weight:600; }}
  .pie {{ color:{TINTA_2}; font-size:12px; }}
</style></head><body>
<h1 style="font-size:18px;margin:0 0 8px">{p.titulo} — estimación en línea vs compósito</h1>
<p style="font-weight:600;margin:0 0 8px">{html.escape(p.modelo or modelo_concentrado())}</p>
{aviso}
<img src="{p.png.name}?v={datetime.now():%Y%m%d%H%M%S}" alt="Tendencia de Cu, Fe y Mo: modelo, Kalman y laboratorio">
<div style="overflow-x:auto"><table><thead>
<tr><th rowspan="2">compósito</th><th rowspan="2">bloques</th><th rowspan="2">estado</th>
<th colspan="3" class="g">Cu</th><th colspan="3" class="g">Fe</th><th colspan="3" class="g">Mo</th></tr>
<tr>{'<th>modelo</th><th>Kalman</th><th>lab (Kalman − lab)</th>' * 3}</tr></thead>
<tbody>{''.join(filas)}</tbody></table></div>
<p class="pie">Modelo = promedio de la estimación por lectura sobre las 12 h centradas del compósito.
Kalman = modelo + sesgo aprendido de los compósitos ya publicados (escala balde: la cifra que se reporta). Última lectura del courier {html.escape(f'{ultima:%Y-%m-%d %H:%M}')}.
La página se recarga cada minuto; la figura cambia cuando {p.monitor} termina un ciclo.</p>
</body></html>""", encoding="utf-8")


def generar(dias: float = 7, p: Perfil = CONCENTRADO) -> None:
    C = pd.read_csv(p.en_linea, parse_dates=["ts", "ultima_lectura"])
    ultima = C["ultima_lectura"].iloc[-1]
    desde = pd.Timestamp.now() - pd.Timedelta(days=dias)
    C = C[C["ts"] + MEDIA >= desde].reset_index(drop=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    figura(C, p.lecturas(desde), desde, ultima, p)
    pagina(C, ultima, p)


def _main():
    ap = argparse.ArgumentParser(description="Tendencia de la estimación en línea vs compósito")
    ap.add_argument("--dias", type=float, default=7, help="días hacia atrás que muestra la figura")
    args = ap.parse_args()
    generar(args.dias)
    print(f"Figura: {PNG}\nPágina: {HTML}")


if __name__ == "__main__":
    _main()
