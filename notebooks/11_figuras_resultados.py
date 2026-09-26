#!/usr/bin/env python3
# ============================================================
# CELDA 11 — FIGURAS Y TABLA DE RESULTADOS (3 elementos)
# ============================================================
# Genera las figuras del capítulo de resultados y la tabla de métricas, todo
# desde los artefactos ya entrenados. Es reproducible: mismas entradas -> mismas
# figuras y mismos números.
#
# EVALUACIÓN UNIFICADA — cada modelo se aplica COMO FUE VALIDADO
# --------------------------------------------------------------
# No se puede evaluar los tres elementos con el mismo procedimiento, porque no
# se calibraron igual:
#
#   pFe  -> modelo RECALIBRADO contra el compósito (celda 10). Se entrenó con
#           intensidades promediadas en la ventana de 12 h, así que se aplica a
#           intensidades promediadas en la ventana. Viene del backtest
#           walk-forward, que nunca usa el futuro.
#   pCu  -> modelo ORIGINAL (etapa 5). Se entrenó con muestras puntuales
#   pMo     emparejadas al instante exacto de una lectura, así que se aplica por
#           lectura y las predicciones se promedian sobre la ventana.
#
# Ambos se comparan contra el MISMO compósito de 12 h y sobre LAS MISMAS 228
# ventanas, que es lo que hace comparables las cifras.
#
# Referencia de posicionamiento: el techo de correlación por elemento viene de
# colocación triple (modelo / calibración de fábrica / compósito, n=184).
#
# Uso:
#   pixi run python notebooks/11_figuras_resultados.py
# ============================================================

import sys, os                                            # utilidades del sistema
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # independiente del CWD

import argparse                                           # argumentos de línea de comandos
import warnings                                            # silenciar avisos cosméticos de sklearn
warnings.filterwarnings("ignore")

import joblib                                             # cargar artefactos
import numpy as np                                        # cálculo numérico
import pandas as pd                                       # DataFrames
import matplotlib                                         # backend sin pantalla
matplotlib.use("Agg")                                     # no requiere display (corre en servidor)
import matplotlib.pyplot as plt                           # figuras
from sklearn.metrics import r2_score, mean_absolute_error  # métricas

from src.pipeline import config as CFG                    # configuración central
from src.pipeline.calibracion_composito import cargar_intensidades, cargar_composito

# ============================================================
# Paleta — slots 1-3 de la paleta categórica validada
# ============================================================
# Validada con el script de la guía de visualización, en modo all-pairs (que es
# el que corresponde a scatter) y en ambos modos:
#   light  CVD ΔE 9.2  normal-vision ΔE 24.0   -> PASS
#   dark   CVD ΔE 9.4  normal-vision ΔE 20.9   -> PASS
# El aqua en modo claro queda bajo 3:1 de contraste, así que aplica la regla de
# relieve: cada panel lleva etiqueta directa y existe la tabla de métricas.
COLOR = {"Fe": "#2a78d6",                                  # slot 1 - azul
         "Cu": "#eb6834",                                  # slot 2 - naranja
         "Mo": "#1baf7a"}                                  # slot 3 - aqua
TINTA = "#0b0b0b"                                          # texto primario
TINTA2 = "#52514e"                                         # texto secundario
GRIS = "#8a8a85"                                           # marcas recesivas (laboratorio, identidad 1:1)
SUPERFICIE = "#fcfcfb"                                     # superficie del gráfico
REJILLA = "#e6e5e1"                                        # rejilla recesiva

# Techo de correlación por colocación triple (n=184) y desempeño de fábrica
TECHO = {"Fe": 0.743, "Cu": 0.634, "Mo": 0.961}            # correlación máxima alcanzable
FABRICA = {"Fe": 0.470, "Cu": 0.517, "Mo": 0.948}          # calibración del fabricante
ANTES = {"Fe": 0.278, "Cu": 0.505, "Mo": 0.958}            # modelo antes de la recalibración

ELEMENTOS = [("pFe", "fe", "Fe"), ("pCu", "cu", "Cu"), ("pMo", "mo", "Mo")]  # (col modelo, col lab, nombre)


# ============================================================
# 1. Construcción del dataset de evaluación unificado
# ============================================================

def construir_evaluacion(ruta_intensidades: str) -> pd.DataFrame:
    """Arma la tabla de evaluación con los tres elementos sobre las mismas ventanas.

    pFe sale del backtest (modelo recalibrado); pCu y pMo se scorean con el
    modelo original por lectura y se promedian en la misma ventana de 12 h.
    """
    # --- pFe: backtest walk-forward de la celda 10 (fuera de muestra) ---
    bt = pd.read_csv(CFG.DATA_PROCESSED / "backtest_recalibracion.csv", parse_dates=["ts"])

    # --- pCu y pMo: modelo original (etapa 5), aplicado por lectura ---
    bundle = joblib.load(CFG.MODELS_DIR / CFG.ART_REGRESION)       # modelos locales/globales
    ortho = joblib.load(CFG.MODELS_DIR / CFG.ART_ORTHO)            # regresiones de ortogonalización
    power = joblib.load(CFG.MODELS_DIR / CFG.ART_POWER)            # PowerTransformer
    escala = joblib.load(CFG.MODELS_DIR / CFG.ART_SCALER)          # StandardScaler del clustering
    gmm = joblib.load(CFG.MODELS_DIR / CFG.ART_GMM)                # GMM de clustering

    inten = cargar_intensidades(ruta_intensidades)                 # stream de intensidades
    inten = inten.loc[bt.ts.min() - pd.Timedelta(hours=7):]        # solo lo necesario para las ventanas

    X = pd.DataFrame(index=inten.index)                            # features de la etapa 5
    for metal in CFG.METALES:                                      # ortogonaliza cada metal vs n6sc
        X[f"{metal}_ortho"] = inten[metal].values - ortho[metal].predict(inten[["n6sc"]].values)
    X["n6sc"] = inten["n6sc"].values                               # conserva n6sc

    cluster = gmm.predict(escala.transform(power.transform(X[CFG.FEATS_CLUSTER].values)))  # cluster por lectura
    V = X[bundle["features"]].values                               # matriz en el orden congelado

    for ley in ("pCu", "pMo"):                                     # las leyes que NO se recalibraron
        pred = np.empty(len(V))                                    # vector de predicciones
        for c in np.unique(cluster):                                # se rutea por cluster
            mascara = cluster == c                                 # lecturas de ese cluster
            modelo = (bundle["modelos_locales"][(ley, c)]          # local o global según la tabla de ruteo
                      if bundle["tabla_ruteo"][(ley, c)] == "local"
                      else bundle["modelos_globales"][ley])
            pred[mascara] = np.ravel(modelo.predict(V[mascara]))   # predice ese subconjunto
        inten[ley] = pred                                          # guarda la columna

    # --- Alinear todo sobre las ventanas del backtest ---
    lab = cargar_composito()                                       # compósito de 12 h
    filas = []                                                     # acumulador
    media_h = CFG.VENTANA_COMPOSITO_H / 2                          # medio ancho de ventana (6 h)
    for _, r in bt.iterrows():                                     # una fila por ventana del backtest
        w = inten.loc[r.ts - pd.Timedelta(hours=media_h):          # lecturas de la ventana
                      r.ts + pd.Timedelta(hours=media_h)]
        ensayo = lab[lab.ts == r.ts]                               # ensayo de laboratorio de esa ventana
        if len(w) < 3 or ensayo.empty:                             # ventana insuficiente o sin ensayo
            continue                                               # se descarta
        filas.append({"ts": r.ts, "turno": r.turno,
                      "pFe": r.pFe_corr, "fe": r.pFe_real,         # Fe recalibrado (ya corregido de sesgo)
                      "pCu": w.pCu.mean(), "cu": float(ensayo.cu.iloc[0]),  # Cu promediado
                      "pMo": w.pMo.mean(), "mo": float(ensayo.mo.iloc[0])})  # Mo promediado

    E = pd.DataFrame(filas).replace([np.inf, -np.inf], np.nan).dropna()  # limpia no finitos
    return E.sort_values("ts").reset_index(drop=True)              # ordenado por tiempo


def tabla_metricas(E: pd.DataFrame) -> pd.DataFrame:
    """Calcula la tabla de métricas por elemento."""
    filas = []                                                     # una fila por elemento
    for col_mod, col_lab, nombre in ELEMENTOS:                     # recorre los tres elementos
        y = E[col_lab].values                                      # laboratorio (compósito)
        p = E[col_mod].values                                      # estimación del modelo
        filas.append({
            "elemento": nombre,                                    # Fe / Cu / Mo
            "n": len(y),                                           # nº de ventanas comparadas
            "R2": r2_score(y, p),                                  # coeficiente de determinación
            "MAE": mean_absolute_error(y, p),                      # error absoluto medio (%)
            "RMSE": float(np.sqrt(((p - y) ** 2).mean())),          # error cuadrático medio
            "corr": float(np.corrcoef(y, p)[0, 1]),                # correlación (métrica del techo)
            "pendiente": float(np.polyfit(y, p, 1)[0]),            # pendiente predicho vs real
            "sesgo": float(p.mean() - y.mean()),                   # sesgo medio
            "sd_modelo": float(p.std()),                           # dispersión de la estimación
            "sd_lab": float(y.std()),                              # dispersión del laboratorio
            "antes": ANTES[nombre],                                # correlación previa (referencia)
            "fabrica": FABRICA[nombre],                            # correlación de la calibración de fábrica
            "techo": TECHO[nombre],                                # techo por colocación triple
        })
    return pd.DataFrame(filas)                                     # tabla de métricas


# ============================================================
# 2. Estilo común de las figuras
# ============================================================

def _estilo_eje(ax):
    """Aplica el estilo recesivo: sin marco superior/derecho, rejilla suave."""
    for lado in ("top", "right"):                                  # quita el marco innecesario
        ax.spines[lado].set_visible(False)                         # menos tinta, más datos
    for lado in ("left", "bottom"):                                # ejes que sí se conservan
        ax.spines[lado].set_color(REJILLA)                         # en color recesivo
    ax.tick_params(colors=TINTA2, labelsize=8, length=3)           # marcas pequeñas y discretas
    ax.grid(True, color=REJILLA, linewidth=0.8, zorder=0)          # rejilla detrás de los datos
    ax.set_axisbelow(True)                                         # garantiza que la rejilla quede detrás


# ============================================================
# 3. FIGURA 1 — dispersión estimado vs laboratorio + serie temporal
# ============================================================

def figura_validacion(E: pd.DataFrame, T: pd.DataFrame, ruta_base):
    """Figura principal: una columna por elemento, dos filas.

    Fila 1 (dispersión): la línea discontinua gris es la identidad 1:1 (acuerdo
    perfecto); la línea de color es el ajuste real. La distancia entre ambas ES
    la compresión de la pendiente, que es el hallazgo central: con correlación
    limitada, el estimador óptimo en error cuadrático comprime hacia la media.

    Fila 2 (serie temporal): laboratorio en gris, modelo en color, para ver si
    el modelo sigue las variaciones del proceso o solo su nivel medio.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.4), facecolor=SUPERFICIE)  # 2 filas x 3 elementos

    for j, (col_mod, col_lab, nombre) in enumerate(ELEMENTOS):     # una columna por elemento
        color = COLOR[nombre]                                      # color del elemento
        m = T[T.elemento == nombre].iloc[0]                         # métricas de ese elemento
        y = E[col_lab].values                                      # laboratorio
        p = E[col_mod].values                                      # modelo

        # ---------- Fila 1: dispersión estimado vs laboratorio ----------
        ax = axes[0, j]                                            # panel superior
        ax.set_facecolor(SUPERFICIE)                               # superficie explícita
        _estilo_eje(ax)                                            # estilo recesivo

        lo = min(y.min(), p.min())                                 # límite inferior común
        hi = max(y.max(), p.max())                                 # límite superior común
        pad = (hi - lo) * 0.06                                     # margen visual
        lo, hi = lo - pad, hi + pad                                # límites con margen

        ax.plot([lo, hi], [lo, hi], "--", color=GRIS,              # identidad 1:1 (acuerdo perfecto)
                linewidth=1.4, zorder=2, label="1:1 (acuerdo perfecto)")

        ax.scatter(y, p, s=26, color=color, alpha=0.75,             # puntos: >=8px, anillo de superficie 2px
                   edgecolors=SUPERFICIE, linewidths=0.8, zorder=3)

        b, a = np.polyfit(y, p, 1)                                 # ajuste lineal (pendiente, intercepto)
        xs = np.array([lo, hi])                                    # extremos para dibujar la recta
        ax.plot(xs, b * xs + a, "-", color=color, linewidth=2.0,    # recta ajustada (2px)
                zorder=4, label=f"ajuste (pend. {b:.2f})")

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)                   # ejes cuadrados (1:1 visualmente honesto)
        ax.set_aspect("equal", adjustable="box")                   # misma escala en ambos ejes
        ax.set_xlabel(f"{nombre} laboratorio (compósito 12 h)  [%]", fontsize=8.5, color=TINTA2)
        if j == 0:                                                 # solo la primera columna rotula el eje Y
            ax.set_ylabel("Estimación del modelo  [%]", fontsize=8.5, color=TINTA2)

        # Etiqueta directa: cumple la regla de relieve (el aqua no alcanza 3:1)
        ax.set_title(nombre, fontsize=13, color=color, fontweight="bold", loc="left", pad=8)
        ax.text(0.035, 0.965,                                       # caja de métricas dentro del panel
                f"R² = {m.R2:.3f}\nMAE = {m.MAE:.3f} %\nr = {m["corr"]:.3f}\npend. = {m.pendiente:.3f}",
                transform=ax.transAxes, fontsize=8, color=TINTA,
                va="top", ha="left", linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.42", facecolor=SUPERFICIE,
                          edgecolor=REJILLA, linewidth=0.9))
        ax.legend(loc="lower right", fontsize=7.2, frameon=False, labelcolor=TINTA2)

        # ---------- Fila 2: serie temporal ----------
        ax = axes[1, j]                                            # panel inferior
        ax.set_facecolor(SUPERFICIE)                               # superficie explícita
        _estilo_eje(ax)                                            # estilo recesivo

        ax.plot(E.ts, y, "-", color=GRIS, linewidth=1.5,            # laboratorio en gris (referencia)
                alpha=0.9, zorder=2, label="laboratorio")
        ax.plot(E.ts, p, "-", color=color, linewidth=1.8,           # modelo en color
                zorder=3, label="modelo")

        ax.set_xlabel("fecha", fontsize=8.5, color=TINTA2)
        if j == 0:                                                 # solo la primera columna rotula el eje Y
            ax.set_ylabel("Ley  [%]", fontsize=8.5, color=TINTA2)
        ax.legend(loc="upper right", fontsize=7.2, frameon=False, labelcolor=TINTA2, ncol=2)
        for etiqueta in ax.get_xticklabels():                      # fechas legibles sin colisionar
            etiqueta.set_rotation(30); etiqueta.set_ha("right")

    fig.suptitle("Estimación de leyes vs laboratorio químico (compósito 12 h, fuera de muestra)",
                 fontsize=13, color=TINTA, y=0.985)
    fig.text(0.5, 0.945,                                            # subtítulo con el protocolo
             f"n = {len(E)} ventanas  ·  {E.ts.min():%Y-%m-%d} a {E.ts.max():%Y-%m-%d}  ·  "
             "Fe: modelo recalibrado (backtest walk-forward)  ·  Cu y Mo: modelo original",
             ha="center", fontsize=8.5, color=TINTA2)
    fig.tight_layout(rect=[0, 0, 1, 0.93])                          # deja espacio al título

    for ext in ("pdf", "png"):                                      # PDF para LaTeX, PNG para revisar
        fig.savefig(f"{ruta_base}.{ext}", dpi=200, facecolor=SUPERFICIE, bbox_inches="tight")
    plt.close(fig)                                                  # libera memoria


# ============================================================
# 4. FIGURA 2 — posición de cada elemento respecto de su techo
# ============================================================

def figura_techo(T: pd.DataFrame, ruta_base):
    """Compara la correlación lograda contra la de fábrica y contra el techo.

    El techo NO se dibuja como una barra más: es un límite, no un competidor,
    así que va como marca vertical. Las barras son las dos cosas comparables
    entre sí (el modelo y la calibración del fabricante).
    """
    fig, ax = plt.subplots(figsize=(9.2, 4.8), facecolor=SUPERFICIE)  # una sola vista
    ax.set_facecolor(SUPERFICIE)                                    # superficie explícita
    _estilo_eje(ax)                                                 # estilo recesivo
    ax.grid(True, axis="x", color=REJILLA, linewidth=0.8)           # rejilla solo en el eje de magnitud
    ax.grid(False, axis="y")                                        # sin rejilla horizontal (categorías)

    alto = 0.3                                                      # alto de cada barra
    posiciones = np.arange(len(T))[::-1]                            # Fe arriba, Mo abajo

    for i, (_, r) in enumerate(T.iterrows()):                       # una fila por elemento
        pos = posiciones[i]                                         # posición vertical
        color = COLOR[r.elemento]                                   # color del elemento

        # Techo: banda recesiva de fondo (el espacio disponible)
        ax.barh(pos, r.techo, height=alto * 2.35, color=REJILLA,    # fondo = techo alcanzable
                zorder=1, edgecolor=SUPERFICIE, linewidth=2)

        # Barras comparables: modelo (color) y fábrica (gris)
        ax.barh(pos + alto * 0.58, r["corr"], height=alto, color=color,  # modelo de la tesis
                zorder=3, edgecolor=SUPERFICIE, linewidth=2)
        ax.barh(pos - alto * 0.58, r.fabrica, height=alto, color=GRIS,  # calibración de fábrica
                zorder=3, edgecolor=SUPERFICIE, linewidth=2)

        # Etiquetas directas de valor (nunca un número en cada punto, sí en cada barra)
        ax.text(r["corr"] + 0.012, pos + alto * 0.58, f"{r["corr"]:.3f}",
                va="center", fontsize=8.5, color=TINTA, fontweight="bold")
        ax.text(r.fabrica + 0.012, pos - alto * 0.58, f"{r.fabrica:.3f}",
                va="center", fontsize=8.5, color=TINTA2)

        # Marca del techo: límite, no competidor
        ax.plot([r.techo, r.techo], [pos - alto * 1.18, pos + alto * 1.18],
                color=TINTA2, linewidth=1.8, zorder=4)
        ax.text(r.techo + 0.012, pos, f"techo {r.techo:.3f}",
                va="center", fontsize=7.8, color=TINTA2, style="italic")

    ax.set_yticks(posiciones)                                       # una marca por elemento
    ax.set_yticklabels([f"{r.elemento}" for _, r in T.iterrows()],  # nombres de los elementos
                       fontsize=12, color=TINTA, fontweight="bold")
    ax.set_xlim(0, 1.16)                                            # deja aire para la etiqueta del techo
    ax.set_xlabel("Correlación con el laboratorio (compósito 12 h)", fontsize=9, color=TINTA2)

    # Leyenda: identidad nunca por color solo -> etiquetas de texto explícitas.
    # Va FUERA del área de datos: dentro colisionaba con la fila de Mo, cuyas
    # barras llegan casi al borde derecho.
    from matplotlib.patches import Patch                            # parches para la leyenda
    ax.legend(handles=[Patch(facecolor=COLOR["Fe"], label="modelo (este trabajo)"),
                       Patch(facecolor=GRIS, label="calibración de fábrica del analizador"),
                       Patch(facecolor=REJILLA, label="techo alcanzable (colocación triple)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.17),      # una fila, debajo del eje
              ncol=3, fontsize=8, frameon=False, labelcolor=TINTA2)

    ax.set_title("Desempeño respecto del límite de información del sistema de medición",
                 fontsize=12.5, color=TINTA, loc="left", pad=12)
    fig.tight_layout()                                              # ajusta márgenes

    for ext in ("pdf", "png"):                                      # PDF para LaTeX, PNG para revisar
        fig.savefig(f"{ruta_base}.{ext}", dpi=200, facecolor=SUPERFICIE, bbox_inches="tight")
    plt.close(fig)                                                  # libera memoria


# ============================================================
# CLI
# ============================================================
def _main():
    ap = argparse.ArgumentParser(description="Celda 11: figuras y tabla de resultados")
    ap.add_argument("--intensidades", default=str(CFG.DATA_PROCESSED / "intensidades_cobre.csv"),
                    help="CSV del stream de intensidades del analizador")
    args, _ = ap.parse_known_args()                                # tolera flags de Jupyter

    CFG.REPORTS_DIR.mkdir(parents=True, exist_ok=True)             # asegura que exista reports/

    E = construir_evaluacion(args.intensidades)                    # dataset de evaluación unificado
    T = tabla_metricas(E)                                          # tabla de métricas

    # --- Tabla en consola ---
    print("=" * 96)
    print(f"TABLA DE MÉTRICAS — estimación vs laboratorio (compósito 12 h), n = {len(E)} ventanas")
    print(f"período: {E.ts.min():%Y-%m-%d} a {E.ts.max():%Y-%m-%d}")
    print("=" * 96)
    cols = ["elemento", "n", "R2", "MAE", "RMSE", "corr", "pendiente", "sesgo", "sd_modelo", "sd_lab"]
    print(T[cols].round(3).to_string(index=False))
    print()
    print("Posición respecto del límite del sistema de medición (correlación):")
    print(T[["elemento", "antes", "corr", "fabrica", "techo"]]
          .rename(columns={"corr": "ahora"}).round(3).to_string(index=False))

    # --- Guardar tabla y datos ---
    ruta_tabla = CFG.REPORTS_DIR / "tabla_metricas_3elementos.csv"  # tabla para el documento
    T.to_csv(ruta_tabla, index=False)                              # se guarda completa
    ruta_datos = CFG.DATA_PROCESSED / "evaluacion_final_3elementos.csv"  # datos de la figura
    E.to_csv(ruta_datos, index=False)                              # auditable

    # --- Figuras ---
    f1 = CFG.REPORTS_DIR / "fig_validacion_3elementos"             # figura principal
    f2 = CFG.REPORTS_DIR / "fig_techo_vs_fabrica"                  # figura de posicionamiento
    figura_validacion(E, T, str(f1))                               # dispersión + serie temporal
    figura_techo(T, str(f2))                                       # modelo vs fábrica vs techo

    print(f"\nTabla   : {ruta_tabla}")
    print(f"Datos   : {ruta_datos}")
    print(f"Figura 1: {f1}.pdf / .png   (dispersión + serie temporal por elemento)")
    print(f"Figura 2: {f2}.pdf / .png   (modelo vs fábrica vs techo)")


if __name__ == "__main__":
    _main()
