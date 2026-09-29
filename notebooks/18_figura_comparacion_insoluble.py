#!/usr/bin/env python3
# ============================================================
# CELDA 18 — Figura: R2 local por ley, ruteo oficial vs +insoluble vs control
# ============================================================
# Resume en una figura el cierre del hilo 5.4 (celdas 13-17): tres barras por
# elemento (ruteo oficial / +ins_t real / control barajado) para que se vea de
# un vistazo que el insoluble real no supera al ruido. Mismos datos que las
# tablas impresas en sesión, recalculados aquí (nada hardcodeado) para que la
# figura sea reproducible desde los mismos artefactos.
#
# Uso:
#   pixi run python notebooks/18_figura_comparacion_insoluble.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.preprocessing import PowerTransformer, StandardScaler

from src.pipeline import config as CFG
from src.pipeline.clustering import entrenar_clustering
from src.pipeline.ruteo_insoluble import cargar_composito_crudo

mod16 = __import__("16_modelos_locales_con_insoluble")

# --- Paleta: reusa los 3 colores ya validados del proyecto (fig_validacion_3elementos) ---
COLOR = {"pFe": "#2a78d6", "pCu": "#eb6834", "pMo": "#1baf7a",
         "pZn": "#8a5fbf"}   # Zn no forma parte de la paleta validada de 3 -- añadido aquí, sin validar contraste
TINTA, TINTA2 = "#0b0b0b", "#52514e"
GRIS, SUPERFICIE, REJILLA = "#8a8a85", "#fcfcfb", "#e6e5e1"


def _preparar_datos():
    df = pd.read_csv(CFG.DATA_PROCESSED / "intensidad_cobre_24_completo_filtrado.csv")
    regresiones_ortho = joblib.load(CFG.MODELS_DIR / CFG.ART_ORTHO)
    X_ref = df[["n6sc"]].values
    for metal in CFG.METALES:
        df[f"{metal}_ortho"] = df[metal].values - regresiones_ortho[metal].predict(X_ref)

    power = joblib.load(CFG.MODELS_DIR / CFG.ART_POWER)
    scaler = joblib.load(CFG.MODELS_DIR / CFG.ART_SCALER)
    gmm_of = joblib.load(CFG.MODELS_DIR / CFG.ART_GMM)
    df["cluster_oficial"] = gmm_of.predict(scaler.transform(power.transform(df[CFG.FEATS_CLUSTER].values)))

    stream = pd.read_csv(CFG.DATA_PROCESSED / "intensidad_cobre_24_orthogonalized.csv")
    stream["ts"] = mod16.construir_ts(stream)
    lab = cargar_composito_crudo()
    stream = pd.merge_asof(stream.sort_values("ts"),
                           lab[["ts", "ins"]].rename(columns={"ins": "ins_t"}),
                           on="ts", direction="backward").dropna(subset=["ins_t"])

    feats_aug = CFG.FEATS_CLUSTER + ["ins_t"]
    p_aug = PowerTransformer(method="yeo-johnson", standardize=False).fit(stream[feats_aug].values)
    Xp_aug = p_aug.transform(stream[feats_aug].values)
    s_aug = StandardScaler().fit(Xp_aug)
    _, gmm_aug, _, _ = entrenar_clustering(s_aug.transform(Xp_aug), k=3)

    df["ts"] = mod16.construir_ts(df)
    df = pd.merge_asof(df.sort_values("ts"),
                       lab[["ts", "ins"]].rename(columns={"ins": "ins_t"}),
                       on="ts", direction="backward").dropna(subset=["ins_t"]).reset_index(drop=True)
    df["cluster_aug"] = gmm_aug.predict(s_aug.transform(p_aug.transform(df[feats_aug].values)))

    rng = np.random.default_rng(CFG.RANDOM_STATE)
    df["cluster_ctrl"] = rng.permutation(df["cluster_aug"].values)
    return df


def _main():
    df = _preparar_datos()

    _, r2l_of = mod16.entrenar_y_medir(df, "cluster_oficial")
    _, r2l_aug = mod16.entrenar_y_medir(df, "cluster_aug")
    _, r2l_ctrl = mod16.entrenar_y_medir(df, "cluster_ctrl")

    leyes = CFG.TARGETS   # ["pFe", "pCu", "pMo", "pZn"]
    vals_of = [mod16.r2_local_ponderado(r2l_of, t) for t in leyes]
    vals_aug = [mod16.r2_local_ponderado(r2l_aug, t) for t in leyes]
    vals_ctrl = [mod16.r2_local_ponderado(r2l_ctrl, t) for t in leyes]

    print("Datos de la figura:")
    for t, o, a, c in zip(leyes, vals_of, vals_aug, vals_ctrl):
        print(f"  {t}: oficial={o:.3f}  +ins_t={a:.3f}  control={c:.3f}")

    fig, ax = plt.subplots(figsize=(8.5, 5.0), facecolor=SUPERFICIE)
    ax.set_facecolor(SUPERFICIE)
    ax.grid(True, axis="y", color=REJILLA, linewidth=0.8, zorder=0)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(REJILLA)
    ax.tick_params(colors=TINTA2, labelsize=9, length=3)

    x = np.arange(len(leyes))
    ancho = 0.29

    for i, (etiqueta, vals, hatch) in enumerate([
        ("ruteo oficial (sin insoluble)", vals_of, None),
        ("+ insoluble real (ins_t, turno anterior)", vals_aug, None),
        ("control (cluster barajado, sin info real)", vals_ctrl, "//"),
    ]):
        offset = (i - 1) * ancho
        colores = [COLOR[t] if i == 1 else (GRIS if i == 2 else TINTA2) for t in leyes]
        barras = ax.bar(x + offset, vals, width=ancho * 0.82, color=colores,
                        edgecolor=SUPERFICIE, linewidth=1.2, zorder=3,
                        alpha=(1.0 if i == 1 else 0.55), hatch=hatch)
        for b, v in zip(barras, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                   ha="center", va="bottom", fontsize=7.3, color=TINTA2)

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("p", "") for t in leyes], fontsize=11, color=TINTA, fontweight="bold")
    ax.set_ylabel("R² local ponderado (validación cruzada, etapa 5)", fontsize=9.5, color=TINTA2)
    ax.set_ylim(0, max(vals_of + vals_aug + vals_ctrl) * 1.18)

    ax.legend(handles=[
        Patch(facecolor=TINTA2, alpha=1.0, label="ruteo oficial (sin insoluble)"),
        Patch(facecolor=COLOR["pFe"], alpha=1.0, label="+ insoluble real (ins_t)"),
        Patch(facecolor=GRIS, alpha=0.55, hatch="//", label="control: cluster barajado (sin info real)"),
    ], loc="upper right", fontsize=8, frameon=False, labelcolor=TINTA2)

    ax.set_title("El insoluble en el ruteo de modelos locales no supera al control aleatorio",
                fontsize=12.5, color=TINTA, loc="left", pad=12)
    fig.text(0.01, -0.02,
             "R² local ponderado por tamaño de celda, mismo KFold(5, shuffle=True) en las tres barras.\n"
             "El control gana en 3 de 4 leyes -- la ganancia real no se distingue del ruido de partición.",
             ha="left", fontsize=8, color=TINTA2)

    CFG.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = CFG.REPORTS_DIR / "fig_insoluble_vs_control"
    for ext in ("png", "pdf"):
        fig.savefig(f"{ruta}.{ext}", dpi=200, facecolor=SUPERFICIE, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigura guardada en: {ruta}.png / .pdf")


if __name__ == "__main__":
    _main()
