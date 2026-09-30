#!/usr/bin/env python3
# ============================================================
# src/pipeline/ruteo_insoluble.py — Experimento: ruteo de pFe por insoluble
# ============================================================
# Pregunta que responde (docs/bitacora_analisis.md, hilo abierto 5.4):
# ¿el ruteo por insoluble reproduce el +0.144 en R2(pFe) medido en la sección
# 2.6, si se evalúa con el backtest walk-forward honesto en vez del CV 5-fold
# sobre las 314 muestras puntuales?
#
# Dos diferencias deliberadas respecto a la medición original de 2.6:
#   1. Se usa el insoluble del COMPÓSITO ANTERIOR (ins_lag = shift(1)), nunca
#      el del compósito que se está evaluando -- en producción no se conoce
#      todavía. La medición de 2.6 usó el insoluble REAL simultáneo (optimista
#      para ese fin, por eso ahí mismo se llama "estimación del límite superior").
#   2. Se evalúa con walk-forward (reentreno periódico, solo pasado) en vez de
#      CV 5-fold, que es lo que este proyecto ya identificó como optimista en
#      otras partes del pipeline (ver decisiones.md de data2Tesis).
#
# Diseño del ruteo: terciles de ins_lag, calculados SOLO con la ventana móvil
# de entrenamiento vigente en ese momento (nunca con datos futuros ni con todo
# el histórico). Un modelo Ridge por tercil, mismo regresor que el pipeline
# base (_construir_regresor). Si la ventana no alcanza para 3 tercios de al
# menos `min_por_bin` cada uno, ese paso NO rutea: usa el modelo global (queda
# registrado en la columna `bin`='global_fallback', no se oculta).
# ============================================================

import numpy as np
import pandas as pd

from .config import (RANDOM_STATE, LAB_COMPOSITO, MAPA_LEY_COMPOSITO,
                     DIAS_VENTANA_MOVIL, PASO_REENTRENO_DIAS)
from .features import features_regresion, columnas_regresion
from .calibracion_composito import _construir_regresor, ventana_de_entrenamiento, cargar_composito


# ============================================================
# 1. CARGA DEL COMPÓSITO CON INSOLUBLE (composito_pi.csv)
# ============================================================

def cargar_composito_crudo(ruta=None) -> pd.DataFrame:
    """Compósito de 12 h (data/raw/composito_pi.csv, etapa 0) con la columna
    'ins'. Mismo formato que calibracion_composito.cargar_composito() -- ts
    naive en hora local + fe/cu/ins/mo --, descartando además las filas con
    insoluble en cero (dato inválido del lab)."""
    lab = cargar_composito(ruta or LAB_COMPOSITO)
    if "ins" in lab.columns:
        lab = lab[lab["ins"] > 0]
    return lab.reset_index(drop=True)


def agregar_ins_lag(D: pd.DataFrame) -> pd.DataFrame:
    """Agrega ins_lag = insoluble del compósito ANTERIOR (shift 1, ~12h atrás).

    Es la información real disponible en producción: cuando se predice el
    compósito de las 07:30, el de las 19:30 del día anterior ya se conoce.
    La primera fila de toda la serie queda sin ins_lag (NaN) y se descarta.
    """
    D = D.sort_values("ts").reset_index(drop=True)
    D["ins_lag"] = D["ins"].shift(1)
    return D


# ============================================================
# 2. BACKTEST WALK-FORWARD CON RUTEO POR TERCIL DE ins_lag
# ============================================================

def backtest_ruteo(D: pd.DataFrame, ley: str = "pFe",
                   dias_ventana: int = DIAS_VENTANA_MOVIL,
                   paso_reentreno_dias: int = PASO_REENTRENO_DIAS,
                   frac_test: float = 0.30,
                   n_bins: int = 3, min_por_bin: int = 15,
                   control_aleatorio: bool = False,
                   random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Backtest walk-forward de ruteo por tercil de ins_lag, para una ley.

    Mismo protocolo temporal que calibracion_composito.backtest(): reentreno
    cada `paso_reentreno_dias` con ventana móvil de `dias_ventana`, sin mirar
    nunca el futuro. La diferencia es que en cada reentreno se ajustan
    n_bins modelos (uno por tercil de ins_lag EN ESA VENTANA), en vez de uno
    global.

    control_aleatorio=True baraja ins_lag dentro de cada ventana antes de
    formar los tercios -- control negativo: si el ruteo real gana pero el
    aleatorio no, la ganancia es de la variable, no de tener más parámetros
    (mismo chequeo que en la sección 2.6 de la bitácora).

    Retorna
    -------
    DataFrame: ts, turno, ins_lag, bin, {ley}_global, {ley}_ruteo, {ley}_real.
    """
    col_lab = MAPA_LEY_COMPOSITO[ley]
    D = D.sort_values("ts").reset_index(drop=True)
    corte = D["ts"].quantile(1 - frac_test)
    idx_test = D.index[D["ts"] > corte]

    rng = np.random.default_rng(random_state)

    bundle_global = None
    bundle_bins = None       # dict bin -> pipeline entrenado, o None si el paso no ruteó
    edges = None
    ultimo_reentreno = None

    filas = []
    for i in idx_test:
        fila = D.loc[i]
        t = fila["ts"]
        if pd.isna(fila["ins_lag"]):        # la primerísima fila de la serie completa
            continue

        # --- Reentrenar si toca ---
        if (ultimo_reentreno is None or
                (t - ultimo_reentreno).days >= paso_reentreno_dias):
            ventana = ventana_de_entrenamiento(D, t, dias_ventana)
            ventana = ventana.dropna(subset=["ins_lag"])

            if len(ventana) < 40:
                ultimo_reentreno = None      # sigue intentando en el siguiente punto
                continue

            # --- modelo global (mismo regresor, mismos datos que el pipeline base) ---
            X_g = features_regresion(ventana)[columnas_regresion()]
            bundle_global = _construir_regresor().fit(X_g.values, ventana[col_lab].values)

            # --- terciles de ins_lag, SOLO con esta ventana ---
            ins_bin = ventana["ins_lag"].sample(frac=1.0, random_state=rng.integers(1 << 31)).values \
                if control_aleatorio else ventana["ins_lag"].values
            try:
                _, edges_raw = pd.qcut(ins_bin, n_bins, retbins=True, duplicates="drop")
            except ValueError:
                edges_raw = None

            bundle_bins = None
            if edges_raw is not None and len(edges_raw) == n_bins + 1:
                edges = edges_raw.copy()
                edges[0], edges[-1] = -np.inf, np.inf   # cualquier ins_lag futuro cae en el bin extremo
                bin_idx = pd.cut(ins_bin, edges, labels=False, include_lowest=True)
                ventana = ventana.assign(_bin=bin_idx)
                conteo = ventana["_bin"].value_counts()
                if (conteo >= min_por_bin).all() and len(conteo) == n_bins:
                    bundle_bins = {}
                    for b in range(n_bins):
                        sub = ventana[ventana["_bin"] == b]
                        Xb = features_regresion(sub)[columnas_regresion()]
                        bundle_bins[b] = _construir_regresor().fit(Xb.values, sub[col_lab].values)

            ultimo_reentreno = t

        if bundle_global is None:
            continue

        # --- Predicción global ---
        Xt = features_regresion(D.loc[[i]])[columnas_regresion()]
        pred_global = float(bundle_global.predict(Xt.values)[0])

        # --- Predicción ruteada (o fallback si el paso no pudo formar tercios) ---
        if bundle_bins is not None and edges is not None:
            b_test = int(pd.cut([fila["ins_lag"]], edges, labels=False, include_lowest=True)[0])
            pred_ruteo = float(bundle_bins[b_test].predict(Xt.values)[0])
            etiqueta_bin = str(b_test)
        else:
            pred_ruteo = pred_global
            etiqueta_bin = "global_fallback"

        filas.append({
            "ts": t, "turno": fila["turno"], "ins_lag": float(fila["ins_lag"]),
            "bin": etiqueta_bin,
            f"{ley}_global": pred_global,
            f"{ley}_ruteo": pred_ruteo,
            f"{ley}_real": float(fila[col_lab]),
        })

    return pd.DataFrame(filas)


def resumen(R: pd.DataFrame, ley: str = "pFe") -> pd.DataFrame:
    """R2 y correlación de la variante global vs. ruteada, sobre los MISMOS
    puntos de prueba (comparación pareada)."""
    from sklearn.metrics import r2_score
    y = R[f"{ley}_real"].values
    filas = []
    for variante in ("global", "ruteo"):
        p = R[f"{ley}_{variante}"].values
        filas.append({
            "variante": variante, "n": len(y),
            "R2": r2_score(y, p),
            "corr": float(np.corrcoef(y, p)[0, 1]),
        })
    return pd.DataFrame(filas)
