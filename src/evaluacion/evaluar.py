# ============================================================
# src/evaluacion/evaluar.py — Evaluación de la estimación vs laboratorio químico
# ============================================================
# Etapas 8 y 9. Toma la salida de la etapa 6 (courier_estimado.csv) y la
# compara contra las dos referencias de laboratorio:
#
#   compósito de 12 h (etapa 8): media de la estimación en la ventana CENTRADA
#       [t-6h, t+6h] del corte, en bloques de 30 min (autocorrelación 0.907 a
#       lag-1) con un mínimo de 8 bloques. Mismo criterio validado en
#       calibracion_composito.alinear_con_composito. Fe, Cu y Mo (sin Zn).
#   muestras puntuales de la BD (etapa 9): la estimación de la lectura exacta
#       de cada muestra (calce por valor de la etapa 1b). Las 4 leyes.
#
# Corrector de sesgo por Kalman: mismo protocolo causal del backtest del
# 2026-09-29 (cada compósito se corrige con los anteriores y recién después
# se revela su valor). q_por_dia se elige SOLO con compósitos del periodo train
# y se aplica fijo a test y producción.
# ============================================================

import numpy as np                                         # cálculo numérico
import pandas as pd                                         # DataFrames
from sklearn.metrics import r2_score, mean_absolute_error   # métricas

from ..pipeline.config import (VENTANA_COMPOSITO_H, MIN_BLOQUES_VENTANA, BLOQUE_RESAMPLE,
                               FECHA_CORTE_TRAIN, FECHA_INICIO_PRODUCCION)
from ..pipeline.limpieza import fin_de_train
from ..pipeline.corrector_kalman import CorrectorKalman

LEYES_COMPOSITO = {"pFe": "fe", "pCu": "cu", "pMo": "mo"}   # ley del modelo -> columna del compósito
Q_GRID = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]      # barrido de q_por_dia (r = 1 fijo)


def periodo_de(ts: pd.Series) -> pd.Series:
    """Mismo corte que la etapa 1c: train hasta el día de corte entero."""
    return pd.Series(np.select([ts < fin_de_train(FECHA_CORTE_TRAIN),
                                ts < pd.Timestamp(FECHA_INICIO_PRODUCCION)],
                               ["train", "test"], "produccion"), index=ts.index)


def mascara_lecturas(df: pd.DataFrame, con_anomalias: bool = False) -> pd.Series:
    """Lecturas continuas del courier (PI) usables para promediar.
    Siempre fuera: congeladas y sostenidas. Las anomalías del IsolationForest
    se excluyen salvo con_anomalias=True (la etapa 1c no permite decidir si
    son mineral o instrumento: se reporta con y sin)."""
    m = (df["fuente"] != "bd") & ~df["congelada"] & ~df["sostenida"]
    return m if con_anomalias else m & ~df["anomalia"]


def alinear_composito(est: pd.DataFrame, comp: pd.DataFrame, mascara: pd.Series) -> pd.DataFrame:
    """Una fila por compósito: media de la estimación en su ventana centrada +
    ley del compósito. Descarta ventanas con menos de MIN_BLOQUES_VENTANA."""
    cols = [f"{l}_est" for l in LEYES_COMPOSITO]
    bloques = (est.loc[mascara].set_index("ts")[cols]
               .resample(BLOQUE_RESAMPLE).mean().dropna())  # submuestreo a 30 min
    media_h = pd.Timedelta(hours=VENTANA_COMPOSITO_H / 2)
    comp = comp[(comp["ts"] >= bloques.index.min()) & (comp["ts"] <= bloques.index.max())]

    filas = []
    for _, c in comp.iterrows():
        v = bloques.loc[c["ts"] - media_h: c["ts"] + media_h]   # ventana centrada
        if len(v) < MIN_BLOQUES_VENTANA:
            continue
        fila = {"ts": c["ts"], "turno": "dia" if c["ts"].hour < 12 else "noche", "n_bloques": len(v)}
        fila.update(v.mean().to_dict())                         # estimación media de la ventana
        fila.update({ley: c[col] for ley, col in LEYES_COMPOSITO.items()})  # ley del compósito
        filas.append(fila)
    A = pd.DataFrame(filas).sort_values("ts").reset_index(drop=True)
    A["periodo"] = periodo_de(A["ts"])
    return A


def kalman_serie(A: pd.DataFrame, ley: str, q: float) -> tuple[pd.Series, CorrectorKalman]:
    """Corrige '<ley>_est' compósito a compósito, en orden temporal: primero se
    corrige con el sesgo de los anteriores, después se revela el real."""
    k = CorrectorKalman(q_por_dia=q)
    out = np.full(len(A), np.nan)
    for i, f in enumerate(A.itertuples(index=False)):
        cruda, real = getattr(f, f"{ley}_est"), getattr(f, ley)
        out[i] = k.corregir(ley, cruda, f.turno)            # sesgo conocido ANTES de este compósito
        if not np.isnan(real):
            k.actualizar(ley, cruda, real, f.turno, ts=f.ts)  # se revela después
    return pd.Series(out, index=A.index), k


def elegir_q(A: pd.DataFrame, ley: str, grid=Q_GRID) -> tuple[float, pd.DataFrame]:
    """q_por_dia con mejor R² sobre los compósitos de train (nunca test)."""
    tr = (A["periodo"] == "train") & A[ley].notna()
    filas = []
    for q in grid:
        corr, _ = kalman_serie(A, ley, q)
        filas.append({"q": q, **metricas(A.loc[tr, ley], corr[tr])})
    T = pd.DataFrame(filas)
    return float(T.loc[T["R2"].idxmax(), "q"]), T


def metricas(y, p) -> dict:
    """n, corr, R², MAE, sesgo (est - lab) y pendiente de est vs lab."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[ok], p[ok]
    if len(y) < 3:
        return {"n": len(y)}
    return {"n": len(y), "corr": float(np.corrcoef(y, p)[0, 1]), "R2": r2_score(y, p),
            "MAE": mean_absolute_error(y, p), "sesgo": float(np.mean(p - y)),
            "pendiente": float(np.polyfit(y, p, 1)[0])}


def kalman_en_instantes(A: pd.DataFrame, ley: str, q: float, est: pd.DataFrame) -> np.ndarray:
    """Corrige estimaciones puntuales (muestras de la BD) con el sesgo que el
    Kalman tenía en ese instante: solo compósitos cuya ventana ya cerró
    (ts + 6 h <= instante). El turno es el del compósito cuya ventana contiene
    la muestra (07:30 -> 'dia', 19:30 -> 'noche')."""
    k = CorrectorKalman(q_por_dia=q)
    media_h = pd.Timedelta(hours=VENTANA_COMPOSITO_H / 2)
    comp = A.sort_values("ts").reset_index(drop=True)
    out = np.full(len(est), np.nan)
    j = 0
    for i in np.argsort(est["ts"].to_numpy()):            # muestras en orden temporal
        t = est["ts"].iloc[i]
        while j < len(comp) and comp["ts"].iloc[j] + media_h <= t:   # revela compósitos ya cerrados
            f = comp.iloc[j]
            if not np.isnan(f[ley]):
                k.actualizar(ley, f[f"{ley}_est"], f[ley], f["turno"], ts=f["ts"])
            j += 1
        centro = (t - pd.Timedelta(hours=1, minutes=30)).floor("12h") + pd.Timedelta(hours=7, minutes=30)
        turno = "dia" if centro.hour < 12 else "noche"    # compósito al que pertenece la muestra
        out[i] = k.corregir(ley, est[f"{ley}_est"].iloc[i], turno)
    return out
