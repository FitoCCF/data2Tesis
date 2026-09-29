#!/usr/bin/env python3
# ============================================================
# CELDA 13 — EXPERIMENTO: ¿la cabeza (rebose hidrociclones L2) mejora el
# clustering de régimen de mineral?
# ============================================================
# Pregunta: el clustering actual (etapa 4) corre solo sobre intensidades del
# Concentrado Colectivo (features n1fe_ortho..n4mo_ortho). La bitácora (2.6)
# ya midió que esos clusters "parpadean" y no separan química real -- la
# hipótesis a probar es si agregar la CABEZA (composición ANTES del circuito,
# rebose de hidrociclones L2) ayuda, porque es una señal causalmente más
# cercana al régimen de mineral que el concentrado ya procesado.
#
# Se usa L2 (no L1): L1 casi no tiene lectura reciente (24 filas/año),
# mientras L2 tiene actividad reciente razonable (85 filas/año). La
# correlación cruda de L2 contra el laboratorio fue mala (Cu~0.04) -- por eso
# aquí se usan FRACCIONES DE CIERRE de la cabeza, no intensidad absoluta,
# igual que el resto del pipeline: es la representación que ya neutraliza
# cambios de calibración y deriva del instrumento.
#
# Limitación declarada: L2 tiene cadencia mediana ~72h. Alineado por
# merge_asof hacia atrás (última cabeza conocida) sobre el stream de 15 min,
# cada valor de cabeza se repite ~280 veces seguidas. Es la misma limitación
# de frecuencia que ya se documentó para el insoluble (hilo 5.4).
#
# Uso:
#   pixi run python notebooks/13_experimento_cluster_cabeza.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import psycopg2
from sklearn.preprocessing import StandardScaler

from src.pipeline import config as CFG
from src.pipeline.clustering import diagnostico_k

SAMPLE_ID_CABEZA_L2 = 18  # "Rebose Hidrociclones L2" en works4cdp_sample


def cargar_cabeza_l2() -> pd.DataFrame:
    """Intensidades crudas de la cabeza (L2) directo de la BD, + sus
    fracciones de cierre (mismo criterio que features.features_cierre)."""
    conn = psycopg2.connect(host="localhost", port=5432, user="myuser",
                            password="mypassword", dbname="mydb")
    q = f"""
        SELECT timestamp AS ts, n1fe, n2cu, n3zn, n4mo
        FROM works4cdp_assay
        WHERE sample_id = {SAMPLE_ID_CABEZA_L2}
          AND n1fe IS NOT NULL AND n2cu IS NOT NULL
          AND n3zn IS NOT NULL AND n4mo IS NOT NULL
        ORDER BY timestamp
    """
    cab = pd.read_sql(q, conn, parse_dates=["ts"])
    conn.close()
    cab["ts"] = cab["ts"].dt.tz_localize(None)   # naive, comparable con el stream del courier

    suma = cab[["n1fe", "n2cu", "n3zn", "n4mo"]].sum(axis=1).replace(0, np.nan)
    for m in ["n1fe", "n2cu", "n3zn", "n4mo"]:
        cab[f"cab_{m}_f"] = cab[m] / suma
    return cab.dropna(subset=[f"cab_{m}_f" for m in ["n1fe", "n2cu", "n3zn", "n4mo"]])


def _main():
    print("=" * 62)
    print("PASO 1 — Cargar clustering actual + cabeza L2")
    print("=" * 62)
    df = pd.read_csv(CFG.DATA_PROCESSED / "intensidad_cobre_24_scaled.csv")
    df["ts"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    df = df.sort_values("ts").reset_index(drop=True)
    print(f"Filas del stream clusterizable : {len(df)}  ({df.ts.min()} -> {df.ts.max()})")

    cab = cargar_cabeza_l2()
    print(f"Lecturas de cabeza L2          : {len(cab)}  ({cab.ts.min()} -> {cab.ts.max()})")

    cols_cab = ["cab_n1fe_f", "cab_n2cu_f", "cab_n3zn_f", "cab_n4mo_f"]
    df = pd.merge_asof(df, cab[["ts"] + cols_cab], on="ts", direction="backward")
    n_sin_cabeza = df[cols_cab[0]].isna().sum()
    print(f"Filas sin cabeza previa conocida (se descartan): {n_sin_cabeza}")
    df = df.dropna(subset=cols_cab).reset_index(drop=True)

    # cuántas lecturas de cabeza L2 distintas quedan efectivamente representadas
    n_bloques_cabeza = df[cols_cab[0]].ne(df[cols_cab[0]].shift()).sum()
    print(f"Bloques distintos de cabeza en el stream final : {n_bloques_cabeza} "
          f"(cada uno se repite ~{len(df) / max(n_bloques_cabeza, 1):.0f} filas seguidas)")

    df[cols_cab] = StandardScaler().fit_transform(df[cols_cab])   # misma escala que el resto

    print("\n" + "=" * 62)
    print("PASO 2 — BASELINE: solo Concentrado Colectivo (features actuales)")
    print("=" * 62)
    X_base = df[CFG.FEATS_CLUSTER].values
    diagnostico_k(X_base)

    print("\n" + "=" * 62)
    print("PASO 3 — AUMENTADO: Concentrado Colectivo + fracciones de cierre de cabeza L2")
    print("=" * 62)
    X_aum = df[CFG.FEATS_CLUSTER + cols_cab].values
    diagnostico_k(X_aum)

    print("\n" + "=" * 62)
    print("PASO 4 — Control negativo: cabeza BARAJADA (rompe la relación real)")
    print("=" * 62)
    df_ctrl = df.copy()
    rng = np.random.default_rng(CFG.RANDOM_STATE)
    orden = rng.permutation(len(df_ctrl))
    df_ctrl[cols_cab] = df_ctrl[cols_cab].values[orden]
    X_ctrl = df_ctrl[CFG.FEATS_CLUSTER + cols_cab].values
    diagnostico_k(X_ctrl)


if __name__ == "__main__":
    _main()
