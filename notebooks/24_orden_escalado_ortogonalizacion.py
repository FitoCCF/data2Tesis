#!/usr/bin/env python3
# ============================================================
# CELDA 24 — ¿Escalar antes o después de ortogonalizar?
# ============================================================
# Solo importa para CLUSTERING (§2.5: para la regresión es un no-op algebraico
# probado, n6sc va como feature aparte y el span es idéntico). Aquí se compara
# a nivel de features de clustering:
#
#   A (actual)   ortho(metal_crudo, n6sc_crudo) -> escalar(residuo)
#   B (invertido) escalar(metal_crudo) -> ortho(metal_escalado, n6sc_crudo) -> escalar
#
# B necesita un StandardScaler final porque el residuo de regresionar un
# metal YA escalado (var=1) contra n6sc crudo no mantiene var=1 exacta -- para
# comparar clustering en igualdad de condiciones ambas variantes deben llegar
# a la misma escala final.
#
# Uso:
#   pixi run python notebooks/24_orden_escalado_ortogonalizacion.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer, StandardScaler

from src.pipeline import config as CFG
from src.pipeline.ortogonalizacion import ortogonalizar
from src.pipeline.escalado import escalar
from src.pipeline.clustering import diagnostico_k


def _main():
    print("=" * 62)
    print("PASO 1 — Cargar intensidades limpias (etapa 1, sin cambios)")
    print("=" * 62)
    clean = pd.read_csv(CFG.DATA_PROCESSED / "intensidad_cobre_24_clean.csv")
    print(f"Filas: {len(clean)}")

    print("\n" + "=" * 62)
    print("PASO 2 — Variante A (actual): ortogonalizar -> escalar")
    print("=" * 62)
    df_ortho_a, _ = ortogonalizar(clean)
    X_a, _, _ = escalar(df_ortho_a)
    print(f"X_a: {X_a.shape}")

    print("\n" + "=" * 62)
    print("PASO 3 — Variante B (invertido): escalar metales -> ortogonalizar -> escalar residuo")
    print("=" * 62)
    p_pre = PowerTransformer(method="yeo-johnson", standardize=False).fit(clean[CFG.METALES].values)
    Xp_pre = p_pre.transform(clean[CFG.METALES].values)
    s_pre = StandardScaler().fit(Xp_pre)
    metales_escalados = s_pre.transform(Xp_pre)

    df_pre = clean.copy()
    df_pre[CFG.METALES] = metales_escalados          # metales YA escalados, n6sc queda crudo (mismo regresor que A)
    df_ortho_b, _ = ortogonalizar(df_pre)             # residuo = metal_escalado - beta*n6sc_crudo

    s_post = StandardScaler().fit(df_ortho_b[CFG.FEATS_CLUSTER].values)  # normaliza el residuo (no vuelve a hacer YJ)
    X_b = s_post.transform(df_ortho_b[CFG.FEATS_CLUSTER].values)
    print(f"X_b: {X_b.shape}")

    print("\n" + "=" * 62)
    print("PASO 4 — Diagnóstico k: A (actual) vs B (invertido)")
    print("=" * 62)
    print("\n--- A: ortogonalizar -> escalar (actual) ---")
    diagnostico_k(X_a)
    print("\n--- B: escalar -> ortogonalizar (invertido) ---")
    diagnostico_k(X_b)


if __name__ == "__main__":
    _main()
