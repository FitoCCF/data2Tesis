#!/usr/bin/env python3
# ============================================================
# CELDA 20 — Validación vs laboratorio, solo desde 2026-05-01, mejor estrategia
# ============================================================
# "Mejor estrategia" = la que ya está adoptada en config.LEYES_RECALIBRAR:
#   pFe -> modelo RECALIBRADO contra el compósito (backtest walk-forward)
#   pCu, pMo -> modelo ORIGINAL de la etapa 5 (la recalibración los empeora)
# Reusa construir_evaluacion/tabla_metricas/figura_validacion de la celda 11
# tal cual, solo filtra el período de evaluación a partir de 2026-05-01.
#
# Uso:
#   pixi run python notebooks/20_figura_validacion_desde_mayo.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from src.pipeline import config as CFG

mod11 = __import__("11_figuras_resultados")

DESDE = "2026-05-01"


def _main():
    CFG.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    E_todo = mod11.construir_evaluacion(str(CFG.DATA_PROCESSED / "intensidades_cobre.csv"))
    print(f"Evaluación completa: {len(E_todo)} ventanas ({E_todo.ts.min():%Y-%m-%d} a {E_todo.ts.max():%Y-%m-%d})")

    E = E_todo[E_todo.ts >= DESDE].reset_index(drop=True)
    print(f"Desde {DESDE}: {len(E)} ventanas ({E.ts.min():%Y-%m-%d} a {E.ts.max():%Y-%m-%d})")

    T = mod11.tabla_metricas(E)
    print()
    cols = ["elemento", "n", "R2", "MAE", "RMSE", "corr", "pendiente", "sesgo"]
    print(T[cols].round(3).to_string(index=False))

    ruta_tabla = CFG.REPORTS_DIR / "tabla_metricas_desde_mayo.csv"
    T.to_csv(ruta_tabla, index=False)
    ruta_datos = CFG.DATA_PROCESSED / "evaluacion_desde_mayo.csv"
    E.to_csv(ruta_datos, index=False)

    f1 = CFG.REPORTS_DIR / "fig_validacion_desde_mayo"
    mod11.figura_validacion(E, T, str(f1))

    print(f"\nTabla : {ruta_tabla}")
    print(f"Datos : {ruta_datos}")
    print(f"Figura: {f1}.png / .pdf")


if __name__ == "__main__":
    _main()
