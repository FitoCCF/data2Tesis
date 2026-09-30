#!/usr/bin/env python3
# ============================================================
# CELDA 26 — "La mejor estrategia" actualizada: Fe con Kalman, desde mayo 2026
# ============================================================
# Combina la celda 20 (filtro desde 2026-05-01) con la celda 22 (Fe corregido
# por Kalman en vez de media movil). Cu y Mo: modelo original, sin cambios.
#
# Uso:
#   pixi run python notebooks/26_figura_mejor_estrategia_kalman_mayo.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore")

from src.pipeline import config as CFG
from src.pipeline.calibracion_composito import cargar_intensidades, cargar_composito, alinear_con_composito, backtest
from src.pipeline.corrector_kalman import CorrectorKalman

mod11 = __import__("11_figuras_resultados")

DESDE = "2026-05-01"
Q_ELEGIDO = 0.1


def _main():
    print("=" * 62)
    print(f"PASO 1 — Backtest de pFe con Kalman (q={Q_ELEGIDO}), luego filtrar desde {DESDE}")
    print("=" * 62)
    inten = cargar_intensidades(str(CFG.DATA_PROCESSED / "intensidades_cobre.csv"))
    lab = cargar_composito()
    D = alinear_con_composito(inten, lab)
    R_kal = backtest(D, corrector=CorrectorKalman(q_por_dia=Q_ELEGIDO))
    print(f"Backtest completo: {len(R_kal)} ventanas ({R_kal.ts.min():%Y-%m-%d} a {R_kal.ts.max():%Y-%m-%d})")

    print("\n" + "=" * 62)
    print("PASO 2 — Evaluación unificada (Fe-Kalman, Cu/Mo original), filtrada desde mayo")
    print("=" * 62)
    E_todo = mod11.construir_evaluacion(str(CFG.DATA_PROCESSED / "intensidades_cobre.csv"), bt=R_kal)
    E = E_todo[E_todo.ts >= DESDE].reset_index(drop=True)
    T = mod11.tabla_metricas(E)

    print(f"n = {len(E)} ventanas  ·  {E.ts.min():%Y-%m-%d} a {E.ts.max():%Y-%m-%d}")
    cols = ["elemento", "n", "R2", "MAE", "RMSE", "corr", "pendiente", "sesgo"]
    print(T[cols].round(3).to_string(index=False))

    ruta_tabla = CFG.REPORTS_DIR / "tabla_metricas_kalman_desde_mayo.csv"
    T.to_csv(ruta_tabla, index=False)
    ruta_datos = CFG.DATA_PROCESSED / "evaluacion_kalman_desde_mayo.csv"
    E.to_csv(ruta_datos, index=False)

    f1 = CFG.REPORTS_DIR / "fig_validacion_kalman_desde_mayo"
    mod11.figura_validacion(E, T, str(f1))

    print(f"\nTabla : {ruta_tabla}")
    print(f"Datos : {ruta_datos}")
    print(f"Figura: {f1}.png / .pdf")


if __name__ == "__main__":
    _main()
