#!/usr/bin/env python3
# ============================================================
# CELDA 22 — Resultado del pipeline con corrector Kalman vs. laboratorio
# ============================================================
# Misma figura canonica que la celda 11/20 (dispersion + serie temporal, Fe +
# Cu + Mo contra el mismo compósito), pero con pFe corregido por
# CorrectorKalman(q_por_dia=0.1) en vez de la media movil N=20. Cu y Mo no
# cambian (no los toca el corrector de Fe).
#
# Uso:
#   pixi run python notebooks/22_figura_validacion_kalman.py
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

Q_ELEGIDO = 0.1  # elegido por R2 en notebooks/21_backtest_corrector_kalman.py


def _main():
    CFG.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("PASO 1 — Backtest de pFe con CorrectorKalman (q_por_dia = %.2g)" % Q_ELEGIDO)
    print("=" * 62)
    inten = cargar_intensidades(str(CFG.DATA_PROCESSED / "intensidades_cobre.csv"))
    lab = cargar_composito()
    D = alinear_con_composito(inten, lab)
    R_kal = backtest(D, corrector=CorrectorKalman(q_por_dia=Q_ELEGIDO))
    print(f"Compósitos evaluados: {len(R_kal)}  ({R_kal.ts.min():%Y-%m-%d} a {R_kal.ts.max():%Y-%m-%d})")

    print("\n" + "=" * 62)
    print("PASO 2 — Evaluación unificada (Fe con Kalman, Cu/Mo modelo original)")
    print("=" * 62)
    E = mod11.construir_evaluacion(str(CFG.DATA_PROCESSED / "intensidades_cobre.csv"), bt=R_kal)
    T = mod11.tabla_metricas(E)

    print(f"n = {len(E)} ventanas  ·  {E.ts.min():%Y-%m-%d} a {E.ts.max():%Y-%m-%d}")
    cols = ["elemento", "n", "R2", "MAE", "RMSE", "corr", "pendiente", "sesgo"]
    print(T[cols].round(3).to_string(index=False))

    # comparación explícita del renglón de Fe contra el corrector actual
    fe = T[T.elemento == "Fe"].iloc[0]
    print(f"\nFe con Kalman: corr={fe["corr"]:.3f}  (corrector actual/media móvil: corr=0.508)")

    ruta_tabla = CFG.REPORTS_DIR / "tabla_metricas_kalman.csv"
    T.to_csv(ruta_tabla, index=False)
    ruta_datos = CFG.DATA_PROCESSED / "evaluacion_kalman.csv"
    E.to_csv(ruta_datos, index=False)

    f1 = CFG.REPORTS_DIR / "fig_validacion_kalman"
    mod11.figura_validacion(E, T, str(f1))

    print(f"\nTabla : {ruta_tabla}")
    print(f"Datos : {ruta_datos}")
    print(f"Figura: {f1}.png / .pdf")


if __name__ == "__main__":
    _main()
