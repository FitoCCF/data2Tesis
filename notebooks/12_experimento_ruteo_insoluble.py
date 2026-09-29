#!/usr/bin/env python3
# ============================================================
# CELDA 12 — EXPERIMENTO: ¿ruteo por insoluble mejora pFe con validación honesta?
# ============================================================
# Responde el hilo abierto 5.4 de docs/bitacora_analisis.md: la sección 2.6
# midió +0.144 en R2(pFe) ruteando por tercil de insoluble REAL simultáneo,
# con CV 5-fold sobre las 314 muestras puntuales. Aquí se remide con:
#   - insoluble del compósito ANTERIOR (ins_lag), no el simultáneo -- es lo
#     único que existe en producción.
#   - backtest walk-forward (reentreno cada 15 d, ventana móvil de 270 d),
#     no CV 5-fold -- el mismo protocolo que ya valida la recalibración de Fe.
#   - control negativo: mismo procedimiento con ins_lag barajado dentro de
#     cada ventana, para descartar que la ganancia sea solo de tener más
#     parámetros (3 modelos en vez de 1).
#
# Uso:
#   pixi run python notebooks/12_experimento_ruteo_insoluble.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import config as CFG
from src.pipeline.calibracion_composito import cargar_intensidades, alinear_con_composito
from src.pipeline.ruteo_insoluble import cargar_composito_crudo, agregar_ins_lag, backtest_ruteo, resumen


def _main():
    print("=" * 62)
    print("PASO 1 — Cargar y alinear (con insoluble)")
    print("=" * 62)
    inten = cargar_intensidades(CFG.DATA_PROCESSED / "intensidades_cobre.csv")
    lab = cargar_composito_crudo()   # data/raw/LABCOMPOSITO.csv, ya con columna 'ins'
    print(f"Lecturas de intensidad válidas : {len(inten):>7}")
    print(f"Ensayos del compósito (crudo)  : {len(lab):>7}  ({lab.ts.min()} -> {lab.ts.max()})")

    D = alinear_con_composito(inten, lab)
    D = agregar_ins_lag(D)
    n_sin_lag = D["ins_lag"].isna().sum()
    print(f"Ventanas alineadas             : {len(D):>7}  ({D.ts.min()} -> {D.ts.max()})")
    print(f"  sin ins_lag (primera fila)   : {n_sin_lag}")

    print("\n" + "=" * 62)
    print("PASO 2 — Backtest walk-forward: ruteo REAL por tercil de ins_lag")
    print("=" * 62)
    R = backtest_ruteo(D, ley="pFe", control_aleatorio=False)
    tabla = resumen(R, "pFe")
    print(f"Compósitos evaluados: {len(R)}")
    print(tabla.round(4).to_string(index=False))
    ganancia = tabla.loc[tabla.variante == "ruteo", "R2"].iloc[0] - tabla.loc[tabla.variante == "global", "R2"].iloc[0]
    print(f"\nGanancia R2 (ruteo - global) = {ganancia:+.4f}")

    dist_bin = R["bin"].value_counts()
    print(f"\nDistribución de bins usados (incluye 'global_fallback' si la ventana no alcanzó para tercios):")
    print(dist_bin.to_string())

    print("\n" + "=" * 62)
    print("PASO 3 — CONTROL NEGATIVO: mismo procedimiento con ins_lag barajado")
    print("=" * 62)
    R_ctrl = backtest_ruteo(D, ley="pFe", control_aleatorio=True)
    tabla_ctrl = resumen(R_ctrl, "pFe")
    print(f"Compósitos evaluados: {len(R_ctrl)}")
    print(tabla_ctrl.round(4).to_string(index=False))
    ganancia_ctrl = (tabla_ctrl.loc[tabla_ctrl.variante == "ruteo", "R2"].iloc[0]
                     - tabla_ctrl.loc[tabla_ctrl.variante == "global", "R2"].iloc[0])
    print(f"\nGanancia R2 control (ruteo - global) = {ganancia_ctrl:+.4f}")

    print("\n" + "=" * 62)
    print("RESUMEN FINAL")
    print("=" * 62)
    print(f"{'':20s}{'R2 global':>12s}{'R2 ruteo':>12s}{'ganancia':>12s}")
    g = tabla.loc[tabla.variante == 'global', 'R2'].iloc[0]
    r = tabla.loc[tabla.variante == 'ruteo', 'R2'].iloc[0]
    gc = tabla_ctrl.loc[tabla_ctrl.variante == 'global', 'R2'].iloc[0]
    rc = tabla_ctrl.loc[tabla_ctrl.variante == 'ruteo', 'R2'].iloc[0]
    print(f"{'ruteo real (ins_lag)':20s}{g:12.4f}{r:12.4f}{r-g:+12.4f}")
    print(f"{'control (barajado)':20s}{gc:12.4f}{rc:12.4f}{rc-gc:+12.4f}")
    print(f"\nReferencia sección 2.6 (CV 5-fold, insoluble simultáneo, n=314):")
    print(f"{'':20s}{'global':>12s}{'local pIns':>12s}{'ganancia':>12s}")
    print(f"{'pFe (histórico)':20s}{0.352:12.4f}{0.496:12.4f}{0.144:+12.4f}")


if __name__ == "__main__":
    _main()
