#!/usr/bin/env python3
# ============================================================
# src/pipeline/limpieza.py — Etapa 1: limpieza y detección de anomalías
# ============================================================
# Convierte la celda 01_limpieza en funciones reutilizables. Separa el
# "entrenar" (ajustar el IsolationForest) del "aplicar" (usar uno ya
# entrenado), para que entrenamiento e inferencia usen la MISMA lógica.
#
# Ejecutable de forma independiente:
#   python -m src.pipeline.limpieza --input data/processed/intensidades_cobre.csv
#   python -m src.pipeline.limpieza --input nuevas.csv --artifact-in models/anomaly_detector.joblib
# ============================================================

import pandas as pd                                       # manejo de DataFrames
from sklearn.ensemble import IsolationForest              # detector de anomalías multivariadas

from .config import (CANALES, RANDOM_STATE,               # constantes centralizadas
                     CONTAMINACION_ANOMALIA)
from .features import features_cierre                     # fracciones de cierre (invariantes a la deriva)


def _filtrar_invalidos(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica los tres filtros deterministas de calidad (no requieren modelo).

    1. Elimina códigos de error del analizador (-9999).
    2. Elimina lecturas en cero simultáneo (planta/analizador detenido).
    3. Elimina 'sensor congelado' (vector idéntico repetido >= 3 veces).
    """
    df = df[~(df[CANALES] == -9999).any(axis=1)]          # quita filas con algún -9999
    df = df[~(df[CANALES] == 0).all(axis=1)].copy()       # quita filas con los 5 canales en 0

    repeticiones = df.groupby(CANALES)[CANALES[0]].transform("size")  # nº de repeticiones de cada vector
    df = df[repeticiones < 3].copy()                      # conserva vectores que aparecen menos de 3 veces
    return df                                             # devuelve el DataFrame filtrado


def _matriz_anomalia(df: pd.DataFrame, robusto: bool) -> pd.DataFrame:
    """Arma la matriz sobre la que trabaja el IsolationForest.

    robusto=True  -> fracciones de cierre (CAMBIO 4, recomendado).
    robusto=False -> intensidades absolutas (comportamiento histórico, se
                     conserva solo para poder reproducir resultados antiguos).
    """
    if robusto:                                           # --- modo robusto a la deriva ---
        return features_cierre(df).dropna()               # fracciones m/Σm, sin NaN
    return df[CANALES].dropna()                           # --- modo histórico: intensidades absolutas ---


def limpiar(df: pd.DataFrame, detector: IsolationForest | None = None,
            robusto: bool = True):
    """Limpia intensidades crudas y detecta anomalías multivariadas.

    CAMBIO 4 — detector robusto a la deriva del instrumento
    -------------------------------------------------------
    Antes el IsolationForest se ajustaba sobre las intensidades ABSOLUTAS. Como
    el analizador pierde ~22%/año de cuentas, las lecturas recientes caen fuera
    de la nube de entrenamiento y se marcan como anómalas aunque sean válidas.
    Tasa de rechazo medida con el detector histórico (entrenado hasta 2025-09):

        2025Q3  0.1%   2025Q4  0.2%   2026Q1  0.6%   2026Q2 15.8%   2026Q3 16.0%

    Con el mismo detector sobre fracciones de cierre (robusto=True):

        2025Q3  1.7%   2025Q4  0.9%   2026Q1  1.0%   2026Q2  3.3%   2026Q3  3.5%

    El detector deja de confundir "el equipo envejeció" con "esta lectura está mala".

    Parámetros
    ----------
    df : DataFrame con los canales de intensidad crudos.
    detector : IsolationForest ya entrenado. Si es None, se entrena uno nuevo
        (modo ENTRENAMIENTO); si se pasa, se aplica el existente (modo INFERENCIA).
    robusto : True (default) ajusta/aplica sobre fracciones de cierre; False
        reproduce el comportamiento histórico sobre intensidades absolutas.

    Retorna
    -------
    df_limpio : DataFrame sin filas inválidas ni anómalas.
    detector : el IsolationForest (nuevo o el mismo que se recibió).
    """
    df = _filtrar_invalidos(df)                           # aplica los filtros deterministas

    x = _matriz_anomalia(df, robusto)                     # matriz de trabajo (cierre o absolutas)

    if detector is None:                                  # --- modo ENTRENAMIENTO ---
        detector = IsolationForest(                       # instancia el detector
            n_estimators=200,                             # 200 árboles (más estable que 100, costo despreciable)
            contamination=CONTAMINACION_ANOMALIA,         # proporción esperada de anomalías reales (2%)
            random_state=RANDOM_STATE,                    # reproducibilidad
        )
        detector.fit(x)                                   # ajusta sobre la matriz elegida
        detector._robusto = robusto                       # marca el modo en el artefacto (lo lee la inferencia)

    etiquetas = detector.predict(x)                       # -1 = anomalía, 1 = normal
    idx_normales = x.index[etiquetas == 1]                # índices marcados como normales (== 1, no != -1)

    df_limpio = df.loc[idx_normales].copy()               # conserva solo las filas normales
    return df_limpio, detector                            # devuelve datos limpios + el detector


# ============================================================
# CLI — ejecución independiente de la etapa
# ============================================================
def _main():
    import argparse
    import joblib

    from .config import DATA_PROCESSED, MODELS_DIR, ART_ANOMALIAS

    ap = argparse.ArgumentParser(description="Etapa 1: limpieza + detección de anomalías")
    ap.add_argument("--input", default=str(DATA_PROCESSED / "intensidades_cobre_train.csv"),
                    help="CSV de intensidades crudas de entrada. Default: el split TRAIN de "
                         "merge_cobre_data.py (calibración) -- NO el _cobre.csv completo ni el _test.csv "
                         "(subconjunto de validación del mismo período histórico, no se re-entrena con él). "
                         "Los datos de PRODUCCIÓN (p.ej. septiembre en adelante) son un extracto aparte, "
                         "nunca pasan por esta etapa: se scorean con la etapa 6 (src.pipeline.inferencia)")
    ap.add_argument("--output", default=str(DATA_PROCESSED / "intensidad_cobre_24_clean.csv"),
                    help="Ruta del CSV limpio de salida")
    ap.add_argument("--artifact-in", default=None,
                    help="IsolationForest .joblib ya entrenado (modo INFERENCIA; si se omite, se entrena uno nuevo)")
    ap.add_argument("--artifact-out", default=str(MODELS_DIR / ART_ANOMALIAS),
                    help="Ruta donde guardar el detector entrenado (ignorado en modo INFERENCIA)")
    ap.add_argument("--absolutas", action="store_true",
                    help="CAMBIO 4: por defecto el detector usa fracciones de cierre (robusto a la "
                         "deriva del instrumento). Esta bandera vuelve al comportamiento histórico "
                         "sobre intensidades absolutas -- solo para reproducir resultados antiguos.")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    detector = joblib.load(args.artifact_in) if args.artifact_in else None

    robusto = not args.absolutas                          # robusto salvo que se pida explícitamente lo contrario
    if detector is not None:                              # en modo INFERENCIA manda el modo con que se entrenó
        robusto = getattr(detector, "_robusto", robusto)  # (evita aplicar un detector de cierre a absolutas)

    df_limpio, detector = limpiar(df, detector, robusto=robusto)
    print(f"Modo del detector: {'cierre (robusto a deriva)' if robusto else 'intensidades absolutas'}")

    df_limpio.to_csv(args.output, index=False)
    print(f"Filas finales tras limpieza: {len(df_limpio)}")
    print(f"CSV limpio guardado en: {args.output}")

    if args.artifact_in is None:
        joblib.dump(detector, args.artifact_out)
        print(f"Detector de anomalías guardado en: {args.artifact_out}")


if __name__ == "__main__":
    _main()
