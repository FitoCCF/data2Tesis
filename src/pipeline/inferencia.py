# ============================================================
# src/pipeline/inferencia.py — Etapa 6: estimación de leyes por lectura
# ============================================================
# Aplica el bundle de la etapa 5 a cada lectura del courier y agrega una
# columna '<ley>_est' por ley. Con la tabla de ruteo actual (ningún local supera
# al control aleatorio) todo sale del modelo global; el grupo de mineral viaja
# como columna informativa (régimen), no decide la estimación.
#
# Modo lote sobre la salida de la etapa 4. Las banderas (valida, anomalia,
# congelada, sostenida) se conservan: quien evalúa decide qué lecturas usa.
# La versión en línea (lectura a lectura, para producción) se rehace sobre
# esta cuando haga falta; la anterior (EstimadorHibrido) dependía de la
# ortogonalización contra n6sc, que ya no existe.
#
#   python -m src.pipeline.inferencia
# ============================================================

import joblib                                             # carga del bundle
import pandas as pd                                        # DataFrames

from .config import CLUSTERIZADO, ESTIMADO, MODELS_DIR, ART_REGRESION
from .modelos import predecir                              # aplica la tabla de ruteo


def estimar(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Devuelve df con una columna '<ley>_est' por ley del bundle."""
    return pd.concat([df, predecir(bundle, df)], axis=1)  # estimación para TODAS las lecturas


def _main():
    import argparse

    ap = argparse.ArgumentParser(description="Etapa 6: estimación de leyes por lectura")
    ap.add_argument("--input", default=str(CLUSTERIZADO))
    ap.add_argument("--bundle", default=str(MODELS_DIR / ART_REGRESION))
    ap.add_argument("--output", default=str(ESTIMADO))
    args = ap.parse_args()

    df = pd.read_csv(args.input, parse_dates=["ts"])
    out = estimar(df, joblib.load(args.bundle))
    out.to_csv(args.output, index=False)
    print(f"{len(out)} lecturas estimadas -> {args.output}")


if __name__ == "__main__":
    _main()
