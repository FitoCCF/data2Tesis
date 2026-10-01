# ============================================================
# src/pipeline/modelos.py — Etapa 5: modelos por grupo de mineral + tabla de ruteo
# ============================================================
# Entrena un modelo de regresión por ley (global) y uno por (ley, grupo) donde
# hay datos suficientes, y decide para cada celda si se usa el local o el global.
#
# Validación (reemplaza a KFold(shuffle=True), que filtraba en serie de tiempo):
#   - walk-forward con purga y embargo por marca de tiempo (split_purgado),
#     folds calculados UNA vez sobre las muestras de train y compartidos por
#     todas las leyes, el global y los locales;
#   - global y local se comparan sobre LOS MISMOS puntos fuera de fold;
#   - control aleatorio: el mismo procedimiento con grupos al azar del mismo
#     tamaño. El local solo se rutea si su mejora supera el percentil
#     PCTL_CONTROL del control (bitácora §8-9).
#
# Entrada: courier_clusterizado.csv (etapa 4). El grupo ya viene asignado por
# el GMM congelado y FEATS_REGRESION son transformaciones fijas, así que aquí
# no hay transformadores que reajustar dentro del fold.
#
#   python -m src.pipeline.modelos
# ============================================================

import numpy as np                                        # arrays numéricos
import pandas as pd                                        # DataFrames
from sklearn.linear_model import Ridge                     # regresión regularizada
from sklearn.svm import SVR                                # regresión de vectores soporte
from sklearn.cross_decomposition import PLSRegression      # mínimos cuadrados parciales
from sklearn.preprocessing import StandardScaler           # estandarización interna
from sklearn.pipeline import Pipeline                      # encadena escala + modelo
from sklearn.metrics import r2_score, mean_absolute_error  # métricas

from .config import (FEATS_REGRESION, TARGETS, MODELO_POR_TARGET, RANDOM_STATE,
                     N_SPLITS, PURGA, EMBARGO, N_MIN_LOCAL, MIN_TRAIN_FOLD_LOCAL,
                     N_CONTROL, PCTL_CONTROL)
from .split_temporal import split_purgado                  # walk-forward con purga y embargo


def _construir_modelo(nombre: str) -> Pipeline:
    """Devuelve un pipeline (estandarizar -> modelo) según la familia elegida."""
    if nombre == "SVR":                                   # cobre
        modelo = SVR(kernel="rbf", C=10, epsilon=0.3)     # SVR con kernel radial
    elif nombre == "Ridge":                               # hierro / molibdeno
        modelo = Ridge(alpha=1.0)                         # Ridge con regularización moderada
    else:                                                 # zinc
        modelo = PLSRegression(n_components=2)            # PLS con 2 componentes
    return Pipeline([("scaler", StandardScaler()), ("modelo", modelo)])  # escala + modelo


def _predecir(modelo: Pipeline, X: np.ndarray) -> np.ndarray:
    """predict() aplanado: PLSRegression devuelve (n, 1)."""
    return np.asarray(modelo.predict(X)).ravel()


def conjunto_supervisado(df: pd.DataFrame, periodo: str = "train") -> pd.DataFrame:
    """Muestras con ley válida del periodo pedido, ordenadas por ts y con índice
    posicional (requisito de split_purgado)."""
    s = df[df["valida_ley"] & (df["periodo"] == periodo)]  # solo muestras de laboratorio usables
    return s.sort_values("ts").reset_index(drop=True)     # orden temporal + índice 0..n-1


def _oof(X: np.ndarray, y: np.ndarray, folds: list, familia: str,
         mascara: np.ndarray, min_train: int = 1) -> np.ndarray:
    """Predicciones fuera de fold restringidas a `mascara` (filas con ley y, si
    es local, del grupo). NaN donde no hubo predicción: el primer bloque nunca
    es prueba, y un fold con menos de `min_train` filas de train no predice."""
    pred = np.full(len(y), np.nan)                        # sin predicción por defecto
    for tr, te in folds:                                  # folds compartidos
        tr, te = tr[mascara[tr]], te[mascara[te]]         # filtro por ley / grupo DESPUÉS del split
        if len(te) == 0 or len(tr) < min_train:           # nada que predecir o train insuficiente
            continue
        m = _construir_modelo(familia).fit(X[tr], y[tr])  # ajuste solo con el pasado
        pred[te] = _predecir(m, X[te])                    # predicción del bloque de prueba
    return pred


def _mejora(y, pred_g, pred_l, puntos) -> tuple:
    """R² global, R² local y su diferencia sobre los MISMOS puntos (ambos predichos)."""
    p = puntos & ~np.isnan(pred_g) & ~np.isnan(pred_l)    # puntos comparables
    if p.sum() < 3:                                       # R² sin sentido con tan pocos puntos
        return np.nan, np.nan, np.nan, int(p.sum())
    r2g, r2l = r2_score(y[p], pred_g[p]), r2_score(y[p], pred_l[p])
    return r2g, r2l, r2l - r2g, int(p.sum())


def entrenar_modelos_locales(df: pd.DataFrame, n_control: int = N_CONTROL) -> dict:
    """Valida, rutea y ajusta los modelos finales.

    Parámetros
    ----------
    df : salida de la etapa 4 (courier_clusterizado.csv), con ts, grupo,
        valida_ley, periodo, FEATS_REGRESION y TARGETS.
    n_control : particiones aleatorias del control (0 lo desactiva y entonces
        ningún local se rutea: la regla exige superar el control).

    Retorna
    -------
    bundle : dict con modelos, tabla de ruteo, tabla de validación y folds.
    """
    s = conjunto_supervisado(df)                          # train con ley válida, ordenado
    folds = split_purgado(s["ts"], n_splits=N_SPLITS, purga=PURGA, embargo=EMBARGO)  # una sola vez
    X = s[FEATS_REGRESION].to_numpy()                     # features (transformaciones fijas)
    grupo = s["grupo"].to_numpy()                         # grupo de mineral (GMM congelado)
    grupos = sorted(s["grupo"].unique())                  # nombres de los grupos
    rng = np.random.default_rng(RANDOM_STATE)             # azar reproducible del control

    filas, tabla_ruteo = [], {}                           # validación por celda, decisiones
    oof = s[["ts", "grupo"] + TARGETS].copy()             # predicciones fuera de fold (trazabilidad, etapa 9)
    modelos_globales, modelos_locales = {}, {}            # modelos finales

    for tgt in TARGETS:                                   # por cada ley
        familia = MODELO_POR_TARGET[tgt]                  # familia de modelo para esa ley
        y = s[tgt].to_numpy(dtype=float)                  # ley (NaN donde no se analizó)
        con_ley = ~np.isnan(y)                            # dropna POR ELEMENTO, sin tocar los folds

        pred_g = _oof(X, y, folds, familia, con_ley)      # global fuera de fold
        oof[f"{tgt}_oof"] = pred_g
        modelos_globales[tgt] = _construir_modelo(familia).fit(X[con_ley], y[con_ley])  # ajuste final

        # Control: grupos al azar con los mismos tamaños, mismo procedimiento.
        # Se permuta la etiqueta entre las filas con ley, así cada pseudo-grupo
        # tiene exactamente el N del grupo real.
        elegibles = [g for g in grupos if (con_ley & (grupo == g)).sum() >= N_MIN_LOCAL]
        ctrl = {g: [] for g in elegibles}                 # mejora del control por grupo
        for _ in range(n_control if elegibles else 0):
            perm = grupo.copy()
            perm[con_ley] = rng.permutation(grupo[con_ley])  # partición aleatoria, tamaños intactos
            for g in elegibles:
                m = con_ley & (perm == g)
                pl = _oof(X, y, folds, familia, m, MIN_TRAIN_FOLD_LOCAL)
                ctrl[g].append(_mejora(y, pred_g, pl, m)[2])

        for g in grupos:                                  # por cada grupo
            m = con_ley & (grupo == g)                    # filas del grupo con ley
            fila = {"ley": tgt, "grupo": g, "familia": familia, "N": int(m.sum())}
            if fila["N"] < N_MIN_LOCAL:                   # regla a priori: sin modelo local
                p = m & ~np.isnan(pred_g)
                fila.update(N_eval=int(p.sum()),
                            r2_global=r2_score(y[p], pred_g[p]) if p.sum() >= 3 else np.nan,
                            mae_global=mean_absolute_error(y[p], pred_g[p]) if p.sum() else np.nan)
                tabla_ruteo[(tgt, g)] = "global (N insuficiente)"
                filas.append(fila)
                continue

            pred_l = _oof(X, y, folds, familia, m, MIN_TRAIN_FOLD_LOCAL)  # local fuera de fold
            r2g, r2l, mejora, n_eval = _mejora(y, pred_g, pred_l, m)
            p = m & ~np.isnan(pred_g) & ~np.isnan(pred_l)
            umbral = np.nanpercentile(ctrl[g], PCTL_CONTROL) if ctrl[g] else np.nan
            pctl = (100 * np.mean(np.asarray(ctrl[g]) < mejora)) if ctrl[g] else np.nan
            usa_local = bool(mejora > 0 and mejora > umbral)  # NaN en cualquiera -> False
            fila.update(N_eval=n_eval, r2_global=r2g, r2_local=r2l, mejora=mejora,
                        mae_global=mean_absolute_error(y[p], pred_g[p]),
                        mae_local=mean_absolute_error(y[p], pred_l[p]),
                        umbral_control=umbral, pctl_en_control=pctl)
            tabla_ruteo[(tgt, g)] = "local" if usa_local else "global"
            modelos_locales[(tgt, g)] = _construir_modelo(familia).fit(X[m], y[m])  # ajuste local final
            filas.append(fila)

        # R² global de la ley sobre todos sus puntos fuera de fold (referencia)
        p = con_ley & ~np.isnan(pred_g)
        filas.append({"ley": tgt, "grupo": "(todos)", "familia": familia, "N": int(con_ley.sum()),
                      "N_eval": int(p.sum()), "r2_global": r2_score(y[p], pred_g[p]),
                      "mae_global": mean_absolute_error(y[p], pred_g[p])})

    for k, v in tabla_ruteo.items():                      # solo se guardan locales que se usan
        if v != "local":
            modelos_locales.pop(k, None)

    return {                                              # bundle a serializar
        "modelos_locales": modelos_locales,               # (ley, grupo) -> pipeline, solo los ruteados
        "modelos_globales": modelos_globales,             # ley -> pipeline
        "features": FEATS_REGRESION,                      # orden de features esperado
        "targets": TARGETS,                               # leyes estimadas
        "cluster_col": "grupo",                           # columna de grupo usada
        "tabla_ruteo": tabla_ruteo,                       # decisiones local/global
        "validacion": pd.DataFrame(filas),                # trazabilidad: N, R², MAE, control por celda
        "folds": folds,                                   # índices sobre conjunto_supervisado(df)
        "oof": oof,                                       # ts, grupo, ley real y '<ley>_oof' (modelo global)
        "parametros": {"n_splits": N_SPLITS, "purga": PURGA, "embargo": EMBARGO,
                       "n_min_local": N_MIN_LOCAL, "min_train_fold_local": MIN_TRAIN_FOLD_LOCAL,
                       "n_control": n_control, "pctl_control": PCTL_CONTROL,
                       "random_state": RANDOM_STATE},
    }


def predecir(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la tabla de ruteo: devuelve una columna '<ley>_est' por ley."""
    X = df[bundle["features"]].to_numpy()                 # features en el orden del bundle
    out = pd.DataFrame(index=df.index)
    for tgt in bundle["targets"]:
        est = _predecir(bundle["modelos_globales"][tgt], X)  # global por defecto
        for (t, g), modelo in bundle["modelos_locales"].items():
            if t != tgt:
                continue
            m = (df[bundle["cluster_col"]] == g).to_numpy()
            if m.any():
                est[m] = _predecir(modelo, X[m])          # local donde está ruteado
        out[f"{tgt}_est"] = est
    return out


def evaluar_test(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Chequeo sobre las muestras de laboratorio del periodo test (pocas: es un
    control de cordura, la evaluación externa es contra el compósito)."""
    t = conjunto_supervisado(df, "test")
    est = predecir(bundle, t)
    filas = []
    for tgt in bundle["targets"]:
        p = t[tgt].notna().to_numpy()
        y, yhat = t[tgt].to_numpy()[p], est[f"{tgt}_est"].to_numpy()[p]
        filas.append({"ley": tgt, "N": int(p.sum()),
                      "r2": r2_score(y, yhat) if p.sum() >= 3 else np.nan,
                      "mae": mean_absolute_error(y, yhat) if p.sum() else np.nan,
                      "sesgo": float(np.mean(yhat - y)) if p.sum() else np.nan})
    return pd.DataFrame(filas)


# ============================================================
# CLI — ejecución independiente de la etapa
# ============================================================
def _main():
    import argparse
    import joblib

    from .config import CLUSTERIZADO, MODELS_DIR, ART_REGRESION

    ap = argparse.ArgumentParser(description="Etapa 5: modelos por grupo + tabla de ruteo")
    ap.add_argument("--input", default=str(CLUSTERIZADO))
    ap.add_argument("--output-bundle", default=str(MODELS_DIR / ART_REGRESION))
    ap.add_argument("--n-control", type=int, default=N_CONTROL)
    args = ap.parse_args()

    df = pd.read_csv(args.input, parse_dates=["ts"])
    bundle = entrenar_modelos_locales(df, args.n_control)
    print(bundle["validacion"].round(3).to_string(index=False))
    joblib.dump(bundle, args.output_bundle)
    print(f"\nBundle guardado en: {args.output_bundle}")


if __name__ == "__main__":
    _main()
