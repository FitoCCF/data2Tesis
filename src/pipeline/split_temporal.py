# ============================================================
# src/pipeline/split_temporal.py — Divisor walk-forward con purga y embargo
# ============================================================
# Reemplaza al KFold(shuffle=True) que se usaba sobre una serie de tiempo.
# Ventana expansiva, purga y embargo POR MARCA DE TIEMPO: el parametro `gap` de
# TimeSeriesSplit cuenta MUESTRAS y el muestreo del dataset canonico es
# irregular (mediana ~3 d, minimo ~9.5 h, maximo 28 d), de modo que un gap fijo
# en numero de muestras no equivale a una separacion fija en horas.
# Ver ../../docs/recalculo-validacion-temporal.md, secciones 2.2 y 2.3.
# ============================================================

import numpy as np                                        # arrays de indices
import pandas as pd                                        # timestamps y Timedelta


def _eje_temporal(ts) -> pd.Series:
    """Normaliza `ts` a una Series datetime con indice 0..n-1 y valida el orden."""
    t = pd.to_datetime(pd.Series(np.asarray(ts)).reset_index(drop=True))  # a datetime posicional
    if t.isna().any():                                    # un NaT rompe toda comparacion
        raise ValueError("split_purgado: `ts` contiene valores nulos (NaT).")
    if not t.is_monotonic_increasing:                     # ordenar aqui devolveria indices
        raise ValueError(                                 # contra un orden que el llamador no ve
            "split_purgado: `ts` debe venir ordenado ascendentemente. "
            "Ordena el DataFrame por marca de tiempo y haz reset_index(drop=True) antes."
        )
    return t                                              # eje temporal validado


def split_purgado(ts, n_splits: int = 5, purga: str = "12h", embargo=None) -> list:
    """Walk-forward de ventana expansiva con purga y embargo por marca de tiempo.

    Parametros
    ----------
    ts : array-like de timestamps, longitud n, ORDENADO ASCENDENTEMENTE.
        Se construye con `to_datetime(date + ' ' + time)` sobre el dataset
        supervisado, ordenando y haciendo reset_index (seccion 2.1 del doc).
    n_splits : numero de folds (bloques de prueba). Por defecto 5.
    purga : ventana ANTERIOR al bloque de prueba que se excluye del train.
        Cualquier cosa que acepte `pd.Timedelta` ('12h', '30min', ...).
    embargo : ventana POSTERIOR a cada bloque de prueba que se excluye de los
        trains de los folds siguientes. Si es None, toma el valor de `purga`.

    Retorna
    -------
    list[(train_idx, test_idx)] : arrays posicionales de enteros, usable
        directamente como `cv=` en sklearn.

    Semantica
    ---------
    Los bloques se cortan POR NUMERO DE MUESTRAS, igual que TimeSeriesSplit:
    n_splits+1 bloques contiguos, los ultimos n_splits son los de prueba. Lo que
    es por tiempo es la purga y el embargo, que es donde estaba el problema.

    Para el fold k, con t0 = ts de la primera fila de prueba y t1 = la de la
    ultima:
      - expansiva : el train arranca con todas las filas anteriores al bloque;
      - purga     : se quita del train toda fila con ts >= t0 - purga;
      - embargo   : se quita del train toda fila en (t1_j, t1_j + embargo] para
                    cualquier bloque de prueba anterior j < k, de modo que el
                    arranque posterior a una prueba no se reincorpora nunca.
    """
    t = _eje_temporal(ts)                                 # valida y normaliza el eje
    n = len(t)                                            # numero de muestras

    if n_splits < 1:                                      # al menos un fold
        raise ValueError(f"split_purgado: n_splits debe ser >= 1, recibido {n_splits}.")
    if n < n_splits + 1:                                  # hace falta un bloque por fold + train
        raise ValueError(
            f"split_purgado: n={n} muestras es insuficiente para n_splits={n_splits}."
        )

    d_purga = pd.Timedelta(purga)                         # purga como Timedelta
    d_embargo = pd.Timedelta(purga if embargo is None else embargo)  # embargo (por defecto = purga)
    if d_purga < pd.Timedelta(0) or d_embargo < pd.Timedelta(0):     # ventanas negativas no tienen sentido
        raise ValueError("split_purgado: `purga` y `embargo` deben ser no negativos.")

    # --- Bloques identicos a los de TimeSeriesSplit(n_splits) ---
    tam_test = n // (n_splits + 1)                        # tamano de cada bloque de prueba
    inicios = range(n - n_splits * tam_test, n, tam_test)  # inicio de cada bloque (el resto va al 1er train)

    folds = []                                            # lista de (train_idx, test_idx)
    fines_previos = []                                    # t1 de los bloques de prueba ya vistos

    for k, ini in enumerate(inicios):                     # por cada fold, en orden temporal
        test_idx = np.arange(ini, ini + tam_test)         # bloque de prueba contiguo
        t0 = t.iloc[ini]                                  # primer instante de la prueba
        t1 = t.iloc[ini + tam_test - 1]                   # ultimo instante de la prueba

        candidatos = np.arange(ini)                       # ventana expansiva: todo lo anterior
        t_cand = t.iloc[:ini]                             # sus marcas de tiempo

        mantener = (t_cand < t0 - d_purga).to_numpy(copy=True)  # PURGA: fuera lo que este a menos de `purga`
        for t1_prev in fines_previos:                     # EMBARGO: por cada prueba anterior
            mantener &= ~(                                # fuera la ventana (t1_prev, t1_prev + embargo]
                (t_cand > t1_prev) & (t_cand <= t1_prev + d_embargo)
            ).to_numpy()

        train_idx = candidatos[mantener]                  # train final del fold
        if train_idx.size == 0:                           # un fold sin train no se devuelve callando
            raise ValueError(
                f"split_purgado: el fold {k} queda sin train tras aplicar "
                f"purga={d_purga} y embargo={d_embargo} sobre {ini} candidatos."
            )

        folds.append((train_idx, test_idx))               # fold listo
        fines_previos.append(t1)                          # su t1 embarga a los folds siguientes

    return folds                                          # lista de folds, en orden temporal
