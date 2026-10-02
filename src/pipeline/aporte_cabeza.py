# ============================================================
# src/pipeline/aporte_cabeza.py — ¿La cabeza mejora la estimación del concentrado?
# ============================================================
# Pregunta: añadir las leyes del compósito de CABEZA a las intensidades del
# concentrado, ¿mejora la estimación de las leyes del concentrado?
#
# DOS ESCENARIOS, y la diferencia importa más que el número:
#   lag 0  — cabeza de LA MISMA ventana. Mide cuánta información lleva la
#            alimentación sobre el producto, pero NO es usable en línea: ese
#            ensayo llega junto con el del concentrado, así que si lo tienes ya
#            tienes el laboratorio. Es resultado científico, no operativo.
#   lag 1  — cabeza de la ventana ANTERIOR (12 h antes). Esto sí está disponible
#            cuando hay que estimar, así que es el único escenario con valor
#            operativo. §15.3 ya midió que mo_cab->mo cae de 0.848 a 0.758 al
#            retrasar un turno, así que se espera menos aporte pero real.
#
# PROTOCOLO: el mismo que mató la arquitectura por grupo en §14, dos filtros.
#   1. Walk-forward con split_purgado sobre las ventanas de train, purga de
#      PURGA_VENTANAS (días, no 12 h: las ventanas son contiguas, §14.1), y
#      comparación base vs base+cabeza SOBRE LOS MISMOS PUNTOS fuera de fold.
#   2. Control por PERMUTACIÓN de las columnas de cabeza: las mismas columnas,
#      misma distribución, orden barajado -> sin información real. Es el análogo
#      del control de particiones aleatorias: si la cabeza no le gana a su propia
#      versión barajada, lo que se midió es capacidad extra del modelo, no señal.
#   3. Confirmación fuera de muestra en las ventanas de test.
#
#   python -m src.pipeline.aporte_cabeza
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .config import (CLUSTERIZADO, LIMPIO_COMPOSITO, RAW_CABEZA, FEATS_REGRESION,
                     MODELO_POR_TARGET, N_SPLITS, EMBARGO, RANDOM_STATE)
from .modelos import _construir_modelo, _oof, _predecir
from .split_temporal import split_purgado
from .gate_grupos import ventanas_composito, LEYES, PURGA_VENTANAS

CABEZA = ["cu_cab", "fe_cab", "zn_cab", "mo_cab"]
N_PERM = 200                       # permutaciones del control
PCTL = 95                          # percentil del control que hay que superar
# Piso de magnitud, además del control. Hace falta porque el control por
# permutación añade 4 columnas de ruido y por eso su p95 queda en ~0 o negativo:
# con esa vara sola, una mejora de +0.002 de R² "supera al control" y se
# reportaría como aporte, lo que sería engañoso. Un aporte tiene que ser, además,
# de un tamaño que signifique algo.
MEJORA_MIN = 0.01


def dataset(lag_turnos: int = 0) -> pd.DataFrame:
    """Ventanas de compósito con las leyes de cabeza de `lag_turnos` turnos antes.

    lag 0 = misma ventana; lag 1 = la cabeza del turno anterior (12 h), que es la
    que está disponible causalmente al momento de estimar.
    """
    v = ventanas_composito(pd.read_csv(CLUSTERIZADO, parse_dates=["ts"]),
                           pd.read_csv(LIMPIO_COMPOSITO, parse_dates=["ts"]))
    cab = pd.read_csv(RAW_CABEZA, parse_dates=["ts"])
    cab = cab.rename(columns={c: f"{c}_cab" for c in ["fe", "cu", "zn", "mo"]})
    cab["ts"] = cab["ts"] + pd.Timedelta(hours=12 * lag_turnos)   # desplaza la cabeza hacia adelante
    return v.merge(cab[["ts"] + CABEZA], on="ts", how="inner").sort_values("ts").reset_index(drop=True)


def _r2_oof(X, y, folds, familia, mascara):
    """R² fuera de fold sobre los puntos donde hubo predicción."""
    pred = _oof(X, y, folds, familia, mascara)
    p = mascara & ~np.isnan(pred)
    return (r2_score(y[p], pred[p]) if p.sum() >= 3 else np.nan), p


def evaluar(d: pd.DataFrame, purga: str = PURGA_VENTANAS, n_perm: int = N_PERM) -> pd.DataFrame:
    """Base (intensidades) vs base+cabeza, mismos puntos, con control por permutación."""
    s = d[d.periodo == "train"].reset_index(drop=True)
    folds = split_purgado(s["ts"], n_splits=N_SPLITS, purga=purga, embargo=EMBARGO)
    rng = np.random.default_rng(RANDOM_STATE)

    Xb = s[FEATS_REGRESION].to_numpy()
    Xc = s[FEATS_REGRESION + CABEZA].to_numpy()
    filas = []
    for ley in LEYES:
        familia = MODELO_POR_TARGET[ley]
        y = s[ley].to_numpy(dtype=float)
        m = ~np.isnan(y)

        r2_base, _ = _r2_oof(Xb, y, folds, familia, m)
        r2_cab, p = _r2_oof(Xc, y, folds, familia, m)
        mejora = r2_cab - r2_base

        ctrl = []                                         # control: cabeza barajada
        for _ in range(n_perm):
            Xp = Xc.copy()
            Xp[:, len(FEATS_REGRESION):] = Xc[rng.permutation(len(Xc)), len(FEATS_REGRESION):]
            ctrl.append(_r2_oof(Xp, y, folds, familia, m)[0] - r2_base)
        umbral = float(np.nanpercentile(ctrl, PCTL))
        pctl = 100 * float(np.mean(np.asarray(ctrl) < mejora))

        filas.append({"ley": ley, "N": int(m.sum()), "N_eval": int(p.sum()),
                      "r2_base": r2_base, "r2_cabeza": r2_cab, "mejora": mejora,
                      "p95_control": umbral, "pctl_en_control": pctl,
                      "veredicto": ("APORTA" if (mejora > umbral and mejora >= MEJORA_MIN)
                                    else "no aporta")})
    return pd.DataFrame(filas)


def confirmar_en_test(d: pd.DataFrame) -> pd.DataFrame:
    """Ajusta en todo train y compara base vs base+cabeza en test."""
    tr, te = d[d.periodo == "train"], d[d.periodo == "test"]
    filas = []
    for ley in LEYES:
        familia = MODELO_POR_TARGET[ley]
        fila = {"ley": ley, "N_train": len(tr), "N_test": len(te)}
        for nombre, feats in [("base", FEATS_REGRESION), ("cabeza", FEATS_REGRESION + CABEZA)]:
            mod = _construir_modelo(familia).fit(tr[feats], tr[ley])
            fila[f"r2_{nombre}"] = r2_score(te[ley], _predecir(mod, te[feats]))
        fila["mejora"] = fila["r2_cabeza"] - fila["r2_base"]
        filas.append(fila)
    return pd.DataFrame(filas)


def main() -> None:
    for lag, rotulo in [(0, "lag 0 — cabeza de la MISMA ventana (no usable en línea)"),
                        (1, "lag 1 — cabeza del turno ANTERIOR (usable en línea)")]:
        d = dataset(lag)
        print(f"\n{'=' * 80}\n{rotulo}   n={len(d)}  "
              f"(train {int((d.periodo == 'train').sum())}, test {int((d.periodo == 'test').sum())})\n{'=' * 80}")
        t = evaluar(d)
        print("\nFiltro 1 — walk-forward en train, control de 200 permutaciones de la cabeza:")
        print(f"  {'ley':<5}{'N_ev':>6}{'R2 base':>10}{'R2 +cab':>10}{'mejora':>9}{'p95 ctrl':>10}{'pctl':>7}  veredicto")
        for _, r in t.iterrows():
            print(f"  {r.ley:<5}{int(r.N_eval):>6}{r.r2_base:>10.3f}{r.r2_cabeza:>10.3f}"
                  f"{r.mejora:>+9.3f}{r.p95_control:>10.3f}{r.pctl_en_control:>7.0f}  {r.veredicto}")
        c = confirmar_en_test(d)
        print("\nFiltro 2 — fuera de muestra (test):")
        print(f"  {'ley':<5}{'N_te':>6}{'R2 base':>10}{'R2 +cab':>10}{'mejora':>9}")
        for _, r in c.iterrows():
            print(f"  {r.ley:<5}{int(r.N_test):>6}{r.r2_base:>10.3f}{r.r2_cabeza:>10.3f}{r.mejora:>+9.3f}")
    print()


if __name__ == "__main__":
    main()
