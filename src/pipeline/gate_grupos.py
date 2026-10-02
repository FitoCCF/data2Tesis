# ============================================================
# src/pipeline/gate_grupos.py — Modelos por grupo de mineral sobre ventanas de
# compósito, en reemplazo de la tabla de ruteo
# ============================================================
# Arquitectura pedida por el usuario (2026-10-01): en vez de una tabla de ruteo
# que decide offline "local o global" por celda, cada grupo de mineral tiene sus
# propios modelos de Fe, Cu y Mo, y un trigger conmuta de modelo cuando el
# régimen cambia.
#
# POR QUÉ VENTANAS DE COMPÓSITO Y NO MUESTRAS DE LA BD (medido, §13 bitácora):
#   con las muestras puntuales de la BD dos de los cuatro grupos no alcanzan
#   N >= 60 (Mo 54, Zn 53) y el test supervisado son 18 muestras con Fe = 0, así
#   que la arquitectura no se puede ni entrenar ni validar. Las ventanas de
#   compósito dan 589 de train (Cu 225 / Mo 175 / Zn 120 / Fe 69) y 153 de test.
#   El compósito mide exactamente Fe, Cu y Mo — las tres leyes del encargo.
#   Costo declarado: el modelo aprende sobre promedios de 12 h; aplicarlo por
#   lectura tiene un precio ya medido en Cu (§4).
#
# POR QUÉ LA PURGA AQUÍ ES DE DÍAS Y NO DE 12 h:
#   las muestras de la BD están a 2.9 d una de otra y los cortes del
#   walk-forward caen en huecos de 4.8-10.2 d, así que la purga de 12 h era
#   inerte (§13.5). Las ventanas de compósito son CONTIGUAS cada 12 h: en el
#   corte de cada fold hay una ventana de train pegada al bloque de test. Como el
#   régimen de mineral se descorrelaciona en 3-5 d (§13.3), aquí la purga sí
#   muerde y PURGA_VENTANAS se fija en esa escala, no en 12 h.
#
# El orden de este módulo es deliberado: primero se MIDE si los modelos por grupo
# le ganan al global, con DOS filtros independientes —control de particiones
# aleatorias dentro de train (`validar_por_grupo`) y supervivencia fuera de
# muestra (`confirmar_en_test`)—, y solo si alguna celda pasa los dos tiene
# sentido escribir el trigger. Seis pruebas previas murieron en el control (§8-9).
#
# RESULTADO (2026-10-01, §14 de la bitácora): **el trigger no se implementó.**
# De las 12 celdas (3 leyes x 4 grupos), 4 pasan el control en train pero solo
# UNA sobrevive fuera de muestra (pMo en el grupo Cu, +0.103 de R²), y 1 de 12 es
# lo que produce el azar al probar 12 celdas. La celda queda como pista, no como
# resultado. Este módulo se conserva porque es el aparato de medición
# reproducible, no porque la arquitectura esté construida.
#
#   python -m src.pipeline.gate_grupos
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

from .config import (CLUSTERIZADO, LIMPIO_COMPOSITO, FEATS_REGRESION,
                     MAPA_LEY_COMPOSITO, MODELO_POR_TARGET, VENTANA_COMPOSITO_H,
                     MIN_BLOQUES_VENTANA, BLOQUE_RESAMPLE, N_SPLITS, EMBARGO,
                     N_MIN_LOCAL, MIN_TRAIN_FOLD_LOCAL, N_CONTROL, PCTL_CONTROL,
                     RANDOM_STATE, FECHA_CORTE_TRAIN, FECHA_INICIO_PRODUCCION)
from .modelos import _construir_modelo, _oof, _mejora, _predecir
from .split_temporal import split_purgado

# La escala de descorrelación del régimen de mineral (§13.3), no la tolerancia
# de emparejamiento: ver la cabecera.
PURGA_VENTANAS = "5d"

GRUPOS = ["Cu", "Mo", "Zn", "Fe"]
LEYES = list(MAPA_LEY_COMPOSITO)                   # pFe, pCu, pMo (el compósito no mide Zn)


def ventanas_composito(clusterizado: pd.DataFrame, composito: pd.DataFrame) -> pd.DataFrame:
    """Una fila por ensayo de compósito: features promediados en su ventana, la
    composición de grupos de mineral de esa ventana, y las leyes de laboratorio.

    Mismo protocolo que la ruta de recalibración (`alinear_con_composito`):
    ventana CENTRADA de VENTANA_COMPOSITO_H horas, submuestreo previo a bloques
    de 30 min para descorrelacionar, y al menos MIN_BLOQUES_VENTANA bloques.
    La alineación centrada está verificada ley contra ley (§12.2).
    """
    c = clusterizado[clusterizado.fuente.isin(["pi", "ambos"])].sort_values("ts")
    c = c[c["valida"].astype(bool)]                 # lecturas usables
    idx = c.set_index("ts")

    bloques = idx[FEATS_REGRESION].resample(BLOQUE_RESAMPLE).mean().dropna()
    grupos_b = idx["grupo"].resample(BLOQUE_RESAMPLE).agg(
        lambda s: s.mode().iat[0] if len(s) else np.nan).dropna()

    medio = pd.Timedelta(hours=VENTANA_COMPOSITO_H / 2)
    filas = []
    for _, ens in composito.sort_values("ts").iterrows():
        v = bloques.loc[ens.ts - medio: ens.ts + medio]
        if len(v) < MIN_BLOQUES_VENTANA:            # ventana sin cobertura suficiente
            continue
        g = grupos_b.loc[ens.ts - medio: ens.ts + medio]
        frac = g.value_counts(normalize=True) if len(g) else pd.Series(dtype=float)
        filas.append({
            "ts": ens.ts, "n_bloques": len(v),
            **v.mean().to_dict(),                                   # features de la ventana
            **{f"frac_{x}": frac.get(x, 0.0) for x in GRUPOS},      # composición de mineral
            "grupo": frac.idxmax() if len(frac) else np.nan,        # grupo dominante
            "pureza": frac.max() if len(frac) else np.nan,
            **{ley: ens[col] for ley, col in MAPA_LEY_COMPOSITO.items()},  # leyes del lab
        })

    v = pd.DataFrame(filas).dropna(subset=["grupo"])
    corte = pd.Timestamp(FECHA_CORTE_TRAIN) + pd.Timedelta(days=1)      # train: <= corte
    prod = pd.Timestamp(FECHA_INICIO_PRODUCCION)                        # produccion: >= inicio
    v["periodo"] = np.where(v.ts < corte, "train",
                            np.where(v.ts < prod, "test", "produccion"))
    return v.sort_values("ts").reset_index(drop=True)


def validar_por_grupo(ventanas: pd.DataFrame, purga: str = PURGA_VENTANAS,
                      n_control: int = N_CONTROL, pureza_min: float = 0.0) -> pd.DataFrame:
    """¿Los modelos por grupo le ganan al global sobre los MISMOS puntos, y le
    ganan también a una partición aleatoria del mismo tamaño?

    Es la pregunta que decide si la arquitectura de 12 modelos se sostiene. No
    construye tabla de ruteo: devuelve la evidencia por celda (ley, grupo).
    """
    s = ventanas[ventanas.periodo == "train"]
    if pureza_min > 0:                               # opcional: solo ventanas de régimen nítido
        s = s[s.pureza >= pureza_min]
    s = s.sort_values("ts").reset_index(drop=True)

    folds = split_purgado(s["ts"], n_splits=N_SPLITS, purga=purga, embargo=EMBARGO)
    X = s[FEATS_REGRESION].to_numpy()
    grupo = s["grupo"].to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)

    filas = []
    for ley in LEYES:
        familia = MODELO_POR_TARGET[ley]
        y = s[ley].to_numpy(dtype=float)
        con_ley = ~np.isnan(y)

        pred_g = _oof(X, y, folds, familia, con_ley)                  # global fuera de fold
        p_all = con_ley & ~np.isnan(pred_g)
        filas.append({"ley": ley, "grupo": "(todos)", "N": int(con_ley.sum()),
                      "N_eval": int(p_all.sum()),
                      "r2_global": r2_score(y[p_all], pred_g[p_all]),
                      "mae_global": mean_absolute_error(y[p_all], pred_g[p_all])})

        # Control: mismas etiquetas permutadas -> pseudo-grupos sin información
        # real, con exactamente los mismos tamaños.
        elegibles = [g for g in GRUPOS if (con_ley & (grupo == g)).sum() >= N_MIN_LOCAL]
        ctrl = {g: [] for g in elegibles}
        for _ in range(n_control if elegibles else 0):
            perm = grupo.copy()
            perm[con_ley] = rng.permutation(grupo[con_ley])
            for g in elegibles:
                m = con_ley & (perm == g)
                ctrl[g].append(_mejora(y, pred_g, _oof(X, y, folds, familia, m,
                                                       MIN_TRAIN_FOLD_LOCAL), m)[2])

        for g in GRUPOS:
            m = con_ley & (grupo == g)
            fila = {"ley": ley, "grupo": g, "N": int(m.sum())}
            if fila["N"] < N_MIN_LOCAL:                               # regla a priori
                fila["veredicto"] = "global (N insuficiente)"
                filas.append(fila)
                continue
            pred_l = _oof(X, y, folds, familia, m, MIN_TRAIN_FOLD_LOCAL)
            r2g, r2l, mejora, n_eval = _mejora(y, pred_g, pred_l, m)
            umbral = np.nanpercentile(ctrl[g], PCTL_CONTROL) if ctrl[g] else np.nan
            pctl = (100 * np.mean(np.asarray(ctrl[g]) < mejora)) if ctrl[g] else np.nan
            pasa = bool(mejora > 0 and mejora > umbral)
            fila.update(N_eval=n_eval, r2_global=r2g, r2_local=r2l, mejora=mejora,
                        umbral_control=umbral, pctl_en_control=pctl,
                        veredicto="LOCAL" if pasa else "global")
            filas.append(fila)
    return pd.DataFrame(filas)


def confirmar_en_test(ventanas: pd.DataFrame, n_min_test: int = 5) -> pd.DataFrame:
    """Segundo filtro, independiente del control aleatorio: ¿la mejora del modelo
    por grupo sobrevive fuera de muestra?

    Ajusta el global y cada local en TODO train y los compara sobre las ventanas
    de test del grupo. Una celda solo es creíble si pasa el control en train
    (`validar_por_grupo`) **y** mantiene la mejora aquí.
    """
    tr, te = ventanas[ventanas.periodo == "train"], ventanas[ventanas.periodo == "test"]
    filas = []
    for ley in LEYES:
        familia = MODELO_POR_TARGET[ley]
        glob = _construir_modelo(familia).fit(tr[FEATS_REGRESION], tr[ley])
        for g in GRUPOS:
            a, b = tr[tr.grupo == g], te[te.grupo == g]
            fila = {"ley": ley, "grupo": g, "N_train": len(a), "N_test": len(b)}
            if len(b) < n_min_test:                       # test demasiado chico para concluir
                fila["nota"] = "test insuficiente"
            elif len(a) < N_MIN_LOCAL:
                fila.update(r2_global=r2_score(b[ley], _predecir(glob, b[FEATS_REGRESION])),
                            nota="N train insuficiente")
            else:
                loc = _construir_modelo(familia).fit(a[FEATS_REGRESION], a[ley])
                r2g = r2_score(b[ley], _predecir(glob, b[FEATS_REGRESION]))
                r2l = r2_score(b[ley], _predecir(loc, b[FEATS_REGRESION]))
                fila.update(r2_global=r2g, r2_local=r2l, mejora=r2l - r2g)
            filas.append(fila)
    return pd.DataFrame(filas)


def _imprimir(tabla: pd.DataFrame, titulo: str) -> None:
    print(f"\n{titulo}")
    print(f"  {'ley':<5}{'grupo':<10}{'N':>5}{'N_ev':>6}{'R2 glob':>9}{'R2 loc':>9}"
          f"{'mejora':>9}{'p95 ctrl':>10}{'pctl':>7}  veredicto")
    for _, r in tabla.iterrows():
        f = lambda k, d=2: ("" if pd.isna(r.get(k, np.nan)) else f"{r[k]:.{d}f}")
        print(f"  {r.ley:<5}{r.grupo:<10}{int(r.N):>5}"
              f"{'' if pd.isna(r.get('N_eval', np.nan)) else int(r.N_eval):>6}"
              f"{f('r2_global', 3):>9}{f('r2_local', 3):>9}{f('mejora', 3):>9}"
              f"{f('umbral_control', 3):>10}{f('pctl_en_control', 0):>7}  "
              f"{r.get('veredicto', '')}")


def main() -> None:
    cl = pd.read_csv(CLUSTERIZADO, parse_dates=["ts"])
    comp = pd.read_csv(LIMPIO_COMPOSITO, parse_dates=["ts"])
    v = ventanas_composito(cl, comp)

    print(f"ventanas de compósito con cobertura: {len(v)}")
    print(v.groupby(["periodo", "grupo"]).size().unstack().reindex(columns=GRUPOS).fillna(0).astype(int).to_string())
    print(f"\npureza del grupo dominante: mediana {v.pureza.median():.2f}, "
          f">=0.7 en {(v.pureza >= 0.7).mean():.0%} de las ventanas")

    print(f"\n{'=' * 78}\n¿LOS MODELOS POR GRUPO LE GANAN AL GLOBAL Y AL CONTROL ALEATORIO?"
          f"\n{'=' * 78}")
    for purga in ["12h", PURGA_VENTANAS]:
        t = validar_por_grupo(v, purga=purga)
        _imprimir(t, f"-- purga {purga} (n_control={N_CONTROL})")

    print(f"\n{'=' * 78}\nSEGUNDO FILTRO — ¿SOBREVIVE FUERA DE MUESTRA? (153 ventanas de test)"
          f"\n{'=' * 78}\n")
    c = confirmar_en_test(v)
    print(f"  {'ley':<5}{'grupo':<8}{'N_tr':>6}{'N_te':>6}{'R2 glob':>10}{'R2 loc':>9}{'mejora':>9}  nota")
    for _, r in c.iterrows():
        f = lambda k: ("" if pd.isna(r.get(k, np.nan)) else f"{r[k]:.3f}")
        print(f"  {r.ley:<5}{r.grupo:<8}{int(r.N_train):>6}{int(r.N_test):>6}"
              f"{f('r2_global'):>10}{f('r2_local'):>9}{f('mejora'):>9}  {r.get('nota', '')}")
    print()


if __name__ == "__main__":
    main()
