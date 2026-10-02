# ============================================================
# scripts/verificar_escala_temporal_regimen.py — ¿cada cuánto cambia el mineral?
# ============================================================
# Reproduce las cifras de la §13 de docs/bitacora_analisis.md. Responde tres
# preguntas, en este orden:
#
#   1. ¿Cada cuánto cambia el grupo de mineral dominante, a resolución semanal?
#   2. ¿Las intensidades se mueven con él? (cuánta varianza explica la semana)
#   3. ¿Cuál es la escala de tiempo propia del fenómeno? (autocorrelación)
#
# Y mide la consecuencia metodológica: si los features se descorrelacionan en
# 3-5 días, la purga de 12 h de `split_purgado` no cierra la fuga entre muestras
# vecinas de la BD, que están a ~2.9 días una de otra.
#
# Solo lectura. Usa la secuencia real de lecturas del courier (fuente pi/ambos);
# las filas solo-BD quedan fuera porque no son lecturas del PI en secuencia.
#
#   pixi run python scripts/verificar_escala_temporal_regimen.py
# ============================================================
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.config import CLUSTERIZADO, FEATURES, LIMPIO_COURIER_BD

GRUPOS = ["Cu", "Mo", "Zn", "Fe"]
FEATS = ["lr_fe_cu", "lr_zn_cu", "lr_mo_cu", "logSumI"]
LAGS = [1, 2, 3, 5, 7, 10, 14, 21, 30]
MIN_LECTURAS_SEMANA = 50
MIN_LECTURAS_DIA = 20


def solo_pi(df):
    """La secuencia real de lecturas del courier, en orden."""
    return df[df.fuente.isin(["pi", "ambos"])].sort_values("ts")


def regimen_semanal():
    c = solo_pi(pd.read_csv(CLUSTERIZADO, parse_dates=["ts"]))
    c["semana"] = c.ts.dt.to_period("W-SUN")

    frac = (c.groupby("semana").grupo.value_counts(normalize=True)
            .unstack().reindex(columns=GRUPOS).fillna(0))
    frac = frac[c.groupby("semana").size() >= MIN_LECTURAS_SEMANA]

    dom = frac.idxmax(axis=1)
    rachas = dom.groupby((dom != dom.shift()).cumsum()).size()

    print("=" * 64)
    print("1. ¿CADA CUÁNTO CAMBIA EL GRUPO DOMINANTE? (resolución semanal)")
    print("=" * 64)
    print(f"\n  {len(frac)} semanas con >= {MIN_LECTURAS_SEMANA} lecturas "
          f"({frac.index.min()} .. {frac.index.max()})")
    print(f"  cambia de dominante de una semana a la siguiente: "
          f"{(dom != dom.shift()).iloc[1:].mean():.0%}")
    print(f"  duración de una racha: mediana {rachas.median():.0f} sem, "
          f"media {rachas.mean():.1f}, máx {rachas.max()}")
    print(f"  rachas de 1-2 semanas: {(rachas <= 2).mean():.0%}")
    print(f"  pureza del dominante: mediana {frac.max(axis=1).median():.2f}")
    return c


def intensidades_por_semana(clusterizado):
    f = solo_pi(pd.read_csv(FEATURES, parse_dates=["ts"]))
    f = f[f.valida.astype(bool)]
    f["semana"] = f.ts.dt.to_period("W-SUN")

    print("\n" + "=" * 64)
    print("2. ¿LAS INTENSIDADES SE MUEVEN CON EL RÉGIMEN?")
    print("=" * 64)
    print(f"\n  {len(f)} lecturas válidas, {f['semana'].nunique()} semanas\n")
    print(f"  {'feature':<10}{'var. que explica la semana':>28}"
          f"{'medianas semanales':>24}{'sd intra':>10}")
    for v in FEATS:
        g = f.groupby("semana")[v]
        n, medias = g.size(), g.mean()
        entre = (n * (medias - f[v].mean()) ** 2).sum() / (n.sum() - 1)
        print(f"  {v:<10}{entre / f[v].var():>28.2f}"
              f"{f'{g.median().min():+.2f} .. {g.median().max():+.2f}':>24}"
              f"{g.std().median():>10.2f}")
    print("\n  La semana es un agrupamiento real de las intensidades: en lr_mo_cu")
    print("  el recorrido entre semanas es ~5x la dispersión dentro de la semana.")
    return f


def escala_de_tiempo(clusterizado, features):
    clusterizado["dia"] = clusterizado.ts.dt.floor("D")
    frac = (clusterizado.groupby("dia").grupo.value_counts(normalize=True)
            .unstack().reindex(columns=GRUPOS).fillna(0))
    frac = frac[clusterizado.groupby("dia").size() >= MIN_LECTURAS_DIA]

    features["dia"] = features.ts.dt.floor("D")
    med = features.groupby("dia")[FEATS].median()
    med = med[features.groupby("dia").size() >= MIN_LECTURAS_DIA]

    print("\n" + "=" * 64)
    print("3. ESCALA DE TIEMPO PROPIA — autocorrelación de la mediana diaria")
    print("=" * 64)
    print(f"\n  ({len(frac)} días con >= {MIN_LECTURAS_DIA} lecturas)\n")
    cols = [f"frac {g}" for g in GRUPOS] + FEATS
    print("  lag(d)" + "".join(f"{c:>11}" for c in cols))
    for L in LAGS:
        fila = [frac[g].autocorr(L) for g in GRUPOS] + [med[v].autocorr(L) for v in FEATS]
        print(f"  {L:>5} " + "".join(f"{x:>11.2f}" for x in fila))
    print("\n  Mineral y log-cocientes se descorrelacionan hacia el día 5-7:")
    print("  la escala del régimen es ~3-5 días, MÁS RÁPIDA que semanal.")
    print("  logSumI es la excepción (deriva lenta de dilución/instrumento,")
    print("  no mineral): sigue correlacionado a 30 días.")


def consecuencia_para_la_purga():
    b = pd.read_csv(LIMPIO_COURIER_BD, parse_dates=["ts"]).sort_values("ts")
    b = b[b.leyes_sospechosas.isna()]
    sep = b.ts.diff().dt.total_seconds() / 86400

    print("\n" + "=" * 64)
    print("4. CONSECUENCIA — la purga de 12 h no cierra esta fuga")
    print("=" * 64)
    print(f"\n  separación entre muestras consecutivas de la BD: "
          f"mediana {sep.median():.1f} d; <= 3 d en {(sep <= 3).mean():.0%} de los pares")
    print("\n  autocorrelación de la ley entre muestras consecutivas:")
    for ley in ["pFe", "pCu", "pMo", "pZn"]:
        s = b[ley].dropna().reset_index(drop=True)
        print(f"    {ley}: lag1 {s.autocorr(1):+.2f}  lag2 {s.autocorr(2):+.2f}  "
              f"lag3 {s.autocorr(3):+.2f}   (n={len(s)})")
    print("\n  Una muestra de test puede estar a 2 días de su vecina de train")
    print("  COMPARTIENDO régimen de mineral. La purga de 12 h no lo impide, y")
    print("  por eso no eliminaba ninguna fila. Zn es la excepción (no persiste).")


def main():
    clusterizado = regimen_semanal()
    features = intensidades_por_semana(clusterizado)
    escala_de_tiempo(clusterizado, features)
    consecuencia_para_la_purga()
    print()


if __name__ == "__main__":
    main()
