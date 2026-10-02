# ============================================================
# src/pipeline/tipos_mineral.py — Tipos de mineral en CABEZA y en RELAVE
# ============================================================
# Define los tipos de mineral sobre los compósitos de 12 h de los dos extremos
# del circuito: la alimentación a flotación (rebose de hidrociclones) y el relave
# final. Hasta ahora los "grupos de mineral" del proyecto se definían sobre las
# intensidades del CONCENTRADO (etapa 4), que es el producto de la flotación, no
# el mineral que entra. Con los compósitos de cabeza y relave hallados el
# 2026-10-01 (§15) se pueden definir donde corresponde geológicamente.
#
# POR QUÉ LOG-COCIENTES Y NO LEYES ABSOLUTAS:
#   el usuario advierte que la muestra de relave viene con mucha dilución de
#   agua. Medido: la mayor parte de la varianza de las leyes absolutas es un
#   factor de NIVEL común, no composición -- el CV cae de 0.671 a 0.094 en el Zn
#   de cabeza y de 0.256 a 0.105 en el Fe de relave al pasar a log-cocientes.
#   Los log-cocientes (alr de Aitchison contra Cu) son invariantes a cualquier
#   efecto multiplicativo de nivel, así que el tipo de mineral queda definido por
#   la COMPOSICIÓN y no por cuánta agua traía la muestra. Es la misma decisión
#   que se tomó en la etapa 2 para las intensidades (ADR 2026-09-30).
#
# PROTOCOLO (idéntico al de la etapa 4, para que los resultados sean comparables):
#   - ajuste solo sobre periodo train (ts <= FECHA_CORTE_TRAIN);
#   - StandardScaler, sin Yeo-Johnson (en la etapa 3 daba lambda degenerado);
#   - GMM full; k elegido por silhouette + ESTABILIDAD (ARI entre dos GMM
#     ajustados en mitades disjuntas de train), nunca por BIC;
#   - cada tipo se nombra por su centroide y se confirma contra las leyes
#     absolutas del laboratorio.
#
#   python -m src.pipeline.tipos_mineral
# ============================================================

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .config import (RAW_CABEZA, RAW_RELAVE, TIPOS_MINERAL, FECHA_CORTE_TRAIN,
                     FECHA_INICIO_PRODUCCION, RANDOM_STATE,
                     K_TIPO_CABEZA, K_TIPO_RELAVE)

K_POR_STREAM = {"cabeza": K_TIPO_CABEZA, "relave": K_TIPO_RELAVE}

# Elementos de cada stream. Cu es el denominador del alr: es el elemento de
# interés económico y el único presente en los tres compósitos.
ELEMENTOS = {"cabeza": ["fe", "zn", "mo"], "relave": ["fe", "mo"]}
UMBRAL_NOMBRE = 0.5          # z del centroide por encima del cual el elemento entra en el nombre


def cargar(stream: str) -> pd.DataFrame:
    """Compósito crudo del stream con sus log-cocientes contra Cu y su periodo."""
    d = pd.read_csv({"cabeza": RAW_CABEZA, "relave": RAW_RELAVE}[stream], parse_dates=["ts"])
    d = d.sort_values("ts").reset_index(drop=True)

    cols = ELEMENTOS[stream] + ["cu"]
    if (d[cols] <= 0).any().any():            # el lab no reporta ceros aquí, pero no se asume
        raise ValueError(f"{stream}: hay leyes <= 0, el log-cociente no está definido")

    for e in ELEMENTOS[stream]:
        d[f"lr_{e}_cu"] = np.log(d[e] / d["cu"])

    corte, prod = pd.Timestamp(FECHA_CORTE_TRAIN) + pd.Timedelta(days=1), pd.Timestamp(FECHA_INICIO_PRODUCCION)
    d["periodo"] = np.where(d.ts < corte, "train", np.where(d.ts < prod, "test", "produccion"))
    return d


def _feats(stream: str) -> list:
    return [f"lr_{e}_cu" for e in ELEMENTOS[stream]]


def diagnostico_k(d: pd.DataFrame, stream: str, k_min: int = 2, k_max: int = 8,
                  n_mitades: int = 3) -> pd.DataFrame:
    """Silhouette y estabilidad por k sobre train. El BIC se reporta pero no decide."""
    X = StandardScaler().fit_transform(d.loc[d.periodo == "train", _feats(stream)])
    filas = []
    for k in range(k_min, k_max + 1):
        g = GaussianMixture(k, covariance_type="full", random_state=RANDOM_STATE, n_init=5).fit(X)
        lab = g.predict(X)
        aris = []
        for s in range(n_mitades):                       # estabilidad: mitades disjuntas de train
            idx = np.random.RandomState(s).permutation(len(X))
            a, b = idx[: len(X) // 2], idx[len(X) // 2:]
            la = GaussianMixture(k, covariance_type="full", random_state=s, n_init=3).fit(X[a]).predict(X)
            lb = GaussianMixture(k, covariance_type="full", random_state=s + 10, n_init=3).fit(X[b]).predict(X)
            aris.append(adjusted_rand_score(la, lb))
        filas.append({"k": k, "bic": round(g.bic(X)),
                      "silhouette": round(silhouette_score(X, lab), 3),
                      "estabilidad_ari": round(float(np.mean(aris)), 3),
                      "grupo_min_pct": round(100 * np.bincount(lab, minlength=k).min() / len(lab), 1)})
    return pd.DataFrame(filas)


def ajustar(d: pd.DataFrame, stream: str, k: int) -> tuple:
    """Escalador + GMM ajustados SOLO en train. Devuelve (scaler, gmm, nombres)."""
    feats = _feats(stream)
    sc = StandardScaler().fit(d.loc[d.periodo == "train", feats])
    gmm = GaussianMixture(k, covariance_type="full", random_state=RANDOM_STATE,
                          n_init=10).fit(sc.transform(d.loc[d.periodo == "train", feats]))

    nombres = {}
    for i, centro in enumerate(gmm.means_):               # nombre por lo que sobre-representa el centroide
        altos = sorted(((z, e) for e, z in zip(ELEMENTOS[stream], centro) if z > UMBRAL_NOMBRE), reverse=True)
        nombres[i] = "+".join(e.capitalize() for _, e in altos) if altos else "Cu"
    for n in list(nombres.values()):                      # desempate si dos centroides dan el mismo nombre
        if list(nombres.values()).count(n) > 1:
            c = 0
            for i, v in nombres.items():
                if v == n:
                    c += 1
                    nombres[i] = f"{n}-{c}"
    return sc, gmm, nombres


def asignar(d: pd.DataFrame, stream: str, sc, gmm, nombres) -> pd.DataFrame:
    """Añade tipo_<stream> y prob_<stream> a todo el período (no solo train)."""
    X = sc.transform(d[_feats(stream)])
    p = gmm.predict_proba(X)
    out = d.copy()
    out[f"tipo_{stream}"] = [nombres[i] for i in p.argmax(axis=1)]
    out[f"prob_{stream}"] = p.max(axis=1).round(3)
    return out


def caracterizar(d: pd.DataFrame, stream: str) -> pd.DataFrame:
    """Perfil de cada tipo: n, y medianas de las leyes ABSOLUTAS del laboratorio.

    Las absolutas no definen los tipos (los definen los log-cocientes), pero son
    la lectura metalúrgica que permite confirmar el nombre.
    """
    cols = ["cu"] + ELEMENTOS[stream]
    g = d.groupby(f"tipo_{stream}")
    t = g[cols].median().round(4)
    t.insert(0, "n", g.size())
    t["pct"] = (100 * t.n / len(d)).round(1)
    t["prob_med"] = g[f"prob_{stream}"].median().round(3)
    return t.sort_values("n", ascending=False)


def main() -> None:
    salida = None
    for stream in ["cabeza", "relave"]:
        d = cargar(stream)
        print(f"\n{'=' * 72}\n{stream.upper()} — n={len(d)}  ({d.ts.min().date()} → {d.ts.max().date()})"
              f"  train={int((d.periodo == 'train').sum())}\n{'=' * 72}")
        print(f"features (alr contra Cu): {', '.join(_feats(stream))}\n")
        diag = diagnostico_k(d, stream)
        print(diag.to_string(index=False))

        # k FIJO por config, no elegido por la métrica: la justificación está en
        # config.K_TIPO_CABEZA / K_TIPO_RELAVE y en §17 de la bitácora. Se imprime
        # también el k que daría la regla automática, para que la diferencia quede
        # visible y auditable en vez de escondida.
        k = K_POR_STREAM[stream]
        k_auto = int(diag.loc[diag.estabilidad_ari.idxmax(), "k"])
        print(f"\n  k = {k} (fijado en config). La regla de máxima estabilidad ARI daría "
              f"k={k_auto}; ver la justificación en config.")

        sc, gmm, nombres = ajustar(d, stream, k)
        d = asignar(d, stream, sc, gmm, nombres)
        print(f"\nperfil de los tipos de mineral en {stream} (medianas de ley %):")
        print(caracterizar(d, stream).to_string())

        cols = ["ts", "periodo", f"tipo_{stream}", f"prob_{stream}"]
        salida = d[cols] if salida is None else salida.merge(d[cols], on=["ts", "periodo"], how="outer")

    salida.to_csv(TIPOS_MINERAL, index=False)
    print(f"\n-> {TIPOS_MINERAL}  ({len(salida)} filas)")


if __name__ == "__main__":
    main()
