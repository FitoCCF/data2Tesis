# ============================================================
# tests/test_split_purgado.py — bateria de src/pipeline/split_temporal.py
# ============================================================
# La purga no se comprueba mirando: se comprueba (a) afirmando la propiedad
# temporal que la define, (b) contra un oraculo ingenuo escrito aparte, y
# (c) anclando el caso purga=0 contra TimeSeriesSplit de sklearn.
# ============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

RAIZ = Path(__file__).resolve().parents[1]                 # .../data2TesisV2
sys.path.insert(0, str(RAIZ))                              # para importar src.pipeline

from src.pipeline.split_temporal import split_purgado      # noqa: E402

CSV_CANONICO = RAIZ / "data" / "processed" / "courier_clusterizado.csv"   # salida de la etapa 4
PURGA = pd.Timedelta("12h")


# ------------------------------------------------------------------
# Utilidades de los tests
# ------------------------------------------------------------------

def ts_regular(n=240, paso="1h"):
    """Serie regular: el muestreo uniforme hace comparable purga y conteo."""
    return pd.date_range("2024-01-01", periods=n, freq=paso).to_series().reset_index(drop=True)


def ts_irregular(n_denso=20, n_ralo=100):
    """Serie deliberadamente irregular: un tramo denso (20 muestras en ~3 h)
    intercalado en un tramo ralo (1 muestra cada 5 dias). Es el caso que
    distingue una purga por tiempo de un `gap` por numero de muestras."""
    ralo_a = pd.date_range("2024-01-01", periods=n_ralo // 2, freq="5D")
    denso = pd.date_range(ralo_a[-1] + pd.Timedelta("5D"), periods=n_denso, freq="9min")
    ralo_b = pd.date_range(denso[-1] + pd.Timedelta("5D"), periods=n_ralo // 2, freq="5D")
    return pd.Series(ralo_a.union(denso).union(ralo_b)).reset_index(drop=True)


def ts_canonico():
    """Eje temporal del conjunto supervisado de la etapa 5 en V2: muestras con
    ley valida (valida_ley) del periodo train, ordenadas por ts."""
    df = pd.read_csv(CSV_CANONICO, usecols=["ts", "valida_ley", "periodo"], parse_dates=["ts"])
    t = df.loc[df["valida_ley"] & (df["periodo"] == "train"), "ts"]
    return t.sort_values().reset_index(drop=True)


def oraculo_ingenuo(t, n_splits, purga, embargo):
    """Implementacion tonta y literal de la definicion, escrita aparte a
    proposito: recorre TODOS los indices y decide uno por uno. Sirve de oraculo
    contra la implementacion vectorizada de split_temporal.py."""
    t = list(pd.to_datetime(pd.Series(t)).reset_index(drop=True))
    n = len(t)
    tam = n // (n_splits + 1)
    inicios = list(range(n - n_splits * tam, n, tam))
    d_p, d_e = pd.Timedelta(purga), pd.Timedelta(embargo)
    folds = []
    for k, ini in enumerate(inicios):
        t0 = t[ini]
        train = []
        for i in range(n):                                 # uno por uno, sin trucos
            if i >= ini:                                   # expansiva: solo el pasado
                continue
            if t[i] >= t0 - d_p:                           # purga
                continue
            embargado = False
            for j in inicios[:k]:                          # embargo de cada prueba anterior
                t1_prev = t[j + tam - 1]
                if t1_prev < t[i] <= t1_prev + d_e:
                    embargado = True
            if embargado:
                continue
            train.append(i)
        folds.append((train, list(range(ini, ini + tam))))
    return folds


# ------------------------------------------------------------------
# 1. La propiedad que define la purga
# ------------------------------------------------------------------

def test_hueco_de_12h_garantizado():
    t = ts_regular(240, "1h")
    for k, (tr, te) in enumerate(split_purgado(t, n_splits=5, purga="12h")):
        assert len(tr) > 0 and len(te) > 0
        assert t.iloc[tr].max() < t.iloc[te].min() - PURGA, f"fold {k}: la purga no dejo el hueco de 12 h"


# ------------------------------------------------------------------
# 2. Contra el oraculo ingenuo
# ------------------------------------------------------------------

@pytest.mark.parametrize("t", [ts_regular(240, "1h"), ts_regular(180, "3h"), ts_irregular()],
                         ids=["horaria", "cada-3h", "irregular"])
def test_contra_oraculo_ingenuo(t):
    obtenido = split_purgado(t, n_splits=5, purga="12h", embargo="12h")
    esperado = oraculo_ingenuo(t, n_splits=5, purga="12h", embargo="12h")
    assert len(obtenido) == len(esperado)
    for k, ((tr_o, te_o), (tr_e, te_e)) in enumerate(zip(obtenido, esperado)):
        assert list(tr_o) == tr_e, f"fold {k}: el train difiere del oraculo"
        assert list(te_o) == te_e, f"fold {k}: el test difiere del oraculo"


# ------------------------------------------------------------------
# 3. Por tiempo, no por muestras — la razon de escribir este divisor
# ------------------------------------------------------------------

def test_purga_por_tiempo_no_por_muestras():
    t = ts_irregular()
    folds = split_purgado(t, n_splits=5, purga="12h", embargo="0s")

    # Nº de filas purgadas por fold = candidatos del pasado - train efectivo.
    purgadas = [int(te.min()) - len(tr) for tr, te in folds]

    # En el tramo denso (9 min entre muestras) la purga de 12 h se lleva muchas
    # filas; en el ralo (5 dias entre muestras) no se lleva ninguna.
    assert max(purgadas) >= 10, f"la purga no mordio el tramo denso: {purgadas}"
    assert min(purgadas) == 0, f"la purga mordio el tramo ralo: {purgadas}"
    assert len(set(purgadas)) > 1, (
        f"purgadas={purgadas}: un numero constante por fold significaria que se "
        "esta purgando por conteo de muestras y no por marca de tiempo"
    )

    # Y ningun `gap` fijo de TimeSeriesSplit reproduce ese reparto: gap=g quita
    # exactamente g muestras en todos los folds, y aqui el reparto es desigual.
    for g in range(0, max(purgadas) + 1):
        assert purgadas != [g] * len(purgadas)

    # Comprobacion directa de la propiedad en cada fold.
    for k, (tr, te) in enumerate(folds):
        assert t.iloc[tr].max() < t.iloc[te].min() - PURGA, f"fold {k}"


# ------------------------------------------------------------------
# 4. Embargo
# ------------------------------------------------------------------

def test_embargo_excluye_ventana_posterior():
    t = ts_regular(240, "1h")
    emb = pd.Timedelta("12h")
    folds = split_purgado(t, n_splits=5, purga="12h", embargo="12h")

    fines = [t.iloc[te].max() for _, te in folds]               # t1 de cada bloque de prueba
    for k, (tr, _) in enumerate(folds):
        for j in range(k):                                 # cada prueba anterior
            dentro = ((t.iloc[tr] > fines[j]) & (t.iloc[tr] <= fines[j] + emb)).sum()
            assert dentro == 0, (
                f"fold {k}: {dentro} filas del train caen en el embargo del bloque {j}"
            )


def test_embargo_cero_devuelve_mas_train_que_embargo_activo():
    t = ts_regular(240, "1h")
    sin_emb = split_purgado(t, n_splits=5, purga="12h", embargo="0s")
    con_emb = split_purgado(t, n_splits=5, purga="12h", embargo="12h")
    # El embargo nunca puede anadir filas al train, solo quitarlas.
    for k in range(5):
        assert set(con_emb[k][0]).issubset(set(sin_emb[k][0])), f"fold {k}"

    # Fold 0: no hay prueba anterior, no hay nada que embargar.
    assert list(sin_emb[0][0]) == list(con_emb[0][0])

    # Fold 1: tampoco cambia, y no es un fallo sino geometria. Con bloques
    # contiguos, la ventana (t1_0, t1_0 + embargo] cae DENTRO del bloque de
    # prueba 1, que no es train de nadie en el fold 1. El embargo del bloque j
    # solo puede morder el train a partir del fold j+2.
    assert list(sin_emb[1][0]) == list(con_emb[1][0])

    # Fold 2 en adelante: el embargo del bloque 0 (y siguientes) si esta dentro
    # de la ventana expansiva y tiene que quitar filas.
    for k in range(2, 5):
        assert len(con_emb[k][0]) < len(sin_emb[k][0]), f"fold {k}: el embargo no quito nada"


# ------------------------------------------------------------------
# 5. Ancla contra sklearn
# ------------------------------------------------------------------

@pytest.mark.parametrize("n", [336, 337, 239, 100, 101],
                         ids=["336-divisible", "337", "239", "100", "101"])
def test_purga_cero_equivale_a_timeseriessplit(n):
    # n no divisible por n_splits+1 es el caso interesante: el resto de muestras
    # va al primer train, y ahi es facil desviarse de sklearn sin notarlo.
    t = ts_regular(n, "1D")
    nuestro = split_purgado(t, n_splits=5, purga="0s", embargo="0s")
    sklearn_ = list(TimeSeriesSplit(n_splits=5).split(np.zeros((len(t), 1))))
    assert len(nuestro) == len(sklearn_)
    for k, ((tr_n, te_n), (tr_s, te_s)) in enumerate(zip(nuestro, sklearn_)):
        assert list(tr_n) == list(tr_s), f"fold {k}: train distinto a TimeSeriesSplit"
        assert list(te_n) == list(te_s), f"fold {k}: test distinto a TimeSeriesSplit"


# ------------------------------------------------------------------
# 6. Invariantes estructurales
# ------------------------------------------------------------------

def test_sin_solape_ni_futuro():
    t = ts_regular(240, "1h")
    folds = split_purgado(t, n_splits=5, purga="12h")
    vistos = set()
    tam_train_previo = -1
    for k, (tr, te) in enumerate(folds):
        assert set(tr).isdisjoint(set(te)), f"fold {k}: train y test se solapan"
        assert vistos.isdisjoint(set(te)), f"fold {k}: bloque de prueba repetido"
        vistos |= set(te)
        assert t.iloc[tr].max() < t.iloc[te].min(), f"fold {k}: hay train posterior al test"
        assert len(tr) > tam_train_previo, f"fold {k}: la ventana no es expansiva"
        tam_train_previo = len(tr)
    assert len(vistos) == 5 * (240 // 6)                   # los 5 bloques cubren la cola


# ------------------------------------------------------------------
# 7-8. Fallos explicitos
# ------------------------------------------------------------------

def test_ts_desordenado_falla():
    t = ts_regular(240, "1h")
    revuelto = pd.concat([t.iloc[100:], t.iloc[:100]]).reset_index(drop=True)
    with pytest.raises(ValueError, match="ordenado"):
        split_purgado(revuelto, n_splits=5, purga="12h")


def test_ts_con_nulos_falla():
    t = ts_regular(240, "1h").copy()
    t.iloc[7] = pd.NaT
    with pytest.raises(ValueError, match="nulos"):
        split_purgado(t, n_splits=5, purga="12h")


def test_train_vacio_falla():
    t = ts_regular(240, "1h")
    with pytest.raises(ValueError, match="fold 0"):
        split_purgado(t, n_splits=5, purga="9999D")


def test_n_insuficiente_falla():
    t = ts_regular(4, "1h")
    with pytest.raises(ValueError, match="insuficiente"):
        split_purgado(t, n_splits=5, purga="12h")


# ------------------------------------------------------------------
# 9. Conjunto supervisado de V2 (342 muestras train con valida_ley)
# ------------------------------------------------------------------

def test_dataset_canonico():
    t = ts_canonico()
    assert len(t) == 342, "el supervisado de train dejo de tener 342 filas: revisar etapas 1c-4"

    folds = split_purgado(t, n_splits=5, purga="12h", embargo="12h")
    assert len(folds) == 5
    assert all(len(te) == 57 for _, te in folds)           # 342 // 6

    tam_previo = -1
    for k, (tr, te) in enumerate(folds):
        assert set(tr).isdisjoint(set(te)), f"fold {k}"
        assert t.iloc[tr].max() < t.iloc[te].min() - PURGA, f"fold {k}: hueco de 12 h no respetado"
        assert len(tr) > tam_previo, f"fold {k}: la ventana no es expansiva"
        tam_previo = len(tr)

    # Cifra que va al documento: cuanto quita realmente la purga.
    # El muestreo es ralo (mediana ~2.8 d) y solo hay 1 intervalo < 12 h en las 342
    # filas, de modo que la purga es metodologicamente necesaria pero casi inerte
    # en volumen. La caida de R2 de B2 vendra del orden temporal y de los folds
    # compartidos (doc, 2.4), no de la purga.
    purgadas = [int(te.min()) - len(tr) for tr, te in folds]
    assert purgadas == [0, 0, 0, 0, 0], f"cambio el reparto de purga del canonico: {purgadas}"
