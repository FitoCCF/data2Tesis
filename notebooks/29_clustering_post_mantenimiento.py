#!/usr/bin/env python3
# ============================================================
# CELDA 29 — AGRUPAMIENTO NO SUPERVISADO, UN SOLO ESTADO DE INSTRUMENTO
# ============================================================
# El mantenimiento de noviembre 2025 (confirmado con planta) cambió la
# respuesta física del courier: después se rehicieron las ecuaciones del
# Outocal con una campaña de muestreo nueva. Las intensidades de antes y
# después NO son la misma medición, así que agrupar atravesando esa fecha es
# agrupar sobre una discontinuidad -- y el notebook 28 mostró que eso es
# exactamente lo que domina el k=2 sobre los 15 meses.
#
# Acá se agrupa solo con lo posterior: 2025-12-15 -> 2026-09-30. El escalón
# está completo para el 2025-12-14 (mediana semanal de n6sc 1780 -> 1436 ->
# 1377 -> 1326 y estable después); el "cambio de n2cu en enero" que detectó
# el notebook 28 era ruido semanal.
#
# Features: las 5 intensidades CRUDAS, sin corrección de ningún tipo. Solo
# StandardScaler, ajustado en el train de la ventana.
#
# Validación externa por DOS vías, porque la BD queda corta en esta ventana:
#   BD        ~60 muestras con ley
#   COMPOSITO ~570 ventanas de 12 h con fe/cu/ins/mo del laboratorio
# En las dos, eta2 contra control aleatorio del mismo tamaño de grupo.
#
#   pixi run python notebooks/29_clustering_post_mantenimiento.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, SpectralClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, davies_bouldin_score,
                             adjusted_rand_score, adjusted_mutual_info_score)

from src.pipeline.config import (LIMPIO, LIMPIO_COMPOSITO, CANALES, METALES,
                                 FECHA_CORTE_TRAIN, VENTANA_COMPOSITO_H)

import argparse

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--desde", default="2025-12-15", help="inicio de la ventana (una sola epoca de instrumento)")
_ap.add_argument("--hasta", default=None, help="fin de la ventana (exclusivo)")
_ap.add_argument("--etiqueta", default=None, help="nombre para el CSV de salida")
_args, _ = _ap.parse_known_args()
INICIO, FIN = _args.desde, _args.hasta
ETIQ = _args.etiqueta or INICIO.replace("-", "")
SEMILLA = 42
KS = range(2, 9)
LEYES_BD = ["pFe", "pCu", "pZn", "pMo", "pIns"]
QUIM_COMP = ["fe", "cu", "ins", "mo"]
N_SIL, N_JERARQ = 5000, 6000
rng = np.random.default_rng(SEMILLA)
pd.set_option("display.width", 230)


# ============================================================
# Datos: una sola época de instrumento
# ============================================================
df = pd.read_csv(LIMPIO, parse_dates=["ts"])
df = df[df["valida"] & (df["ts"] >= INICIO)]
if FIN:
    df = df[df["ts"] < FIN]
df = df.sort_values("ts").reset_index(drop=True)
tr = (df["periodo"] == "train").to_numpy()
esc = StandardScaler().fit(df.loc[tr, CANALES])
X = esc.transform(df[CANALES])
Xtr = X[tr]
df["logSumI"] = np.log(df[METALES].sum(axis=1))
for a, i in [("fe", 1), ("zn", 3), ("mo", 4)]:
    df[f"lr_{a}_cu"] = np.log(df[f"n{i}{a}"] / df["n2cu"])
LR = ["lr_fe_cu", "lr_zn_cu", "lr_mo_cu"]
mes = df["ts"].dt.to_period("M").astype(str).to_numpy()

print(f"ventana {INICIO} -> {df['ts'].max().date()}  |  lecturas {len(df)}  train {tr.sum()}")
print(f"(train = periodo<=FECHA_CORTE_TRAIN={FECHA_CORTE_TRAIN}; en una ventana enteramente anterior, train = todo)")
print(f"features {CANALES} (CRUDAS, sin correccion)")
print(f"muestras BD con ley en la ventana: {int(df['valida_ley'].sum())}")

comp = pd.read_csv(LIMPIO_COMPOSITO, parse_dates=["ts"])
medio = pd.Timedelta(hours=VENTANA_COMPOSITO_H / 2)
comp = comp[(comp["ts"] >= df["ts"].min() + medio) & (comp["ts"] <= df["ts"].max() - medio)]
print(f"compositos con ventana cubierta: {len(comp)}\n")


def submuestra(A, n):
    if len(A) <= n:
        return A, np.arange(len(A))
    i = rng.choice(len(A), n, replace=False)
    return A[i], i


_cache = {}


_mse_ae = {}


def ae_latente(seed=SEMILLA):
    if seed not in _cache:
        red = MLPRegressor(hidden_layer_sizes=(3, 2, 3), activation="tanh", solver="adam",
                           max_iter=400, random_state=seed, early_stopping=True).fit(Xtr, Xtr)
        A = X
        for W, b in zip(red.coefs_[:2], red.intercepts_[:2]):
            A = np.tanh(A @ W + b)
        _cache[seed] = A
        _mse_ae[seed] = float(np.mean((red.predict(Xtr) - Xtr) ** 2))
    return _cache[seed]


def som_codebook(filas=6, cols=6, epocas=12):
    if "som" in _cache:
        return _cache["som"]
    cod = Xtr[rng.choice(len(Xtr), filas * cols, replace=False)].copy()
    rej = np.array([[i, j] for i in range(filas) for j in range(cols)], float)
    d = ((rej[:, None, :] - rej[None]) ** 2).sum(-1)
    for e in range(epocas):
        s2 = (max(filas, cols) / 2 * (1 - e / epocas)) ** 2 + 1e-3
        vec = np.argmin(((Xtr[:, None, :] - cod[None]) ** 2).sum(-1), axis=1)
        W = np.exp(-d / (2 * s2))[vec]
        cod = (W.T @ Xtr) / np.maximum(W.sum(axis=0)[:, None], 1e-9)
    _cache["som"] = cod
    return cod


def por_centroide(A, lab, k):
    cen = np.vstack([A[lab == j].mean(axis=0) for j in range(k)])
    return np.argmin(((X[:, None, :] - cen[None]) ** 2).sum(-1), axis=1)


def etiquetar(nombre, k):
    if nombre == "KMeans":
        return KMeans(k, n_init=10, random_state=SEMILLA).fit(Xtr).predict(X)
    if nombre == "GMM":
        return GaussianMixture(k, covariance_type="full", random_state=SEMILLA).fit(Xtr).predict(X)
    if nombre == "Birch":
        return Birch(n_clusters=k).fit(Xtr).predict(X)
    if nombre == "Ward":
        A, _ = submuestra(Xtr, N_JERARQ)
        return por_centroide(A, AgglomerativeClustering(k, linkage="ward").fit_predict(A), k)
    if nombre == "Spectral":
        A, _ = submuestra(Xtr, N_JERARQ)
        lab = SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=10,
                                 random_state=SEMILLA).fit_predict(A)
        return por_centroide(A, lab, k)
    if nombre == "AE+KMeans":
        L = ae_latente()
        return KMeans(k, n_init=10, random_state=SEMILLA).fit(L[tr]).predict(L)
    if nombre == "SOM+KMeans":
        cod = som_codebook()
        nodo = np.argmin(((X[:, None, :] - cod[None]) ** 2).sum(-1), axis=1)
        return KMeans(k, n_init=10, random_state=SEMILLA).fit(cod).labels_[nodo]
    raise ValueError(nombre)


def estabilidad(nombre, k):
    idx = rng.permutation(len(Xtr)); h = len(idx) // 2
    labs = []
    for A in (Xtr[idx[:h]], Xtr[idx[h:]]):
        if nombre in ("KMeans", "AE+KMeans", "SOM+KMeans"):
            labs.append(KMeans(k, n_init=10, random_state=SEMILLA).fit(A).predict(Xtr))
        elif nombre == "GMM":
            labs.append(GaussianMixture(k, covariance_type="full",
                                        random_state=SEMILLA).fit(A).predict(Xtr))
        elif nombre == "Birch":
            labs.append(Birch(n_clusters=k).fit(A).predict(Xtr))
        else:
            B, _ = submuestra(A, N_JERARQ // 2)
            lab = (AgglomerativeClustering(k, linkage="ward").fit_predict(B) if nombre == "Ward"
                   else SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=10,
                                           random_state=SEMILLA).fit_predict(B))
            cen = np.vstack([B[lab == j].mean(axis=0) for j in range(k)])
            labs.append(np.argmin(((Xtr[:, None, :] - cen[None]) ** 2).sum(-1), axis=1))
    return adjusted_rand_score(labs[0], labs[1])


# ============================================================
# Validación externa: eta2 + control aleatorio del mismo tamaño
# ============================================================
def _eta2(H, n, g):
    return max((H - g + 1) / (n - g), 0.0)


def eta2_tabla(grupo, tabla, cols, min_n=5):
    """eta2 de Kruskal-Wallis por columna de 'tabla', agrupando por 'grupo'."""
    out = {}
    for c in cols:
        s = pd.DataFrame({"g": grupo, "y": tabla[c].to_numpy()}).dropna()
        gr = [v["y"].to_numpy() for _, v in s.groupby("g") if len(v) >= min_n]
        out[c] = _eta2(stats.kruskal(*gr)[0], len(s), len(gr)) if len(gr) >= 2 else np.nan
    return out


def control(grupo, tabla, cols, n_rep=200):
    """p95 del eta2 medio bajo particiones aleatorias con los mismos tamaños."""
    tam = np.bincount(grupo)
    res = []
    for _ in range(n_rep):
        p = np.concatenate([np.full(t, j) for j, t in enumerate(tam)])
        rng.shuffle(p)
        v = [x for x in eta2_tabla(p, tabla, cols).values() if not np.isnan(x)]
        res.append(np.mean(v) if v else np.nan)
    return np.nanpercentile(res, 95)


# grupo dominante del compósito: moda de las etiquetas en su ventana centrada
ts_lect = df["ts"].to_numpy()
idx_vent = [np.where((ts_lect >= t - medio.to_timedelta64()) &
                     (ts_lect <= t + medio.to_timedelta64()))[0]
            for t in comp["ts"].to_numpy()]
ok_vent = np.array([len(i) >= 10 for i in idx_vent])
comp_ok = comp[ok_vent].reset_index(drop=True)
idx_ok = [i for i, k in zip(idx_vent, ok_vent) if k]
print(f"compositos con >=10 lecturas en su ventana: {len(comp_ok)}\n")


def grupo_composito(l):
    return np.array([np.bincount(l[i]).argmax() for i in idx_ok])


# ============================================================
# 1. Métricas internas, estabilidad, época y validación externa
# ============================================================
print("=" * 118)
print("1. TODOS LOS ALGORITMOS — una sola época de instrumento")
print("=" * 118)
Asil, isil = submuestra(Xtr, N_SIL)
filas, etiquetas = [], {}
for nombre in ["KMeans", "GMM", "Birch", "Ward", "Spectral", "AE+KMeans", "SOM+KMeans"]:
    for k in KS:
        l = etiquetar(nombre, k)
        etiquetas[(nombre, k)] = l
        ltr = l[tr]
        if len(np.unique(ltr[isil])) < 2:
            continue
        gc = grupo_composito(l)
        e_bd = eta2_tabla(l[df["valida_ley"].to_numpy()], df[df["valida_ley"]], LEYES_BD)
        e_co = eta2_tabla(gc, comp_ok, QUIM_COMP)
        filas.append({
            "algoritmo": nombre, "k": k,
            "silhouette": silhouette_score(Asil, ltr[isil]),
            "calinski": calinski_harabasz_score(Asil, ltr[isil]),
            "davies_b": davies_bouldin_score(Asil, ltr[isil]),
            "estab_ari": estabilidad(nombre, k),
            "min_pct": 100 * np.bincount(ltr).min() / len(ltr),
            "AMI_mes": adjusted_mutual_info_score(mes, l),
            "dlogSumI": (pd.Series(df["logSumI"]).groupby(l).mean().max()
                         - pd.Series(df["logSumI"]).groupby(l).mean().min()),
            "eta2_bd": np.nanmean(list(e_bd.values())),
            "eta2_comp": np.nanmean(list(e_co.values())),
            **{f"c_{c}": e_co[c] for c in QUIM_COMP}})
M = pd.DataFrame(filas)
print(M[["algoritmo", "k", "silhouette", "estab_ari", "min_pct", "AMI_mes", "dlogSumI",
         "eta2_bd", "eta2_comp"]].to_string(index=False, float_format=lambda v: f"{v:9.3f}"))

print("\n  -- HDBSCAN (sin k) --")
for mcs in (100, 300, 1000):
    hb = HDBSCAN(min_cluster_size=mcs).fit(Xtr)
    l = hb.labels_
    n_gr = len(set(l)) - (1 if -1 in l else 0)
    sil = (silhouette_score(Xtr[l != -1], l[l != -1], sample_size=min(N_SIL, int((l != -1).sum())),
                            random_state=SEMILLA) if n_gr > 1 else np.nan)
    print(f"  min_cluster_size={mcs:5d}: grupos={n_gr:3d}  ruido={100*(l==-1).mean():5.1f}%  sil={sil:.3f}")

print("\n" + "=" * 118)
print("2. CONTROL ALEATORIO sobre el compósito (200 particiones del mismo tamaño)")
print("=" * 118)
print(f"{'algoritmo':<12}{'k':>3}{'eta2_comp':>11}{'p95_azar':>10}{'pasa?':>7}"
      + "".join(f"{'c_'+c:>9}" for c in QUIM_COMP))
for nombre in ["KMeans", "GMM", "AE+KMeans"]:
    for k in [2, 3, 4, 6, 8]:
        if (nombre, k) not in etiquetas:
            continue
        gc = grupo_composito(etiquetas[(nombre, k)])
        e = eta2_tabla(gc, comp_ok, QUIM_COMP)
        em = np.nanmean(list(e.values()))
        p95 = control(gc, comp_ok, QUIM_COMP)
        print(f"{nombre:<12}{k:>3}{em:11.3f}{p95:10.3f}{'SI' if em > p95 else 'no':>7}"
              + "".join(f"{e[c]:9.3f}" for c in QUIM_COMP))

print("\n" + "=" * 118)
print("3. ¿MAGNITUD o COMPOSICIÓN? sd global: "
      f"logSumI {df['logSumI'].std():.3f} | " + " ".join(f"{c} {df[c].std():.3f}" for c in LR))
print("=" * 118)
print(f"{'algoritmo':<12}{'k':>3}{'rango_logSumI':>14}" + "".join(f"{'rango_'+c[3:]:>13}" for c in LR))
for nombre in ["KMeans", "GMM", "AE+KMeans"]:
    for k in [2, 4, 6]:
        l = etiquetas[(nombre, k)]
        r = [pd.Series(df[c]).groupby(l).mean() for c in ["logSumI"] + LR]
        print(f"{nombre:<12}{k:>3}" + "".join(f"{(s.max()-s.min()):>13.3f}  " for s in r))

M.to_csv(f"reports/clustering_{ETIQ}.csv", index=False)
print(f"\ntabla completa -> reports/clustering_{ETIQ}.csv")


# ============================================================
# 4. ¿GRADIENTE o GRUPOS? la prueba que decide
# ============================================================
# Si lo que hay es un gradiente, cualquier corte correlaciona con él pero los
# cortes no concuerdan entre sí, y la variable CONTINUA le gana a cualquier
# etiqueta discreta. Se compara el R2 de `lr_mo_cu` promediado en la ventana
# del compósito contra el R2 de la etiqueta tratada como factor (ANOVA).
print("\n" + "=" * 118)
print("4. GRADIENTE vs GRUPOS — variable continua contra etiqueta discreta")
print("=" * 118)
for a, i in [("fe", 1), ("zn", 3)]:
    pass
cont = {}
for c in LR:
    cont[c] = np.array([df[c].to_numpy()[j].mean() for j in idx_ok])
print("  variable continua (promediada en la ventana del compósito) vs química del lab:")
for c in LR:
    for q in QUIM_COMP:
        r = np.corrcoef(cont[c], comp_ok[q])[0, 1]
        if abs(r) > 0.25:
            rho = stats.spearmanr(cont[c], comp_ok[q])
            print(f"    {c:10s} vs {q:4s}: corr {r:+.3f}  R2 {r**2:.3f}  "
                  f"Spearman {rho.statistic:+.3f} (p={rho.pvalue:.1e})")
r_mo = np.corrcoef(cont["lr_mo_cu"], comp_ok["mo"])[0, 1]
print(f"\n  >> CONTINUA  lr_mo_cu vs mo:  R2 = {r_mo**2:.3f}   (n={len(comp_ok)})")
print("  >> DISCRETA, mejor semilla del AE, mismo Mo y mismas ventanas:")
L = ae_latente()
for k in [2, 3, 4, 6, 8]:
    l = KMeans(k, n_init=10, random_state=SEMILLA).fit(L[tr]).predict(L)
    gc = grupo_composito(l)
    s = pd.DataFrame({"g": gc, "y": comp_ok["mo"].to_numpy()})
    gm = s.groupby("g")["y"].transform("mean")
    r2 = 1 - ((s["y"] - gm) ** 2).sum() / ((s["y"] - s["y"].mean()) ** 2).sum()
    print(f"       k={k}: R2_factor = {r2:.3f}   {'(no supera a la continua)' if r2 < r_mo**2 else '(SUPERA)'}")


# ============================================================
# 5. ESTABILIDAD DEL AUTOENCODER entre semillas
# ============================================================
# El latente de un AE no está identificado: los ejes rotan y permutan según la
# inicialización. Lo que hay que comprobar es si la PARTICIÓN se sostiene.
print("\n" + "=" * 118)
print("5. ESTABILIDAD DEL AE ENTRE SEMILLAS (lo que puede invalidar todo lo anterior)")
print("=" * 118)
SEMILLAS = [42, 1, 7, 13, 99, 21]
for k in [2, 3, 4]:
    labs, filas = {}, []
    for sd in SEMILLAS:
        L = ae_latente(sd)
        l = KMeans(k, n_init=10, random_state=SEMILLA).fit(L[tr]).predict(L)
        labs[sd] = l
        gc = grupo_composito(l)
        filas.append({"semilla": sd, "MSE": _mse_ae[sd],
                      "c_mo": eta2_tabla(gc, comp_ok, ["mo"])["mo"],
                      "AMI_mes": adjusted_mutual_info_score(mes, l),
                      "rango_logSumI": (pd.Series(df["logSumI"]).groupby(l).mean().max()
                                        - pd.Series(df["logSumI"]).groupby(l).mean().min())})
    T = pd.DataFrame(filas)
    pares = [adjusted_rand_score(labs[a], labs[b])
             for i, a in enumerate(SEMILLAS) for b in SEMILLAS[i + 1:]]
    print(f"\n  k={k}:  ARI entre semillas mediana {np.median(pares):+.3f} "
          f"(min {min(pares):+.3f}, max {max(pares):+.3f})")
    print(f"        c_mo mediana {T['c_mo'].median():.3f}  rango {T['c_mo'].min():.3f}-{T['c_mo'].max():.3f}")
    print("        " + T.to_string(index=False, float_format=lambda v: f"{v:7.3f}").replace("\n", "\n        "))
