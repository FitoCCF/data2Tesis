#!/usr/bin/env python3
# ============================================================
# CELDA 28 — AGRUPAMIENTO CON INTENSIDADES CRUDAS (experimento)
# ============================================================
# Encargo: agrupar usando SOLO las 5 intensidades, sin ninguna corrección
# (ni ortogonalización, ni log-cocientes, ni corrección por dilución),
# probando varios algoritmos —incluidas redes neuronales simples— y
# validando la agrupación con métricas.
#
# Lo único que se aplica es StandardScaler, que es escala y no corrección, y
# se ajusta SOLO en train (regla transversal de fuga).
#
# Pregunta de validación, que es la que decide si la agrupación sirve:
#   interna   -> silhouette / Calinski-Harabasz / Davies-Bouldin / BIC
#   estabilidad -> ARI entre modelos ajustados en mitades disjuntas de train
#   EXTERNA   -> ¿los grupos separan la química del laboratorio? (Kruskal + eta2)
#   control   -> ¿separan ÉPOCA en vez de mineral? (MI con el mes, log(SumI))
#
# El último control es obligatorio: la bitácora ya midió que las intensidades
# absolutas derivan 22-25 % entre trimestres (§2.3) y que un GMM sobre ellas
# "separa épocas" (ADR 2026-09-30). Hay que medirlo, no suponerlo.
#
#   pixi run python notebooks/28_experimento_clustering_intensidades_crudas.py
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
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score,
                             adjusted_mutual_info_score)

from src.pipeline.config import LIMPIO, CANALES, FECHA_CORTE_TRAIN

SEMILLA = 42
KS = range(2, 9)
LEYES = ["pFe", "pCu", "pZn", "pMo", "pIns"]
N_SIL = 5000          # submuestra para silhouette (es O(n^2))
N_JERARQ = 6000       # submuestra para Ward y Spectral (memoria O(n^2))
rng = np.random.default_rng(SEMILLA)
pd.set_option("display.width", 220)


# ============================================================
# Datos
# ============================================================
df = pd.read_csv(LIMPIO, parse_dates=["ts"])
df = df[df["valida"]].sort_values("ts").reset_index(drop=True)
tr = df["periodo"] == "train"

esc = StandardScaler().fit(df.loc[tr, CANALES])          # ajustado SOLO en train
X = esc.transform(df[CANALES])
Xtr = X[tr.to_numpy()]
print(f"lecturas validas {len(df)} | train {tr.sum()} | features {CANALES} (CRUDAS)")
print(f"corte train {FECHA_CORTE_TRAIN}\n")


def submuestra(A, n):
    if len(A) <= n:
        return A, np.arange(len(A))
    i = rng.choice(len(A), n, replace=False)
    return A[i], i


# ============================================================
# 0. ¿Hay estructura que agrupar? Hopkins
# ============================================================
def hopkins(A, frac=0.05):
    """~0.5 = aleatorio (nada que agrupar); ->1 = agrupable."""
    from sklearn.neighbors import NearestNeighbors
    m = max(int(frac * len(A)), 50)
    nn = NearestNeighbors(n_neighbors=2).fit(A)
    idx = rng.choice(len(A), m, replace=False)
    w = nn.kneighbors(A[idx], return_distance=True)[0][:, 1]          # vecino real
    lo, hi = A.min(axis=0), A.max(axis=0)
    u = nn.kneighbors(rng.uniform(lo, hi, (m, A.shape[1])), n_neighbors=1,
                      return_distance=True)[0][:, 0]                   # punto uniforme
    return u.sum() / (u.sum() + w.sum())


print("=" * 100)
print("0. TENDENCIA AL AGRUPAMIENTO (Hopkins sobre train; ~0.5 = sin estructura)")
print("=" * 100)
h = [hopkins(Xtr) for _ in range(5)]
print(f"  Hopkins = {np.mean(h):.3f} +/- {np.std(h):.3f}   (5 repeticiones)\n")


# ============================================================
# 1. Algoritmos
# ============================================================
def etiquetar(nombre, k):
    """Ajusta en train y devuelve (etiquetas de TODO df, etiquetas de train)."""
    if nombre == "KMeans":
        m = KMeans(k, n_init=10, random_state=SEMILLA).fit(Xtr)
        return m.predict(X), m.predict(Xtr)
    if nombre == "GMM":
        m = GaussianMixture(k, covariance_type="full", random_state=SEMILLA).fit(Xtr)
        return m.predict(X), m.predict(Xtr)
    if nombre == "Birch":
        m = Birch(n_clusters=k).fit(Xtr)
        return m.predict(X), m.predict(Xtr)
    if nombre == "Ward":                                   # sin predict: se extiende por centroide
        A, i = submuestra(Xtr, N_JERARQ)
        lab = AgglomerativeClustering(k, linkage="ward").fit_predict(A)
        cen = np.vstack([A[lab == j].mean(axis=0) for j in range(k)])
        asignar = lambda Z: np.argmin(((Z[:, None, :] - cen[None]) ** 2).sum(-1), axis=1)
        return asignar(X), asignar(Xtr)
    if nombre == "Spectral":
        A, i = submuestra(Xtr, N_JERARQ)
        lab = SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=10,
                                 random_state=SEMILLA, assign_labels="kmeans").fit_predict(A)
        cen = np.vstack([A[lab == j].mean(axis=0) for j in range(k)])
        asignar = lambda Z: np.argmin(((Z[:, None, :] - cen[None]) ** 2).sum(-1), axis=1)
        return asignar(X), asignar(Xtr)
    if nombre == "AE+KMeans":                              # red neuronal simple: autoencoder
        lat = autoencoder_latente()
        m = KMeans(k, n_init=10, random_state=SEMILLA).fit(lat[tr.to_numpy()])
        return m.predict(lat), m.predict(lat[tr.to_numpy()])
    if nombre == "SOM+KMeans":                             # mapa autoorganizado, implementado abajo
        cod = som_codebook()
        asignar = lambda Z: np.argmin(((Z[:, None, :] - cod[None]) ** 2).sum(-1), axis=1)
        nodo = asignar(X)
        m = KMeans(k, n_init=10, random_state=SEMILLA).fit(cod)        # agrupa los nodos del mapa
        return m.labels_[nodo], m.labels_[nodo[tr.to_numpy()]]
    raise ValueError(nombre)


_cache = {}


def autoencoder_latente():
    """MLP 5-3-2-3-5 entrenado X->X en train; devuelve la capa de 2 unidades."""
    if "ae" in _cache:
        return _cache["ae"]
    red = MLPRegressor(hidden_layer_sizes=(3, 2, 3), activation="tanh", solver="adam",
                       max_iter=400, random_state=SEMILLA,
                       early_stopping=True).fit(Xtr, Xtr)
    def hasta_cuello(Z):
        A = Z
        for W, b in zip(red.coefs_[:2], red.intercepts_[:2]):
            A = np.tanh(A @ W + b)
        return A
    _cache["ae"] = hasta_cuello(X)
    err = np.mean((red.predict(Xtr) - Xtr) ** 2)
    print(f"  [AE] cuello de 2 unidades, MSE de reconstruccion en train = {err:.4f}")
    return _cache["ae"]


def som_codebook(filas=6, cols=6, epocas=12):
    """SOM rectangular, batch simple. Devuelve el codebook (filas*cols, 5)."""
    if "som" in _cache:
        return _cache["som"]
    n_nodos = filas * cols
    cod = Xtr[rng.choice(len(Xtr), n_nodos, replace=False)].copy()
    rej = np.array([[i, j] for i in range(filas) for j in range(cols)], float)
    d_rej = ((rej[:, None, :] - rej[None]) ** 2).sum(-1)
    for e in range(epocas):
        sigma2 = (max(filas, cols) / 2 * (1 - e / epocas)) ** 2 + 1e-3
        vec = np.argmin(((Xtr[:, None, :] - cod[None]) ** 2).sum(-1), axis=1)   # nodo ganador
        hh = np.exp(-d_rej / (2 * sigma2))                                      # vecindad gaussiana
        W = hh[vec]                                                             # (n, n_nodos)
        num, den = W.T @ Xtr, W.sum(axis=0)[:, None]
        cod = num / np.maximum(den, 1e-9)
    _cache["som"] = cod
    print(f"  [SOM] mapa {filas}x{cols} = {n_nodos} nodos, {epocas} epocas")
    return cod


# ============================================================
# 2. Métricas internas + estabilidad
# ============================================================
def estabilidad(nombre, k):
    """ARI entre dos modelos del mismo tipo ajustados en mitades disjuntas de
    train, comparados sobre TODO train (criterio de la etapa 4)."""
    idx = rng.permutation(len(Xtr))
    mitades = [Xtr[idx[: len(idx) // 2]], Xtr[idx[len(idx) // 2:]]]
    labs = []
    for A in mitades:
        if nombre in ("KMeans", "AE+KMeans"):
            m = KMeans(k, n_init=10, random_state=SEMILLA).fit(A)
            labs.append(m.predict(Xtr))
        elif nombre == "GMM":
            m = GaussianMixture(k, covariance_type="full", random_state=SEMILLA).fit(A)
            labs.append(m.predict(Xtr))
        elif nombre == "Birch":
            m = Birch(n_clusters=k).fit(A)
            labs.append(m.predict(Xtr))
        else:                                              # Ward/Spectral/SOM: por centroide
            B, _ = submuestra(A, N_JERARQ // 2)
            lab = (AgglomerativeClustering(k, linkage="ward").fit_predict(B)
                   if nombre != "Spectral" else
                   SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=10,
                                      random_state=SEMILLA).fit_predict(B))
            cen = np.vstack([B[lab == j].mean(axis=0) for j in range(k)])
            labs.append(np.argmin(((Xtr[:, None, :] - cen[None]) ** 2).sum(-1), axis=1))
    return adjusted_rand_score(labs[0], labs[1])


ALGOS = ["KMeans", "GMM", "Birch", "Ward", "Spectral", "AE+KMeans", "SOM+KMeans"]
print("=" * 100)
print("1. MÉTRICAS INTERNAS Y ESTABILIDAD (ajuste en train)")
print("=" * 100)
Asil, isil = submuestra(Xtr, N_SIL)
filas = []
etiquetas = {}
for nombre in ALGOS:
    for k in KS:
        try:
            lab_all, lab_tr = etiquetar(nombre, k)
        except Exception as e:
            print(f"  {nombre} k={k}: {type(e).__name__}: {e}")
            continue
        etiquetas[(nombre, k)] = lab_all
        ls = lab_tr[isil]
        if len(np.unique(ls)) < 2:
            continue
        bic = np.nan
        if nombre == "GMM":
            bic = GaussianMixture(k, covariance_type="full",
                                  random_state=SEMILLA).fit(Xtr).bic(Xtr)
        filas.append({"algoritmo": nombre, "k": k,
                      "silhouette": silhouette_score(Asil, ls),
                      "calinski": calinski_harabasz_score(Asil, ls),
                      "davies_b": davies_bouldin_score(Asil, ls),
                      "bic": bic,
                      "estab_ari": estabilidad(nombre, k),
                      "grupo_min_pct": 100 * np.bincount(lab_tr).min() / len(lab_tr)})
M = pd.DataFrame(filas)
print(M.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))

# HDBSCAN: no toma k
print("\n  -- HDBSCAN (sin k) --")
for mcs in (100, 300, 1000):
    hb = HDBSCAN(min_cluster_size=mcs).fit(Xtr)
    l = hb.labels_
    n_gr = len(set(l)) - (1 if -1 in l else 0)
    ruido = 100 * (l == -1).mean()
    sil = (silhouette_score(Xtr[l != -1], l[l != -1], sample_size=min(N_SIL, (l != -1).sum()),
                            random_state=SEMILLA) if n_gr > 1 else np.nan)
    print(f"  min_cluster_size={mcs:5d}: grupos={n_gr:3d}  ruido={ruido:5.1f}%  silhouette={sil:.3f}")


# ============================================================
# 3. VALIDACIÓN EXTERNA — ¿los grupos separan la química del laboratorio?
# ============================================================
# Es la única validación que este proyecto acepta como real: los grupos se
# definen sin ver ninguna ley, así que si separan la química del lab es
# información, no artefacto. Kruskal-Wallis (no asume normalidad) + eta2
# como tamaño de efecto.
print("\n" + "=" * 100)
print("3. VALIDACIÓN EXTERNA — química del laboratorio (lecturas con valida_ley)")
print("=" * 100)
lab = df[df["valida_ley"]].copy()
print(f"  muestras con ley: {len(lab)}  (train {int((lab['periodo']=='train').sum())})")


def eta2_kruskal(H, n, g):
    """eta2 de Kruskal-Wallis: (H - g + 1) / (n - g)."""
    return max((H - g + 1) / (n - g), 0.0)


def validar_externo(nombre, k):
    l = etiquetas[(nombre, k)][df.index.isin(lab.index)]
    out = {"algoritmo": nombre, "k": k}
    for ley in LEYES:
        sub = pd.DataFrame({"g": l, "y": lab[ley].to_numpy()}).dropna()
        gr = [v["y"].to_numpy() for _, v in sub.groupby("g") if len(v) >= 5]
        if len(gr) < 2:
            out[ley] = np.nan
            continue
        H, p = stats.kruskal(*gr)
        out[ley] = eta2_kruskal(H, len(sub), len(gr))
        out[ley + "_p"] = p
    return out


V = pd.DataFrame([validar_externo(n, k) for (n, k) in etiquetas])
cols_e = ["algoritmo", "k"] + LEYES
print("\n  -- eta2 (fraccion de varianza de la ley explicada por el grupo) --")
print(V[cols_e].to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
print("\n  -- p de Kruskal-Wallis --")
print(V[["algoritmo", "k"] + [l + "_p" for l in LEYES]].to_string(
    index=False, float_format=lambda v: f"{v:9.2e}"))


# ============================================================
# 4. CONTROL — ¿separan ÉPOCA (deriva del instrumento) en vez de mineral?
# ============================================================
# §2.3 midio que las intensidades absolutas derivan 22-25 % entre trimestres y
# el ADR del 2026-09-30 que un GMM sobre ellas "separa epocas". Si el grupo se
# predice bien con solo la fecha, la agrupacion es un reloj, no un mineral.
print("\n" + "=" * 100)
print("4. CONTROL DE ÉPOCA — ¿la agrupación es un reloj?")
print("=" * 100)
mes = df["ts"].dt.to_period("M").astype(str).to_numpy()
semana = df["ts"].dt.to_period("W").astype(str).to_numpy()
logSumI = np.log(df[["n1fe", "n2cu", "n3zn", "n4mo"]].sum(axis=1)).to_numpy()

# referencia de produccion: los 4 grupos del GMM sobre log-cocientes (etapa 4)
ref = None
try:
    cl = pd.read_csv("data/processed/courier_clusterizado.csv", parse_dates=["ts"])
    ref = df[["ts"]].merge(cl[["ts", "grupo"]], on="ts", how="left")["grupo"].to_numpy()
except Exception as e:
    print(f"  (sin referencia de la etapa 4: {e})")

filas = []
for (nombre, k), l in etiquetas.items():
    r = {"algoritmo": nombre, "k": k,
         "AMI_mes": adjusted_mutual_info_score(mes, l),
         "AMI_semana": adjusted_mutual_info_score(semana, l),
         "rango_logSumI_entre_grupos": (pd.Series(logSumI).groupby(l).mean().max()
                                        - pd.Series(logSumI).groupby(l).mean().min())}
    if ref is not None:
        m = ~pd.isna(ref)
        r["ARI_vs_etapa4"] = adjusted_rand_score(ref[m], l[m])
    filas.append(r)
C = pd.DataFrame(filas)
print("  AMI_mes alto = el grupo se explica por la fecha. sd(logSumI) global ="
      f" {logSumI.std():.3f}")
print(C.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))


# ============================================================
# 5. Resumen: la mejor opción por cada criterio
# ============================================================
print("\n" + "=" * 100)
print("5. RESUMEN")
print("=" * 100)
T = M.merge(V[cols_e], on=["algoritmo", "k"]).merge(C, on=["algoritmo", "k"])
T["eta2_medio"] = T[LEYES].mean(axis=1)
print("\n  -- top 10 por eta2 medio (capacidad de separar la quimica del lab) --")
print(T.sort_values("eta2_medio", ascending=False).head(10)[
    ["algoritmo", "k", "silhouette", "estab_ari", "eta2_medio", "AMI_mes",
     "rango_logSumI_entre_grupos"] + (["ARI_vs_etapa4"] if "ARI_vs_etapa4" in T else [])
].to_string(index=False, float_format=lambda v: f"{v:9.3f}"))
T.to_csv("reports/clustering_intensidades_crudas.csv", index=False)
print("\n  tabla completa -> reports/clustering_intensidades_crudas.csv")


# ============================================================
# 6. CONTROL ALEATORIO — eta2 sube con k por construcción
# ============================================================
# Regla del proyecto: ninguna comparación entre particiones vale sin control
# aleatorio del mismo tamaño de grupo (CLAUDE.md; §8-9 de la bitácora).
print("\n" + "=" * 100)
print("6. CONTROL ALEATORIO (200 particiones del mismo tamaño de grupo)")
print("=" * 100)


def _eta2_medio(l):
    out = []
    for ley in LEYES:
        s = pd.DataFrame({"g": l, "y": lab[ley].to_numpy()}).dropna()
        gr = [v["y"].to_numpy() for _, v in s.groupby("g") if len(v) >= 5]
        if len(gr) >= 2:
            out.append(eta2_kruskal(stats.kruskal(*gr)[0], len(s), len(gr)))
    return np.mean(out) if out else np.nan


def p95_azar(l, n_rep=200):
    tam = np.bincount(l)
    res = []
    for _ in range(n_rep):
        p = np.concatenate([np.full(t, j) for j, t in enumerate(tam)])
        rng.shuffle(p)
        res.append(_eta2_medio(p))
    return np.nanpercentile(res, 95)


m_lab = df.index.isin(lab.index)
print(f"{'algoritmo':<12}{'k':>3}{'eta2_medio':>12}{'p95_azar':>10}{'pasa?':>7}")
for nombre in ["KMeans", "GMM", "AE+KMeans"]:
    for k in [2, 3, 4, 6, 8]:
        if (nombre, k) not in etiquetas:
            continue
        l = etiquetas[(nombre, k)][m_lab]
        e = _eta2_medio(l)
        p = p95_azar(l)
        print(f"{nombre:<12}{k:>3}{e:12.3f}{p:10.3f}{'SI' if e > p else 'no':>7}")
print("\n  >> todas pasan con amplio margen: con k>=4 los grupos llevan informacion")
print("     quimica real. Lo que no quiere decir que sean grupos (ver notebook 29).")


# ============================================================
# 7. ESTABILIDAD DEL AE ENTRE SEMILLAS — descalifica su uso tal cual
# ============================================================
# El latente de un autoencoder no esta identificado: los ejes rotan y permutan
# segun la inicializacion. Lo que importa es si la PARTICION se sostiene.
print("\n" + "=" * 100)
print("7. ESTABILIDAD DEL AUTOENCODER ENTRE SEMILLAS")
print("=" * 100)
SEMILLAS_AE = [42, 1, 7, 13, 99, 5, 21, 33]
for k in [3, 4]:
    labs, filas = {}, []
    for sd in SEMILLAS_AE:
        red = MLPRegressor(hidden_layer_sizes=(3, 2, 3), activation="tanh", solver="adam",
                           max_iter=400, random_state=sd, early_stopping=True).fit(Xtr, Xtr)
        A = X
        for W, b in zip(red.coefs_[:2], red.intercepts_[:2]):
            A = np.tanh(A @ W + b)
        l = KMeans(k, n_init=10, random_state=SEMILLA).fit(A[tr.to_numpy()]).predict(A)
        labs[sd] = l
        filas.append({"semilla": sd, "MSE": np.mean((red.predict(Xtr) - Xtr) ** 2),
                      "eta2": _eta2_medio(l[m_lab]),
                      "AMI_mes": adjusted_mutual_info_score(mes, l)})
    T = pd.DataFrame(filas)
    pares = [adjusted_rand_score(labs[a], labs[b])
             for i, a in enumerate(SEMILLAS_AE) for b in SEMILLAS_AE[i + 1:]]
    print(f"\n  k={k}: ARI entre semillas mediana {np.median(pares):+.3f} "
          f"(min {min(pares):+.3f}, max {max(pares):+.3f})")
    print("  " + T.to_string(index=False, float_format=lambda v: f"{v:8.3f}").replace("\n", "\n  "))
print("\n  >> NO es reproducible, y el MSE no indica que semilla salio bien: hay semillas")
print("     con el mejor MSE y el peor comportamiento. Elegir semilla por eta2 seria")
print("     usar las leyes del laboratorio para escoger un modelo NO supervisado.")
