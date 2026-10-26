#!/usr/bin/env python3
# ============================================================
# scripts/cabeza_tipo_combinado.py — ¿se puede reconocer el tipo de cabeza en línea,
#                                    y sirve para el concentrado?
# ============================================================
# Sigue a §23.2 (rama AlimConc). Allí los locales por tipo de cabeza ganaban
# con el tipo REAL del lab (Fe +0.063, Mo +0.032) pero no con el tipo
# predicho desde intensidades (acierto 0.73). El tipo se repite de un
# compósito al siguiente el 78 % de las veces: pista gratis.
#
# PARTE 1 — cabeza. Reconocer el tipo con:
#   I     intensidades (logística, como §23.2)
#   P     último tipo PUBLICADO por el lab (compósito con ts <= t - RETRASO)
#   I+P   ambas en la misma logística
#   y ruteo DURO (un local) o SUAVE (mezcla de locales por probabilidad).
#   Además: tipos definidos DESDE las intensidades (GMM k=3 en train), que se
#   reconocen por construcción. Locales entrenados con el tipo del lab en
#   train (o el del GMM). Control: 200 repeticiones con las etiquetas de train
#   y las probabilidades de test barajadas; adopta si mejora > p95 y >= 0.01.
#
# PARTE 2 — concentrado (idea a, §16.5-2). Base = <ley>_kal (lo que da el
# monitor). Sobre los compósitos del concentrado de la era con cabeza:
#   a) selector: y ~ kal + probabilidades del tipo de cabeza (offset por tipo)
#      contra y ~ kal; control: probabilidades corridas >= 14 días en el tiempo.
#   b) aviso: ¿el error del concentrado es mayor cuando el tipo de cabeza
#      cambió respecto del último publicado? control: misma bandera corrida.
#
#   pixi run python scripts/cabeza_tipo_combinado.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.pipeline.config import DATA_FINAL, RANDOM_STATE
from src.pipeline import tipos_mineral as tm
from scripts.cabeza_intensidades import FEATS
from scripts.cabeza_ley_en_linea import tabla_compositos, _ridge, r2, LEYES, FIN_TRAIN_CABEZA

RETRASOS = {"24h": pd.Timedelta("24h"), "12h": pd.Timedelta("12h")}   # cuándo está publicado el lab
N_CONTROL, MEJORA_MIN, MIN_N_LOCAL, SEP_CONTROL = 200, 0.01, 30, 28


# ---------------------------------------------------------------- parte 1
def tipo_publicado(A, retraso):
    """Último tipo del lab con ts <= t - retraso (serie completa de compósitos de cabeza)."""
    d = tm.cargar("cabeza")
    sc, gmm, nombres = tm.ajustar(d, "cabeza", tm.K_POR_STREAM["cabeza"])
    lab = tm.asignar(d, "cabeza", sc, gmm, nombres)[["ts", "tipo_cabeza"]].rename(columns={"tipo_cabeza": "prev"})
    q = A[["ts"]].assign(t_pub=A["ts"] - retraso).sort_values("t_pub")
    m = pd.merge_asof(q, lab.sort_values("ts"), left_on="t_pub", right_on="ts", suffixes=("", "_lab"))
    return m.set_index("ts")["prev"].reindex(A["ts"]).to_numpy()


def probas(A, tr, tipos, X):
    """Logística en train sobre X -> probabilidades (columnas en el orden de 'tipos')."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X[tr], A.loc[tr, "tipo_cabeza"])
    P = pd.DataFrame(clf.predict_proba(X), columns=clf.classes_)
    return P.reindex(columns=tipos, fill_value=0.0).to_numpy()


def locales(A, tr, etiqueta, G):
    """{tipo -> {ley -> modelo}}; tipo con pocos compósitos en train usa el global."""
    M = {}
    for t in np.unique(etiqueta):
        s = tr & (etiqueta == t)
        M[t] = ({l: _ridge().fit(A.loc[s, FEATS], A.loc[s, l]) for l in LEYES} if s.sum() >= MIN_N_LOCAL else G)
    return M


def mezclar(A, idx, M, tipos, P, duro):
    W = P[idx]
    if duro:
        W = np.eye(len(tipos))[W.argmax(axis=1)]
    X = A.loc[idx, FEATS]
    return {l: sum(W[:, k] * M[t][l].predict(X) for k, t in enumerate(tipos)) for l in LEYES}


def evaluar_variante(A, G, etiqueta, tipos, P, duro, rng):
    tr, te = (A["periodo"] == "train").to_numpy(), (A["periodo"] == "test").to_numpy()
    pg = {l: G[l].predict(A.loc[te, FEATS]) for l in LEYES}
    pl = mezclar(A, te, locales(A, tr, etiqueta, G), tipos, P, duro)
    ctrl = {l: [] for l in LEYES}
    for _ in range(N_CONTROL):
        e = etiqueta.copy()
        e[tr] = rng.permutation(e[tr])
        Pc = P.copy()
        Pc[te] = Pc[te][rng.permutation(te.sum())]
        pc = mezclar(A, te, locales(A, tr, e, G), tipos, Pc, duro)
        for l in LEYES:
            ctrl[l].append(r2(A.loc[te, l], pc[l]) - r2(A.loc[te, l], pg[l]))
    out = {}
    for l in LEYES:
        mej = r2(A.loc[te, l], pl[l]) - r2(A.loc[te, l], pg[l])
        p95 = float(np.quantile(ctrl[l], 0.95))
        out[l] = f"{mej:+.3f} ({p95:.3f}){' ✔' if mej > p95 and mej >= MEJORA_MIN else ''}"
    return out


def parte1(A):
    tr, te = (A["periodo"] == "train").to_numpy(), (A["periodo"] == "test").to_numpy()
    tipos = sorted(A["tipo_cabeza"].unique())
    real = A["tipo_cabeza"].to_numpy()
    XI = A[FEATS].to_numpy()
    G = {l: _ridge().fit(A.loc[tr, FEATS], A.loc[tr, l]) for l in LEYES}
    rng = np.random.RandomState(RANDOM_STATE)

    print("== PARTE 1. Reconocer el tipo de cabeza en línea (test) ==")
    P_var = {"I (intensidades)": probas(A, tr, tipos, XI)}
    for nom, ret in RETRASOS.items():
        prev = tipo_publicado(A, ret)
        oh = (prev[:, None] == np.array(tipos)[None, :]).astype(float)
        P_var[f"P (lab publicado, {nom})"] = oh
        P_var[f"I+P ({nom})"] = probas(A, tr, tipos, np.column_stack([XI, oh]))
    P_var["oráculo (tipo real)"] = (real[:, None] == np.array(tipos)[None, :]).astype(float)
    mayor = pd.Series(real[te]).value_counts(normalize=True).iloc[0]
    filas = []
    for nom, P in P_var.items():
        acierto = (np.array(tipos)[P[te].argmax(axis=1)] == real[te]).mean()
        for duro in (True, False):
            if duro is False and nom.startswith(("P ", "oráculo")):
                continue                                    # one-hot: duro = suave
            filas.append({"pista": nom, "ruteo": "duro" if duro else "suave", "acierto": round(acierto, 2),
                          **evaluar_variante(A, G, real, tipos, P, duro, rng)})
    print(f"(clase mayoritaria {mayor:.2f}); celdas = mejora de R² vs global (p95 control), ✔ = pasa")
    print(pd.DataFrame(filas).to_string(index=False))

    # tipos definidos desde las intensidades: se reconocen por construcción
    sc = StandardScaler().fit(XI[tr])
    gmm = GaussianMixture(3, covariance_type="full", random_state=RANDOM_STATE, n_init=10).fit(sc.transform(XI[tr]))
    Pg = gmm.predict_proba(sc.transform(XI))
    eg = Pg.argmax(axis=1)
    print("\nTipos desde intensidades (GMM k=3 en train), vs tipo del lab (filas):")
    print(pd.crosstab(A["tipo_cabeza"], eg).to_string())
    filas = [{"pista": "GMM intensidades", "ruteo": "duro" if d else "suave",
              **evaluar_variante(A, G, eg, list(range(3)), Pg, d, rng)} for d in (True, False)]
    print(pd.DataFrame(filas).to_string(index=False))
    return P_var, tipos


# ---------------------------------------------------------------- parte 2
def r2_fit(Xtr, ytr, Xte, yte):
    return r2(yte, LinearRegression().fit(Xtr, ytr).predict(Xte))


def parte2(A, P_var, tipos):
    print("\n== PARTE 2. Tipo de cabeza en el CONCENTRADO (idea a) ==")
    ev = pd.read_csv(DATA_FINAL / "eval_composito.csv", parse_dates=["ts"])
    C = ev.merge(A[["ts", "periodo"]].rename(columns={"periodo": "per_cab"}), on="ts")
    C = C.dropna(subset=["pCu", "pFe", "pMo"]).reset_index(drop=True)
    pos = A.set_index("ts").index.get_indexer(C["ts"])
    tr, te = (C["per_cab"] == "train").to_numpy(), (C["per_cab"] == "test").to_numpy()
    print(f"compósitos del concentrado con cabeza: train {tr.sum()}, test {te.sum()}")

    filas = []
    for nom in ("I+P (24h)", "I (intensidades)", "oráculo (tipo real)"):
        P = P_var[nom][pos][:, 1:]                           # k-1 columnas (una es la referencia)
        for ley in ("pCu", "pFe", "pMo"):
            k, y = C[f"{ley}_kal"].to_numpy()[:, None], C[ley].to_numpy()
            base = r2_fit(k[tr], y[tr], k[te], y[te])
            con = r2_fit(np.hstack([k, P])[tr], y[tr], np.hstack([k, P])[te], y[te])
            n = len(C)
            ctrl = [r2_fit(np.hstack([k, np.roll(P, s, 0)])[tr], y[tr], np.hstack([k, np.roll(P, s, 0)])[te], y[te]) - base
                    for s in np.linspace(SEP_CONTROL, n - SEP_CONTROL, N_CONTROL).astype(int)]
            mej, p95 = con - base, float(np.quantile(ctrl, 0.95))
            filas.append({"tipo": nom, "ley": ley, "R2_kal_recal": round(base, 3), "R2_+tipo": round(con, 3),
                          "mejora": round(mej, 3), "p95_control": round(p95, 3),
                          "veredicto": "APORTA" if mej > p95 and mej >= MEJORA_MIN else "no"})
    print("a) selector (offset por tipo sobre el Kalman), test:")
    print(pd.DataFrame(filas).to_string(index=False))

    # b) aviso: tipo predicho ahora (I) distinto del último publicado (24h)
    tipo_I = np.array(tipos)[P_var["I (intensidades)"][pos].argmax(axis=1)]
    prev = np.array(tipos)[P_var["P (lab publicado, 24h)"][pos].argmax(axis=1)]
    cambio = tipo_I != prev
    print(f"\nb) aviso de cambio de mineral (tipo por intensidades != último publicado): "
          f"{cambio.mean():.0%} de los compósitos")
    filas = []
    for ley in ("pCu", "pFe", "pMo"):
        err = np.abs(C[f"{ley}_kal"] - C[ley]).to_numpy()
        razon = err[cambio].mean() / err[~cambio].mean()
        ctrl = [err[np.roll(cambio, s)].mean() / err[~np.roll(cambio, s)].mean()
                for s in range(SEP_CONTROL, len(C) - SEP_CONTROL)]
        filas.append({"ley": ley, "|err| con aviso": round(err[cambio].mean(), 3),
                      "|err| sin aviso": round(err[~cambio].mean(), 3), "razón": round(razon, 2),
                      "p95_control": round(float(np.quantile(ctrl, 0.95)), 2),
                      "veredicto": "AVISA" if razon > np.quantile(ctrl, 0.95) else "no"})
    print(pd.DataFrame(filas).to_string(index=False))
    u = C.assign(aviso=cambio, tipo_I=tipo_I, prev=prev)
    print("\nÚltimos compósitos (aviso, tipo ahora / publicado, error Cu del Kalman):")
    print(u.assign(err_cu=(u.pCu_kal - u.pCu).round(2))[["ts", "aviso", "tipo_I", "prev", "err_cu"]].tail(8)
          .to_string(index=False))


def _main():
    pd.set_option("display.width", 220)
    A, _ = tabla_compositos("L2")
    print(f"Cabeza L2: train {(A.periodo == 'train').sum()} / test {(A.periodo == 'test').sum()} compósitos "
          f"(corte {FIN_TRAIN_CABEZA})\n")
    P_var, tipos = parte1(A)
    parte2(A, P_var, tipos)


if __name__ == "__main__":
    _main()
