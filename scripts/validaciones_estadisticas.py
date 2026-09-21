# ============================================================
# src/evaluacion/validaciones_estadisticas.py  (COMPLETO — Capítulo 5)
# ============================================================
# Reproduce TODOS los cuadros y figuras del Capítulo 5 a partir de los CSV
# y los artefactos .joblib del pipeline. Cada bloque imprime los números de
# su cuadro y (cuando aplica) genera el PDF/PNG de su figura.
#
# CUADROS:
#   5.1  cleaning_results        limpieza de la señal
#   5.2  orthogonalization_pearson  correlación de Pearson con n6sc
#   5.3  breusch_pagan           homocedasticidad ortogonalización vs ratio
#   5.4  seleccion_k             BIC + silueta por K
#   5.5  gmm_full_tied           BIC full vs tied
#   5.6  hypothesis_tests        ANOVA / Kruskal-Wallis
#   5.7  cluster_profiles        perfil químico por clúster
#   5.8  global_local            R2 global vs local agregado
#   5.9  r2_por_cluster          R2 local por (ley, clúster) + enrutamiento
#   5.10 online_audit            desempeño fuera de muestra vs lab real
#   5.11 baseline_analizador     modelo vs ecuaciones del analizador
#   5.12 drift_comparison        drift entrenamiento vs validación (KS)
#
# FIGURAS:
#   fig_5_bic_silhouette, fig_5_global_vs_local, fig_5_r2_heatmap,
#   fig_5_scatter_bland, fig_5_serie_temporal
#
# Además: comparación de la VENTANA HISTÓRICA (2025) contra laboratorio real.
# ============================================================

import os
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, silhouette_score

import matplotlib
matplotlib.use('Agg')                                    # backend sin ventana (guarda a archivo)
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACIÓN (ajusta las rutas si tu estructura difiere)
# ============================================================
DIR_PROC = '../data/processed'
DIR_RAW = '../data/raw'                                    # datos procesados
DIR_MODELS = '../models'                                    # artefactos .joblib
DIR_FIG = '../reports'                                      # salida de figuras (se crea si no existe)

CSV_CLEAN = os.path.join(DIR_PROC, 'intensidad_cobre_24_clean.csv')            # crudas (5.1, 5.2, 5.3)
CSV_CLUSTER = os.path.join(DIR_PROC, 'intensidad_cobre_24_clusterizado.csv')   # _ortho escaladas (5.4-5.6)
CSV_SUPERV = os.path.join(DIR_PROC, 'intensidad_cobre_24_completo_filtrado.csv')  # supervisado (5.7-5.9)
CSV_LAB = os.path.join(DIR_RAW, 'assay_lab_raw.csv')                          # laboratorio real
CSV_ANALIZADOR = os.path.join(DIR_PROC, 'AssayLab_combined_limpio.csv')        # ecuaciones del analizador
CSV_STREAM = os.path.join(DIR_PROC, 'intensidades_cobre.csv')                  # stream histórico (ventana 2025)
CSV_COMP_LAB = os.path.join(DIR_PROC, 'comparacion_vs_lab_real.csv')           # comparación ya alineada (5.10, figs)

METALES = [('n1fe', 'Fe'), ('n2cu', 'Cu'), ('n3zn', 'Zn'), ('n4mo', 'Mo')]    # (columna cruda, nombre)
FC = ['n1fe_ortho', 'n2cu_ortho', 'n3zn_ortho', 'n4mo_ortho']                 # features del clustering
FEATS_REG = ['n1fe_ortho', 'n2cu_ortho', 'n3zn_ortho', 'n4mo_ortho', 'n6sc']  # features de la regresión
TARGETS = ['pFe', 'pCu', 'pMo', 'pZn']                                        # leyes a estimar
MODELO_POR_TARGET = {'pFe': 'Ridge', 'pCu': 'SVR', 'pMo': 'Ridge', 'pZn': 'PLS'}
RANDOM_STATE = 42

MESES = {'ene': '01', 'feb': '02', 'mar': '03', 'abr': '04', 'may': '05', 'jun': '06',
         'jul': '07', 'ago': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dic': '12'}


# ============================================================
# UTILIDADES
# ============================================================

def parse_es(s):
    """Convierte '01-ene-24 06:00:00' a Timestamp."""
    d, t = s.split(' ')
    dd, mm, yy = d.split('-')
    return pd.Timestamp(f'20{yy}-{MESES[mm]}-{dd} {t}')


def cargar_lab_real(ruta=CSV_LAB):
    """Carga el laboratorio real y deja un valor por cada 12 h."""
    lab = pd.read_csv(ruta)
    lab['ts'] = lab['date_time'].apply(parse_es)
    return lab.drop_duplicates(subset=['Cu', 'Mo', 'Fe'], keep='first').sort_values('ts')


def _z(v):
    """Estandariza un vector a media 0, desviación 1."""
    return (v - v.mean()) / v.std()


def bp_lm(y, X):
    """Estadístico LM de Breusch-Pagan (mayor = más heterocedástico)."""
    X1 = np.column_stack([np.ones(len(X)), X])            # diseño con intercepto
    resid = y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]  # residuos de y~X
    r2 = resid ** 2                                       # residuos al cuadrado
    pred = X1 @ np.linalg.lstsq(X1, r2, rcond=None)[0]    # auxiliar resid^2 ~ X
    R2 = 1 - ((r2 - pred) ** 2).sum() / ((r2 - r2.mean()) ** 2).sum()
    return len(y) * R2                                    # LM = n * R2


def hopkins(X, m_frac=0.05, seed=42):
    """Estadístico de Hopkins (H->1 fuerte tendencia de agrupamiento)."""
    n, d = X.shape
    m = max(1, int(m_frac * n))
    rng = np.random.RandomState(seed)
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)
    idx = rng.choice(n, m, replace=False)
    w = nbrs.kneighbors(X[idx])[0][:, 1]
    Y = rng.uniform(X.min(0), X.max(0), (m, d))
    u = NearestNeighbors(n_neighbors=1).fit(X).kneighbors(Y)[0][:, 0]
    return (u ** d).sum() / ((u ** d).sum() + (w ** d).sum())


def preparar_espacio_clustering(df, power, scaler):
    """Matriz escalada para el clustering, evitando el DOBLE escalado: si las
    columnas _ortho del CSV ya vienen estandarizadas (03_escalado.py), se usan
    directamente; si son residuos crudos, se aplica power + scaler."""
    X = df[FC].values
    if np.allclose(X.std(axis=0), 1.0, atol=0.3):        # ya escaladas (std≈1)
        return X
    return scaler.transform(power.transform(X))          # crudas -> escala


def construir_modelo(nombre):
    """Pipeline estandarizar -> modelo, según la familia por ley."""
    if nombre == 'SVR':
        m = SVR(kernel='rbf', C=10, epsilon=0.3)
    elif nombre == 'Ridge':
        m = Ridge(alpha=1.0)
    else:
        m = PLSRegression(n_components=2)
    return Pipeline([('scaler', StandardScaler()), ('modelo', m)])


# ============================================================
# CUADRO 5.1 — Limpieza
# ============================================================

def cuadro_5_1_limpieza(df_clean, df_superv):
    print('\n=== Cuadro 5.1 — Limpieza de la señal ===')
    print(f'Flujo continuo limpio        : {len(df_clean)}')
    print(f'Subconjunto supervisado (M24): {len(df_superv)}')


# ============================================================
# CUADRO 5.2 — Correlación de Pearson con n6sc (ortogonalización)
# ============================================================

def cuadro_5_2_pearson(df_clean, regr_ortho):
    print('\n=== Cuadro 5.2 — Correlación de Pearson con n6sc ===')
    n6v = df_clean['n6sc'].values
    n6 = n6v.reshape(-1, 1)
    print(f'{"Canal":6s} {"r inicial":>10s} {"r ortog.":>12s}')
    for m, nombre in METALES:
        r_ini = np.corrcoef(df_clean[m].values, n6v)[0, 1]           # crudo vs n6sc
        residuo = df_clean[m].values - regr_ortho[m].predict(n6)     # residuo ortogonalizado
        r_ort = np.corrcoef(residuo, n6v)[0, 1]                      # ≈0 por construcción
        print(f'{nombre:6s} {r_ini:>10.3f} {r_ort:>12.6f}')


# ============================================================
# CUADRO 5.3 — Homocedasticidad (Breusch-Pagan)
# ============================================================

def cuadro_5_3_breusch_pagan(df_clean, regr_ortho):
    print('\n=== Cuadro 5.3 — Breusch-Pagan (ortogonalización vs ratio) ===')
    n6 = df_clean['n6sc'].values.reshape(-1, 1)
    print(f'{"Elem":5s} {"LM ortho":>10s} {"LM ratio":>10s} {"reducción":>10s}')
    for m, nombre in METALES:
        o = _z(df_clean[m].values - regr_ortho[m].predict(n6))       # residuo estandarizado
        r = _z(df_clean[m].values / df_clean['n6sc'].values)         # ratio estandarizado
        lm_o, lm_r = bp_lm(o, n6), bp_lm(r, n6)
        print(f'{nombre:5s} {lm_o:>10.1f} {lm_r:>10.1f} {(1 - lm_o / lm_r) * 100:>9.1f}%')


# ============================================================
# CUADRO 5.4 y FIGURA fig_5_bic_silhouette — Selección de K
# ============================================================

def cuadro_5_4_seleccion_k(X, k_max=8):
    print('\n=== Cuadro 5.4 — Selección de K (BIC + silueta) ===')
    mu = np.random.RandomState(0).choice(len(X), min(5000, len(X)), replace=False)
    print(f'{"K":>2} {"BIC_GMM":>12} {"sil_KM":>8} {"sil_GMM":>8}')
    ks, bic, skm, sgm = [], [], [], []
    for k in range(2, k_max + 1):
        km = KMeans(k, random_state=RANDOM_STATE, n_init=10).fit(X)
        gm = GaussianMixture(k, covariance_type='full', random_state=RANDOM_STATE, n_init=10).fit(X)
        b = gm.bic(X)
        sk = silhouette_score(X[mu], km.labels_[mu])
        sg = silhouette_score(X[mu], gm.predict(X)[mu])
        print(f'{k:>2} {b:>12.0f} {sk:>8.3f} {sg:>8.3f}')
        ks.append(k); bic.append(b); skm.append(sk); sgm.append(sg)
    _fig_bic_silhouette(ks, bic, skm, sgm)
    return ks, bic, skm, sgm


def _fig_bic_silhouette(ks, bic, skm, sgm):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ks, bic, 'o-', color='#4a4e69', lw=2, label='BIC (GMM)')
    ax1.set_xlabel('Número de regímenes K'); ax1.set_ylabel('BIC (GMM)')
    ax1.axvline(3, color='#e94560', ls='--', alpha=0.5)
    ax2 = ax1.twinx()
    ax2.plot(ks, skm, 's--', color='#2a9d8f', lw=2, label='Silueta K-Means')
    ax2.plot(ks, sgm, '^--', color='#e94560', lw=2, label='Silueta GMM')
    ax2.set_ylabel('Coeficiente de silueta')
    l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc='center right', fontsize=9)
    ax1.set_title('Selección de K: BIC y silueta'); ax1.grid(alpha=0.3)
    plt.tight_layout()
    _guardar(fig, 'fig_5_bic_silhouette')


# ============================================================
# CUADRO 5.5 — GMM full vs tied
# ============================================================

def cuadro_5_5_full_tied(X):
    print('\n=== Cuadro 5.5 — GMM full vs tied (BIC) ===')
    H = np.mean([hopkins(X) for _ in range(5)])
    print(f'Hopkins H = {H:.3f}')
    print(f'{"K":>2} {"BIC_full":>12} {"BIC_tied":>12}')
    for k in [2, 3, 4, 5]:
        gf = GaussianMixture(k, covariance_type='full', random_state=RANDOM_STATE, n_init=5).fit(X)
        gt = GaussianMixture(k, covariance_type='tied', random_state=RANDOM_STATE, n_init=5).fit(X)
        print(f'{k:>2} {gf.bic(X):>12.0f} {gt.bic(X):>12.0f}')


# ============================================================
# Modelos: recomputar ortho + cluster consistentes (5.6-5.9)
# ============================================================

def _preparar_supervisado(df, regr_ortho, power, scaler, gmm):
    """Recomputa ortho y cluster con los modelos congelados (consistencia)."""
    n6 = df[['n6sc']].values
    for m, _ in METALES:
        df[f'{m}_ortho'] = df[m].values - regr_ortho[m].predict(n6)
    Xc = scaler.transform(power.transform(df[FC].values))
    df['cluster_gmm'] = gmm.predict(Xc)
    return df


# ============================================================
# CUADRO 5.6 — ANOVA / Kruskal-Wallis (significancia química)
# ============================================================

def cuadro_5_6_anova(sup, cluster_col='cluster_gmm', leyes=(('pCu', 'Cu'), ('pFe', 'Fe'), ('pMo', 'Mo'))):
    print('\n=== Cuadro 5.6 — ANOVA / Kruskal-Wallis ===')
    print(f'(n = {len(sup)} muestras)')
    print(f'{"Ley":6s} {"F":>10s} {"p ANOVA":>12s} {"H":>10s} {"p Kruskal":>12s}')
    for col, nombre in leyes:
        if col not in sup.columns:
            continue
        g = [sup[sup[cluster_col] == c][col].dropna().values for c in sorted(sup[cluster_col].unique())]
        g = [x for x in g if len(x) > 1]
        if len(g) < 2:
            continue
        f, pa = stats.f_oneway(*g)
        h, pk = stats.kruskal(*g)
        print(f'{nombre:6s} {f:>10.1f} {pa:>12.2e} {h:>10.1f} {pk:>12.2e}')


# ============================================================
# CUADRO 5.7 — Perfil químico por clúster
# ============================================================

def cuadro_5_7_perfiles(sup, cluster_col='cluster_gmm'):
    print('\n=== Cuadro 5.7 — Perfil químico por clúster ===')
    for c in sorted(sup[cluster_col].unique()):
        s = sup[sup[cluster_col] == c]
        fila = f'Clúster {c} (N={len(s)}): '
        for col, nombre in [('pCu', 'Cu'), ('pFe', 'Fe'), ('pMo', 'Mo')]:
            if col in s.columns:
                fila += f'{nombre}={s[col].mean():.2f}±{s[col].std():.2f}  '
        print(fila)


# ============================================================
# CUADROS 5.8 y 5.9 + FIGURAS global_vs_local y r2_heatmap
# ============================================================

def cuadros_5_8_5_9_modelos(sup, cluster_col='cluster_gmm'):
    print('\n=== Cuadros 5.8 y 5.9 — R2 global/local + enrutamiento ===')
    df = sup.dropna(subset=TARGETS).reset_index(drop=True)
    r2_global, r2_local_agg, r2_celda = {}, {}, {}
    clusters = sorted(df[cluster_col].unique())
    for tgt in TARGETS:
        familia = MODELO_POR_TARGET[tgt]
        y = df[tgt].values
        pg = cross_val_predict(construir_modelo(familia), df[FEATS_REG].values, y,
                               cv=KFold(5, shuffle=True, random_state=RANDOM_STATE))
        r2_global[tgt] = r2_score(y, pg)
        pl = np.zeros(len(df))
        for c in clusters:
            idx = df[cluster_col] == c
            Xc, yc = df.loc[idx, FEATS_REG].values, df.loc[idx, tgt].values
            k = min(5, int(idx.sum()))
            if k < 2:
                r2_celda[(tgt, c)] = float('nan'); pl[idx.values] = yc.mean(); continue
            pcv = cross_val_predict(construir_modelo(familia), Xc, yc,
                                    cv=KFold(k, shuffle=True, random_state=RANDOM_STATE))
            r2_celda[(tgt, c)] = r2_score(yc, pcv)
            pl[idx.values] = pcv
        r2_local_agg[tgt] = r2_score(y, pl)

    # Cuadro 5.8
    print('\n-- Cuadro 5.8: global vs local agregado --')
    print(f'{"Ley":6s} {"Modelo":6s} {"R2_global":>10s} {"R2_local":>10s}')
    for tgt in TARGETS:
        print(f'{tgt:6s} {MODELO_POR_TARGET[tgt]:6s} {r2_global[tgt]:>10.3f} {r2_local_agg[tgt]:>10.3f}')

    # Cuadro 5.9 + enrutamiento
    print('\n-- Cuadro 5.9: R2 local por (ley, clúster) + enrutamiento --')
    header = f'{"Ley":6s}' + ''.join([f'{"C"+str(c):>9s}' for c in clusters]) + '   Enrutamiento'
    print(header)
    ruteo = {}
    for tgt in TARGETS:
        fila = f'{tgt:6s}'
        decisiones = []
        for c in clusters:
            r2c = r2_celda[(tgt, c)]
            fila += f'{r2c:>9.3f}'
            usa_local = (not np.isnan(r2c)) and r2c > r2_global[tgt]
            ruteo[(tgt, c)] = 'local' if usa_local else 'global'
            decisiones.append(ruteo[(tgt, c)])
        print(fila + '   ' + ' / '.join(decisiones))

    _fig_global_vs_local(r2_global, r2_local_agg)
    _fig_r2_heatmap(r2_celda, r2_global, clusters)
    return r2_global, r2_celda, clusters


def _fig_global_vs_local(r2_global, r2_local_agg):
    leyes = [t[1:] for t in TARGETS]                     # Fe, Cu, Mo, Zn -> sin la 'p'
    g = [r2_global[t] for t in TARGETS]; l = [r2_local_agg[t] for t in TARGETS]
    x = np.arange(len(TARGETS)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, g, w, label='Global', color='#4a4e69')
    ax.bar(x + w / 2, l, w, label='Local agregado', color='#e94560')
    ax.set_xticks(x); ax.set_xticklabels(leyes)
    ax.set_ylabel('$R^2$ (validación cruzada)'); ax.set_title('$R^2$ global vs local agregado')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    _guardar(fig, 'fig_5_global_vs_local')


def _fig_r2_heatmap(r2_celda, r2_global, clusters):
    M = np.array([[r2_celda[(t, c)] for c in clusters] for t in TARGETS])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(M, cmap='RdYlGn', vmin=-0.2, vmax=0.9, aspect='auto')
    ax.set_xticks(range(len(clusters))); ax.set_xticklabels([f'Clúster {c}' for c in clusters])
    ax.set_yticks(range(len(TARGETS))); ax.set_yticklabels([t[1:] for t in TARGETS])
    for i, t in enumerate(TARGETS):
        for j, c in enumerate(clusters):
            usa_local = (not np.isnan(M[i, j])) and M[i, j] > r2_global[t]
            ax.text(j, i, f'{M[i, j]:.3f}', ha='center', va='center', fontsize=10,
                    fontweight='bold' if usa_local else 'normal')
            if usa_local:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='black', lw=3))
    ax.set_title('$R^2$ local por ley y clúster\n(borde grueso = local enrutado)')
    plt.colorbar(im, label='$R^2$'); plt.tight_layout()
    _guardar(fig, 'fig_5_r2_heatmap')


# ============================================================
# VENTANA HISTÓRICA (2025) vs LABORATORIO REAL — genera la comparación
# ============================================================

def evaluar_ventana_historica(fecha_ini='2025-01-01', fecha_fin='2026-01-01', guardar=True):
    """Corre la inferencia sobre el stream del período, alinea en ventanas de
    12 h centradas en el ensayo de laboratorio (excluyendo fuga) y devuelve la
    tabla de comparación. Reproduce comparacion_vs_lab_real.csv."""
    print(f'\n=== Ventana histórica {fecha_ini[:4]} vs laboratorio real ===')
    # Import diferido para no exigir el estimador si solo se quieren las tablas estáticas
    try:
        from src.pipeline.inferencia import EstimadorHibrido
    except Exception:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from pipeline.inferencia import EstimadorHibrido

    CANALES = ['n1fe', 'n2cu', 'n3zn', 'n4mo', 'n6sc']
    est = EstimadorHibrido()
    lab = cargar_lab_real()
    lab = lab[(lab['ts'] >= fecha_ini) & (lab['ts'] < fecha_fin)]

    inten = pd.read_csv(CSV_STREAM)
    inten['ts'] = pd.to_datetime(inten['date'].astype(str) + ' ' + inten['time'].astype(str), errors='coerce')
    inten = inten[(inten['ts'] >= fecha_ini) & (inten['ts'] < fecha_fin)].dropna(subset=['ts'])

    cal = pd.read_csv(CSV_SUPERV)                         # muestras de calibración (para excluir fuga)
    cal_keys = set(map(tuple, cal[CANALES].round(3).values))
    inten['es_calib'] = [tuple(np.round(r, 3)) in cal_keys for r in inten[CANALES].values]

    filas = []
    for _, f in inten.iterrows():
        r = est.predecir({c: f[c] for c in CANALES})
        if r['leyes']:
            filas.append({'ts': f['ts'], 'es_calib': f['es_calib'], **r['leyes']})
    pred = pd.DataFrame(filas).set_index('ts').sort_index()
    pred = pred[~pred['es_calib']]                        # excluye fuga

    rows = []
    for _, L in lab.iterrows():
        c = L['ts']
        w = pred[(pred.index > c - pd.Timedelta('6h')) & (pred.index <= c + pd.Timedelta('6h'))]
        if len(w) == 0:
            continue
        rows.append({'fecha': c, 'pCu': w['pCu'].mean(), 'Cu': L['Cu'],
                     'pFe': w['pFe'].mean(), 'Fe': L['Fe'],
                     'pMo': w['pMo'].mean(), 'Mo': L['Mo']})
    M = pd.DataFrame(rows)
    if guardar and len(M):
        M.to_csv(CSV_COMP_LAB, index=False)
        print(f'Comparación guardada en: {CSV_COMP_LAB}')
    return M


# ============================================================
# CUADRO 5.10 — Auditoría fuera de muestra (usa la comparación)
# ============================================================

def cuadro_5_10_online_audit(M):
    print('\n=== Cuadro 5.10 — Desempeño fuera de muestra vs lab real ===')
    print(f'{"Elem":6s} {"R2":>8s} {"MAE":>8s} {"corr":>8s} {"n":>6s}')
    for e, l, n in [('pCu', 'Cu', 'Cu'), ('pFe', 'Fe', 'Fe'), ('pMo', 'Mo', 'Mo')]:
        print(f'{n:6s} {r2_score(M[l], M[e]):>8.3f} {mean_absolute_error(M[l], M[e]):>8.3f} '
              f'{M[e].corr(M[l]):>8.3f} {len(M):>6d}')
    _fig_scatter_bland(M)
    _fig_serie_temporal(M)


def _fig_scatter_bland(M):
    elem = [('pCu', 'Cu', 'Cu'), ('pFe', 'Fe', 'Fe'), ('pMo', 'Mo', 'Mo')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for j, (e, l, n) in enumerate(elem):
        y, yh = M[l].values, M[e].values
        ax = axes[0, j]
        ax.scatter(y, yh, alpha=0.4, s=20, color='#e94560', edgecolor='none')
        lo, hi = min(y.min(), yh.min()), max(y.max(), yh.max())
        ax.plot([lo, hi], [lo, hi], '--', color='black', lw=1.3, label='identidad')
        ax.set_xlabel(f'{n} laboratorio (%)'); ax.set_ylabel(f'{n} modelo (%)')
        ax.set_title(f'{n} — R²={r2_score(y, yh):.3f}'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax = axes[1, j]
        prom, dif = (y + yh) / 2, yh - y; md, sd = dif.mean(), dif.std()
        ax.scatter(prom, dif, alpha=0.4, s=20, color='#4a4e69', edgecolor='none')
        ax.axhline(md, color='black', lw=1.4, label=f'sesgo={md:+.2f}')
        ax.axhline(md + 1.96 * sd, color='gray', ls='--', lw=1)
        ax.axhline(md - 1.96 * sd, color='gray', ls='--', lw=1)
        ax.set_xlabel(f'{n} promedio (%)'); ax.set_ylabel(f'{n} dif. (modelo-lab)')
        ax.set_title(f'Bland-Altman — {n}'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle('Acuerdo soft-sensor vs laboratorio real', fontsize=13, y=1.0)
    plt.tight_layout()
    _guardar(fig, 'fig_5_scatter_bland')


def _fig_serie_temporal(M):
    if 'fecha' not in M.columns:
        return
    Ms = M.copy(); Ms['fecha'] = pd.to_datetime(Ms['fecha']); Ms = Ms.sort_values('fecha')
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for ax, (e, l, n) in zip(axes, [('pCu', 'Cu', 'Cu'), ('pMo', 'Mo', 'Mo')]):
        ax.plot(Ms['fecha'], Ms[l], '-', color='#1a1a2e', lw=1, label='Laboratorio real')
        ax.plot(Ms['fecha'], Ms[e], '-', color='#e94560', lw=1, alpha=0.8, label='Soft-sensor')
        ax.set_ylabel(f'{n} (%)'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    axes[-1].set_xlabel('Fecha')
    plt.suptitle('Trayectoria temporal: soft-sensor vs laboratorio', y=0.98)
    plt.tight_layout()
    _guardar(fig, 'fig_5_serie_temporal')


# ============================================================
# CUADRO 5.11 — Baseline del analizador vs laboratorio real
# ============================================================

def cuadro_5_11_baseline(lab, fecha_ini='2025-01-01', fecha_fin='2026-01-01'):
    print('\n=== Cuadro 5.11 — Modelo vs ecuaciones del analizador ===')
    analiz = pd.read_csv(CSV_ANALIZADOR, parse_dates=['fecha_hora']).rename(columns={'fecha_hora': 'ts'}).sort_values('ts')
    comp = pd.merge_asof(analiz, lab[['ts', 'Cu', 'Fe', 'Mo']], on='ts',
                         tolerance=pd.Timedelta('6h'), direction='nearest').dropna()
    comp = comp[(comp['ts'] >= fecha_ini) & (comp['ts'] < fecha_fin)]
    print(f'{"Elem":6s} {"R2 analizador":>14s}  (modelo: Cu=0.069, Fe=-0.080, Mo=0.707)')
    for a, r in [('cu', 'Cu'), ('fe', 'Fe'), ('mo', 'Mo')]:
        print(f'{r:6s} {r2_score(comp[r], comp[a]):>14.3f}')


# ============================================================
# CUADRO 5.12 — Drift entrenamiento vs validación (KS)
# ============================================================

def cuadro_5_12_drift(lab, corte='2025-01-01'):
    print('\n=== Cuadro 5.12 — Drift (entrenamiento vs validación, KS) ===')
    tr = lab[lab['ts'] < corte]
    va = lab[(lab['ts'] >= corte) & (lab['ts'] < '2026-01-01')]
    print(f'n_train={len(tr)}  n_valid={len(va)}')
    print(f'{"Ley":6s} {"μ_tr":>7s} {"σ_tr":>6s} {"μ_va":>7s} {"σ_va":>6s} {"Δμ":>7s} {"KS p":>10s}')
    for ley in ['Cu', 'Fe', 'Mo']:
        ks, pks = stats.ks_2samp(tr[ley], va[ley])
        print(f'{ley:6s} {tr[ley].mean():>7.2f} {tr[ley].std():>6.2f} {va[ley].mean():>7.2f} '
              f'{va[ley].std():>6.2f} {va[ley].mean() - tr[ley].mean():>+7.2f} {pks:>10.2e}')


# ============================================================
# GUARDAR FIGURAS
# ============================================================

def _guardar(fig, nombre):
    os.makedirs(DIR_FIG, exist_ok=True)
    fig.savefig(os.path.join(DIR_FIG, nombre + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(DIR_FIG, nombre + '.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  [figura] {nombre}.pdf / .png')


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

if __name__ == '__main__':
    # --- Carga base ---
    df_clean = pd.read_csv(CSV_CLEAN)                     # crudas (5.1-5.3)
    df_cluster = pd.read_csv(CSV_CLUSTER)                 # _ortho escaladas (5.4-5.5)
    df_superv = pd.read_csv(CSV_SUPERV)                   # supervisado (5.7-5.9)
    regr_ortho = joblib.load(os.path.join(DIR_MODELS, 'orthogonalization_regressions.joblib'))
    power = joblib.load(os.path.join(DIR_MODELS, 'power_transformer.joblib'))
    scaler = joblib.load(os.path.join(DIR_MODELS, 'scaler.joblib'))
    gmm = joblib.load(os.path.join(DIR_MODELS, 'gmm_model.joblib'))
    lab = cargar_lab_real()

    X = preparar_espacio_clustering(df_cluster, power, scaler)  # espacio de clustering (sin doble escala)

    # --- Cuadros 5.1 a 5.5 ---
    cuadro_5_1_limpieza(df_clean, df_superv)
    cuadro_5_2_pearson(df_clean, regr_ortho)
    cuadro_5_3_breusch_pagan(df_clean, regr_ortho)
    cuadro_5_4_seleccion_k(X)
    cuadro_5_5_full_tied(X)

    # --- Conjunto supervisado con ortho + cluster consistentes ---
    sup = _preparar_supervisado(df_superv.copy(), regr_ortho, power, scaler, gmm)

    # --- Cuadros 5.6 a 5.9 ---
    cuadro_5_6_anova(sup)
    cuadro_5_7_perfiles(sup)
    cuadros_5_8_5_9_modelos(sup)

    # --- Ventana histórica 2025 vs laboratorio real (comparación + figuras) ---
    if os.path.exists(CSV_COMP_LAB):
        M = pd.read_csv(CSV_COMP_LAB)                     # usa la comparación ya generada
    else:
        M = evaluar_ventana_historica()                  # o la genera corriendo la inferencia

    # --- Cuadros 5.10 a 5.12 ---
    cuadro_5_10_online_audit(M)
    cuadro_5_11_baseline(lab)
    cuadro_5_12_drift(lab)

    print('\n=== Validación completa. Figuras en:', DIR_FIG, '===')
