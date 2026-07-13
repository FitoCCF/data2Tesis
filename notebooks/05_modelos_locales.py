# ============================================================
# CELDA 5 — MODELOS LOCALES (una regresión de leyes por cluster)
# ============================================================
# Entrena un modelo de regresión por cada (ley, cluster) y construye la TABLA
# DE RUTEO (local vs global). CLAVE: usa EXACTAMENTE las mismas transformaciones
# guardadas (ortogonalización de la celda 2, escala de la 3, cluster de la 4)
# que la celda 6 aplicará en inferencia -> entrenamiento e inferencia coinciden.
# ============================================================

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score

# --- 1. Cargar dataset de calibración (intensidades CRUDAS + leyes de lab) ---
file_path = 'data/processed/intensidad_cobre_24_completo_filtrado.csv'
df = pd.read_csv(file_path)

# --- Cargar las transformaciones YA GUARDADAS por las celdas 2, 3 y 4 ---
regresiones_ortho = joblib.load('models/orthogonalization_regressions.joblib')  # celda 2
power = joblib.load('models/power_transformer.joblib')                           # celda 3
scaler_cluster = joblib.load('models/scaler.joblib')                             # celda 3
gmm = joblib.load('models/gmm_model.joblib')                                     # celda 4

METALES = ['n1fe', 'n2cu', 'n3zn', 'n4mo']                       # canales a ortogonalizar
FEATS_CLUSTER = ['n1fe_ortho', 'n2cu_ortho', 'n3zn_ortho', 'n4mo_ortho']  # features del clustering

# --- 2. RECOMPUTAR el ortho desde las intensidades crudas (mismo cálculo que la celda 6) ---
for m in METALES:                                               # por cada metal
    pred_n6sc = regresiones_ortho[m].predict(df[['n6sc']].values)  # parte explicada por n6sc
    df[f'{m}_ortho'] = df[m].values - pred_n6sc                 # residuo CRUDO (misma escala que inferencia)

# --- 3. RE-ASIGNAR el cluster con el GMM guardado (mismo que usará la celda 6) ---
X_cluster = scaler_cluster.transform(power.transform(df[FEATS_CLUSTER].values))  # ortho -> power -> escala
df['cluster'] = gmm.predict(X_cluster)                          # etiqueta de cluster consistente
CLUSTER_COL = 'cluster'                                         # columna de cluster a usar

# --- Features de la regresión: ortho (crudo) + n6sc ---
feats = ['n1fe_ortho', 'n2cu_ortho', 'n3zn_ortho', 'n4mo_ortho', 'n6sc']  # se conserva n6sc

# --- Modelo por elemento y leyes a estimar ---
MODELO_POR_TARGET = {'pFe': 'Ridge', 'pCu': 'SVR', 'pMo': 'Ridge', 'pZn': 'PLS'}
targets = list(MODELO_POR_TARGET.keys())
df = df.dropna(subset=targets).reset_index(drop=True)           # quita filas sin ley medida


def construir_modelo(nombre):
    """Pipeline: estandarizar -> modelo (la escala interna maneja el ortho crudo + n6sc)."""
    if nombre == 'SVR':
        m = SVR(kernel='rbf', C=10, epsilon=0.3)
    elif nombre == 'Ridge':
        m = Ridge(alpha=1.0)
    else:
        m = PLSRegression(n_components=2)
    return Pipeline([('scaler', StandardScaler()), ('modelo', m)])


# --- 4. Entrenar modelos globales y locales + medir R2 ---
modelos_locales = {}
modelos_globales = {}
reporte = []
r2_global_por_ley = {}
r2_local_por_celda = {}

for tgt in targets:
    familia = MODELO_POR_TARGET[tgt]
    y_all = df[tgt].values

    # Global (referencia): CV + ajuste final
    pred_global = cross_val_predict(construir_modelo(familia), df[feats].values, y_all,
                                    cv=KFold(5, shuffle=True, random_state=42))
    r2_global = r2_score(y_all, pred_global)
    r2_global_por_ley[tgt] = r2_global
    modelos_globales[tgt] = construir_modelo(familia).fit(df[feats].values, y_all)

    # Locales: uno por cluster
    pred_local = np.zeros(len(df))
    for cl in sorted(df[CLUSTER_COL].unique()):
        idx = df[CLUSTER_COL] == cl
        n = int(idx.sum())
        Xc = df.loc[idx, feats].values
        yc = df.loc[idx, tgt].values
        k = min(5, n)
        pred_cv = cross_val_predict(construir_modelo(familia), Xc, yc,
                                    cv=KFold(k, shuffle=True, random_state=42))
        r2_cl = r2_score(yc, pred_cv)
        pred_local[idx.values] = pred_cv
        modelos_locales[(tgt, cl)] = construir_modelo(familia).fit(Xc, yc)
        reporte.append({'target': tgt, 'cluster': cl, 'n': n, 'R2_local': round(r2_cl, 3)})
        r2_local_por_celda[(tgt, cl)] = r2_cl

    r2_local_agg = r2_score(y_all, pred_local)
    print(f'{tgt}: global R2={r2_global:.3f}  |  local agregado R2={r2_local_agg:.3f}  ({familia})')

# --- 5. Tabla de R2 por cluster ---
print('\nR2 por cluster (validación cruzada interna):')
print(pd.DataFrame(reporte).pivot(index='cluster', columns='target', values='R2_local'))

# --- 6. Construir la TABLA DE RUTEO (local si supera al global; si no, global) ---
tabla_ruteo = {}
for tgt in targets:
    for cl in sorted(df[CLUSTER_COL].unique()):
        tabla_ruteo[(tgt, cl)] = 'local' if r2_local_por_celda[(tgt, cl)] > r2_global_por_ley[tgt] else 'global'

print('\nTabla de ruteo (local vs global) por (ley, cluster):')
print(pd.DataFrame(
    [{'cluster': cl, **{tgt: tabla_ruteo[(tgt, cl)] for tgt in targets}}
     for cl in sorted(df[CLUSTER_COL].unique())]
).set_index('cluster'))

# --- 7. Guardar modelos + tabla de ruteo en el bundle ---
os.makedirs('models', exist_ok=True)
bundle = {
    'modelos_locales': modelos_locales,
    'modelos_globales': modelos_globales,
    'features': feats,
    'targets': targets,
    'cluster_col': CLUSTER_COL,
    'tabla_ruteo': tabla_ruteo,
    'r2_global_por_ley': r2_global_por_ley,
    'r2_local_por_celda': r2_local_por_celda,
}
joblib.dump(bundle, 'models/modelos_locales_por_cluster.joblib')
print('\nModelos locales + tabla de ruteo guardados en: models/modelos_locales_por_cluster.joblib')
