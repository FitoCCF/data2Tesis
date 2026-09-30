# data2TesisV2

Reimplementación **desacoplada y escalable** de `data2Tesis` (V1). Mismo
pipeline, mismos resultados — pero cada etapa vive en **un solo lugar**
(`src/pipeline/*.py`) y es ejecutable **de forma independiente o encadenada**.

## Qué cambió respecto a V1

En V1, la lógica de cada etapa existía **duplicada dos veces**: una vez en
`src/pipeline/*.py` (reutilizable, con modo entrenar/aplicar) y otra vez
reimplementada a mano en `notebooks/0N_*.py` (con rutas hardcodeadas). La
celda 6 de inferencia llegó a cargarse en las celdas 8 y 9 vía
`exec(open('06_inferencia_hibrida.py').read()...)` — frágil, depende del CWD
exacto y triplica el código.

En V2:
- Toda la lógica vive **una sola vez** en `src/`. Los scripts de `notebooks/`
  son ahora wrappers delgados que importan de `src.pipeline` / `src.evaluacion`
  — nunca reimplementan nada.
- Cada etapa tiene un **CLI propio** (`python -m src.pipeline.<etapa> --help`)
  y puede correr sola, apuntando a cualquier CSV de entrada/salida.
- Un **orquestador** (`src/run_pipeline.py`) encadena las etapas en un solo
  comando, con `--hasta-etapa` para detenerse donde se quiera (p. ej. dejar el
  pipeline "en línea" solo hasta clustering, sin necesitar la etapa 5).
- Nuevo `EstimadorCluster` (en `src/pipeline/inferencia.py`) para escenarios
  donde solo se necesita el régimen de mineral (cluster) de una lectura nueva,
  sin cargar el bundle de modelos de leyes de la etapa 5.
- Nuevo `src/acquisition/`: adquisición explícita, separada del resto del
  pipeline, un crudo por fuente (ver "Tres fuentes de datos" abajo):
  - `from_db.py` — Postgres (`works4cdp_assay`): intensidades + leyes de
    laboratorio (fuente A).
  - `from_pi.py` — PI vía la pasarela de `data4cdpv1_local`: recorded del
    courier (fuente B) y compósito de 12 h (fuente C).
  - `from_pi_afsdk.py` — AF SDK directo en Windows, solo referencia.
- Se corrigió el puerto de la BD hardcodeado en 5432 (V1: `scripts/assay_int.py`)
  que en realidad está en 5433 (`docker ps` confirma el contenedor
  `postgres_db` publicado en `5433->5432/tcp`). Ahora es una sola constante
  (`src/database/connection.py:DB_CONFIG_DEFAULT`), configurable por variables
  de entorno `DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME`.
- Se corrigió el path absoluto roto en `scripts/split_date_time.py` (apuntaba
  a `/home/fito/Proyects/data2Tesis/...`, de otra máquina/usuario) por un
  argumento `--file`.

## Tres fuentes de datos (A, B, C), un crudo por fuente

La etapa 0 (`notebooks/00_getdata.py`) extrae cada fuente a su propio CSV en
`data/raw/`, por separado: si la pasarela PI no responde, la BD igual se
guarda. Los crudos no se modifican después; todo sale en **hora local naive**
(America/Lima).

| | Fuente | Crudo | Qué trae | Cadencia |
|---|---|---|---|---|
| **A** | Postgres `works4cdp_assay`, `sample_id=24` (`from_db.py`) | `courier_bd.csv` | 7 canales + leyes de laboratorio `pFe..pSol` en la misma fila | ~cada 3 días, desde 2021-07 (leyes desde 2022-12) |
| **B** | PI, tags `_296290_ConcFinal_Canal{Fe,Cu,Zn,Mo,Sc}_ABB` (`from_pi.py`) | `courier_pi.csv` | 5 canales (sin `n5ech5`/`n7ech7`), sin leyes | eventos ~cada 100 s; lecturas reales ~cada 21 min — s, desde 2025-07-13 |
| **C** | PI, tags `7100AIP10{1..4}MAN` (`from_pi.py`) | `composito_pi.csv` | compósito de lab de 12 h: `fe/cu/ins/mo` (sin Zn ni Sol) | 07:30 / 19:30 |

**A y B son el mismo instrumento.** El courier mide igual en muestreo y en
operación normal; en A además se corta una muestra para el laboratorio.
Verificado el 2026-09-30: 108 de 126 muestras de A desde 2025-07-13 aparecen
con los 5 valores **idénticos** en el recorded de B, a ~20 s de diferencia de
reloj (ni `date+time` ni `instance` coinciden exactamente). Las 18 que no
aparecen caen todas en 2026-03-23 .. 2026-07-01. Por eso:

- B se extrae con `recorded`, **nunca interpolado**: interpolar inventa valores
  entre lecturas y rompe esa correspondencia.
- El courier da una lectura del concentrado cada ~21 min y la sostiene hasta
  la siguiente; el PI la vuelve a archivar cada 100 s (`ExcMax=100` en los 5
  tags): ~14 eventos por lectura real. No es un sensor congelado. La etapa 1a
  las colapsa a una fila por lectura (~23,600 hasta agosto).
- Concatenar A y B duplica lecturas. La unificación (etapa 1b) pega las
  leyes de A a la lectura de B que calza por valor de los 4 metales: con las
  lecturas ya reconstruidas en la 1a calzan **122/122**, `n6sc` incluido (las
  18 que no calzaban eran lecturas partidas por el desfase de `n6sc`). La
  hora de la BD va ~6 min detrás del inicio de la lectura en el PI, y 3 filas
  de jun-2026 están corridas exactamente 12 h (error AM/PM en la BD).

**Conexión.** BD: `DB_PORT=5432` si el contenedor no publica en 5433 (default
del código). PI: pasarela en `config.PI_HOST`/`PI_PUERTO` (o
`PI_GATEWAY_HOST`/`PI_GATEWAY_PORT`) y token en `PI_TOKEN` o `~/.pi_token`
— nunca en el código.

## Estructura

```
src/          lógica reutilizable: toda etapa vive aquí una sola vez
notebooks/    celdas numeradas: el pipeline paso a paso (00-11) y los experimentos (12-26)
scripts/      utilidades sueltas: extracción, fusión de fuentes, validaciones del Cap. 5
data/         raw (extracciones), processed (salida de cada etapa), final (vacío)
models/       artefactos entrenados (.joblib) + manifiesto de reproducibilidad
reports/      figuras y tablas de resultados
docs/         bitácora de análisis y flujo del pipeline
references/   vacío
```

`data/`, `models/*.joblib` y `reports/` están en `.gitignore` por diseño (tamaño
y datos industriales). Los archivos de esos directorios que sí aparecen en git
se subieron a propósito con `git add -f`.

### `src/` — la lógica del pipeline

| Archivo | Qué hace |
|---|---|
| `run_pipeline.py` | Orquestador: encadena las etapas 1-5 en un comando; `--hasta-etapa` permite detenerse en cualquiera |
| **`src/acquisition/`** | **Extracción de datos crudos** |
| `from_db.py` | `extraer()`: intensidades del courier + leyes de laboratorio desde Postgres (`works4cdp_assay`) en un solo DataFrame. Baja frecuencia |
| `from_pi.py` | `extraer_courier()` (fuente B, recorded) y `extraer_composito()` (fuente C) desde PI vía la pasarela; token por `PI_TOKEN`/`~/.pi_token` |
| `from_pi_afsdk.py` | La misma extracción PI por AF SDK directo en Windows. Solo referencia, no se usa |
| `pi_client.py` | Cliente de la pasarela PI (con token), copia tal cual de `data4cdpv1_local/scripts/pi_client.py` |
| **`src/database/`** | **Acceso a Postgres** |
| `connection.py` | Configuración de conexión (puerto, usuario, BD), sobrescribible por variables de entorno |
| `extractor.py` | `ExtractorBD`: una sola consulta, armada desde los grupos de columnas de `config.COLS_BD`, con el filtro de fechas en SQL |
| **`src/pipeline/`** | **Las etapas, cada una con función reutilizable + CLI propio** |
| `config.py` | Rutas, tabla y columnas de la BD (`COLS_BD`), tags del compósito PI y todas las constantes. Un solo lugar que editar |
| `limpieza_fuentes.py` | **Etapa 1a.** Limpieza determinista por fuente: PI -> una fila por lectura real (colapsa republicaciones, reconstruye lecturas partidas, banderas `sostenida`/`n6sc_desfasado`); BD -> une `instance` duplicadas, ley en 0 -> NaN, bandera `leyes_sospechosas`; compósito -> orden y duplicados |
| `unificacion.py` | **Etapa 1b.** Tabla maestra A + B: pega cada muestra de la BD a su lectura del PI por calce de valor de los 4 metales (no por hora); columna `fuente` = `pi`/`ambos`/`bd` |
| `limpieza.py` | **Etapa 1c.** Sobre la tabla maestra: banderas `congelada`, `anomalia` (IsolationForest sobre fracciones de cierre, ajustado solo hasta `FECHA_CORTE_TRAIN`), `valida` y `periodo` (train/test). No borra filas |
| `escalado.py` | **Etapa 3.** StandardScaler de los 3 log-cocientes (columnas `_z`), ajustado solo con `train & valida`. Sin Yeo-Johnson |
| `clustering.py` | **Etapa 4.** GMM full con `K_GRUPOS` (4), ajustado solo con `train & valida`; `diagnostico_k` (silhouette + estabilidad ARI), nombre de cada grupo por su centroide (`Cu`/`Fe`/`Mo`/`Zn`), `caracterizar` contra BD y compósito |
| `dataset_supervisado.py` | Puente: une intensidades clusterizadas con las leyes de laboratorio |
| `modelos.py` | **Etapa 5.** Un modelo por (ley, cluster) + uno global + tabla de ruteo. Usa `KFold(shuffle=True)`, ver bitácora §9.4 |
| `inferencia.py` | **Etapa 6.** Estimación en producción: `EstimadorCluster` (etapas 1-4) y `EstimadorHibrido` (1-6) |
| `features.py` | **Etapa 2.** Features robustas a la deriva (reemplaza a la ortogonalización contra `n6sc`): log-cocientes contra Cu para el clustering (`lr_fe_cu`, `lr_zn_cu`, `lr_mo_cu`) y fracciones de cierre + `logSumI` para la regresión. Transformación fija, sin artefacto |
| `calibracion_composito.py` | Recalibración de Fe contra el compósito de 12h (cambios 1-3) y su backtest walk-forward |
| `corrector_kalman.py` | Corrector de sesgo por filtro de Kalman, reemplazo propuesto de la media móvil de `CorrectorSesgo` |
| `ruteo_insoluble.py` | Experimento cerrado: ruteo de pFe por insoluble (no se sostuvo, bitácora §8.2) |
| **`src/evaluacion/`** | |
| `evaluar.py` | Evaluación contra laboratorio real en ventanas de 12h centradas en el ensayo |

### `notebooks/` — celdas numeradas

Se corren con `pixi run python notebooks/NN_*.py`. No reimplementan lógica:
llaman a `src/`.

**Pipeline, en orden de ejecución**

| Celda | Qué hace |
|---|---|
| `00_getdata.py` | Etapa 0: extrae las tres fuentes (A BD, B PI courier, C PI compósito), un crudo por fuente (por defecto hasta agosto) |
| `01a_limpieza_fuentes.py` | Etapa 1a: limpieza por fuente (`courier_pi_lecturas.csv`, `courier_bd_limpio.csv`, `composito_limpio.csv`) |
| `01b_unificacion.py` | Etapa 1b: unificación BD + PI -> `courier_unificado.csv` |
| `01c_limpieza.py` | Etapa 1c: limpieza general -> `courier_limpio.csv` + `models/anomaly_detector.joblib` |
| `02_features.py` | Etapa 2: features robustas a la deriva -> `courier_features.csv` |
| `03_escalado.py` | Etapa 3: escalado -> `courier_escalado.csv` + `models/scaler.joblib` |
| `04_clusterizado.py` | Etapa 4: diagnóstico de k, GMM, caracterización y grupos de producción por semana -> `courier_clusterizado.csv` + `models/gmm_model.joblib` |
| `05_modelos_locales.py` | Etapa 5: modelos por (ley, cluster) y tabla de ruteo |
| `06_inferencia_hibrida.py` | Etapa 6: demo de estimación en producción |
| `08_evaluacion_laboratorio.py` | Evaluación fuera de muestra contra laboratorio de alta frecuencia |
| `09_evaluacion_historica.py` | La evaluación de la celda 8 generalizada a cualquier período |
| `09_evaluacion_vs_lab_real.py` | Evaluación contra el reporte de laboratorio químico real |
| `10_recalibracion_composito.py` | **Recalibración de Fe contra el compósito** (cambios 1-4): backtest, bundle, corrector de sesgo, detector y manifiesto |
| `11_figuras_resultados.py` | Figuras y tabla del capítulo de resultados (Fe recalibrado + Cu/Mo original) |

**Experimentos (sesiones 2026-09-28/29)** — detalle y cifras en `docs/bitacora_analisis.md` §8-9

| Celda | Pregunta | Resultado |
|---|---|---|
| `12_experimento_ruteo_insoluble.py` | ¿Rutear pFe por el insoluble del turno anterior mejora? | No (control aleatorio empata) |
| `13_experimento_cluster_cabeza.py` | ¿La cabeza (rebose hidrociclones L2) mejora el clustering? | No (artefacto de bloques repetidos) |
| `14_experimento_cluster_cabeza_epocas.py` | Lo mismo, a la granularidad de la cabeza | Inconcluso (n=58) |
| `15_pipeline_insoluble_clustering.py` | ¿El insoluble como feature del clustering separa mejor la química? | No, separa peor Fe/Cu/Mo |
| `16_modelos_locales_con_insoluble.py` | ¿Y en R² de los modelos locales? | Parecía que sí... |
| `17_control_ruteo_insoluble.py` | Control negativo de la celda 16 | ...el control aleatorio gana en 3 de 4 leyes |
| `18_figura_comparacion_insoluble.py` | Figura de barras: oficial vs insoluble vs control | `reports/fig_insoluble_vs_control` |
| `19_figura_tendencia_insoluble.py` | Figura de tendencia de lo mismo | `reports/fig_tendencia_insoluble` |
| `20_figura_validacion_desde_mayo.py` | Validación vs laboratorio desde 2026-05-01 (media móvil) | `reports/fig_validacion_desde_mayo` |
| `21_backtest_corrector_kalman.py` | ¿Kalman mejora el corrector de sesgo de Fe? | **Sí**: corr 0.508→0.576 |
| `22_figura_validacion_kalman.py` | Validación completa con Fe corregido por Kalman | `reports/fig_validacion_kalman` |
| `23_backtest_corrector_kalman_cu.py` | ¿Y aplicado a Cu? | Mixto: R²/MAE mejoran, corr empeora |
| `24_orden_escalado_ortogonalizacion.py` | ¿Escalar antes de ortogonalizar? (métricas de clustering) | Diferencia marginal |
| `25_impacto_orden_en_cu_mo.py` | Lo mismo, medido en R² de Cu/Mo con control | No (el control gana en 3 de 4) |
| `26_figura_mejor_estrategia_kalman_mayo.py` | **Mejor estrategia vigente**: Fe con Kalman, desde mayo | `reports/fig_validacion_kalman_desde_mayo` |

### `scripts/`

| Archivo | Qué hace |
|---|---|
| `merge_cobre_data.py` | Fusiona la extracción de BD y la de PI, ordena por tiempo y parte en train/test |
| `assay_lab_test.py` | Combina y limpia exportaciones de AssayLab |
| `split_date_time.py` | Separa una columna datetime en `date` + `time` |
| `validaciones_estadisticas.py` | Reproduce los cuadros y figuras del Capítulo 5. Usa `KFold(shuffle=True)` |

### `data/`

**`data/raw/`** — extracciones, sin procesar

| Archivo | Qué es |
|---|---|
| `courier_bd.csv` | **A**: intensidades + leyes de laboratorio desde Postgres (etapa 0) |
| `courier_pi.csv` | **B**: intensidades del courier, recorded del PI (etapa 0; ~390k filas) |
| `composito_pi.csv` | **C**: compósito de laboratorio de 12 h, `ts` + `fe/cu/ins/mo` (etapa 0) |
| `assay_lab_AIQ_pi.csv` | Salida del analizador con su calibración de fábrica. **No es laboratorio**: úsese como línea base, nunca como referencia |
| `produccion_sept.csv` | Extracción de producción de septiembre, se scorea aparte con la etapa 6 |

**`data/processed/`** — salida de cada etapa

| Archivo | Qué es |
|---|---|
| `courier_pi_lecturas.csv` | Etapa 1a: una fila por lectura real del courier (~23,600 hasta agosto) + `sostenida_s`, `sostenida`, `n6sc_desfase_s`, `n6sc_desfasado` |
| `courier_bd_limpio.csv` | Etapa 1a: BD con `ts`, leyes en 0 -> NaN, `instance_duplicada`, `leyes_sospechosas` |
| `composito_limpio.csv` | Etapa 1a: compósito ordenado, sin duplicados |
| `courier_clusterizado.csv` | Etapa 4: + `grupo_id`, `grupo` (nombre), `prob_grupo`, para train, test y producción |
| `courier_escalado.csv` | Etapa 3: + columnas `_z` |
| `courier_features.csv` | Etapa 2: `courier_limpio.csv` + features de clustering y de regresión. **Entrada de la etapa 3** |
| `courier_limpio.csv` | Etapa 1c: tabla maestra + `periodo`, `congelada`, `anomalia`, `valida` (para ajustar lo no supervisado), `valida_ley` (muestras para modelos; la anomalía no excluye). |
| `courier_unificado.csv` | Etapa 1b: **tabla maestra**, una fila por lectura del courier (24,044 hasta agosto): `fuente`, 5 canales, leyes (solo donde hubo muestra), `ts_bd`/`dt_bd_s`/`hora_bd_desfasada` y las banderas de 1a |
| `intensidades_cobre.csv` | Fusión completa BD + PI (`merge_cobre_data.py`) |
| `intensidades_cobre_train.csv` | Split de calibración: **entrada de la etapa 1** (28,158 filas) |
| `intensidades_cobre_test.csv` | Split de validación del mismo período histórico |
| `intensidad_cobre_24_clean.csv` | Salida de la etapa 1 (23,058 filas) |
| `intensidad_cobre_24_completo.csv` | Clusterizado + leyes de laboratorio (428 filas) |
| `intensidad_cobre_24_completo_filtrado.csv` | Lo mismo, solo filas con al menos una ley (314): **conjunto supervisado de la etapa 5** |
| `*_test.csv`, `*_sept.csv` | Las mismas etapas aplicadas al split de test y a producción de septiembre |
| `intensidades_cobre_test_leyes.csv` | Salida de la etapa 6 sobre el split test: leyes estimadas, cluster, ruteo y alertas |
| `comparacion_test_vs_lab_AIQ.csv` | Estimación del split test contra la calibración de fábrica, por ventana |
| `composito_12h_alineado.csv` | Intensidades promediadas en cada ventana de 12h del compósito (760 ventanas) |
| `backtest_recalibracion.csv` | Backtest walk-forward de Fe con el corrector de media móvil (228 ventanas) |
| `evaluacion_desde_mayo.csv` | Evaluación Fe/Cu/Mo vs laboratorio desde mayo, media móvil (celda 20) |
| `evaluacion_kalman.csv` | Lo mismo con Fe corregido por Kalman, período completo (celda 22) |
| `evaluacion_kalman_desde_mayo.csv` | Lo mismo desde mayo (celda 26) |

### `models/` — artefactos entrenados

| Archivo | Lo produce | Qué es |
|---|---|---|
| `anomaly_detector.joblib` | etapa 1 / celda 10 | IsolationForest sobre fracciones de cierre |
| `orthogonalization_regressions.joblib` | etapa 2 | Una regresión lineal metal ~ `n6sc` por metal |
| `power_transformer.joblib` | etapa 3 | Yeo-Johnson ajustado |
| `scaler.joblib` | etapa 3 | StandardScaler ajustado |
| `kmeans_model.joblib`, `gmm_model.joblib` | etapa 4 | Modelos de clustering, k=3 |
| `modelos_locales_por_cluster.joblib` | etapa 5 | Modelos locales y globales por ley + tabla de ruteo. Lo usan Cu y Mo |
| `calibracion_composito.joblib` | celda 10 | Modelo de Fe recalibrado contra el compósito (ventana móvil de 270 días) |
| `corrector_sesgo.joblib` | celda 10 | Corrector de sesgo de Fe. **Todavía es la media móvil**, no Kalman |
| `manifiesto_recalibracion.json` | celda 10 | Hashes de entrada, parámetros, métricas y versiones de la última recalibración |

### `reports/` — figuras y tablas

Cada figura existe en `.png` (para revisar) y `.pdf` (para LaTeX).

| Archivo | Qué muestra |
|---|---|
| `fig_validacion_kalman_desde_mayo` | **Resultado vigente**: Fe (Kalman), Cu y Mo vs laboratorio, mayo-agosto 2026 |
| `fig_validacion_kalman` | Lo mismo, período completo (desde 2026-04-25) |
| `fig_validacion_desde_mayo` | La versión anterior, con Fe corregido por media móvil |
| `fig_tendencia_kalman_vs_lab` | Serie de Fe: laboratorio vs media móvil vs Kalman, con panel de error |
| `fig_tendencia_kalman_cu_vs_lab` | Serie de Cu: laboratorio vs sin corrector vs Kalman |
| `fig_insoluble_vs_control` | Barras: R² local oficial vs insoluble vs control aleatorio |
| `fig_tendencia_insoluble` | Lo mismo como gráfica de tendencia |
| `tabla_metricas_*.csv` | Métricas (R², MAE, RMSE, corr, pendiente, sesgo) de cada figura de validación |

**Artifacts publicados en claude.ai** (privados, del dueño del repo), con las
figuras de arriba y su explicación:

| Artifact | Qué contiene |
|---|---|
| [La mejor estrategia](https://claude.ai/artifact/BVDMwWYuGqTTDJWuajdqga) | Resultado vigente: `fig_validacion_kalman_desde_mayo` + métricas + lectura por elemento |
| [Kalman vs. media móvil](https://claude.ai/artifact/8PPM1ySCk3XKXJQTiQ1HGb) | Corrector Kalman en Fe: tendencia, validación completa y barrido de `q` |
| [Insoluble vs. azar](https://claude.ai/artifact/TM1d4whC7be8mj79TwftLS) | Las 4 pruebas del insoluble contra el control aleatorio |
| [Escalar vs. ortogonalizar](https://claude.ai/artifact/LvCoqCSQDudhB4Z17cCAot) | Orden de las etapas 2-3: no ayuda |

### `docs/`

| Archivo | Qué es |
|---|---|
| `bitacora_analisis.md` | **Registro de todo el diagnóstico**, con cifras medidas: qué funciona, qué no, qué se probó y descartó. Leer antes de repetir un experimento |
| `flujo_pipeline.md` | Flujo de entrenamiento/calibración, flujo de inferencia en producción y ciclo operativo |

## Cómo correr cada etapa por separado

```bash
pixi run python notebooks/00_getdata.py            # 0   extracción: courier_bd / courier_pi / composito_pi
pixi run python notebooks/01a_limpieza_fuentes.py  # 1a  limpieza por fuente
pixi run python notebooks/01b_unificacion.py       # 1b  tabla maestra (calce BD<->PI por valor)
pixi run python notebooks/01c_limpieza.py          # 1c  banderas + IsolationForest (ajustado hasta FECHA_CORTE_TRAIN)
pixi run python notebooks/02_features.py           # 2   features robustas a la deriva
pixi run python notebooks/03_escalado.py           # 3   StandardScaler (ajustado con train)
pixi run python notebooks/04_clusterizado.py       # 4   grupos de mineral (GMM) + producción por semana
# etapas 5 en adelante: en revisión
```

## Flujo train/test vs. producción (qué CSV entra a cada etapa)

Son **dos cosas distintas** que no hay que confundir:

1. **train/test** (`scripts/merge_cobre_data.py`) — un split dentro de los
   datos HISTÓRICOS (los que ya se extrajeron, típicamente hasta agosto) para
   calibrar y validar el pipeline. El `test` sigue siendo data histórica
   pasada, no septiembre.
2. **Producción** (septiembre en adelante) — un flujo totalmente aparte: se
   extrae por separado (`from_db`/`from_pi` con `--desde 2026-09-01`) y se
   scorea directo con la etapa 6. **Nunca pasa por `merge_cobre_data.py` ni
   por el split train/test.**

`scripts/merge_cobre_data.py` genera **tres** CSV, pero solo dos entran al
pipeline de calibración:

| CSV | Para qué |
|---|---|
| `intensidades_cobre.csv` (completo) | Archivo histórico. NO se usa como entrada de la etapa 1. Lo consumen scripts que necesitan el stream completo de referencia: `06_inferencia_hibrida.py` (demo) y las evaluaciones (`09_evaluacion_historica.py`, `09_evaluacion_vs_lab_real.py`, `src/evaluacion/evaluar.py`). |
| `intensidades_cobre_train.csv` | **Entrada de la etapa 1** (`src.pipeline.limpieza`, default de `--input`) y de ahí encadena hasta la etapa 5. Con esto se **calibran** (ajustan) todos los artefactos en `models/*.joblib`. |
| `intensidades_cobre_test.csv` | Subconjunto de **validación**, dentro del mismo período histórico que train (no se re-entrena con él: se puede correr por la etapa 6 ya calibrada y comparar contra lo esperado, como chequeo de que el modelo generaliza dentro de su propio período). |

Por eso `src/pipeline/limpieza.py` y `notebooks/01_limpieza.py` apuntan por
default a `intensidades_cobre_train.csv`, no a `intensidades_cobre.csv`.

`--train-hasta` en el merge define el corte train/validación **dentro** del
histórico ya extraído (p.ej. si extrajiste hasta agosto, podrías usar
`--train-hasta 2026-07-31` para dejar todo agosto como test/validación).
Sin ese flag, el split es 70/30 por posición sobre todo lo extraído:

```bash
pixi run scripts/merge_cobre_data.py --train-hasta 2026-07-31
```

Los datos de **producción** (septiembre en adelante) se extraen y scorean
aparte, sin merge ni split — ver "Cómo correr todo junto" abajo.

## Cómo correr todo junto

```bash
# hasta clustering (no requiere leyes de laboratorio):
pixi run python -m src.run_pipeline --input data/processed/intensidades_cobre_train.csv --hasta-etapa clustering

# pipeline completo (requiere un CSV de leyes de laboratorio):
pixi run python -m src.run_pipeline --input data/processed/intensidades_cobre_train.csv \
    --hasta-etapa modelos --assays-csv data/raw/assays.csv

```

## Cómo scorear datos de producción (septiembre en adelante)

Se extraen aparte (nunca por `merge_cobre_data.py`) y se scorean con los
artefactos ya calibrados arriba, vía la etapa 6:

```bash
# 1. extraer solo el período de producción:
# (OJO: sobreescribe los crudos de data/raw/; pendiente de rediseño en la etapa 1b)
pixi run notebooks/00_getdata.py --desde 2026-09-01 --hasta 2026-09-21

# 2. scorear (solo cluster, sin necesitar la etapa 5):
pixi run python -m src.pipeline.inferencia --input data/raw/produccion_sept.csv --hasta-etapa clustering

# o leyes completas (requiere que la etapa 5 ya esté calibrada):
pixi run python -m src.pipeline.inferencia --input data/raw/produccion_sept.csv --hasta-etapa leyes
```

## Adquisición

```bash
# Etapa 0 completa: las tres fuentes, hasta agosto (default)
DB_PORT=5432 PI_TOKEN=... pixi run python notebooks/00_getdata.py
# solo algunas fuentes:
pixi run python notebooks/00_getdata.py --fuentes bd
PI_TOKEN=... pixi run python notebooks/00_getdata.py --fuentes pi composito

# Por módulo, una fuente a la vez:
pixi run python -m src.acquisition.from_db --hasta 2026-08-31              # A -> courier_bd.csv
pixi run python -m src.acquisition.from_db --grupos leyes --out data/raw/leyes.csv
pixi run python -m src.acquisition.from_pi courier --hasta 2026-09-01      # B -> courier_pi.csv
pixi run python -m src.acquisition.from_pi composito --hasta 2026-09-01    # C -> composito_pi.csv
```

`from_pi_afsdk.py` (AF SDK directo en Windows) queda solo como referencia.

## CSV faltantes

**Ninguno de los CSV de datos se copió desde V1** (solo se replicó código +
estructura de carpetas, como se pidió). `data/raw/`, `data/processed/`,
`data/final/` y `models/` están vacíos (solo `.gitkeep`). Para dejar el
pipeline operativo hace falta, en orden:

1. **Extraer intensidades crudas** de las dos fuentes: `src.acquisition.from_db`
   (BD, requiere que el contenedor `postgres_db` esté accesible; en este
   entorno corre en el puerto `5433`) y `src.acquisition.from_pi` (PI, alta
   frecuencia, vía el gateway desde WSL — requiere `PI_GATEWAY_HOST`/el
   gateway de `data4cdpv1_local` accesible).
2. Fusionar ambas con `scripts/merge_cobre_data.py --cobre <BD> --cobre-24
   <PI> --train-hasta <fecha>` -> genera `intensidades_cobre.csv` (completo) +
   `_train.csv` / `_test.csv` (split por esa fecha).
3. Correr las etapas 1-4 sobre el **train** (`src.run_pipeline --input
   .../intensidades_cobre_train.csv --hasta-etapa clustering`). El **test**
   no entra aquí — ver "Flujo train/test" más abajo.
4. Para entrenar la etapa 5 (leyes químicas) hace falta además un CSV/fuente
   de **leyes de laboratorio** (`--assays-csv`, o `--from-db` en
   `dataset_supervisado`).

Además, estos **tres archivos ya faltaban en V1** (nunca estuvieron en el
repo, no es algo que se perdiera al replicar) y bloquean específicamente la
evaluación contra laboratorio:
- `data/raw/AssayLab_*.csv` (dos exportaciones crudas que
  `scripts/assay_lab_test.py` combina en `AssayLab_combined_limpio.csv`,
  usado por `notebooks/08_evaluacion_laboratorio.py` y
  `09_evaluacion_historica.py`).
- `data/raw/assay_lab_raw.csv` (laboratorio real, usado por
  `src/evaluacion/evaluar.py` y `09_evaluacion_vs_lab_real.py`).

Sin estos tres, todo el pipeline de estimación (etapas 1-6) funciona igual;
solo queda bloqueada la **evaluación** contra laboratorio.
