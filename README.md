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
  pipeline, con dos fuentes activas + una de referencia (antes mezcladas a
  mano en `merge_cobre_data.py`):
  - `from_db.py` — Postgres (`works4cdp_assay`), fuente de **baja frecuencia**
    de `n1fe, n2cu, n3zn, n4mo, n6sc`. Soporta filtro `--desde/--hasta`.
  - `from_pi.py` — **RUTA VIGENTE** de extracción PI: desde WSL, a través del
    gateway HTTP (`gw4Pi.exe`) de `data4cdpv1_local`, con los tags del courier
    (`_296290_ConcFinal_Canal{Cu,Fe,Mo,Sc,Zn}_ABB`) ya incorporados por
    defecto. Fuente de **alta frecuencia** (cada 15 min, desde 2025-07-13).
    Es la que alimenta `Intensidades_*.csv`, que `merge_cobre_data.py`
    combina con la BD para tener más datos que cualquiera de las dos fuentes
    por separado. También soporta tags arbitrarios (`--tags`) para otras
    variables de proceso.
  - `from_pi_afsdk.py` — misma fuente PI y mismos tags, pero por AF SDK
    directo (Windows, requiere PI AF Client Tools + pythonnet). Queda como
    **referencia/alternativa**, no como ruta activa, ya que la extracción se
    hace ahora desde WSL vía el gateway.
- Se corrigió el puerto de la BD hardcodeado en 5432 (V1: `scripts/assay_int.py`)
  que en realidad está en 5433 (`docker ps` confirma el contenedor
  `postgres_db` publicado en `5433->5432/tcp`). Ahora es una sola constante
  (`src/database/connection.py:DB_CONFIG_DEFAULT`), configurable por variables
  de entorno `DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME`.
- Se corrigió el path absoluto roto en `scripts/split_date_time.py` (apuntaba
  a `/home/fito/Proyects/data2Tesis/...`, de otra máquina/usuario) por un
  argumento `--file`.

## Dos fuentes reales para los canales del courier (BD + PI, distinta frecuencia)

(Corrección a una nota anterior de este README, que decía que PI no tenía los
canales del courier — sí los tiene, en el servidor correcto, y la extracción
se hace desde WSL vía el gateway, no por AF SDK directo en Windows.)

Hay **dos fuentes independientes** de `n1fe, n2cu, n3zn, n4mo, n6sc`, con
distinta frecuencia de muestreo, y el pipeline las combina para tener más
datos:

1. **Postgres (`works4cdp_assay`)** — `src/acquisition/from_db.py`. El
   analizador expone una API HTTP local (CLB) que otro proyecto (`api2db.py`
   en `data4cdpv1_local`) vuelca a esta tabla. Frecuencia baja (según cadencia
   del analizador).
2. **PI OSIsoft, vía el gateway HTTP desde WSL** — `src/acquisition/from_pi.py`
   (usa `pi_client.PiGateway`, copiado de `data4cdpv1_local`). Tags del
   courier ya incorporados por defecto:
   `_296290_ConcFinal_Canal{Cu,Fe,Mo,Sc,Zn}_ABB` -> columnas `cu/fe/mo/sc/zn`.
   Interpolados cada **15 min desde 2025-07-13 15:00:00** (fecha de inicio
   fija: no hay historia de estos tags antes). El gateway resuelve del lado
   Windows contra el servidor PI de producción (`tpi.southernperu.com.pe`);
   este cliente no necesita saberlo.

`scripts/merge_cobre_data.py` ya fusiona ambas (`--cobre` = BD, `--cobre-24` =
PI) y ordena por timestamp, de ahí el dataset combinado con más filas que
cualquiera de las dos fuentes solas.

`src/acquisition/from_pi_afsdk.py` es la misma fuente PI, mismos tags, pero
por AF SDK directo en Windows — queda como alternativa/referencia, no como
ruta activa (ver nota en el propio archivo).

## Estructura

```
src/
  acquisition/     # extracción: from_db.py (BD), from_pi.py (PI vía gateway WSL, ruta vigente), from_pi_afsdk.py (PI AF SDK, referencia)
  database/        # conexión + Extractor (Postgres)
  pipeline/        # las 6 etapas, cada una con función reutilizable + CLI
    limpieza.py            # etapa 1: filtros deterministas + IsolationForest
    ortogonalizacion.py    # etapa 2: corrección por dilución vs n6sc
    escalado.py            # etapa 3: Yeo-Johnson + StandardScaler
    clustering.py          # etapa 4: KMeans + GMM (régimen de mineral)
    dataset_supervisado.py # fusión: intensidades clusterizadas + leyes de lab
    modelos.py             # etapa 5: modelos locales por cluster + ruteo
    inferencia.py          # etapa 6: EstimadorCluster (1-4) / EstimadorHibrido (1-6)
  evaluacion/      # comparación contra laboratorio real (ventanas de 12h)
  run_pipeline.py  # orquestador: encadena todas las etapas en un comando
notebooks/         # wrappers delgados (documentación/EDA), llaman a src.*
scripts/           # utilidades: merge de extracciones, split train/test, etc.
data/{raw,processed,final}/   # vacíos (solo .gitkeep) — ver "CSV faltantes"
models/            # vacío (solo .gitkeep) — se llena al correr las etapas
```

## Cómo correr cada etapa por separado

```bash
pixi run python -m src.pipeline.limpieza --input data/processed/intensidades_cobre.csv
pixi run python -m src.pipeline.ortogonalizacion --input data/processed/intensidad_cobre_24_clean.csv
pixi run python -m src.pipeline.escalado --input data/processed/intensidad_cobre_24_orthogonalized.csv
pixi run python -m src.pipeline.clustering --input data/processed/intensidad_cobre_24_scaled.csv
# diagnóstico de k sin entrenar nada:
pixi run python -m src.pipeline.clustering --input ... --diagnostico

# aplicar una etapa ya entrenada sobre datos NUEVOS (modo inferencia):
pixi run python -m src.pipeline.ortogonalizacion --input nuevas.csv --artifact-in models/orthogonalization_regressions.joblib
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
pixi run notebooks/00_getdata.py --desde 2026-09-01 --hasta 2026-09-21 --out data/raw/produccion_sept.csv
pixi run notebooks/00_getdata_pi.py --desde 2026-09-01 --hasta 2026-09-21 --out data/raw/produccion_sept_pi.csv

# 2. scorear (solo cluster, sin necesitar la etapa 5):
pixi run python -m src.pipeline.inferencia --input data/raw/produccion_sept.csv --hasta-etapa clustering

# o leyes completas (requiere que la etapa 5 ya esté calibrada):
pixi run python -m src.pipeline.inferencia --input data/raw/produccion_sept.csv --hasta-etapa leyes
```

## Adquisición

```bash
# BD Postgres (baja frecuencia):
pixi run python -m src.acquisition.from_db --sample-id 24 --tabla intensidades \
    --hasta 2026-08-31 --out data/raw/intensidad_cobre_db.csv

# PI OSIsoft, courier, vía gateway desde WSL (RUTA VIGENTE, alta frecuencia):
export PI_GATEWAY_HOST=<ip_windows>   # si el autodescubrimiento no lo encuentra
pixi run python -m src.acquisition.from_pi --hasta 2026-09-21 --out data/raw/Intensidades_nuevo.csv
# ya trae los 5 tags del courier por defecto (cu/fe/mo/sc/zn) -> directo a:
pixi run python scripts/merge_cobre_data.py --cobre-24 data/raw/Intensidades_nuevo.csv

# PI OSIsoft, tags arbitrarios (p.ej. espesadores/relaves, NO los del courier):
pixi run python -m src.acquisition.from_pi --tags _294100_LIT_1011_ABB \
    --desde 2026-08-01 --hasta 2026-09-21 --out data/raw/pi_espesadores.csv

# Alternativa/referencia: PI vía AF SDK directo (requiere Windows + PI AF Client):
python src/acquisition/from_pi_afsdk.py --fecha-fin "2026-09-21 00:00:00" --out Intensidades_nuevo.csv
```

### Wrappers de notebook equivalentes (con --desde/--hasta por defecto "solo hasta agosto")

`notebooks/00_getdata.py` (BD) y `notebooks/00_getdata_pi.py` (PI vía gateway)
son el mismo par de fuentes de arriba, pero como celdas de notebook: valores
por defecto embebidos (`FECHA_HASTA = '2026-08-31'`) para no traer de más en
esta fase, y overrideables por CLI sin romper su uso pegados en Jupyter
(`argparse.parse_known_args`, ignora los argumentos propios del kernel):

```bash
pixi run notebooks/00_getdata.py                        # BD, hasta agosto (default)
pixi run notebooks/00_getdata_pi.py                      # PI/gateway, hasta agosto (default)
pixi run notebooks/00_getdata.py --desde 2026-09-01 --hasta 2026-09-21 --out data/raw/produccion_sept.csv
pixi run notebooks/00_getdata_pi.py --desde 2026-09-01 --hasta 2026-09-21 --out data/raw/produccion_sept_pi.csv
```

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
