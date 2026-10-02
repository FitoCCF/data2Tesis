# data2TesisV2

Branch huérfano de `data2Tesis` (mismo repo remoto, `FitoCCF/data2Tesis`,
sin historia git compartida), clonado en su propia carpeta. Reescritura
completa del pipeline de `data2Tesis`, desacoplada (`src/pipeline/*.py`,
cada etapa con CLI propio + orquestador `src/run_pipeline.py`).

**Registro de investigación completo, con cifras medidas:**
`docs/bitacora_analisis.md`. Léelo antes de repetir un experimento — hay
seis pruebas ya cerradas con control negativo (insoluble, cabeza del
circuito, orden escalado/ortogonalización) que no se sostuvieron.

## Convenciones

- Dependencias con Pixi (`pixi run python notebooks/NN_*.py`), Python 3.14.
- `notebooks/NN_*.py` son wrappers delgados numerados en orden de ejecución
  del pipeline; la lógica real vive en `src/pipeline/`. Un módulo con nombre
  que empieza en dígito no es importable con `import`, pero sí con
  `__import__("NN_nombre")` (varios notebooks lo hacen para reusar funciones
  de otro sin duplicar código).
- `data/`, `reports/` y `models/*.joblib` están gitignorados por diseño —
  el comentario del propio `.gitignore` dice por qué: "no saturar GitHub
  ni filtrar info industrial". Si hace falta subir un CSV o una figura
  puntual, `git add -f`, pero no por defecto.

## Flujo del pipeline (revisado 2026-09-30, etapas 0-6, 8, 9)

```
00_getdata            crudos por fuente: courier_bd / courier_pi (recorded) / composito_pi
01a_limpieza_fuentes  PI -> una fila por lectura real; BD -> leyes 0 = NaN, duplicados unidos
01b_unificacion       tabla maestra: BD pegada al PI por VALOR de los 4 metales (fuente pi/ambos/bd)
01c_limpieza          periodo (train <= FECHA_CORTE_TRAIN / test / produccion), anomalia, valida, valida_ley
02_features           log-cocientes vs Cu (clustering) + fracciones y logSumI (regresión) + dil
03_escalado           corrección de dilución (solo Fe/Cu) + StandardScaler, ajustados en train
04_clusterizado       GMM K_GRUPOS=4 (Cu/Mo/Zn/Fe), diagnóstico de k, caracterización vs lab
05_modelos_locales    walk-forward (split_purgado) + control aleatorio -> bundle (hoy: todo global)
06_inferencia_hibrida <ley>_est por lectura -> courier_estimado.csv
08_evaluacion_lab...  vs compósito 12 h, sin/con Kalman -> data/final/eval_composito*.csv, fig_08_*
09_evaluacion_vs_lab  vs muestras BD (OOF histórico + test) -> eval_bd_metricas.csv, fig_09_*
```

`pixi run test` corre los 18 tests de `split_purgado`. La evaluación vive en
`src/evaluacion/` (`evaluar.py` + `figuras.py`); la 9 lee lo que escribe la 8.

Todo lo que se ajusta usa solo `periodo == 'train'` (también el `q` del
Kalman). Ninguna etapa borra filas por una bandera: marca. `run_pipeline.py`,
el notebook 10 y `scripts/validaciones_estadisticas.py` todavía usan la
interfaz antigua (ortogonalización, `EstimadorHibrido`).

## Trampas ya encontradas

- **Crudos de la etapa 0: uno por fuente** (`courier_bd.csv` A,
  `courier_pi.csv` B, `composito_pi.csv` C), todos en hora local naive. A y B
  son el mismo instrumento: **125/125** muestras de la BD posteriores al inicio
  del PI (2025-07-13) están con valores idénticos en el recorded del PI, con la
  hora de la BD ~6 min detrás (p95 28 min) — **no concatenarlas**
  (duplica lecturas) y **no interpolar el PI**. El courier da una lectura cada ~21 min
  y el PI la re-archiva cada 100 s (`ExcMax=100`): ~14 eventos por lectura,
  no un sensor congelado — la etapa 1a (`limpieza_fuentes.py`) las colapsa. El token del PI va
  en `PI_TOKEN` o `~/.pi_token`, nunca en el código.
- **El courier expone MUCHO más que el concentrado (hallado el 2026-10-01, §15).**
  El PI tiene **66 tags `_296290_*Canal*`** con los 5 canales de intensidad de
  *cada* línea de ambos couriers (`RebHidroL1/L2`, `ConcPrimL1/L2`,
  `Conc1raLimpL1/L2`, `ColaScavL1/L2`, `RelaveFinal`, `ConcFinal`, y los del
  courier moly), y **tres familias de compósito de 12 h**: `7100AIP011-014MAN`
  cabeza (con **%Zn**), `7100AIP031-034MAN` relave, `7100AIP101-104MAN`
  concentrado — las tres con la misma historia, 2022-12-30 → hoy, 2603 ensayos.
  Por eso ninguna de las otras 24 líneas calzaba con el PI: **cada stream tiene
  sus propios tags** y solo se extraían los de `ConcFinal`. Las intensidades de
  cabeza empiezan el 2026-03-01. La IP de la pasarela **cambia**: pedírsela al
  usuario, no está registrada en ningún archivo.
- **Los tipos de mineral se definen en la CABEZA, no en el concentrado**
  (`src/pipeline/tipos_mineral.py`): cabeza k=2 (Cu 62 % / Fe 38 %), relave k=3,
  sobre **log-cocientes** alr contra Cu —nunca leyes absolutas, porque la muestra
  de relave trae mucha dilución de agua y el CV cae de 0.671 a 0.094 al pasar a
  cocientes—. Validado por fase de la mina (p=7.4e-07), por los grupos del
  concentrado (cabeza Cu → Cu+Mo 80.7 %) y por recuperación (4 puntos de Cu entre
  la mejor y la peor combinación). Los grupos del concentrado de la etapa 4
  quedan como caracterización del **producto**, no como tipo de mineral.
- **El relave se lee como tipos de lo que se PIERDE**, no de riqueza: su tipo
  "Cu" es cola relativamente rica en Cu (0.061 vs 0.046) = peor recuperación.
- **Los tags PI `_296290_ConcFinal_*` ("C2 Courier cobre Concentrado Final")
  son el `sample_id=24` "Concentrado Colectivo" (equipo 5) de la BD**, no el
  `sample_id=10` "Concentrado Final" (equipo 1, otro courier). Verificado por
  valor el 2026-09-30: 125/125 muestras del 24 calzan con el PI; 0 de ~2,700
  de las otras 24 líneas de muestreo.
- **Las dos fuentes de ley son dos cortes físicos del mismo courier, y solo una
  tiene intensidad propia.** La BD es toma **puntual Lun/Mie/Vie** (356/365 de
  las muestras con ley) del material que **sí pasó por los rayos X**: existe la
  lectura que le corresponde, y de ahí que el calce por valor sea exacto —es
  identidad del material, no un emparejamiento afortunado. El compósito se corta
  **antes del multiplexor**, 2/día (07:30 y 19:30), y **no tiene intensidad
  propia**: la única alineación posible es promediar la estimación sobre su
  ventana. Eso justifica *físicamente* los dos protocolos de evaluación (etapa 9
  por lectura, etapa 8 por ventana) y que el compósito no audite Zn, que no
  reporta.
- **La ventana centrada del compósito ya está verificada: no volver a
  cuestionarla desde cero.** Que el compósito sea "de turno, cada 12 h" sugiere
  que su marca es el **cierre** (soporte `[t−12h, t]`); está medido que se
  comporta como el **centro** de su ventana. Ley contra ley, sin modelo en
  medio, la centrada gana en las tres leyes y en Mo —el árbitro, porque es la
  única donde ambos laboratorios concuerdan— 0.917 contra 0.801. Concuerda con
  el barrido de lag de `calibracion_composito.py` (óptimo en lag 0). Reproducir
  con `pixi run python scripts/verificar_fuentes_laboratorio.py`; detalle en
  `docs/bitacora_analisis.md` §12.
- **Cualquier comparación de R² entre particiones necesita el control
  aleatorio** (mismo tamaño de grupo, sin información real): con ~340
  muestras casi cualquier reparto parece mejorar. Seis pruebas de §8-9 y la
  propia etapa 5 lo confirmaron; por eso el control es parte de la regla de
  ruteo, y hoy ningún local la pasa.
- **BD y compósito no son la misma referencia en Cu:** la BD lee 1.2
  puntos menos que el compósito de su ventana (corr 0.48; Fe −0.18/0.61, Mo
  +0.07/0.92). Una métrica de Cu o Fe cambia según contra cuál se mida, y
  el Kalman (que aprende del compósito) empeora Cu contra la BD. El −1.23 de Cu
  **no** es artefacto de alineación ni de la hora de toma (09-13 h → −1.12;
  15-19 h → −1.81): es diferencia de ensayo. Lo que sí depende del soporte
  temporal (corte puntual vs integral de 12 h) es la **correlación**, no el
  nivel — promediar no corre la media.
- **La ortogonalización contra `n6sc` se eliminó (2026-09-30).** Sus
  residuos arrastraban la deriva del instrumento (`n2cu_ortho` de +6,600 en
  2022H2 a -5,100 en 2026H2 con pCu casi constante). La etapa 2 es ahora
  `features.py`: log-cocientes contra Cu para clustering, fracciones de
  cierre + `logSumI` para regresión, sin artefacto. Si aparece una columna
  `_ortho` en un CSV viejo, es de antes de este cambio.
- **El courier no mide insolubles.** `a7a7`/`a6sol` en `works4cdp_assay`
  son una estimación del operador a partir de las demás leyes, no una
  lectura del sensor (confirmado con el equipo). `n7ech7` (el canal crudo
  real) está vacío para `sample_id=24` (Concentrado Colectivo). La única
  fuente de insoluble es el compósito de laboratorio vía PI (12h) o `pIns`
  en la BD (misma cadencia baja).
- **`n6sc` no mide el sólido de la muestra** (es el canal de dispersión):
  `corr(n6sc, pSol) = -0.15` sobre 344 muestras; `logSumI` sí (+0.57). `pSol` de
  alta frecuencia no existe en ninguna fuente revisada (ni compósito PI ni
  BD) — si algún día se instrumenta, usarlo como feature de sólido es la vía
  más prometedora que queda sin probar. **Precisión del 2026-10-01:** el hueco no
  es de diseño sino de poblado — la tabla `works4cdp_assayspsi` ya existe en la BD
  con el esquema completo (`psolid`, `aSol`, `aDensity`, `P80`, mallas) y está
  **vacía, 0 filas**; `api2db.py` no la llena. Y `pSol` sí existe a cadencia de
  muestra para la cabeza (`sample_id` 17 y 18: 215 y 243 muestras).

## Estado y decisiones
Ver `docs/estado.md` (sección `data2TesisV2`) y `docs/decisiones.md` en la
raíz de `Proyects/` para cómo esto se relaciona con `tesis_ucspv2/` — **el
encuadre de esta rama (solo Mo defendible) todavía no está reconciliado con
lo ya migrado ahí (4 elementos).**
