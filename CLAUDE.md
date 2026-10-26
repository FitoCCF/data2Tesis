# data2TesisV2

> **Rama `AlimConc`** (worktree `~/AlimConc`, 2026-10-26): copia del pipeline con la etapa 4
> "solo Mo, k=4" (`config.FEATS_GMM`) y la ley de cabeza en línea
> (`scripts/cabeza_ley_en_linea.py`). Ver bitácora §23. El monitor de `~/data2TesisV2` no usa esta rama.

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
01a_limpieza_fuentes  PI -> una fila por lectura real (por evento, 5 canales juntos); BD -> leyes 0 = NaN, duplicados unidos
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
- **Los 5 canales del courier se miden JUNTOS (medido el 2026-10-21, §18).**
  Cada canal tiene exactamente 25,439 instantes de cambio y los 25,179 eventos
  agrupados tienen los 5; cero casos en 15 meses donde `n6sc` se mueva y los
  metales no. Por eso **una fila del PI a la que le falta un canal no es una
  lectura**: es media lectura archivada aparte, y el desfase (p99 = 3 s,
  max 145 s, cuantizado y con signo variable) es del PI, no del equipo.
  La 1a reconstruye por evento (`lecturas_por_evento`): agrupa instantes de
  cambio, no filas, y corta cuando un canal se repite. **No volver a agrupar
  filas por proximidad temporal** (`TOL_EVENTO`): fechaba 1,201 lecturas 100 s
  tarde. En modo automático el mínimo entre muestras es 2 min y rotan 10
  líneas (~20 min para el concentrado final); con menos líneas el ciclo se
  acorta, y de ahí las lecturas legítimas a 4-5 min.
- **NOVIEMBRE 2025 ES LÍMITE DE INSTRUMENTO (confirmado con planta, §19).**
  Hubo mantenimiento y después una campaña de muestreo para rehacer las
  ecuaciones del Outocal. Las intensidades de antes y después **no son la misma
  medición**: los cocientes entre canales se movieron −15.1 % a +14.1 %, y una
  ganancia común (dilución, decaimiento de fuente) no puede cambiar cocientes.
  El escalón de `n6sc` es de un día, −18.9 % el 2025-12-10; las fechas por canal
  están escalonadas (n1fe ~2025-11-30). **Toda ventana de análisis se define
  dentro de una época**; un modelo que atraviese la fecha lleva indicador de
  época. Ventanas limpias: `2025-07-13 → 2025-11-30` y `2025-12-15 → hoy`.
  Para Mo las dos se pueden juntar **con offset** (pendientes 1.9945 y 1.9520,
  salto p=0.203): `scripts/comparar_epocas_mo.py`.
- **El concentrado NO tiene grupos de mineral: tiene un GRADIENTE de Mo (§19).**
  `lr_mo_cu` explica R² 0.667 (pre) y 0.769 (post) del %Mo del compósito **sin
  ajustar un parámetro**, y **ninguna etiqueta discreta le gana con ningún k en
  ninguna de las dos ventanas** (0.486-0.620). Los 4 grupos de la etapa 4 son
  rebanadas arbitrarias de ese gradiente. **No volver a buscar grupos en el
  concentrado**: silhouette ≤ 0.31 en todo k y algoritmo, HDBSCAN da 100 % de
  ruido en la ventana pre, el histograma de `lr_mo_cu` tiene una sola joroba, y
  el autoencoder no es reproducible entre semillas. Reproducible con
  `notebooks/29_clustering_post_mantenimiento.py --desde X --hasta Y`.
- **Trampa específica de este problema: las métricas internas premian el
  artefacto.** El silhouette da 0.25-0.31 a la partición por **dilución** y
  0.07-0.16 a la de **composición**, porque mide distancias en el espacio crudo
  donde la magnitud domina la varianza. Elegir k o algoritmo por silhouette acá
  lleva derecho a la respuesta equivocada. Las únicas métricas que apuntaron
  bien fueron la **validación externa contra el compósito** (524 ventanas, 8×
  las muestras de la BD) y el **control de época** (`AMI_mes`).
- **La deriva de la magnitud son TRES causas, no una (§19; corrige §2.3).**
  Dilución de la muestra (dominante: R² 0.221 → 0.696 al entrar `pSol`,
  t=+19.6, ~1 % de intensidad por punto de %sólido), el escalón de
  recalibración, y decaimiento de fuente de **−2.7 %/año** que sobrevive
  controlando química **y** sólido. `pSol` cae ~2 puntos/año y eso **no** lo
  explica la química (hipótesis de planta: líneas de agua contra espuma y
  atoro; no demostrable, no hay tag de densidad en la línea de concentrado
  final). `scripts/diagnostico_deriva_intensidades.py`.
- **La invarianza a la dilución es física del canal, no una corrección.** El
  agua absorbe según la energía (~E⁻³): `lr_zn_cu` (Zn 8.64 ≈ Cu 8.05 keV) es
  invariante y `lr_fe_cu` (+0.519) / `lr_mo_cu` (−0.278) no pueden serlo, con
  los signos que predice la física. El techo de la corrección actual (0.91 →
  0.73 sd) es ese. Dentro de época el proxy es **mejor** que el +0.659 global de
  §2.4: `corr(logSumI, pSol)` centrado por mes da +0.840 (pre) y +0.684 (post).
- **`dil` NO se toca, y está medido (§19.10).** Es cierto que
  `dil = logSumI − EWMA(90 d)` atraviesa la recalibración de nov-2025 y absorbe
  el escalón (media mensual −0.075 en dic-ene, sobre sd 0.095). Pero **las ocho
  variantes que quitan el escalón empeoran la corrección**: el efecto residual
  de `pSol` sobre `lr_fe_cu` pasa de 0.712 (actual) a 0.729-0.752 al
  re-nivelar o reiniciar por época, y acortar la vida media también empeora.
  El escalón de `logSumI` fue −7.4 %, en la **misma dirección** que la caída
  del `pSol` y en los mismos meses, así que quitarlo le resta señal real: **la
  corrección funciona en parte por coincidencia**. `DIL_HALFLIFE="90D"` y
  `DIL_CORREGIR=["lr_fe_cu"]` se quedan. **Riesgo con disparador:** si un
  mantenimiento futuro mueve las intensidades en dirección contraria a la del
  sólido, `dil` corrompe la corrección con el signo equivocado — la salvaguarda
  es el monitor de cocientes y re-ajustar `b` tras cada mantenimiento, no
  cambiar la fórmula. Reproducible: `scripts/variantes_senal_dilucion.py`.
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
- **Las dos fuentes de ley son dos CORTES del mismo flujo, con el MISMO
  laboratorio y el MISMO método** (confirmado con planta el 2026-10-22; antes
  esta nota y la bitácora decían "dos laboratorios", que es incorrecto):

  ```
  flujo → MULTIPLEXOR ─corte→ balde 12 h          (COMPÓSITO; no pasa por el probe)
               │
               └──────→ PROBE (rayos X) ─corte→ recipiente  (CALIBRACIÓN)
  ```

  El multiplexor está a **<2 m** del probe y en ese tramo **no hay agua**; el
  agua de dilución se aplica **muchos metros antes del multiplexor**, así que
  las dos muestras ven el mismo material ya diluido. Los dos cortadores toman
  la **sección completa** y el balde se **agita y se procesa completo**.
  La toma puntual es **Lun/Mie/Vie** entre las 10 y las 12 h, son las muestras
  de **calibración del Outocal**, y su intensidad y su ensayo quedan en la misma
  fila de la BD — de ahí que el calce por valor sea exacto: es identidad del
  material. El compósito **no tiene intensidad propia**, así que la única
  alineación posible es promediar la estimación sobre su ventana. Eso justifica
  *físicamente* los dos protocolos (etapa 9 por lectura, etapa 8 por ventana) y
  que el compósito no audite Zn, que no reporta.
  **Roles, y no son dos candidatas al mismo puesto:** la puntual es la
  referencia de **calibración** por diseño; el compósito es la **auditoría
  independiente**.
- **Hay un sesgo de muestreo Cu↔ganga de −1.2 puntos entre los dos cortes
  (§19.13), y sigue SIN EXPLICACIÓN.** La puntual lee **−1.200** de Cu
  (IC95 [−1.44, −0.96]) y **+1.160** de insoluble respecto del compósito de su
  ventana; `corr(dcu, dins) = −0.834` y la suma se cancela: es balance de masa.
  Es **aditivo**, no proporcional (Bland-Altman p=0.093). Descartados con
  medición: laboratorio, soporte (solo 0.743 de 2.223), agua
  (`corr(dcu,pSol)=+0.006`), la línea al probe, delimitación de los cortadores,
  sedimentación del balde, hora de toma (explica el 15 %), error de escala,
  deriva temporal y diferencia de base (los 4 relativos no son iguales ni del
  mismo signo). **No proponer más mecanismos sin información de proceso nueva.**
  El contraste que avanzaría: extraer cabeza y relave y ver si el mismo −4.7 %
  relativo aparece ahí.
- **Los límites de concordancia entre los dos cortes son [−5.56, +3.16] en Cu.**
  Par a par las dos fuentes **no son comparables**; solo el promedio lo es. Por
  eso la etapa 9, con 18 muestras de la BD en test, tiene intervalos muy anchos
  — y por eso el Kalman "empeora Cu" ahí (0.52 → 0.24) sin que eso pruebe que
  empeoró algo real.
- **El techo del Cu: su R² bajo es falta de varianza, no falla del modelo
  (§19.12).** `CV` del Cu en el concentrado = **7.5 %** contra **43.8 %** del
  Mo: el Cu casi no se mueve porque es el objetivo controlado. Y el residuo del
  modelo (sd 1.460) ya es ≤ el desacuerdo entre las dos referencias (sd 2.223).
  **No invertir en modelado de Cu**; lo que desbloquearía la pregunta son 30-50
  **duplicados de ensayo** del compósito. El variograma NO sirve para estimar
  el techo acá (no hay datos bajo 24 h y el turno contamina los pares de 12 h).
- **Estructura química del concentrado (§19.14): dos perillas, y el Cu sin
  perilla propia.** El Mo no arrastra al Cu (corr ≈ 0) pero sí al Fe (−0.56 de
  Fe por punto de Mo; **no** es cierre de masa, que daría −1). Cu, Fe y ganga
  se intercambian por balance de masa (parciales −0.65 / −0.77 / −0.72). Es la
  química del lab: que el courier resuelva ese intercambio **no** está
  demostrado. `scripts/estructura_quimica_composito.py`.
- **Escala balde y tres leyes (decidido el 2026-10-22, §19.15).** Las
  estimaciones se reportan en escala del **compósito** para Fe, Cu y Mo: el
  Kalman es parte del modelo, la etapa 8 es el juez y la 9 es control de
  calibración (allí se espera ver ~1.2 de Cu). **`pZn` ya no está en
  `TARGETS`**; el canal `n3zn` sigue como feature. Línea base del modelado en
  §19.15; los datos nuevos del PI se extraen **al final**, como prueba ciega.
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
