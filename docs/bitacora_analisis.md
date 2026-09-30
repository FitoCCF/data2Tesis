# Bitácora de análisis — diagnóstico y recalibración del pipeline

Registro del razonamiento, las mediciones y las decisiones tomadas durante la
sesión de diagnóstico. Sirve para retomar el trabajo en otra máquina sin repetir
los experimentos. **Todos los números de este documento están medidos**, no
estimados; cada uno indica el protocolo con que se obtuvo.

Última actualización: 2026-09-29

---

## 0. Cómo retomar

```bash
git clone git@github.com-ucsp:FitoCCF/data2Tesis.git
cd data2Tesis && git checkout data2TesisV2
pixi install
```

**Los datos NO están en el repositorio** (`.gitignore` excluye `data/`, `reports/`
y `models/*.joblib`). Hay que copiarlos por medio privado. Sin ellos el código
corre pero no hay nada que procesar.

Archivos mínimos para reproducir:

| Archivo | Qué es |
|---|---|
| `data/processed/intensidades_cobre.csv` | stream del analizador, ~40k lecturas |
| `data/raw/assay_lab_courier_pi.csv` | compósito 12 h, 2590 ensayos (referencia de validación) |
| `data/processed/intensidad_cobre_24_completo_filtrado.csv` | 314 muestras puntuales de laboratorio |
| `data/raw/assay_lab_AIQ_pi.csv` | salida de la calibración de fábrica del analizador |

**Si falta `assay_lab_courier_pi.csv` pero existe `data/raw/LABCOMPOSITO.csv`**
(export directo de PI con los tags sin renombrar, vía
`scripts/extraer_labcomposito.py`): regenerarlo con el mapeo `TAGS_COMPOSITO`
de `config.py`, no hay que volver a extraer de PI —

```python
import pandas as pd
from src.pipeline.config import TAGS_COMPOSITO, DATA_RAW
lab = pd.read_csv(DATA_RAW / "LABCOMPOSITO.csv").rename(columns={"t": "idx", **TAGS_COMPOSITO})
lab = lab.set_index("idx"); lab.index.name = None
lab.to_csv(DATA_RAW / "assay_lab_courier_pi.csv")
```

Verificado en sesión (2026-09-28): reproduce el backtest oficial exacto
(corr pFe = 0.508). Ver §8.5.

Luego:

```bash
pixi run python notebooks/10_recalibracion_composito.py   # regenera todos los .joblib
pixi run python notebooks/11_figuras_resultados.py        # regenera figuras y tabla
```

---

## 1. Estado actual: qué funciona y qué no

Evaluación contra el compósito de 12 h, n = 228 ventanas, abr–ago 2026, fuera de muestra.

| elemento | corr | R² | MAE | pendiente | fábrica | techo | veredicto |
|---|---|---|---|---|---|---|---|
| **Mo** | 0.953 | 0.906 | 0.157% | 0.880 | 0.948 | 0.961 | Funciona. En el techo físico |
| **Fe** | 0.508 | 0.244 | 0.914% | 0.317 | 0.470 | 0.743 | Supera a fábrica, no operativo |
| **Cu** | 0.484 | 0.086 | 1.390% | 0.268 | 0.517 | 0.634 | No funciona |
| **Zn** | — | — | — | — | — | — | Sin validar (el compósito no trae Zn) |

`fábrica` = calibración del fabricante del analizador. `techo` = correlación máxima
alcanzable, por colocación triple.

**Lectura operativa de la pendiente**: si el Fe real sube 1 punto, el modelo sube
0.32. Para control de proceso eso no sirve. Mo con 0.88 sí.

### Veredicto por etapa

| etapa | veredicto | evidencia |
|---|---|---|
| 0 adquisición | OK | — |
| 1 filtros deterministas | OK | 8.5% de rechazo legítimo |
| 1 IsolationForest | OK tras el arreglo | 31% → 2.7% de rechazo |
| 1 sensor congelado | **ROTO** | rechaza 6.6% por el forward-fill de PI |
| 2 ortogonalización | **DECORATIVA** | no-op algebraico + premisa falsa |
| 3 escalado | sin valor propio | solo alimenta la etapa 4 |
| 4 clustering | **NO APORTA** | +0.001 de correlación |
| 5 parte global | OK | es lo que produce las estimaciones |
| 5 parte local | **NO APORTA** | +0.001 |
| 6 inferencia | OK | integración probada |
| ruta B (Fe) | OK | 0.278 → 0.508 |

**El pipeline que funciona es: filtros → features → una regresión global.**

---

## 2. Hallazgos principales, en orden de importancia

### 2.1 Techo de información por colocación triple

Con tres estimaciones del mismo material (modelo / calibración de fábrica /
compósito) se despeja el error de cada una sin conocer la verdad. n = 184.

```
el   techo T~comp  fabrica~T  modelo~T | fabrica~comp  modelo~comp   margen
cu          0.634      0.815     0.798          0.517        0.505    0.128
fe          0.743      0.633     0.374          0.470        0.278    0.464
mo          0.961      0.986     0.996          0.948        0.958    0.003
```

Supuesto: errores mutuamente independientes. Modelo y fábrica comparten las
mismas mediciones XRF, así que la correlación entre ellos está inflada y los
techos quedan **subestimados** (son cotas conservadoras).

Validado por el usuario: el compósito lo corta personal de metalurgia justo antes
de la etapa de rayos X, sin proceso intermedio. Mismo material.

### 2.2 Deriva del instrumento: −22% anual

```
periodo   cu_lab  fe_lab    n2cu     n1fe    n6sc
2025Q3     26.15   27.28   39456    37122    1714
2026Q3     24.21   28.77   30929    28291    1365
```

Las intensidades caen 22–25% mientras las leyes reales casi no se mueven.
Sensibilidad `n1fe/pFe`: 1411 (2022) → 1222 (2026).

**Consecuencia**: la correlación intra-trimestre es MAYOR que la agrupada de 4
años. Para `fe_cu` vs Cu la agrupada (−0.207) es peor que cualquier trimestre
individual (−0.22 a −0.50). Agrupar el histórico mete la deriva como confusora.

### 2.3 Las fracciones de cierre son inmunes a la deriva

```
deriva relativa entre trimestres (2025Q3–2026Q3):
  n2cu_f   1.2%     n1fe_f   1.3%     n4mo_f   9.5%
  n3zn_f  17.6%     SumI/n6sc 17.4%
  n2cu    22.8%     n1fe    24.1%     n6sc    25.1%
```

`f_i = I_i / Σ metales` cancela la ganancia común del decaimiento de fuente.
`n6sc` NO decae al mismo ritmo, por eso normalizar por él reintroduce deriva.

### 2.4 `n6sc` no mide el sólido

Contra 309 valores de sólido medidos en laboratorio:

```
corr(n6sc,     pSol) = -0.065      <- esencialmente cero
corr(SumI,     pSol) = +0.659      <- este sí
corr(log SumI, pSol) = +0.655
```

La etapa 2 se llama "corrección por dilución (n6sc)" y su docstring afirma que
n6sc es "el sólido en la muestra / dilución". **Esa premisa es falsa.** No cambia
ningún número (la ortogonalización es un no-op algebraico de todos modos) pero la
justificación escrita en la tesis está mal.

Lo que sí es proxy del sólido es `SumI`, lo que valida retroactivamente `logSumI`
como feature de la ruta B, elegida por backtest sin saber por qué funcionaba.

### 2.5 La ortogonalización es un no-op algebraico

`m_ortho = m − a·n6sc − b`, y `n6sc` está en `FEATS_REGRESION`. Entonces
`span{m_ortho, n6sc} = span{m, n6sc}`. Para Ridge son idénticos:

```
pFe: A(4 ortho + n6sc) R2=0.351  |  B(4 crudas + n6sc) R2=0.352
pCu: A R2=0.690                  |  B R2=0.690
```

Solo afecta al clustering.

### 2.6 Régimen de mineral: el efecto es real, el instrumento no lo ve

**Ruteando por insolubles REALES de laboratorio** (CV 5-fold, n=314):

```
ley     global  local pIns  ganancia  azar(control negativo)
pFe      0.352       0.496    +0.144        -0.022
pCu      0.690       0.770    +0.080        -0.024
pMo      0.345       0.341    -0.004        -0.016
pZn      0.360       0.310    -0.050        -0.036
```

El efecto es grande y aparece justo en Fe y Cu, los dos que no funcionan. Mo no
lo necesita: su señal es directa (canal propio, CV 72%).

**Pero el espectro no puede identificar el régimen:**

```
Mejor asidero de la ganga en el espectro:  |corr| = 0.283  (SumI vs pIns)
Predecir insolubles desde intensidades:     R2 = 0.088
Ruteo por insolubles ESTIMADOS:             0.311  (peor que global 0.352)
Acuerdo tercil real vs estimado:            41.4%  (azar = 33%)
```

Hay 5 canales: Fe, Cu, Zn, Mo, dispersión. Ninguno mide silicatos. El régimen
está codificado en la ganga, y la ganga es invisible para el analizador. **Es un
límite físico, no de método.**

**Por qué los clusters actuales no sirven:**

1. Parpadean: racha mediana de 3 lecturas ≈ 1.1 h. Un régimen dura días.
2. No separan la química: dispersión entre/dentro = 0.21 en pIns, 0.05 en pSol.
3. Los coeficientes por cluster difieren (pFe, `n1fe_ortho`: 0.70 / 0.72 / 1.70)
   pero no replican: el modelo con interacción da −0.007 fuera de muestra. Con
   ~105 muestras por cluster esa dispersión es ruido de estimación.

**Descriptores temporales de régimen tampoco aportan** (medianas móviles de las
fracciones, estrictamente pasadas):

```
base 0.514 | +móvil 3d 0.505 | +móvil 7d 0.505 | +móvil 21d 0.464 | +interacción -0.130
```

Autocorrelación diaria de las fracciones: lag1d 0.37–0.75, lag7d 0.02–0.14,
lag30d ≈ 0. No hay regímenes de semanas visibles en el espectro.

### 2.7 Detección de ruptura en lugar de identificación de régimen

Los operadores del courier generan ecuación nueva cada mes, corrigen offset a
diario contra el lab de 12 h, y generan modelo nuevo cuando cambia el mineral.
Eso es **detección de cambio**, no clasificación. CUSUM sobre los residuos:

```
estrategia                                R2    MAE   corr   pend  n_cal  n_gat
calendario 30 d                        0.233  0.918  0.500  0.314      5      0
calendario 15 d (actual)               0.244  0.914  0.508  0.317      9      0
calendario 7 d                         0.238  0.919  0.504  0.317     19      0
calendario 3 d                         0.252  0.909  0.515  0.326     41      0
CUSUM h=5 + tope 30 d                  0.263  0.895  0.521  0.320      4      1
CUSUM h=3 + tope 30 d                  0.264  0.900  0.523  0.323      3      4
CUSUM h=3 + calendario 15 d            0.261  0.901  0.521  0.324      8      3
CUSUM h=2 + calendario 7 d             0.256  0.905  0.518  0.326     18      4
```

**Mejor desempeño con menos recalibraciones** (7 vs 9). El CUSUM detectó 4 cambios
en 4 meses ≈ una cadencia mensual, que coincide con la práctica de los operadores.

**Honestidad estadística**: +0.015 en correlación con n=228 está dentro del ruido
para una comparación aislada (error estándar ≈ 0.05). Lo que convence es que las
cuatro variantes con CUSUM superan a las cuatro por calendario sin excepción. La
mejora sólida es de **eficiencia operativa**, no de exactitud.

**NO implementado todavía.** Ver hilos abiertos.

### 2.8 Redes neuronales y PINN: descartados con medición

Split temporal 70/30, n_train=538, n_test=231:

```
===== CU =====                              R2    MAE   corr  slope
crudas + Ridge                           0.139  1.333  0.517  0.206
razon I/Isc (fisica XRF)                -0.440  1.785  0.414  0.203
Lachance-Traill (correccion de matriz)  -0.794  2.026  0.376  0.256
MLP(16,)                                -2.463  2.834  0.209  0.231
MLP(64,32)                              -2.923  3.086  0.166  0.165
MLP(128,64,32)                          -3.050  3.136  0.232  0.254

===== FE =====
crudas + Ridge                           0.161  0.929  0.450  0.157
estequiometrico (Cu + Fe_exc)            0.075  0.990  0.454  0.142
Lachance-Traill                         -2.357  1.830  0.062  0.073
MLP(64,32)                              -5.319  2.532 -0.072 -0.110
```

El orden es monótono en simplicidad. Un PINN además es un error de categoría: no
hay EDP que imponer, es una regresión estática. Lachance-Traill falla porque sus
coeficientes asumen instrumento estable y el tuyo deriva 22%/año.

### 2.9 Estequiometría calcopirita/pirita

```
Fe total       : 27.91 +- 1.41
Fe calcopirita : 22.29 +- 1.79    (0.879 * Cu, por CuFeS2)
Fe excedente   :  5.62 +- 2.39    (pirita/otros)
corr(Cu, Fe)     = -0.106
corr(Cu, Fe_exc) = -0.811
```

Interpretación mineralógica válida: la dilución por pirita gobierna la ley de
cobre. **Pero NO explica el mal desempeño del Fe** — si lo explicara, la
calibración de fábrica tampoco podría. Reparametrizar (predecir Cu y Fe_exc,
reconstruir Fe) da 0.075 contra 0.161 del directo: no ayuda.

### 2.10 El sólido de laboratorio no ayuda al hierro

`pSol` es valor de laboratorio, no existe en línea. Sólo se usaría estimándolo
desde las mismas intensidades, y ese estimado es función lineal de las features
que el modelo ya tiene. Backtest:

```
variante de uso de pSol                       R2    MAE   corr   pend
ninguna (ruta B actual)                    0.244  0.914  0.508  0.317
pSol_est como feature adicional            0.243  0.914  0.507  0.316
I_i / pSol_est (correccion de masa)       -0.196  1.063  0.321  0.278
log(pSol_est) como feature                 0.183  0.931  0.482  0.339
```

Además el modelo auxiliar extrapola mal: produjo `pSol_est` de hasta −13.4%.

Calidad del dato: `pSol` tiene un valor de 295%, un 0, y 4 bajo 5%. Limpiando a
`5 < pSol <= 100` quedan 309 de 314.

### 2.11 Los insolubles SÍ están en el PI (hallazgo del 2026-09-26)

Al extraer los tags del compósito directamente del PI se verificó el descriptor
real de cada uno, que hasta entonces se había **deducido, no confirmado**:

```
7100AIP101MAN = %Fe Promedio Concentrado colectivo  Courier Cobre
7100AIP102MAN = %Cu Promedio Concentrado colectivo  Courier Cobre
7100AIP103MAN = %Ins Concentrado colectivo  Courier Cobre      <-- INSOLUBLES
7100AIP104MAN = %Moly Promedio Concentrado colectivo  Courier Cobre
```

Dos consecuencias:

**1. El mapeo documental de `config.py` estaba mal** — tenía Fe y Cu invertidos y
suponía un tag de Zn inexistente. Error solo documental: ninguna etapa lo usaba.
Se verificó que las columnas de `assay_lab_courier_pi.csv` están bien nombradas,
idénticas al PI hasta 1e-6. **Todos los resultados anteriores se mantienen.**

**2. Los insolubles están disponibles para los 2594 compósitos**, no solo para
las 314 muestras puntuales. Media 9.40%, sd 2.75, rango 2.02–23.68, desde
2022-12-30, misma cadencia de 12 h.

Esto cambia el estado del hilo 5.4. En la sección 2.6 se concluyó que el ruteo
por insolubles daba +0.144 en pFe y +0.080 en pCu pero **no era desplegable**,
porque el insoluble solo existía en las 314 puntuales y no se puede predecir
desde el espectro (R² = 0.088). Eso era cierto para la fuente que se conocía
entonces. Con el tag del PI:

- Es **dato de laboratorio**, no de proceso -> permitido en la tesis.
- Llega en la **misma cadencia de 12 h** que la corrección de sesgo y el
  reentreno, así que encaja en el ciclo operativo ya implementado.
- En inferencia no se conoce el insoluble del turno en curso, pero sí el del
  turno anterior. Los regímenes duran días, así que el último valor conocido es
  un indicador razonable del régimen actual — y es exactamente la información
  que tiene el operador cuando decide generar una ecuación nueva.

Correlaciones del insoluble con las leyes del compósito (n=2590):
`corr(Ins, Cu) = -0.565`, `corr(Ins, Fe) = -0.476`. Consistente con la dilución
por ganga bajando la ley.

**PENDIENTE DE MEDIR**: si rutear por el insoluble del turno anterior reproduce
el +0.144 en el backtest walk-forward contra el compósito. Las ganancias de la
sección 2.6 están medidas en CV sobre las 314 puntuales, que ya se comprobó que
es optimista.

Datos en `data/raw/LABCOMPOSITO.csv`, extraídos con
`scripts/extraer_labcomposito.py`.


---

## 3. Correcciones al razonamiento durante la sesión

Tres veces el usuario corrigió una premisa y el diagnóstico cambió. Quedan
registradas para no repetirlas.

| # | Lo que se afirmó | Corrección | Impacto |
|---|---|---|---|
| 1 | "`works4cdp_assay` no es laboratorio; corr 0.436 con el compósito es techo de etiqueta" | Ambas son laboratorio. La discrepancia es grab vs compósito de 12 h: variabilidad de proceso, no error | El consejo "reentrenar contra otra etiqueta" era erróneo |
| 2 | "Fe es irresoluble, la fábrica también falla" | Se miró el R² (−0.485) sin separar sesgo de correlación. En correlación la fábrica saca 0.470 y el modelo 0.278 | Fe pasó de "límite físico" a "el mayor margen disponible" |
| 3 | "La arquitectura local no aporta, la hipótesis de la tesis no se sostiene" | Correcto sobre la implementación, falso sobre la hipótesis: con insolubles reales da +0.144 | La hipótesis se sostiene; falla la variable de ruteo |

**Lección transversal**: separar siempre sesgo de correlación antes de concluir.
Un R² negativo puede ser puro sesgo y esconder una correlación utilizable.

---

## 4. Los cuatro cambios implementados

Ver `notebooks/10_recalibracion_composito.py` y `docs/flujo_pipeline.md`.

1. **Entrenar Fe con el compósito de 12 h** (760 ventanas) en vez de las 314 puntuales.
2. **Ventana móvil de 270 días**, reentreno cada 15 días.
3. **Corrección de sesgo en línea**, últimos 20 compósitos, separada por turno
   (el sesgo de Fe difiere 0.211 entre día y noche).
4. **Detector de anomalías sobre fracciones de cierre** en vez de intensidades
   absolutas: rechazo de 31% → 2.7%.

**Solo se recalibró Fe.** Medido:

```
ley   antes   recalibrado   fábrica   techo   decisión
pFe   0.278       0.508       0.470   0.743   RECALIBRAR
pCu   0.505       0.389       0.517   0.634   NO -- empeora
pMo   0.958       0.951       0.948   0.961   NO -- ya en el techo
```

Cu empeora porque su modelo original se calibra con muestras puntuales
emparejadas al instante exacto de una lectura; para cobre esa señal es más limpia
que el promedio de una ventana de 12 h.

Barridos que fijaron los parámetros (backtest walk-forward sobre pFe):

- ventana: 120 d → 0.465 | 180 d → 0.489 | **270 d → 0.508**
- reentreno: 30 d → 0.500 | **15 d → 0.508**
- features: sin `fe_cu` → **0.508** | con `fe_cu` → 0.488 | `n6sc` indiferente

---

## 5. Hilos abiertos

### 5.1 Integrar el CUSUM como quinto cambio
Reemplazar el reentreno por calendario con gatillo por deriva de residuos.
Medido: 0.508 → 0.523 con 7 reajustes en vez de 9. Código de prueba en la sesión,
**no integrado al pipeline**.

### 5.2 Arreglar el filtro de sensor congelado
Rechaza 6.6% de lecturas válidas. Los datos de PI vienen forward-filled en grilla
de 15 min, así que la repetición es por diseño del tag, no falla del sensor.

### 5.3 Ortogonalizar contra `SumI` en vez de `n6sc`
`SumI` sí correlaciona con el sólido (0.659) y `n6sc` no (−0.065). La ruta A (Cu,
Mo) sí usa ortogonalización. **No probado.**

### 5.4 Ruteo por régimen de mineral — REABIERTO con los insolubles del PI

**Prioridad alta.** Ver sección 2.11: el insoluble está en el tag 7100AIP103MAN
para los 2594 compósitos, es dato de laboratorio (permitido en tesis) y llega
cada 12 h. Medir si rutear por el insoluble del turno anterior reproduce el
+0.144 de pFe en el backtest contra el compósito.

**Actualización 2026-09-28 — CERRADO, no se sostiene.** Se midió, con el
backtest walk-forward real y contra un control negativo (mismo tamaño de
grupos, sin información real): el control iguala o supera al insoluble real en
Fe, Cu y Zn; solo en Mo el insoluble real queda 0.03 por delante, dentro del
ruido que ya muestra ese mismo control entre sus propias celdas (0.15 a 0.97).
El +0.144 de la sección 2.6 era CV 5-fold con el insoluble **simultáneo**
(no disponible en producción); con el insoluble del turno anterior y
validación honesta, el efecto no se distingue de repartir clusters al azar.
Detalle completo, metodología y las otras tres pruebas relacionadas en §8.

### 5.4b Ruteo por variables de planta (solo producción)
Restricción del proyecto: **los datos de proceso son sensibles y no se pueden
usar en la tesis**; las intensidades sí se pueden anonimizar. Para producción sí
está permitido.

Con insolubles reales el ruteo da +0.144 en Fe. Las variables de planta que
deberían rastrear el régimen: potencia de molino y kWh/t, presión y energía
específica del HPGR, carga circulante y de zarandas, torque de espesador y dosis
de floculante, granulometría en línea.

Experimento propuesto: extraer 6–10 tags de PI desde 2025-07, alinear a las
ventanas de 12 h, medir (a) cuánto predicen el insoluble de laboratorio y (b) si
rutear por ellas reproduce el +0.144 en el backtest contra el compósito. Si (a)
da |corr| > 0.6, hay camino para llevar el Fe hacia el techo de 0.743.

**Actualización 2026-09-28.** La premisa (+0.144 reproducible) queda retirada
por §5.4. Este experimento pierde su justificación tal como estaba planteado
--- si se retoma, hay que remedir el punto (b) contra un control aleatorio
antes de invertir en extraer los 6-10 tags de planta.

### 5.5 Validar pZn o sacarlo de resultados
No tiene ninguna validación contra el compósito, que no trae Zn. Contra las 314
puntuales con split temporal: PLS(2) R²=0.349, RidgeCV R²=0.396.

### 5.6 Unificar el conjunto de evaluación
Cu aparece como 0.505 (colocación triple, 184 ventanas) y 0.484 (evaluación
final, 228 ventanas). Es diferencia de muestra, no de modelo. Para la tesis hay
que citar uno solo — se sugiere el de 228.

---

## 6. Restricciones y límites de alcance

1. **Datos de proceso prohibidos en la tesis.** Sensibles. Las intensidades se
   pueden anonimizar, las de proceso no. Sí permitidos para producción.
2. **Todo descansa en 13 meses.** Antes de 2025Q3 hay ~25 lecturas de intensidad
   por trimestre contra miles después. El dataset alineado arranca en 2025-07 y
   la evaluación cubre abr–ago 2026.
3. **La ruta B requiere mantenimiento cada 15 días.** Si nadie lo sostiene en
   planta, el 0.508 se degrada.
4. **`assay_lab_AIQ_pi.csv` NO es laboratorio.** 238 registros/día,
   autocorrelación lag-1 de 0.907. Es la salida del analizador con su calibración
   de fábrica. Úsese como línea base competidora, nunca como referencia.

---

## 7. Encuadre sugerido para la tesis

De "sistema de estimación multielemento" a **"caracterización del límite de
estimación de un analizador en línea, con un caso de éxito en molibdeno"**.

- **Mo** es el resultado principal: 99.6% de la señal disponible capturada,
  supera a la calibración de fábrica.
- **Fe** es el resultado metodológico: recalibración + ventana móvil + corrección
  de sesgo recuperan casi el doble de señal y superan al fabricante.
- **Cu** va como límite documentado, con techo 0.634 y explicación física.
- **Zn** sale de resultados o va explícitamente sin validación independiente.
- **Etapas 2–4** se reportan como hipótesis arquitectónica evaluada que **no** se
  sostuvo con ruteo espectral, con el +0.001 como evidencia. El insoluble real
  del turno anterior **tampoco** sostiene el ruteo bajo validación honesta
  (§5.4, §8): el efecto de régimen sigue siendo real en principio (§2.6), pero
  ninguna variable disponible hoy —ni el espectro, ni el insoluble rezagado, ni
  la cabeza del circuito— permite explotarlo en la práctica.

Aportes defendibles: cuantificación del techo por colocación triple, superar la
calibración comercial en los tres elementos en MAE, diagnóstico de deriva con
detector inmune, formalización de la práctica manual de los operadores (ecuación
mensual + offset diario), y un conjunto de resultados negativos rigurosos.

---

## 8. Sesión 2026-09-28 — cierre del hilo del insoluble y del régimen de mineral

Cuatro pruebas distintas de usar el insoluble (o una fuente similar) para
mejorar clustering/ruteo. Las cuatro veces, un control negativo (mismo tamaño
de grupo, sin información real) igualó o superó a la señal real. Metodología y
cifras completas de cada una, en el orden en que se corrieron.

### 8.1 Bug de datos encontrado y corregido: columnas `_ortho` mal etiquetadas

`data/processed/intensidad_cobre_24_completo_filtrado.csv` (y `_completo.csv`)
traían columnas `n1fe_ortho..n4mo_ortho` que **no son el ortogonalizado crudo**
(salida de la etapa 2) sino el **ya escalado** (salida de la etapa 3) —mismo
nombre en dos archivos, dos significados. Confirmado con un caso concreto
(timestamp 2022-12-05 15:38:30): `n1fe_ortho = 8378.12` en
`intensidad_cobre_24_orthogonalized.csv` (crudo) vs `n1fe_ortho = 3.16` en el
archivo supervisado (escalado).

**Impacto real: ninguno en el pipeline de producción.**
`src/pipeline/modelos.py:_recomputar_features` nunca lee esas columnas —
siempre recalcula ortho y cluster desde las intensidades crudas con los
artefactos congelados, que es como debe ser para que entrenamiento e
inferencia sean idénticos. El riesgo era solo para quien reutilizara esas
columnas a mano asumiendo que eran crudas (aplicar power+scaler otra vez sobre
un valor ya escalado colapsa el GMM a un solo cluster — así se detectó).

**Corregido:** se eliminaron esas 4 columnas de ambos CSV y se actualizó
`src/pipeline/dataset_supervisado.py` para que no vuelva a arrastrarlas. Las
columnas `cluster_kmeans`/`cluster_gmm` sí se conservan (son etiquetas, no
features, y sí las usa el pipeline).

### 8.2 Prueba 1 — Ruteo de pFe por insoluble del turno anterior (walk-forward)

Rediseño honesto de la sección 2.6: `ins_lag` = insoluble del compósito
**anterior** (nunca el simultáneo, que no existe en producción), terciles
calculados **solo con la ventana móvil vigente** en cada reentreno (270 d,
igual que la recalibración de Fe), backtest walk-forward sobre las mismas 228
ventanas fuera de muestra.

```
                  R2 global   R2 ruteo   ganancia
ruteo real (ins_lag)  0.160      0.242     +0.082
control (barajado)    0.160      0.220     +0.061
```

Ganancia real vs. control: **+0.021** — dentro del ruido. El +0.144 de la
sección 2.6 (CV 5-fold, insoluble simultáneo) no sobrevive ni al cambio de
validación ni al cambio a insoluble rezagado; no se separó cuál de los dos
pesa más. Código: `src/pipeline/ruteo_insoluble.py`,
`notebooks/12_experimento_ruteo_insoluble.py`.

### 8.3 Prueba 2 — Cabeza (rebose de hidrociclones) como feature de clustering

Se exploró una segunda fuente de régimen: el rebose de hidrociclones (cabeza
del circuito, antes de flotación), streams `Rebose Hidrociclones L1/L2`
(`works4cdp_assay.sample_id` 17 y 18, mismo `equipment_id=5` que el
Concentrado Colectivo).

- **L1**: 384 filas históricas pero solo **24 en los últimos 12 meses**
  (15 con `pIns`) — casi sin dato reciente.
- **L2**: 406 filas, 85/año recientes, pero `corr(n2cu_L2, pCu_lab) = 0.04`
  crudo — el usuario confirmó que hubo cambios de calibración en el histórico
  de intensidades de esta línea, lo que explica la correlación cruda pobre.

Con L2 alineado al stream de 15 min por último valor conocido (cadencia
mediana 72 h → cada lectura de cabeza se repite ~86 veces seguidas): el BIC se
desploma con la cabeza real, **pero colapsa exactamente igual con la cabeza
barajada** (mismo orden de magnitud) — es el artefacto de bloques repetidos,
no información real.

Repitiendo a la granularidad correcta (una fila por época de cabeza, ~266
filas en vez de 23,058): el traslape real de alta frecuencia es mínimo (solo
58 de 267 épocas con ≥5 lecturas de courier, porque el courier de alta
frecuencia solo existe desde 2025-07 y la cabeza tiene historia desde 2021).
Con n=58 el resultado es ruido en ambas direcciones (real pierde en silhouette
GMM contra el control, gana en BIC) — **inconcluso, no descartado**: falta
traslape de alta frecuencia, no evidencia de que no sirva. Ver §8.4 sobre por
qué no hay una vía de mayor frecuencia disponible para esta variable en
particular (si la hay para la cabeza, no se investigó).

Código: `notebooks/13_experimento_cluster_cabeza.py`,
`notebooks/14_experimento_cluster_cabeza_epocas.py`.

### 8.4 Confirmación: el courier NO mide insolubles

Se investigó si el analizador mismo genera una intensidad de insoluble
(`a7a7`/`a6sol` en `works4cdp_assay`) que sirviera de proxy de alta frecuencia,
evitando la cadencia de 12 h del compósito de laboratorio.

- El canal crudo análogo a `n1fe..n4mo` para insoluble es `n7ech7` — está
  **0/548 no nulo** para `sample_id=24` (Concentrado Colectivo,
  `equipment_id=5`). Sí existe en otros equipos del mismo analizador
  multiplexado (`equipment_id=1`: ~70-80% de llenado), pero no en el stream
  que le interesa a este proyecto.
- `a7a7`/`a6sol` sí tienen valores reales (no son el centinela `-1`) para
  `sample_id=24`, pero **confirmado directamente con el equipo/operador**: son
  una estimación que el operador calcula a partir de las demás leyes (Fe, Cu,
  Zn, Mo), no una lectura del sensor. Medido: `corr(a7a7, pIns_lab) = 0.116` —
  consistente con que es una fórmula, no una medición, y con el R²=0.088 de la
  sección 2.11 (predecir insoluble desde el espectro).

**Conclusión: no hay atajo de alta frecuencia para el insoluble en este
stream.** La única fuente legítima sigue siendo el compósito de laboratorio vía
PI (12 h) o las muestras puntuales (`pIns` en la BD).

### 8.5 Prueba 3 — Insoluble en el clustering (pipeline 0.5→1→2→3→4)

Diseño acordado: `ins_t` (último insoluble conocido, merge_asof hacia atrás)
entra en la etapa 3 (Yeo-Johnson + StandardScaler) junto a las 4 features
ortogonalizadas de siempre; etapas 1-2 sin cambios. Evaluado con
`eta² = SS_entre_clusters / SS_total` contra las leyes reales de las 314
muestras puntuales (recalculando el ortogonalizado crudo por timestamp desde
el stream — ver §8.1, el bug que esto destapó).

```
ley    oficial(prod)  base(refit)  +ins_t real  +ins_t control
pFe        0.194         0.215        0.176          0.212
pCu        0.140         0.135        0.056          0.135
pMo        0.147         0.140        0.117          0.137
pZn        0.065         0.073        0.095          0.074
```

El insoluble real separa **peor** Fe, Cu y Mo que sin él, y el control
(insoluble barajado en el entrenamiento) rinde igual que sin insoluble en las
tres. Solo Zn mejora un poco, con eta² bajo en todos los casos (más
compatible con ruido). Código:
`notebooks/15_pipeline_insoluble_clustering.py`.

### 8.6 Prueba 4 — Insoluble como variable de ruteo en la etapa 5 (modelos locales)

Última prueba, con la métrica que de verdad importa: R² de los modelos locales
contra ley real, mismo `KFold(5, shuffle=True)` que ya usa `modelos.py`, única
diferencia el cluster usado para rutear.

```
ley   R2 local (oficial)  R2 local (+ins_t real)  R2 local (control barajado)
pFe         0.161                 0.218                    0.305
pCu         0.392                 0.500                    0.604
pMo         0.620                 0.659                    0.625
pZn         0.248                 0.228                    0.334
```

El control aleatorio **gana en 3 de 4 leyes**, con margen (hasta −0.10 en Cu).
Solo en Mo el insoluble real gana por +0.033 — dentro del ruido que ya muestra
el propio control entre sus celdas individuales (R² de 0.15 a 0.97 en la misma
corrida). Veredicto: **no rutear por insoluble.** Figuras:
`reports/fig_insoluble_vs_control.png`, `reports/fig_tendencia_insoluble.png`.
Código: `notebooks/16_modelos_locales_con_insoluble.py`,
`notebooks/17_control_ruteo_insoluble.py`, `notebooks/18_*`, `notebooks/19_*`.

### 8.7 Reconstrucción de `assay_lab_courier_pi.csv` y verificación cruzada

Este archivo (esperado por `cargar_composito()`/celda 10) no existía en este
clon del repo — solo `data/raw/LABCOMPOSITO.csv` (export crudo de PI, tags sin
renombrar). Se regeneró aplicando el mapeo `TAGS_COMPOSITO` ya definido en
`config.py` (ver receta en §0). Al correr `notebooks/10_recalibracion_composito.py`
con el archivo regenerado, el backtest reprodujo **exacto** lo ya documentado
(`corr pFe = 0.508`, fábrica 0.470) — verificación cruzada de que
`LABCOMPOSITO.csv` y el `assay_lab_courier_pi.csv` original son la misma
fuente, solo con formato distinto.

### 8.8 Figura de validación filtrada desde 2026-05-01 ("la mejor estrategia")

Con el backtest reconstruido, se regeneró la evaluación unificada de la celda
11 (Fe recalibrado + Cu/Mo modelo original, contra el mismo compósito)
filtrada a partir del 2026-05-01 (216 de las 228 ventanas totales, que ya
cubrían casi por completo ese rango):

```
elemento   n    R2    MAE    corr   pendiente
Fe        216  0.254  0.921  0.516    0.324
Cu        216  0.071  1.417  0.470    0.251
Mo        216  0.905  0.162  0.953    0.882
```

Consistente con las cifras de todo el período abr-ago. Código:
`notebooks/20_figura_validacion_desde_mayo.py`, figura en
`reports/fig_validacion_desde_mayo.png`.

---

## 9. Sesión 2026-09-29 — corrector Kalman, recuperación de filas, y dos hilos más cerrados

### 9.0 Recomendación — la mejor opción para el mejor resultado

De todo lo medido esta sesión (aquí + §8), **una sola cosa vale la pena
adoptar en producción**: el corrector Kalman de sesgo para **pFe**
(§9.2) — mejora corr, R² y MAE a la vez, mismo backtest, sin tocar el
modelo. El resto son negativos ya sea confirmados (insoluble, cabeza,
orden escalado/ortogonalización) o resultados mixtos que requieren una
decisión editorial, no una adopción automática (Kalman en Cu, §9.3). El
filtro de `-9999` recuperable (§9.1) es una corrección de robustez, no
cambia ningún número hoy. `pSol` de alta frecuencia (§9.5) sigue
bloqueado por falta de instrumentación/ingesta.

```
cambio                              estado          accion recomendada
------------------------------------------------------------------------
Kalman en pFe (bias, q=0.1)         MEJORA (3/3)    ADOPTAR
Kalman en pCu (bias, q=0.0001)      MIXTO           decidir por criterio
                                                      (R2/MAE si o corr no)
filtro -9999 recuperable            robustez        ya aplicado, sin riesgo
orden escalado/ortogonalizacion     NO AYUDA         dejar como esta
insoluble (4 pruebas, sesion previa) NO AYUDA        cerrado (S8)
cabeza/rebose hidrociclones          INCONCLUSO      pendiente de mas dato
pSol alta frecuencia                 NO EXISTE        pendiente de ingesta
```

### 9.1 Filtro de limpieza: filas con error parcial ya no se borran

`_filtrar_invalidos()` (`src/pipeline/limpieza.py`) borraba la fila
completa si **cualquiera** de los 5 canales traía el código de error
`-9999`, perdiendo los otros 4 canales válidos por uno malo. Ahora solo
se borra si **los 5** están en error; si es parcial, el canal puntual
pasa a `NaN` explícito y la fila se conserva. Se corrigió también
`limpiar()` para que esas filas recuperadas (no evaluables por el
IsolationForest, por el `NaN`) no se pierdan igual dos pasos después --
antes se descartaban en silencio sin dejar rastro. De paso, un bug
latente en `features_cierre()` (`features.py`): la suma de cierre usaba
`skipna=True` por defecto, así que un metal en `NaN` no anulaba la
fracción de los otros 3 (las calculaba con un denominador entendido de
menos) -- corregido con `min_count=len(METALES)`.

**Con los datos actuales, esto no cambia ningún número**: las 4 filas
con `-9999` de hoy tienen los 5 canales en error, no parcial. Verificado
con 4 casos sintéticos (parcial en `n6sc`, parcial en un metal, total,
y el caso real sin cambios) -- los 4 se comportan como se espera.
Corrección preventiva/estructural, no retroactiva.

### 9.2 Corrector Kalman de sesgo para pFe -- RECOMENDADO ADOPTAR

`CorrectorSesgo` (el "cambio 3" de la recalibración de Fe) promedia los
últimos 20 residuos con ventana fija -- viola la regla del proyecto de
usar EWMA, nunca media móvil simple. Se implementó `CorrectorKalman`
(`src/pipeline/corrector_kalman.py`): sesgo como caminata aleatoria
escalar por `(ley, turno)`, con el ruido de proceso escalado por tiempo
transcurrido (maneja el muestreo irregular, que una ventana de N no
puede). Misma interfaz que `CorrectorSesgo`, inyectable en
`calibracion_composito.backtest(corrector=...)` sin tocar esa función
(comportamiento por defecto sin cambios si no se pasa nada).

Backtest walk-forward, mismos 228 puntos que ya valida el resto del
pipeline:

```
                          corr      R2     MAE
media movil (actual)     0.508   0.244   0.914
kalman q=0.1 (nuevo)     0.576   0.316   0.863
diferencia              +0.068  +0.072  -0.051
```

Barrido de `q_por_dia` (único hiperparámetro libre, r=1 fijo): sube
monótono hasta saturar en 0.1-0.2 y **cae de nuevo** en q>=0.5 pese a
que el corr seguía subiendo un poco ahí -- se eligió por R2, no por la
métrica que más convenía. Las tres métricas mejoran a la vez, a
diferencia de todo lo demás probado esta sesión. Figuras:
`reports/fig_tendencia_kalman_vs_lab.png` (Fe solo, media móvil vs
Kalman vs lab) y `reports/fig_validacion_kalman.png` (los 3 elementos,
mismo formato que la celda 11, con Fe corregido por Kalman: corr Fe
0.576, Cu 0.484 y Mo 0.953 sin cambios). Código:
`notebooks/21_backtest_corrector_kalman.py`,
`notebooks/22_figura_validacion_kalman.py`.

**No implementado en el bundle de producción todavía** --
`models/calibracion_composito.joblib`/`corrector_sesgo.joblib` siguen
con `CorrectorSesgo`. Adoptar significa cambiar
`notebooks/10_recalibracion_composito.py` para usar `CorrectorKalman`
en el paso 4 y volver a correr el ciclo completo.

### 9.3 Corrector Kalman de sesgo para pCu -- resultado mixto, no automático

Cu **nunca tuvo** un corrector de sesgo (el "cambio 3" solo se aplicó a
Fe), pese a que la bitácora ya medía que el sesgo de Cu también
deambula por trimestre incluso en la calibración de fábrica (§2.3:
`sesgo_cu` 0.462 -> 1.109 -> 0.055 -> 0.018). Se probó el mismo
mecanismo, sin tocar el modelo de Cu:

```
                          corr      R2     MAE
sin corrector (actual)   0.484   0.086   1.390
kalman q=0.0001 (nuevo)  0.463   0.207   1.283
diferencia              -0.021  +0.120  -0.107
```

`q` óptimo confirmado en el límite q->0 (no es borde de grilla: 0.0001,
0.0005 y 0.0 dan el mismo resultado, 0.207/0.463/1.281-1.283). R² y MAE
mejoran sustancial -- el modelo de Cu tenía un sesgo negativo
persistente, visible en el panel de error de
`reports/fig_tendencia_kalman_cu_vs_lab.png`, y corregirlo ayuda mucho
a esas dos métricas, que son muy sensibles a sesgo sistemático. Pero
corr **baja** -- el corrector arregla el desplazamiento, no la forma de
la relación (que es lo que corr mide), y reaccionar a la deriva
reciente mete algo de varianza. Interpretación: Cu no falla por un
offset simple (como sí parece ser el caso de Fe), falla por ruido de
medición en el canal -- corregir el offset no resuelve el problema de
fondo. **Decisión pendiente de criterio editorial**: si lo que importa
para la tesis es exactitud absoluta (MAE/R²), adoptarlo; si es el techo
de colocación triple (corr), no. Código:
`notebooks/23_backtest_corrector_kalman_cu.py`.

### 9.4 ¿Escalar antes o después de ortogonalizar? -- no ayuda, y van 6 controles negativos seguidos

Solo podía importar para clustering (para regresión, §2.5 ya prueba que
es un no-op algebraico). Medido con GMM k=3, sobre las 314 muestras con
ley real, contra un control aleatorio del mismo tamaño de grupo:

```
ley     A (actual)   B (invertido)   control (azar, tam. B)
pFe        0.166         0.207              0.310
pCu        0.418         0.482              0.590
pMo        0.620         0.636              0.594
pZn        0.264         0.240              0.303
```

El control supera a B en 3 de 4 leyes -- la aparente mejora de invertir
el orden es ruido de partición, no señal del reordenamiento. **Dejar el
pipeline como está.**

**Hallazgo colateral, más importante que la pregunta original**: esta
es la **sexta** vez en la sesión (insoluble x4, cabeza, y ahora esto)
que un control aleatorio iguala o supera al cambio real bajo el
`KFold(shuffle=True)` de la etapa 5. Con el tamaño de muestra actual
(~307-314), casi cualquier repartición de cluster puede parecer una
mejora si no se contrasta contra el azar. Esto pone en duda no solo
cada cambio probado con esta validación, sino **el método de
evaluación de la etapa 5 en sí** -- que sigue siendo el mismo
`KFold(shuffle=True)` que ya se identificó como optimista en
`data2Tesis` (ADR de `decisiones.md`: "Validación temporal: de `KFold`
barajado a walk-forward con purga y embargo") y que **nunca se corrigió
en V2** para la etapa 5, solo para la recalibración de Fe contra el
compósito (celda 10). Queda como hilo abierto de mayor prioridad que
cualquiera de los específicos de esta sesión. Código:
`notebooks/24_orden_escalado_ortogonalizacion.py`,
`notebooks/25_impacto_orden_en_cu_mo.py`.

### 9.5 `pSol` de alta frecuencia -- no existe en ninguna fuente revisada

Se evaluó si había sólidos de laboratorio a alta frecuencia (para
ortogonalizar contra dilución real en vez de `n6sc`, que §2.4 ya probó
que no mide el sólido). Revisadas las dos fuentes:

- **Compósito de PI** (`LABCOMPOSITO.csv`, cada 12h): solo 4 tags --
  Fe, Cu, Ins, Mo (`7100AIP101-104MAN`). **No trae `pSol`.**
- **Base de datos** (`works4cdp_assay`): `pSol` existe, pero a la
  misma cadencia baja que el insoluble -- mediana de 67.8h entre
  lecturas, 366 muestras desde 2021-07 para `sample_id=24`
  (Concentrado Colectivo, el mejor de los 15 `sample_id` revisados).

**No hay sólidos de alta frecuencia en ningún sistema al que se tenga
acceso hoy.** Falta ingresar ese dato a la base de datos (instrumentación
u otra fuente) antes de poder medir si ayuda -- queda como hilo abierto,
bloqueado por adquisición, no por análisis.

---

## 10. Sesión 2026-09-30 — revisión etapa por etapa (0-4) y grupos de mineral

Pipeline reescrito desde la adquisición hasta el clustering, con datos hasta
2026-09-30. ADR correspondientes en `Proyects/docs/decisiones.md` (4 entradas
del 2026-09-30).

### 10.1 Etapa 0 — tres fuentes, un crudo cada una
`courier_bd.csv` (BD, 548 filas, 366 con ley), `courier_pi.csv` (PI
`recorded`, 413,834 eventos, desde 2025-07-13), `composito_pi.csv` (2,601
ensayos 12 h). Token PI por `PI_TOKEN`/`~/.pi_token`; pasarela 10.25.18.85:5173.
Los tags `_296290_ConcFinal_*` son el `sample_id=24`: 125/125 muestras calzan,
0 de ~2,700 de las otras 24 líneas.

### 10.2 Lo que es el PI, medido
- El courier da **una lectura del concentrado cada ~21 min** (mediana 1,251 s);
  el PI la re-archiva cada 100 s (`ExcMax=100` en los 5 tags) → ~14 eventos
  por lectura. No es sensor congelado.
- 35,925 eventos todo-0 (analizador detenido, 177 días) y 41 todo -9999.
- `n6sc` archivado 1-60 s aparte de los metales, concentrado en mar-jul 2026
  (pico 6,399 filas en mayo). A veces `n1fe` llega 1 s antes que el resto.
  La 1a reconstruye: 25,429 lecturas hasta el 30-sep, 0 con metales incompletos.

### 10.3 BD ↔ PI
Calce por valor de los 4 metales redondeados: **125/125** desde 2025-07-13
(las 18 "perdidas" de mar-jul eran lecturas partidas por `n6sc`). Hora BD
~6 min detrás del inicio de la lectura en el PI (p95 28 min). **3 filas de
jun-2026 con la hora corrida exactamente 12 h** (AM/PM en la BD). 4 `instance`
duplicadas = misma lectura registrada con y sin leyes. `instance` no sirve de llave.

### 10.4 Leyes
Ley en 0 = no analizada (pZn 72 en 2023-06..2025-10: n3zn igual con y sin
pZn=0 → el Zn estaba ahí) → NaN. Ley sospechosa = |z| > 5 y su canal del
courier no la respalda (|z| ≤ 2 o signo opuesto): 4 filas — 2023-02-24
(pFe 4.16/pCu 0.43), 2025-04-18 (pMo 16.77), 2026-02-16 (pSol 295),
2026-05-29 (pFe 39.18). Los pZn/pMo altos reales traen su canal alto.

### 10.5 Etapa 1c — corte y anomalías
Corte train ≤ 2026-05-31, producción ≥ 2026-09-01. IsolationForest (fracciones
de cierre, contamination 2 %): train 2.0 %, test 9.2 % (junio 14.7 %,
Mo/Zn altos), producción 5.9 %. El compósito no decide si es mineral o
instrumento (corr fracción anómala vs Mo lab 0.06). `valida_ley`: 361/366.

### 10.6 Etapa 2 — por qué se eliminó la ortogonalización
Media por semestre de `n2cu_ortho`: +6,607 (2022H2) → −5,133 (2026H2), pCu
24.8 → 22.9. R² metal~n6sc 0.01-0.22. Log-cocientes vs química del lab:
Fe/Cu 0.73, Zn/Cu 0.86, Mo/Cu 0.95. GMM k=2 sobre intensidades absolutas
(lo de V1): silhouette 0.19, log(ΣI) igual en ambos grupos (11.20/11.18) →
**no se reproduce la separación por dilución de V1** con estos datos.

### 10.7 Dilución, medida
OLS `feature ~ química_lab + pSol` (n=343): lr_fe_cu t_pSol=+12.1
(p10→p90 = +0.91 sd), lr_mo_cu −10.5 (−0.37 sd), lr_zn_cu −1.3 (sin efecto).
Corrección con `dil = logSumI − EWMA(90 d, pasado)`: corr(dil, pSol)=0.50-0.55.
Aplicada solo a Fe/Cu (b=+0.104): 0.91 → 0.73 sd, corr química 0.75 → 0.80.
En Zn/Cu la corrección creaba un efecto (t=+3.3): no se aplica.

### 10.8 Etapa 4 — k y grupos

```
k   silhouette  ARI mitades     (con corrección de dilución)
2     0.290       0.958
3     0.230       0.661
4     0.249       0.856   <- elegido
5     0.227       0.541
```

| grupo | BD (n) | compósito (n) | pSol | train | sept |
|---|---|---|---|---|---|
| Cu | pCu 25.64 (117) | cu 25.77 (186) | 23.6 | 37 % | 18 % |
| Mo | pMo 2.41 (60) | mo 2.46 (207) | 21.5 | 30 % | 60 % |
| Zn | pZn 0.40 (61) | — | 20.0 | 19 % | 18 % |
| Fe | pFe 29.53, pCu 22.94 (123) | fe 28.77 (43) | 27.4 | 14 % | 5 % |

Septiembre por semana (% Mo del grupo / Mo compósito): 44/2.13, 95/2.71,
50/1.98, 53/1.73, 36/1.63 (hasta 30-sep 00:00: Cu 42→56 %). Racha mediana en
un grupo: 2 lecturas (~40 min), p90 21. Artifact:
https://claude.ai/artifact/LXzMFBmpFoooh5XwG88hfZ

### 10.9 Abierto
Etapa 5 (features y `valida_ley` nuevos, walk-forward), etapa 6,
`run_pipeline.py` y notebook 10 rotos por la nueva interfaz de 1-4.
Reconciliación con `tesis_ucspv2/` (etapa 2 y K=3 descritos allí ya no valen).
