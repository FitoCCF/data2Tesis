# Flujo del pipeline — estimación de leyes en concentrado de cobre

Diagramas del pipeline **después** de la recalibración contra el compósito de 12 h
(celda 10). Los bloques marcados con borde grueso azul son los cuatro cambios
introducidos; el resto es el pipeline original.

Fuente de verdad del código:

| Etapa | Módulo |
|---|---|
| 1 limpieza | `src/pipeline/limpieza.py` |
| 2 ortogonalización | `src/pipeline/ortogonalizacion.py` |
| 3 escalado | `src/pipeline/escalado.py` |
| 4 clustering | `src/pipeline/clustering.py` |
| 5 modelos locales | `src/pipeline/modelos.py` |
| 6 inferencia | `src/pipeline/inferencia.py` |
| features de cierre | `src/pipeline/features.py` |
| recalibración vs compósito | `src/pipeline/calibracion_composito.py` |

---

## 1. Flujo de entrenamiento / calibración

Dos rutas de calibración que conviven, porque las dos fuentes de laboratorio son
muestras físicamente distintas y cada ley se calibra con la que le conviene.

```mermaid
flowchart TB
    %% ---------- FUENTES ----------
    subgraph FUENTES["FUENTES DE DATOS"]
        direction LR
        PG[("Postgres works4cdp<br/>sample_id = 24")]
        PI[("PI System<br/>tags del courier")]
        INT["Intensidades del analizador<br/>n1fe n2cu n3zn n4mo n6sc<br/>~40k lecturas, 1 cada 22 min"]
        GRAB["Laboratorio PUNTUAL<br/>works4cdp_assay<br/>pFe pCu pZn pMo pIns pOx pSol<br/>n = 314, toma manual 09-13 h"]
        COMP["Laboratorio COMPOSITO 12 h<br/>tags 7100AIP101-104MAN<br/>cu fe mo<br/>n = 2590, turno dia 07:30 / noche 19:30<br/>cortado antes del multiplexor"]
        PG --> INT
        PG --> GRAB
        PI --> COMP
    end

    %% ---------- RUTA A: PIPELINE ORIGINAL ----------
    subgraph RUTA_A["RUTA A - pipeline original - estima pCu, pMo, pZn"]
        direction TB
        E1["ETAPA 1 — Limpieza<br/>1. quita -9999<br/>2. quita 5 canales en cero<br/>3. quita vector repetido 3 o mas veces"]
        E1B["IsolationForest<br/>contaminacion 2%, 200 arboles"]
        E2["ETAPA 2 — Ortogonalizacion<br/>metal_ortho = metal menos LinReg de n6sc<br/>una regresion por metal"]
        E3["ETAPA 3 — Escalado<br/>Yeo-Johnson y luego StandardScaler<br/>sobre las 4 columnas _ortho"]
        E4["ETAPA 4 — Clustering<br/>KMeans y GMM, k = 3<br/>la etapa 5 usa GMM"]
        DS["Dataset supervisado<br/>merge por date + time exacto<br/>intensidades crudas + _ortho + cluster + leyes"]
        E5["ETAPA 5 — Modelos locales<br/>un modelo por ley y cluster + uno global<br/>features: 4 _ortho + n6sc"]
        RUT["Tabla de ruteo<br/>local solo si el R2 CV local supera al global"]
        E1 --> E1B --> E2 --> E3 --> E4 --> DS --> E5 --> RUT
    end

    %% ---------- RUTA B: RECALIBRACION ----------
    subgraph RUTA_B["RUTA B - recalibracion vs composito - estima pFe"]
        direction TB
        C0["Filtros deterministas<br/>-9999, ceros, faltantes"]
        C1["Submuestreo a bloques de 30 min<br/>descorrelaciona: autocorr lag-1 = 0.907"]
        C2["Alineacion con el composito<br/>ventana CENTRADA de 12 h, lag 0<br/>se exigen al menos 8 bloques por ventana<br/>760 ventanas validas"]
        C3["Features de cierre<br/>f_i = I_i / suma de los 4 metales<br/>+ log de la suma"]
        C4["Ventana movil de 270 dias<br/>reentreno cada 15 dias"]
        C5["RidgeCV<br/>alpha por validacion cruzada interna"]
        C6["Backtest walk-forward<br/>predice, corrige, RECIEN revela el real"]
        C7["CorrectorSesgo<br/>ultimos 20 compositos, separado dia/noche"]
        C0 --> C1 --> C2 --> C3 --> C4 --> C5 --> C6 --> C7
    end

    %% ---------- ARTEFACTOS ----------
    subgraph ART["ARTEFACTOS CONGELADOS en models"]
        direction LR
        A1["anomaly_detector<br/>.joblib"]
        A2["orthogonalization<br/>_regressions.joblib"]
        A3["power_transformer<br/>+ scaler.joblib"]
        A4["kmeans_model<br/>+ gmm_model.joblib"]
        A5["modelos_locales<br/>_por_cluster.joblib"]
        A6["calibracion<br/>_composito.joblib"]
        A7["corrector<br/>_sesgo.joblib"]
        A8["manifiesto<br/>_recalibracion.json"]
    end

    INT --> E1
    INT --> C0
    GRAB --> DS
    COMP --> C2

    E1B -.-> A1
    E2 -.-> A2
    E3 -.-> A3
    E4 -.-> A4
    RUT -.-> A5
    C5 -.-> A6
    C7 -.-> A7
    C6 -.-> A8

    classDef fuente fill:#f0efec,stroke:#8a8a85,stroke-width:1px,color:#0b0b0b
    classDef origen fill:#fcfcfb,stroke:#8a8a85,stroke-width:1px,color:#0b0b0b
    classDef cambio fill:#dceafc,stroke:#2a78d6,stroke-width:3px,color:#0b0b0b
    classDef arte fill:#fdf0e8,stroke:#eb6834,stroke-width:1px,color:#0b0b0b

    class PG,PI,INT,GRAB fuente
    class COMP,E1B,C0,C1,C2,C3,C4,C5,C6,C7 cambio
    class E1,E2,E3,E4,DS,E5,RUT origen
    class A1,A2,A3,A4,A5,A6,A7,A8 arte
```

**Por qué dos rutas.** No es redundancia: el backtest walk-forward midió que la
recalibración contra el compósito **mejora Fe pero empeora Cu**.

| ley | antes | recalibrado | fábrica | techo | decisión |
|---|---|---|---|---|---|
| pFe | 0.278 | **0.508** | 0.470 | 0.743 | ruta B |
| pCu | 0.505 | 0.389 | 0.517 | 0.634 | ruta A |
| pMo | 0.958 | 0.951 | 0.948 | 0.961 | ruta A |

Cu se calibra mejor con la muestra puntual porque está emparejada al instante
exacto de una lectura; para cobre esa señal es más limpia que el promedio de una
ventana de 12 h. Mo ya está en el techo de información del sistema.

---

## 2. Flujo de inferencia en producción

```mermaid
flowchart TB
    IN["Lectura nueva del analizador<br/>n1fe n2cu n3zn n4mo n6sc"]

    subgraph GATE["COMPUERTA DE CALIDAD - etapa 1"]
        direction TB
        G1{"-9999 o<br/>todo en cero?"}
        G2{"vector repetido<br/>3 o mas veces en<br/>buffer de 10?"}
        G3{"IsolationForest<br/>sobre fracciones<br/>de cierre?"}
        STOP(["DESCARTA<br/>no se estima nada"])
        FLAG1["marca: sensor congelado<br/>confiable = false"]
        FLAG2["marca: anomalia<br/>confiable = false"]
    end

    subgraph TRANS["TRANSFORMACIONES"]
        direction TB
        T1["Ortogonalizacion vs n6sc<br/>regresiones congeladas"]
        T2["Yeo-Johnson y luego StandardScaler"]
        T3["GMM asigna cluster 0, 1 o 2"]
        T4["Features de cierre<br/>4 fracciones + log de la suma"]
    end

    subgraph EST["ESTIMACION - ruteo por ley"]
        direction TB
        R1{"que ley?"}
        RA["RUTA A — modelos_locales_por_cluster<br/>tabla de ruteo decide local o global<br/>features: 4 _ortho + n6sc"]
        RB["RUTA B — calibracion_composito<br/>RidgeCV entrenado vs composito<br/>features: cierre + log suma"]
        SESGO["CorrectorSesgo<br/>resta el offset del turno<br/>dia o noche segun la hora"]
    end

    OUT["Resultado<br/>leyes: pFe pCu pMo pZn<br/>cluster, ruteo por ley,<br/>alertas, confiable"]

    IN --> G1
    G1 -->|"si"| STOP
    G1 -->|"no"| G2
    G2 -->|"si"| FLAG1
    G2 -->|"no"| G3
    FLAG1 --> G3
    G3 -->|"si"| FLAG2
    G3 -->|"no"| T1
    FLAG2 --> T1

    T1 --> T2 --> T3 --> R1
    T1 --> T4 --> R1

    R1 -->|"pCu pMo pZn"| RA
    R1 -->|"pFe"| RB
    RB --> SESGO
    RA --> OUT
    SESGO --> OUT

    classDef entrada fill:#f0efec,stroke:#8a8a85,stroke-width:1px,color:#0b0b0b
    classDef origen fill:#fcfcfb,stroke:#8a8a85,stroke-width:1px,color:#0b0b0b
    classDef cambio fill:#dceafc,stroke:#2a78d6,stroke-width:3px,color:#0b0b0b
    classDef alto fill:#fbe3e3,stroke:#e34948,stroke-width:1px,color:#0b0b0b
    classDef salida fill:#e3f5ee,stroke:#1baf7a,stroke-width:2px,color:#0b0b0b

    class IN entrada
    class G1,G2,T1,T2,T3,R1,RA origen
    class G3,T4,RB,SESGO cambio
    class STOP,FLAG1,FLAG2 alto
    class OUT salida
```

**Detalle del ruteo.** `pFe` se salta por completo la tabla de ruteo local/global
y va al modelo recalibrado; las otras tres leyes conservan el comportamiento
original. El campo `ruteo` de la salida lo deja explícito: `composito` para Fe,
`local` o `global` para el resto.

**Detalle de la compuerta.** El IsolationForest ahora evalúa en el espacio de
fracciones de cierre, no de intensidades absolutas. El artefacto lleva la marca
`_robusto` y la inferencia la lee para transformar la lectura antes de evaluarla:
aplicar un detector de cierre a intensidades absolutas marcaría todo como anomalía.

---

## 3. Ciclo operativo

La ruta B **no es un modelo congelado**: tiene un ciclo de mantenimiento que hay
que sostener, porque el instrumento pierde ~22% de cuentas al año.

```mermaid
flowchart LR
    subgraph DIARIO["CONTINUO"]
        D1["Analizador<br/>1 lectura cada 22 min"]
        D2["Inferencia<br/>estimacion de leyes"]
        D1 --> D2
    end

    subgraph T12["CADA 12 HORAS"]
        H1["Llega el composito<br/>del turno"]
        H2["corrector.actualizar<br/>ley, prediccion, real, turno"]
        H3["Re-serializar<br/>corrector_sesgo.joblib"]
        H1 --> H2 --> H3
    end

    subgraph D15["CADA 15 DIAS"]
        Q1["Re-alinear intensidades<br/>con los compositos nuevos"]
        Q2["Re-entrenar con la ventana<br/>movil de 270 dias"]
        Q3["Re-serializar<br/>calibracion_composito.joblib"]
        Q1 --> Q2 --> Q3
    end

    subgraph VIG["VIGILANCIA"]
        V1{"tasa de anomalia<br/>supera 10 por ciento?"}
        V2["Re-ajustar el<br/>detector de anomalias"]
        V3{"corr de Fe<br/>cae bajo 0.47?"}
        V4["Revisar: se perdio<br/>la ventaja vs fabrica"]
        V1 -->|"si"| V2
        V3 -->|"si"| V4
    end

    D2 --> H1
    H3 --> D2
    Q3 --> D2
    H2 --> V3
    D2 --> V1

    classDef origen fill:#fcfcfb,stroke:#8a8a85,stroke-width:1px,color:#0b0b0b
    classDef cambio fill:#dceafc,stroke:#2a78d6,stroke-width:3px,color:#0b0b0b
    classDef alerta fill:#fdf6e3,stroke:#eda100,stroke-width:2px,color:#0b0b0b

    class D1,D2 origen
    class H1,H2,H3,Q1,Q2,Q3 cambio
    class V1,V2,V3,V4 alerta
```

**El paso de 15 días es obligatorio.** La ventana móvil es justamente lo que le da
a Fe el 0.508; con una ventana vieja el modelo vuelve al problema original de
aprender la deriva en vez de la química.

---

## Reproducción

```bash
pixi run python notebooks/10_recalibracion_composito.py --solo-backtest   # evaluar sin escribir
pixi run python notebooks/10_recalibracion_composito.py                   # aplicar los 4 cambios
pixi run python notebooks/11_figuras_resultados.py                        # figuras y tabla
```

El manifiesto `models/manifiesto_recalibracion.json` guarda los SHA-256 de las
entradas, todos los parámetros, las versiones de las librerías y las métricas
obtenidas, de modo que cualquier corrida se puede auditar contra la que produjo
los resultados de la tesis.
