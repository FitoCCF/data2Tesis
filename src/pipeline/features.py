#!/usr/bin/env python3
# ============================================================
# src/pipeline/features.py — Etapa 2: features robustas a la deriva del instrumento
# ============================================================
# Reemplaza a la antigua etapa 2 (ortogonalización contra n6sc). Ejecutable:
#   python -m src.pipeline.features        # courier_limpio.csv -> courier_features.csv
#
# PROBLEMA QUE RESUELVE
# --------------------
# El analizador pierde cuentas con el tiempo (decaimiento de la fuente de rayos
# X y envejecimiento del detector). Medido sobre data/processed/intensidades_cobre.csv,
# entre 2025Q3 y 2026Q3:
#
#     canal          media 2025Q3   media 2026Q3   deriva
#     n1fe (abs)          35894          28178      -24.1%
#     n2cu (abs)          38464          30871      -22.8%
#     n6sc (abs)           1743           1368      -25.1%
#
# Mientras tanto la ley real de cobre sólo bajó 7% y la de hierro SUBIÓ. Es decir:
# casi toda la variación de las intensidades absolutas es del instrumento, no del
# mineral. Cualquier modelo entrenado sobre ellas aprende la deriva.
#
# SOLUCIÓN
# --------
# Las FRACCIONES DE CIERRE  f_i = I_i / Σ_j I_j  cancelan cualquier ganancia
# multiplicativa común a todos los canales (que es la forma del decaimiento de
# fuente). Medido sobre los mismos trimestres:
#
#     n2cu_f   deriva  1.2%          <- estable
#     n1fe_f   deriva  1.3%          <- estable
#     n4mo_f   deriva  9.5%
#     n3zn_f   deriva 17.6%          (canal pequeño y ruidoso)
#     SumI/n6sc deriva 17.4%         <- n6sc NO deriva igual que los metales
#
# Por eso el cierre se hace sobre los 4 canales de METAL y n6sc se deja fuera del
# denominador: normalizar por n6sc reintroduce deriva en vez de quitarla.
# ============================================================

import numpy as np                                        # cálculo numérico
import pandas as pd                                       # DataFrames

from .config import METALES, FEATS_CLUSTER, FEATS_REGRESION, DIL_HALFLIFE  # canales, features, dilución


def features_cierre(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula las fracciones de cierre f_i = I_i / Σ I_metales.

    Estas son las features INVARIANTES A LA DERIVA: si el instrumento pierde un
    factor g común en todos los canales, el numerador y el denominador se
    multiplican por g y la fracción no cambia.

    Parámetros
    ----------
    df : DataFrame con los 4 canales de metal (n1fe, n2cu, n3zn, n4mo).

    Retorna
    -------
    DataFrame con las columnas '<metal>_f', mismo índice que df.
    """
    # min_count=len(METALES): si falta CUALQUIER metal (NaN, p.ej. un valor de
    # error -9999 recuperado por limpieza.py), la suma debe ser NaN, no la
    # suma parcial de los 3 restantes -- si no, las otras 3 fracciones
    # saldrían mal (denominador entendido de menos) en vez de indefinidas.
    suma = df[METALES].sum(axis=1, min_count=len(METALES))  # Σ de los 4 canales de metal (el "cierre")
    suma = suma.replace(0, np.nan)                        # evita división por cero (lecturas muertas)

    salida = pd.DataFrame(index=df.index)                 # DataFrame de features, mismo índice
    for metal in METALES:                                 # por cada canal de metal
        salida[f"{metal}_f"] = df[metal] / suma           # fracción de cierre (adimensional, suma 1)
    return salida                                         # devuelve solo las fracciones


def features_regresion(df: pd.DataFrame) -> pd.DataFrame:
    """Construye el set de features para la regresión contra el compósito.

    Combina dos bloques con roles distintos:
      1. Fracciones de cierre  -> forma del espectro, invariante a la deriva.
         Es de donde sale la señal química real.
      2. log(Σ intensidades)   -> magnitud total. SÍ deriva, pero se conserva
         porque lleva información de densidad/carga de la pulpa que las
         fracciones no pueden ver. La ventana móvil (cambio 2) se encarga de
         que su deriva no contamine: dentro de la ventana es casi constante.

    FEATURES DESCARTADAS Y POR QUÉ (backtest walk-forward sobre pFe, n=228):
      - fe_cu = n1fe/n2cu : razón inter-elemento. Conceptualmente atractiva
        (es el proxy de dilución por pirita; medido corr(Cu, Fe_exc) = -0.811
        con Fe_exc = Fe - 0.879*Cu, el hierro no ligado a calcopirita CuFeS2).
        Pero EMPEORA el backtest: corr 0.508 sin ella contra 0.488 con ella.
        Es redundante con las fracciones y agrega varianza.
      - n6sc : indiferente (0.508 sin ella, 0.507 con ella) -> se omite por
        parsimonia.
      - ΣI/n6sc : deriva 17.4% entre trimestres, o sea NO es invariante; n6sc
        no decae al mismo ritmo que los canales de metal.

    Parámetros
    ----------
    df : DataFrame con los 4 canales de metal.

    Retorna
    -------
    DataFrame de features listo para el regresor (sin NaN por construcción,
    salvo que la entrada los tenga).
    """
    salida = features_cierre(df)                          # bloque 1: las 4 fracciones de cierre

    suma = df[METALES].sum(axis=1).replace(0, np.nan)     # Σ de canales de metal
    salida["logSumI"] = np.log(suma)                      # bloque 2: magnitud total en escala log

    return salida                                         # features de regresión


def columnas_regresion() -> list[str]:
    """Nombres de las features de regresión, en orden fijo (config.FEATS_REGRESION).

    Se usa para congelar el orden entre entrenamiento e inferencia: si el orden
    cambiara, el modelo recibiría las columnas permutadas y predeciría basura.
    """
    return list(FEATS_REGRESION)


def features_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """Features del clustering: log-cocientes aditivos (alr) contra el cobre.

        lr_fe_cu = log(n1fe/n2cu)   lr_zn_cu = log(n3zn/n2cu)   lr_mo_cu = log(n4mo/n2cu)

    Por qué así y no las 4 fracciones de cierre:
      - Una ganancia común del instrumento se cancela en el cociente, igual que
        en las fracciones (invariantes a la deriva de la fuente).
      - Las 4 fracciones suman 1: una es combinación lineal de las otras y la
        covarianza de un GMM 'full' queda singular. Los 3 log-cocientes son
        libres (coordenadas de Aitchison para datos composicionales).
      - Las fracciones viven acotadas en (0, 1) y, para Zn y Mo, pegadas a 0;
        el logaritmo las lleva a una escala donde una gaussiana es razonable.
    Cu como denominador: es el canal más grande y estable (deriva de su
    fracción 1.2%, ver cabecera).
    """
    cu = df["n2cu"].where(df["n2cu"] > 0)
    salida = pd.DataFrame(index=df.index)
    for col, metal in zip(FEATS_CLUSTER, ["n1fe", "n3zn", "n4mo"]):
        salida[col] = np.log(df[metal].where(df[metal] > 0) / cu)
    return salida


def senal_dilucion(df: pd.DataFrame, halflife: str = DIL_HALFLIFE) -> pd.Series:
    """dil = logSumI - EWMA(logSumI), con la EWMA calculada SOLO con lecturas
    válidas ANTERIORES (causal: sirve igual en producción, sin fuga).

    La EWMA por tiempo (no por nº de lecturas, el muestreo es irregular: cada
    3 días en la BD, cada 21 min en el PI) sigue la deriva lenta de la fuente
    de rayos X; lo que queda es la variación rápida de magnitud, que es la
    dilución de la pulpa. Una fila sin historia previa queda con dil = 0.
    Requiere df ordenado por ts y la columna 'valida' (etapa 1c).
    """
    log_sum = np.log(df[METALES].sum(axis=1).replace(0, np.nan))
    ok = df["valida"] & log_sum.notna()
    ewma = log_sum[ok].ewm(halflife=pd.Timedelta(halflife), times=df.loc[ok, "ts"]).mean()
    base = ewma.shift(1).reindex(df.index).ffill()        # último nivel conocido ANTES de cada fila
    # una fila no válida entre dos válidas toma la base de la válida anterior
    # (ffill del valor ya desplazado), que también es pasado
    return (log_sum - base).fillna(0.0).rename("dil")


def agregar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Etapa 2: agrega a la tabla las features de clustering, la señal de
    dilución y las de regresión. No ajusta nada: no hay artefacto ni fuga."""
    df = df.sort_values("ts").reset_index(drop=True)
    return pd.concat([df, features_cluster(df), senal_dilucion(df),
                      features_regresion(df)[columnas_regresion()]], axis=1)


# ============================================================
# CLI — etapa 2
# ============================================================
def _main():
    import argparse
    from .config import LIMPIO, FEATURES

    ap = argparse.ArgumentParser(description="Etapa 2: features robustas a la deriva")
    ap.add_argument("--input", default=str(LIMPIO))
    ap.add_argument("--output", default=str(FEATURES))
    args = ap.parse_args()

    df = agregar_features(pd.read_csv(args.input, parse_dates=["ts"]))
    df.to_csv(args.output, index=False)
    feats = FEATS_CLUSTER + FEATS_REGRESION
    print(f"{args.input} -> {args.output}  ({len(df)} filas)")
    print(f"  features con NaN: {df[feats].isna().any(axis=1).sum()}")


if __name__ == "__main__":
    _main()
