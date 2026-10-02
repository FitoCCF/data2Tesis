# ============================================================
# scripts/verificar_fuentes_laboratorio.py — Caracteriza las dos fuentes de
# laboratorio y verifica la alineación temporal del compósito
# ============================================================
# Reproduce todas las cifras de la §12 de docs/bitacora_analisis.md:
#
#   1. Cadencia de cada fuente (BD: lunes/miércoles/viernes; compósito: 2/día).
#   2. Prueba de alineación A vs B. El compósito se acumula a lo largo de un
#      turno de 12 h, así que cabe preguntarse si su marca de tiempo es el
#      CIERRE del turno (soporte [t-12h, t], hipótesis B) o el CENTRO de la
#      ventana que integra (soporte [t-6h, t+6h], hipótesis A, que es la que
#      usa todo el pipeline vía VENTANA_COMPOSITO_H).
#
#      La prueba compara ley de laboratorio contra ley de laboratorio -- sin
#      modelo en medio -- emparejando cada muestra de la BD con el compósito
#      que le asigna cada hipótesis. El árbitro es Mo: es la única ley en que
#      ambos laboratorios concuerdan (§11.3), de modo que la pérdida de
#      correlación al desalinear se lee limpia.
#
# Solo lectura: no escribe nada. Consume las tablas ya limpias de la etapa 1a,
# para no reimplementar aquí las reglas de limpieza de leyes (§10.4: ley en
# 0 = no analizada -> NaN, y la bandera `leyes_sospechosas`).
#
#   pixi run python scripts/verificar_fuentes_laboratorio.py
# ============================================================
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.config import LIMPIO_COURIER_BD, LIMPIO_COMPOSITO, VENTANA_COMPOSITO_H

LEYES = [("Fe", "pFe", "fe"), ("Cu", "pCu", "cu"), ("Mo", "pMo", "mo")]
DIAS = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]


def cargar():
    """Las dos fuentes limpias: muestras de la BD con ley, y compósito."""
    bd = pd.read_csv(LIMPIO_COURIER_BD, parse_dates=["ts"])
    comp = pd.read_csv(LIMPIO_COMPOSITO, parse_dates=["ts"])

    con_ley = bd[bd[[c for _, c, _ in LEYES]].notna().any(axis=1)]
    return bd, con_ley, comp


def cadencia(con_ley, comp):
    print("=" * 62)
    print("1. CADENCIA DE CADA FUENTE")
    print("=" * 62)

    por_dia = con_ley["ts"].dt.dayofweek.map(lambda i: DIAS[i]).value_counts()
    lmv = int(por_dia.reindex(["Lun", "Mie", "Vie"]).sum())
    print(f"\nBD (sample_id=24, muestra puntual): {len(con_ley)} muestras con ley")
    print(f"  rango {con_ley.ts.min().date()} -> {con_ley.ts.max().date()}")
    print(f"  Lun/Mie/Vie: {lmv}/{len(con_ley)} "
          f"({100 * lmv / len(con_ley):.1f} %), resto {len(con_ley) - lmv}")
    print("  " + por_dia.reindex(DIAS).to_string().replace("\n", "\n  "))

    h = con_ley["ts"].dt.hour
    manana, tarde = h.between(9, 13).sum(), h.between(15, 19).sum()
    print(f"  hora de toma: 09-13 h {manana} | 15-19 h {tarde} | "
          f"otras {len(con_ley) - manana - tarde}")

    dif = con_ley["ts"].dt.normalize().diff().dt.days.value_counts()
    print("  días entre muestras consecutivas: "
          + ", ".join(f"{int(k)}d x{v}" for k, v in dif.head(4).items()))

    print(f"\nCompósito (PI, integra {VENTANA_COMPOSITO_H} h): {len(comp)} ensayos")
    print(f"  rango {comp.ts.min().date()} -> {comp.ts.max().date()}")
    print(f"  marcas: {comp.ts.dt.strftime('%H:%M').value_counts().to_dict()}")
    print(f"  cobertura: ~{len(comp) / ((comp.ts.max() - comp.ts.min()).days * 2):.0%} "
          "de los dos turnos de cada día")


def emparejar(con_ley, comp, hipotesis):
    """Empareja cada muestra de la BD con el compósito que le asigna la hipótesis.

    A (centro): el compósito cuya ventana centrada de 12 h cubre la muestra,
                es decir el más cercano a menos de 6 h.
    B (cierre): el compósito que cierra el turno que contiene a la muestra,
                es decir la siguiente marca a menos de 12 h.
    """
    direccion, tolerancia = {
        "A": ("nearest", f"{VENTANA_COMPOSITO_H / 2:.0f}h"),
        "B": ("forward", f"{VENTANA_COMPOSITO_H:.0f}h"),
    }[hipotesis]

    return pd.merge_asof(
        con_ley[con_ley.leyes_sospechosas.isna()].sort_values("ts"),
        comp.rename(columns={"ts": "ts_comp"}).sort_values("ts_comp"),
        left_on="ts", right_on="ts_comp",
        direction=direccion, tolerance=pd.Timedelta(tolerancia),
    ).dropna(subset=["cu"])


def alineacion(con_ley, comp):
    print("\n" + "=" * 62)
    print("2. ALINEACIÓN DEL COMPÓSITO — ¿su marca es el centro o el cierre?")
    print("=" * 62)
    print("\nBD - compósito, sesgo / correlación (las 4 filas con ley sospechosa fuera):\n")
    print(f"  {'hipótesis':<26}   {'n':>4}   " + "   ".join(f"{n:^14}" for n, _, _ in LEYES))

    resultados = {}
    for hip, rotulo in [("A", "A centro [t-6h, t+6h]"), ("B", "B cierre [t-12h, t]")]:
        D = emparejar(con_ley, comp, hip)
        celdas, resultados[hip] = [], {}
        for nombre, col_bd, col_comp in LEYES:
            d = D[[col_bd, col_comp]].dropna()
            sesgo = (d[col_bd] - d[col_comp]).mean()
            corr = d.corr().iloc[0, 1]
            resultados[hip][nombre] = corr
            celdas.append(f"{sesgo:+.3f} / {corr:.3f}".center(14))
        print(f"  {rotulo:<26}   {len(D):>4}   " + "   ".join(celdas))

    gana_a = sum(resultados["A"][n] > resultados["B"][n] for n, _, _ in LEYES)
    d_mo = resultados["A"]["Mo"] - resultados["B"]["Mo"]
    print(f"\n  A gana en {gana_a} de {len(LEYES)} leyes; en Mo (el árbitro) "
          f"{resultados['A']['Mo']:.3f} vs {resultados['B']['Mo']:.3f} "
          f"({d_mo:+.3f}).")
    print("  => la marca del compósito se comporta como el CENTRO de su ventana,")
    print("     no como el cierre del turno. Confirma el barrido de lag de")
    print("     src/pipeline/calibracion_composito.py por una vía independiente.")


def sesgo_cu_por_hora(con_ley, comp):
    print("\n" + "=" * 62)
    print("3. EL SESGO DE Cu NO SE EXPLICA POR LA HORA DE TOMA")
    print("=" * 62 + "\n")

    D = emparejar(con_ley, comp, "A")
    hora = D["ts"].dt.hour
    for rotulo, mascara in [("09-13 h", hora.between(9, 13)), ("15-19 h", hora.between(15, 19))]:
        d = D[mascara][["pCu", "cu"]].dropna()
        print(f"  BD tomada {rotulo}: n={len(d):>3}  sesgo Cu "
              f"{(d.pCu - d.cu).mean():+.3f}  corr {d.corr().iloc[0, 1]:.3f}")
    print("\n  El sesgo persiste en ambas franjas (y empeora por la tarde): es")
    print("  diferencia sistemática entre los dos ensayos, no artefacto horario.")


def main():
    bd, con_ley, comp = cargar()
    print(f"\nFuentes: {LIMPIO_COURIER_BD.name} ({len(bd)} lecturas de la BD) y "
          f"{LIMPIO_COMPOSITO.name} ({len(comp)} ensayos)")
    cadencia(con_ley, comp)
    alineacion(con_ley, comp)
    sesgo_cu_por_hora(con_ley, comp)
    print()


if __name__ == "__main__":
    main()
