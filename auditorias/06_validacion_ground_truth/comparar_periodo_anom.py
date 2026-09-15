# -*- coding: utf-8 -*-

r"""
======================================================================
comparar_periodo_anom.py - AISLAR LA CONTRIBUCION DE 'period' AL AUC
======================================================================

Responde la primera de las dos tareas pendientes de whitepaper.md,
seccion 4.9: ¿cuanto aporta `period` al AUC=0.6608 obtenido en
`validar_contra_anom.py`?

Reutiliza las funciones de `validar_contra_anom.py` (debe estar en el
mismo directorio) para no pagar dos veces el coste de cargar y
extraer features de TRAIN (25k) + TEST (8k) + ANOM (1.087) -- eso se
hace UNA sola vez aqui, y luego se entrena `detectar_ood_multiclase()`
dos veces: una con las 12 variables base, otra con las 12 + `period`.

Coloca este script en el mismo sitio que `validar_contra_anom.py`
(`auditorias\06_validacion_ground_truth\`).
======================================================================
"""

import os
import sys
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validar_contra_anom import (
    RUTA_TRAIN, RUTA_TEST, RUTA_ANOM, MAX_TRAIN, MAX_TEST,
    CARPETA_SALIDA, cargar_y_extraer, evaluar, calcular_metricas, banner,
)
from anomaly_detector import VARIABLES_BASE


def main():

    inicio = time.time()

    banner("COMPARACION CON / SIN 'period' -- CARGA DE DATOS (una sola vez)")

    X_train = cargar_y_extraer(RUTA_TRAIN, MAX_TRAIN, "TRAIN")
    X_test = cargar_y_extraer(RUTA_TEST, MAX_TEST, "TEST")
    X_anom = cargar_y_extraer(RUTA_ANOM, None, "ANOM")

    tiene_period = (
        "period" in X_train.columns
        and "period" in X_test.columns
        and "period" in X_anom.columns
    )

    if not tiene_period:
        print(
            "AVISO: 'period' no esta disponible en los tres splits a la "
            "vez -- no se puede hacer la comparacion. Revisa la extraccion "
            "de features."
        )
        return

    y_train = X_train["clase"].astype(str)

    resultados = {}

    for etiqueta, variables in [
        ("SIN_period", list(VARIABLES_BASE)),
        ("CON_period", list(VARIABLES_BASE) + ["period"]),
    ]:

        banner(f"EVALUANDO: {etiqueta}  (variables={variables})")

        resultado = evaluar(X_train, y_train, X_test, X_anom, variables)
        auc, tabla_umbrales, resumen_clase = calcular_metricas(resultado)

        resultados[etiqueta] = {
            "auc": auc,
            "tabla_umbrales": tabla_umbrales,
            "resumen_clase": resumen_clase,
            "resultado_completo": resultado,
        }

        resultado.to_csv(
            os.path.join(CARPETA_SALIDA, f"validacion_anom_scores_{etiqueta}.csv"),
            index=False,
        )
        tabla_umbrales.to_csv(
            os.path.join(CARPETA_SALIDA, f"validacion_anom_umbrales_{etiqueta}.csv"),
            index=False,
        )
        resumen_clase.to_csv(
            os.path.join(CARPETA_SALIDA, f"validacion_anom_recall_por_clase_{etiqueta}.csv"),
            index=False,
        )

    # ------------------------------------------------------------------
    # COMPARACION FINAL
    # ------------------------------------------------------------------

    banner("COMPARACION FINAL: CON vs SIN 'period'")

    auc_sin = resultados["SIN_period"]["auc"]
    auc_con = resultados["CON_period"]["auc"]

    print(f"AUC sin period: {auc_sin:.4f}")
    print(f"AUC con period: {auc_con:.4f}")
    print(f"Diferencia    : {auc_con - auc_sin:+.4f}")
    print()

    if auc_con - auc_sin > 0.02:
        print(
            "-> 'period' aporta una mejora real y no trivial al AUC. "
            "Confirma que es una feature valiosa tambien para OOD, no "
            "solo para clasificacion supervisada."
        )
    elif auc_con - auc_sin < -0.02:
        print(
            "-> 'period' EMPEORA el AUC. Contraintuitivo -- posible causa: "
            "el rango de periodo de las clases OOD podria solapar mucho "
            "con el de las clases conocidas (a diferencia de la forma de "
            "la curva), añadiendo una dimension que no ayuda a separar y "
            "sí añade ruido a la distancia euclidiana tras escalar."
        )
    else:
        print(
            "-> Diferencia pequeña, 'period' no es el factor determinante "
            "del AUC=0.66 obtenido -- la limitacion esta en las otras 12 "
            "variables (estadisticos agregados), no en la ausencia/presencia "
            "de esta."
        )

    print()
    print("Recall a FP=5% por clase OOD, comparado:")

    comp = resultados["SIN_period"]["resumen_clase"][["clase_real", "recall"]].rename(
        columns={"recall": "recall_sin_period"}
    ).merge(
        resultados["CON_period"]["resumen_clase"][["clase_real", "recall"]].rename(
            columns={"recall": "recall_con_period"}
        ),
        on="clase_real",
    )
    comp["diferencia"] = comp["recall_con_period"] - comp["recall_sin_period"]
    comp = comp.sort_values("diferencia", ascending=False)

    print(comp.to_string(index=False))

    ruta_comparacion = os.path.join(CARPETA_SALIDA, "comparacion_con_sin_period.csv")
    comp.to_csv(ruta_comparacion, index=False)

    print()
    print("Archivos guardados (6 de detalle + 1 de comparacion):")
    for etiqueta in ["SIN_period", "CON_period"]:
        print(f"  validacion_anom_scores_{etiqueta}.csv")
        print(f"  validacion_anom_umbrales_{etiqueta}.csv")
        print(f"  validacion_anom_recall_por_clase_{etiqueta}.csv")
    print(f"  {ruta_comparacion}")

    print()
    print(f"Tiempo total: {time.time() - inicio:.2f} segundos")


if __name__ == "__main__":
    main()
