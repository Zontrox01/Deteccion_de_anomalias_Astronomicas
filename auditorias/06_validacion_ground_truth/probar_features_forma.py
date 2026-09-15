# -*- coding: utf-8 -*-

r"""
======================================================================
probar_features_forma.py - ¿AYUDAN LAS FEATURES DE FORMA DE CURVA?
======================================================================

Responde la ultima tarea pendiente de whitepaper.md, seccion 4.9:
¿mejora el AUC/recall de deteccion OOD si se añaden features de forma
de curva (Fourier del plegado en fase, ver
anomaly_detector/features_forma.py) al espacio de 12 variables + period
ya validado?

Compara tres configuraciones sobre los mismos datos (cargados una sola
vez, igual que en comparar_periodo_anom.py):
  1. BASE            : las 12 variables + period (referencia conocida,
                       AUC=0.6608).
  2. BASE + FOURIER  : las 12 + period + las 6 features de Fourier.
  3. SOLO_FOURIER    : solo las 6 features de Fourier (para ver si por
                       si solas ya aportan señal, sin apoyarse en las
                       12 base -- diagnostico, no una opcion real a
                       adoptar).

Coloca este script en el mismo sitio que validar_contra_anom.py y
comparar_periodo_anom.py (auditorias\06_validacion_ground_truth\, o
crea auditorias\07_features_forma_curva\ si prefieres mantener esta
fase separada -- ver FILES.md).
======================================================================
"""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ======================================================================
# AÑADIR EL DIRECTORIO RAIZ AL PATH PARA IMPORTAR anomaly_detector
# ======================================================================

# Ruta del script
RUTA_SCRIPT = Path(__file__).resolve()

# Ruta raiz del proyecto (Astronomia)
BASE_PROYECTO = RUTA_SCRIPT.parents[2]  # Sube 2 niveles hasta Astronomia/

# Añadir BASE_PROYECTO al path para poder importar anomaly_detector
sys.path.insert(0, str(BASE_PROYECTO))

from anomaly_detector.adapters.starembed import AdaptadorStarEmbed
from anomaly_detector import VARIABLES_BASE, CLAVES_FOURIER
from anomaly_detector.features import extraer_features_dataset
from anomaly_detector.features_forma import extraer_features_fourier_dataset


# ======================================================================
# CONFIGURACION
# ======================================================================

BASE_DATASET = BASE_PROYECTO / "dataset" / "ZTF_40k_StarEmbed" / "data"

RUTA_TRAIN = [
    BASE_DATASET / "train-00000-of-00002.parquet",
    BASE_DATASET / "train-00001-of-00002.parquet",
]
RUTA_TEST = [BASE_DATASET / "test-00000-of-00001.parquet"]
RUTA_ANOM = [BASE_DATASET / "anom-00000-of-00001.parquet"]

MAX_TRAIN = 25000
MAX_TEST = 8000

CARPETA_SALIDA = RUTA_SCRIPT.parent / "resultados"
CARPETA_SALIDA.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "forma_"

N_ARMONICOS = 3


def banner(texto):
    print()
    print("=" * 70)
    print(texto)
    print("=" * 70)


# ======================================================================
# FUNCIONES DE CARGA
# ======================================================================

def cargar_y_extraer(rutas, max_objetos, etiqueta):
    """
    Carga datos y extrae features base (12 variables + period).
    """

    print(f"Cargando {etiqueta}...")

    # Asegurarnos de que rutas sea una lista
    if not isinstance(rutas, list):
        rutas = [rutas]

    adaptador = AdaptadorStarEmbed(rutas, max_objetos=max_objetos)
    dataset = adaptador.cargar()

    print(f"  {len(dataset.ids_unicos())} objetos unicos cargados "
          f"({adaptador.n_descartados} descartados por curva invalida)")

    print(f"  Extrayendo features base de {etiqueta}...")
    X = extraer_features_dataset(dataset)

    return X


def cargar_y_extraer_con_fourier(rutas, max_objetos, etiqueta):
    """
    Igual que cargar_y_extraer(), pero ademas calcula las features de
    Fourier y las cruza (merge por id_objeto) con las 12 base + period.
    """

    print(f"Cargando {etiqueta} (base + Fourier)...")

    # Asegurarnos de que rutas sea una lista
    if not isinstance(rutas, list):
        rutas = [rutas]

    adaptador = AdaptadorStarEmbed(rutas, max_objetos=max_objetos)
    dataset = adaptador.cargar()

    print(f"  {len(dataset.ids_unicos())} objetos unicos cargados "
          f"({adaptador.n_descartados} descartados por curva invalida)")

    print(f"  Extrayendo features base de {etiqueta}...")
    X_base = extraer_features_dataset(dataset)

    print(f"  Extrayendo features de Fourier de {etiqueta} (n_armonicos={N_ARMONICOS})...")
    X_fourier = extraer_features_fourier_dataset(dataset, n_armonicos=N_ARMONICOS)

    # Verificar que los DataFrames tienen la columna id_objeto
    if "id_objeto" not in X_base.columns:
        print("  AVISO: X_base no tiene 'id_objeto'. Añadiendo desde dataset...")
        ids = dataset.ids_unicos()
        # Si las longitudes no coinciden, algo ha ido mal
        if len(ids) != len(X_base):
            print(f"  ERROR: len(ids)={len(ids)} != len(X_base)={len(X_base)}")
            # Intentar usar el índice como id
            X_base["id_objeto"] = X_base.index.astype(str)
        else:
            X_base["id_objeto"] = ids

    X = X_base.merge(X_fourier, on="id_objeto", how="left")

    n_sin_fourier = X[CLAVES_FOURIER[0]].isna().sum() if len(CLAVES_FOURIER) > 0 else 0
    if n_sin_fourier > 0:
        print(
            f"  AVISO: {n_sin_fourier}/{len(X)} objetos de {etiqueta} sin "
            f"features de Fourier validas (periodo ausente o muy pocos "
            f"puntos) -- quedaran como NaN, imputados con la mediana de "
            f"TRAIN mas adelante como el resto de features."
        )

    return X


# ======================================================================
# EVALUACION (copia local para no depender de validar_contra_anom)
# ======================================================================

def evaluar(X_train, y_train, X_test, X_anom, variables):

    banner("ENTRENANDO DETECTORES POR CLASE Y PUNTUANDO TEST + ANOM")

    # Importar detectar_ood_multiclase
    from anomaly_detector import detectar_ood_multiclase
    from sklearn.metrics import roc_auc_score

    X_eval = pd.concat([X_test, X_anom], ignore_index=True)
    es_ood = np.concatenate([
        np.zeros(len(X_test), dtype=int),
        np.ones(len(X_anom), dtype=int),
    ])

    resultado = detectar_ood_multiclase(X_train, y_train, X_eval, variables)

    resultado["es_ood_real"] = es_ood
    resultado["clase_real"] = pd.concat(
        [X_test["clase"], X_anom["clase"]], ignore_index=True
    ).values
    resultado["id_objeto"] = pd.concat(
        [X_test["id_objeto"], X_anom["id_objeto"]], ignore_index=True
    ).values

    # Calcular métricas
    auc = roc_auc_score(resultado["es_ood_real"], resultado["score_min"])

    scores_test = resultado.loc[resultado["es_ood_real"] == 0, "score_min"]
    scores_anom = resultado.loc[resultado["es_ood_real"] == 1, "score_min"]

    umbral_referencia = scores_test.quantile(0.95)

    anom_rows = resultado[resultado["es_ood_real"] == 1].copy()
    anom_rows["detectado"] = anom_rows["score_min"] >= umbral_referencia

    resumen_clase = anom_rows.groupby("clase_real").agg(
        n=("clase_real", "size"),
        detectados=("detectado", "sum"),
    ).reset_index()
    resumen_clase["recall"] = resumen_clase["detectados"] / resumen_clase["n"]
    resumen_clase = resumen_clase.sort_values("recall", ascending=False)

    print(f"AUC-ROC: {auc:.4f}")

    return resultado, auc, resumen_clase


# ======================================================================
# MAIN
# ======================================================================

def main():

    inicio = time.time()

    # Verificar que los archivos existen
    for ruta in RUTA_TRAIN:
        if not ruta.exists():
            print(f"ERROR: No existe el archivo: {ruta}")
            return

    for ruta in RUTA_TEST:
        if not ruta.exists():
            print(f"ERROR: No existe el archivo: {ruta}")
            return

    for ruta in RUTA_ANOM:
        if not ruta.exists():
            print(f"ERROR: No existe el archivo: {ruta}")
            return

    banner("EXPERIMENTO: FEATURES DE FORMA DE CURVA (FOURIER) -- CARGA DE DATOS")

    # Cargar datos con Fourier
    X_train = cargar_y_extraer_con_fourier(RUTA_TRAIN, MAX_TRAIN, "TRAIN")
    X_test = cargar_y_extraer_con_fourier(RUTA_TEST, MAX_TEST, "TEST")
    X_anom = cargar_y_extraer_con_fourier(RUTA_ANOM, None, "ANOM")

    # Verificar que CLAVES_FOURIER existe y no está vacío
    if len(CLAVES_FOURIER) == 0:
        print("ERROR: CLAVES_FOURIER está vacío. Verifica anomaly_detector/__init__.py")
        return

    # Imputacion de NaN en las features de Fourier (objetos sin periodo
    # valido o con muy pocos puntos): mediana de TRAIN, mismo criterio
    # que el resto del pipeline (ver features.py, limpiar()).
    medianas_fourier = X_train[CLAVES_FOURIER].median()
    for X in (X_train, X_test, X_anom):
        X[CLAVES_FOURIER] = X[CLAVES_FOURIER].fillna(medianas_fourier)

    y_train = X_train["clase"].astype(str)

    # Asegurar que todas las variables existen en los DataFrames
    todas_variables = list(VARIABLES_BASE) + ["period"] + CLAVES_FOURIER
    for X in (X_train, X_test, X_anom):
        for var in todas_variables:
            if var not in X.columns:
                print(f"  AVISO: '{var}' no está en el DataFrame, añadiendo columna con NaN")
                X[var] = np.nan

    configuraciones = {
        "BASE": list(VARIABLES_BASE) + ["period"],
        "BASE_MAS_FOURIER": list(VARIABLES_BASE) + ["period"] + CLAVES_FOURIER,
        "SOLO_FOURIER": list(CLAVES_FOURIER),
    }

    resultados = {}

    for etiqueta, variables in configuraciones.items():

        banner(f"EVALUANDO: {etiqueta}  ({len(variables)} variables)")

        # Verificar que todas las variables existen
        vars_validas = [v for v in variables if v in X_train.columns]
        if len(vars_validas) < len(variables):
            print(f"  AVISO: Solo {len(vars_validas)} de {len(variables)} variables encontradas")
            variables = vars_validas

        resultado, auc, resumen_clase = evaluar(X_train, y_train, X_test, X_anom, variables)

        resultados[etiqueta] = {
            "auc": auc,
            "resumen_clase": resumen_clase,
        }

        resultado.to_csv(
            CARPETA_SALIDA / f"{PREFIJO}scores_{etiqueta}.csv", index=False
        )
        resumen_clase.to_csv(
            CARPETA_SALIDA / f"{PREFIJO}recall_por_clase_{etiqueta}.csv",
            index=False,
        )

    # ------------------------------------------------------------------
    # COMPARACION FINAL
    # ------------------------------------------------------------------

    banner("COMPARACION FINAL")

    print("AUC por configuracion:")
    for etiqueta in configuraciones:
        print(f"  {etiqueta:20s}: {resultados[etiqueta]['auc']:.4f}")

    print()
    diferencia_clave = resultados["BASE_MAS_FOURIER"]["auc"] - resultados["BASE"]["auc"]
    print(f"Diferencia BASE_MAS_FOURIER - BASE: {diferencia_clave:+.4f}")
    print()

    if diferencia_clave > 0.03:
        print(
            "-> Mejora clara. Las features de forma de curva aportan senal "
            "real que las 12 variables base + period no capturaban. "
            "Candidatas a promoverse a VARIABLES_BASE tras revisar el "
            "detalle por clase."
        )
    elif diferencia_clave > 0.0:
        print(
            "-> Mejora pequeña pero positiva. Revisar si se concentra en "
            "Blazhko/HADS/ELL (las clases que motivaron el experimento) "
            "o esta repartida sin patron -- eso decide si merece la pena "
            "adoptarlas permanentemente."
        )
    else:
        print(
            "-> No mejora (o empeora). La hipotesis de que la forma del "
            "pliegue en fase explicaba el problema de Blazhko/HADS/ELL "
            "no se sostiene con este metodo concreto. Posibles causas: "
            "el numero de armonicos (N_ARMONICOS={}) no es el adecuado, "
            "o el problema real esta en otro sitio (p. ej. calidad/numero "
            "de puntos disponibles para plegar bien la curva en esas "
            "clases especificas -- revisar cuantos objetos de cada clase "
            "OOD tuvieron que ser imputados por falta de periodo o "
            "puntos suficientes).".format(N_ARMONICOS)
        )

    print()
    print("Recall a FP=5% por clase OOD, las tres configuraciones:")

    comp = resultados["BASE"]["resumen_clase"][["clase_real", "recall"]].rename(
        columns={"recall": "recall_BASE"}
    )
    for etiqueta in ["BASE_MAS_FOURIER", "SOLO_FOURIER"]:
        comp = comp.merge(
            resultados[etiqueta]["resumen_clase"][["clase_real", "recall"]].rename(
                columns={"recall": f"recall_{etiqueta}"}
            ),
            on="clase_real",
        )

    comp["diferencia_BASE_MAS_FOURIER"] = comp["recall_BASE_MAS_FOURIER"] - comp["recall_BASE"]
    comp = comp.sort_values("diferencia_BASE_MAS_FOURIER", ascending=False)

    print(comp.to_string(index=False))

    ruta_comparacion = CARPETA_SALIDA / f"{PREFIJO}comparacion_final.csv"
    comp.to_csv(ruta_comparacion, index=False)

    print()
    print("Fijate especialmente en Blazhko, HADS y ELL en la tabla de arriba "
          "-- son las tres clases que motivaron este experimento.")

    print()
    print("Archivos guardados:")
    for etiqueta in configuraciones:
        print(f"  {PREFIJO}scores_{etiqueta}.csv")
        print(f"  {PREFIJO}recall_por_clase_{etiqueta}.csv")
    print(f"  {ruta_comparacion}")

    print()
    print(f"Tiempo total: {time.time() - inicio:.2f} segundos")


if __name__ == "__main__":
    main()