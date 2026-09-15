# -*- coding: utf-8 -*-

"""
======================================================================
validar_contra_anom.py - VALIDACION CON GROUND TRUTH REAL (split anom)
======================================================================

Primera validacion de todo el proyecto con verdad fundamental real
(ver whitepaper.md, seccion 4.8). Usa `detectar_ood_multiclase()`
(anomaly_detector/deteccion.py) -- el metodo de referencia de los
propios autores de StarEmbed (Isolation Forest por clase, score
minimo) -- para responder una pregunta muy concreta:

    De los 1.087 objetos de `anom` (10 clases OOD reales, nunca
    vistas en TRAIN), ¿cuantos quedan correctamente senalados como
    anomalos, y a que coste en falsos positivos sobre objetos
    normales (el split `test`, clases conocidas)?

Metodo
------
1. Entrena un detector por clase sobre TRAIN (7 clases conocidas).
2. Puntua TEST (in-distribution, "normal") y ANOM (out-of-distribution,
   "anomalo") con el mismo conjunto de detectores.
3. Calcula AUC-ROC (separabilidad global) y, mas util en la practica,
   el recall sobre ANOM a varios umbrales de falsos positivos fijados
   sobre TEST (p. ej., "si aceptamos marcar como sospechoso al 5% de
   los objetos normales, ¿que porcentaje de las 10 clases OOD reales
   detectamos?").
4. Desglosa el recall por cada una de las 10 clases OOD -- es
   esperable que unas se detecten mejor que otras.

Requiere que `anomaly_detector` este instalado/accesible (mismo
directorio o en PYTHONPATH) y que el dataset este en la ruta
configurada abajo.
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

# Ruta del script (auditorias/06_validacion_ground_truth)
RUTA_SCRIPT = Path(__file__).resolve()

# Ruta raiz del proyecto (Astronomia)
BASE_PROYECTO = RUTA_SCRIPT.parents[2]  # Sube 2 niveles hasta Astronomia/
print(f"BASE_PROYECTO: {BASE_PROYECTO}")

# Añadir BASE_PROYECTO al path para poder importar anomaly_detector
sys.path.insert(0, str(BASE_PROYECTO))

# Verificar que anomaly_detector existe
RUTA_ANOMALY_DETECTOR = BASE_PROYECTO / "anomaly_detector"
print(f"Verificando existencia de anomaly_detector en: {RUTA_ANOMALY_DETECTOR}")
if not RUTA_ANOMALY_DETECTOR.exists():
    print(f"ADVERTENCIA: No se encontró anomaly_detector en {RUTA_ANOMALY_DETECTOR}")
    print("Intentando buscar en ubicaciones alternativas...")
    # Intentar subir un nivel más (por si el script está en otra ubicación)
    BASE_PROYECTO_ALT = RUTA_SCRIPT.parents[2]
    RUTA_ANOMALY_DETECTOR_ALT = BASE_PROYECTO_ALT / "anomaly_detector"
    if RUTA_ANOMALY_DETECTOR_ALT.exists():
        sys.path.insert(0, str(BASE_PROYECTO_ALT))
        print(f"  -> Encontrado en: {RUTA_ANOMALY_DETECTOR_ALT}")
    else:
        print("  -> NO encontrado. Asegúrate de que anomaly_detector esté en el PYTHONPATH.")

from sklearn.metrics import roc_auc_score

from anomaly_detector.adapters.starembed import AdaptadorStarEmbed
from anomaly_detector import extraer_features_dataset, detectar_ood_multiclase, VARIABLES_BASE


# ======================================================================
# CONFIGURACION
# ======================================================================

BASE_DATASET = BASE_PROYECTO / "dataset" / "ZTF_40k_StarEmbed" / "data"
print(f"BASE_DATASET: {BASE_DATASET}")

# IMPORTANTE: El adaptador espera UNA LISTA de rutas, incluso para un solo archivo
RUTA_TRAIN = [
    BASE_DATASET / "train-00000-of-00002.parquet",
    BASE_DATASET / "train-00001-of-00002.parquet",
]
RUTA_TEST = [BASE_DATASET / "test-00000-of-00001.parquet"]  # <-- LISTA, no Path simple
RUTA_ANOM = [BASE_DATASET / "anom-00000-of-00001.parquet"]  # <-- LISTA, no Path simple
print(f"RUTA_TRAIN: {RUTA_TRAIN}")
print(f"RUTA_TEST : {RUTA_TEST}")
print(f"RUTA_ANOM : {RUTA_ANOM}")

MAX_TRAIN = 25000
MAX_TEST = 8000
# anom tiene 1.087 objetos en total -- se usan todos, sin truncar.

CARPETA_SALIDA = RUTA_SCRIPT.parent / "resultados"
CARPETA_SALIDA.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "validacion_"

# Umbrales de falsos positivos (sobre TEST) a los que medir el recall
# sobre ANOM. 0.05 = "si aceptamos marcar como sospechoso al 5% de los
# objetos normales, ¿que recall conseguimos en las clases OOD reales?"
UMBRALES_FALSOS_POSITIVOS = [0.01, 0.02, 0.05, 0.10, 0.20]


def banner(texto):
    print()
    print("=" * 70)
    print(texto)
    print("=" * 70)


# ======================================================================
# CARGA Y EXTRACCION DE FEATURES
# ======================================================================

def cargar_y_extraer(rutas, max_objetos, etiqueta):

    print(f"Cargando {etiqueta}...")

    # Asegurarnos de que rutas sea una lista
    if not isinstance(rutas, list):
        rutas = [rutas]

    adaptador = AdaptadorStarEmbed(rutas, max_objetos=max_objetos)
    dataset = adaptador.cargar()

    print(f"  {len(dataset.ids_unicos())} objetos unicos cargados "
          f"({adaptador.n_descartados} descartados por curva invalida)")

    print(f"  Extrayendo features de {etiqueta}...")
    X = extraer_features_dataset(dataset)

    return X


# ======================================================================
# EVALUACION
# ======================================================================

def evaluar(X_train, y_train, X_test, X_anom, variables):

    banner("ENTRENANDO DETECTORES POR CLASE Y PUNTUANDO TEST + ANOM")

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

    return resultado


def calcular_metricas(resultado):

    banner("METRICAS GLOBALES")

    auc = roc_auc_score(resultado["es_ood_real"], resultado["score_min"])
    print(f"AUC-ROC (separabilidad TEST vs ANOM): {auc:.4f}")
    print(
        "  (1.0 = separacion perfecta, 0.5 = no mejor que azar, "
        "el score_min NO distingue nada)"
    )

    scores_test = resultado.loc[resultado["es_ood_real"] == 0, "score_min"]
    scores_anom = resultado.loc[resultado["es_ood_real"] == 1, "score_min"]

    print()
    print(f"score_min medio en TEST (normal): {scores_test.mean():.4f}")
    print(f"score_min medio en ANOM (OOD)   : {scores_anom.mean():.4f}")

    banner("RECALL SOBRE ANOM A VARIOS NIVELES DE FALSOS POSITIVOS (EN TEST)")

    filas_umbral = []

    for fp_objetivo in UMBRALES_FALSOS_POSITIVOS:

        umbral = scores_test.quantile(1 - fp_objetivo)

        fp_real = (scores_test >= umbral).mean()
        recall_anom = (scores_anom >= umbral).mean()

        print(
            f"  FP objetivo {fp_objetivo*100:5.1f}%  ->  umbral={umbral:.4f}  "
            f"FP real={fp_real*100:5.2f}%  recall en ANOM={recall_anom*100:5.2f}%"
        )

        filas_umbral.append({
            "fp_objetivo": fp_objetivo,
            "umbral_score": umbral,
            "fp_real": fp_real,
            "recall_anom": recall_anom,
        })

    tabla_umbrales = pd.DataFrame(filas_umbral)

    banner("RECALL POR CLASE OOD (a FP=5% en TEST, umbral de referencia)")

    umbral_referencia = scores_test.quantile(0.95)

    anom_rows = resultado[resultado["es_ood_real"] == 1].copy()
    anom_rows["detectado"] = anom_rows["score_min"] >= umbral_referencia

    resumen_clase = anom_rows.groupby("clase_real").agg(
        n=("clase_real", "size"),
        detectados=("detectado", "sum"),
    ).reset_index()
    resumen_clase["recall"] = resumen_clase["detectados"] / resumen_clase["n"]
    resumen_clase = resumen_clase.sort_values("recall", ascending=False)

    print(resumen_clase.to_string(index=False))

    return auc, tabla_umbrales, resumen_clase


# ======================================================================
# MAIN
# ======================================================================

def main():

    inicio = time.time()

    banner("VALIDACION CONTRA GROUND TRUTH REAL: split anom de StarEmbed")

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

    X_train = cargar_y_extraer(RUTA_TRAIN, MAX_TRAIN, "TRAIN")
    X_test = cargar_y_extraer(RUTA_TEST, MAX_TEST, "TEST")
    X_anom = cargar_y_extraer(RUTA_ANOM, None, "ANOM")

    print()
    print(f"TRAIN: {X_train.shape}, clases: {sorted(X_train['clase'].unique())}")
    print(f"TEST : {X_test.shape}")
    print(f"ANOM : {X_anom.shape}, clases: {sorted(X_anom['clase'].unique())}")

    variables = list(VARIABLES_BASE)
    if "period" in X_train.columns and "period" in X_test.columns and "period" in X_anom.columns:
        variables = variables + ["period"]
        print()
        print("'period' disponible en los tres splits -- incluida en el espacio de features.")
    else:
        print()
        print("AVISO: 'period' no disponible en los tres splits a la vez -- se omite.")

    y_train = X_train["clase"].astype(str)

    resultado = evaluar(X_train, y_train, X_test, X_anom, variables)

    auc, tabla_umbrales, resumen_clase = calcular_metricas(resultado)

    # ------------------------------------------------------------------
    # GUARDAR RESULTADOS
    # ------------------------------------------------------------------

    ruta_resultado = CARPETA_SALIDA / f"{PREFIJO}anom_scores.csv"
    ruta_umbrales = CARPETA_SALIDA / f"{PREFIJO}anom_umbrales.csv"
    ruta_por_clase = CARPETA_SALIDA / f"{PREFIJO}anom_recall_por_clase.csv"

    resultado.to_csv(ruta_resultado, index=False)
    tabla_umbrales.to_csv(ruta_umbrales, index=False)
    resumen_clase.to_csv(ruta_por_clase, index=False)

    banner("RESUMEN E INTERPRETACION")

    print(f"AUC-ROC global: {auc:.4f}")
    print()
    print(
        "Como leerlo:\n"
        "  - AUC > 0.9: el detector separa muy bien lo normal de lo "
        "anomalo -- el enfoque de deteccion de anomalias del proyecto "
        "esta bien fundamentado, no solo sobre datos sinteticos.\n"
        "  - AUC entre 0.7 y 0.9: separa razonablemente pero con "
        "solape -- revisar que clases OOD concretas cuestan mas "
        "(tabla por clase) para entender por que.\n"
        "  - AUC cercano a 0.5: el espacio de 12 variables no "
        "distingue estas clases OOD de las conocidas -- señal de que "
        "haria falta ampliar el espacio de features (quiza con "
        "informacion de forma de la curva mas alla de estadisticos "
        "agregados) antes de confiar en el detector para descubrimiento "
        "real."
    )
    print()
    print(
        "La tabla de recall por clase es la mas importante para "
        "decidir que hacer despues: las clases OOD con recall bajo "
        "son precisamente los casos mas dificiles -- las que un "
        "astronomo real necesitaria que el sistema NO se le escapen."
    )

    print()
    print("Archivos guardados:")
    print(" ", ruta_resultado)
    print(" ", ruta_umbrales)
    print(" ", ruta_por_clase)

    print()
    print(f"Tiempo total: {time.time() - inicio:.2f} segundos")


if __name__ == "__main__":
    main()