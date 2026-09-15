# -*- coding: utf-8 -*-

"""
04_clasificador_datasets.py

PRUEBA DE SESGO ENTRE DATASETS
==============================

Objetivo:
    Determinar hasta qué punto podemos identificar si un objeto
    procede de DEEP, DISK o M31 utilizando únicamente sus 42 features.

Interpretación científica:
    Si un clasificador consigue una precisión muy elevada,
    significa que existen diferencias estadísticas importantes
    entre los datasets.

    Esto NO significa necesariamente que existan diferencias
    astronómicas. Pueden existir efectos de selección,
    instrumentación, campo observado, magnitud, cadencia, etc.

    Esta prueba sirve para detectar posibles sesgos antes de
    construir el detector de anomalías.

Entrada esperada:
    feature_m31.dat
    feature_m31.name
    oid_m31.dat

    feature_deep.dat
    feature_deep.name
    oid_deep.dat

    feature_disk.dat
    feature_disk.name
    oid_disk.dat

Salida:
    04_resultados_clasificador.txt
    04_importancia_features.csv
    04_matriz_confusion_random_forest.png
    04_matriz_confusion_logistica.png
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ==============================================================
# CONFIGURACIÓN
# ==============================================================

# Ruta relativa desde la ubicación del script
BASE_DIR = Path(__file__).resolve().parents[3]  # Sube 3 niveles hasta Astronomia/

# Directorio de resultados dentro de la misma carpeta que el script
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "04_"

MUESTRA_POR_DATASET = 50_000

TEST_SIZE = 0.20

RANDOM_STATE = 42

RF_N_ESTIMATORS = 150

RF_N_JOBS = -1

DATASETS = {
    "M31": {
        "feature": BASE_DIR / "dataset" / "ZTF_DR3" / "feature_m31.dat",
        "name": BASE_DIR / "dataset" / "ZTF_DR3" / "feature_m31.name",
        "oid": BASE_DIR / "dataset" / "ZTF_DR3" / "oid_m31.dat"
    },
    "DEEP": {
        "feature": BASE_DIR / "dataset" / "ZTF_DR3" / "feature_deep.dat",
        "name": BASE_DIR / "dataset" / "ZTF_DR3" / "feature_deep.name",
        "oid": BASE_DIR / "dataset" / "ZTF_DR3" / "oid_deep.dat"
    },
    "DISK": {
        "feature": BASE_DIR / "dataset" / "ZTF_DR3" / "feature_disk.dat",
        "name": BASE_DIR / "dataset" / "ZTF_DR3" / "feature_disk.name",
        "oid": BASE_DIR / "dataset" / "ZTF_DR3" / "oid_disk.dat"
    }
}


# ==============================================================
# UTILIDADES
# ==============================================================

def imprimir(texto="", archivo=None):
    print(texto)

    if archivo is not None:
        archivo.write(str(texto) + "\n")


def cargar_nombres(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().split()


def construir_dtype(nombres):
    return [(nombre, np.float32) for nombre in nombres]


def cargar_dataset(nombre_dataset):
    """
    Carga un dataset mediante memmap.
    No copia todo el dataset a RAM.
    """

    info = DATASETS[nombre_dataset]

    feature_path = info["feature"]
    name_path = info["name"]
    oid_path = info["oid"]

    if not feature_path.exists():
        raise FileNotFoundError(feature_path)

    if not name_path.exists():
        raise FileNotFoundError(name_path)

    if not oid_path.exists():
        raise FileNotFoundError(oid_path)

    nombres = cargar_nombres(name_path)

    dtype = construir_dtype(nombres)

    oid = np.memmap(
        oid_path,
        mode="r",
        dtype=np.uint64
    )

    feature = np.memmap(
        feature_path,
        mode="r",
        dtype=dtype,
        shape=oid.shape
    )

    return feature, oid, nombres


def obtener_muestra(feature, nombres, n_objetos, rng):
    """
    Extrae una muestra aleatoria y la convierte a ndarray normal.
    """

    total = len(feature)

    n = min(n_objetos, total)

    indices = rng.choice(
        total,
        size=n,
        replace=False
    )

    # Convertimos solamente la muestra a RAM
    X = np.empty(
        (n, len(nombres)),
        dtype=np.float32
    )

    for i, nombre in enumerate(nombres):
        X[:, i] = feature[nombre][indices]

    return X


def matriz_confusion_png(cm, clases, titulo, filename):
    """
    Genera matriz de confusión.
    """

    fig, ax = plt.subplots(figsize=(7, 6))

    imagen = ax.imshow(cm)

    ax.set_title(titulo)

    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")

    ax.set_xticks(range(len(clases)))
    ax.set_yticks(range(len(clases)))

    ax.set_xticklabels(clases)
    ax.set_yticklabels(clases)

    # Valores dentro de las celdas
    for i in range(len(clases)):
        for j in range(len(clases)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    fig.colorbar(imagen, ax=ax)

    plt.tight_layout()

    path = RESULTADOS_DIR / filename

    plt.savefig(
        path,
        dpi=150
    )

    plt.close()

    return path


# ==============================================================
# PROGRAMA PRINCIPAL
# ==============================================================

def main():

    inicio = time.time()

    salida_txt = RESULTADOS_DIR / f"{PREFIJO}resultados_clasificador.txt"

    with open(
        salida_txt,
        "w",
        encoding="utf-8"
    ) as log:

        imprimir("=" * 78, log)
        imprimir("CLASIFICADOR DE ORIGEN DE DATASETS", log)
        imprimir("=" * 78, log)

        imprimir("", log)

        imprimir(
            "Objetivo:",
            log
        )

        imprimir(
            "Determinar si las 42 features permiten identificar",
            log
        )

        imprimir(
            "si un objeto pertenece a M31, DEEP o DISK.",
            log
        )

        imprimir("", log)

        imprimir(
            f"Muestra por dataset: {MUESTRA_POR_DATASET:,}",
            log
        )

        imprimir(
            f"Random State: {RANDOM_STATE}",
            log
        )

        # ------------------------------------------------------
        # CARGA
        # ------------------------------------------------------

        rng = np.random.default_rng(RANDOM_STATE)

        muestras = []
        etiquetas = []

        nombres_globales = None

        imprimir("", log)
        imprimir("=" * 78, log)
        imprimir("CARGA DE DATOS", log)
        imprimir("=" * 78, log)

        for nombre_dataset in ["M31", "DEEP", "DISK"]:

            imprimir("", log)
            imprimir(
                f"Dataset: {nombre_dataset}",
                log
            )

            feature, oid, nombres = cargar_dataset(
                nombre_dataset
            )

            if nombres_globales is None:
                nombres_globales = nombres
            else:
                if nombres != nombres_globales:
                    raise RuntimeError(
                        "Los datasets no tienen las mismas features."
                    )

            imprimir(
                f"Objetos disponibles: {len(feature):,}",
                log
            )

            imprimir(
                f"Features: {len(nombres)}",
                log
            )

            X = obtener_muestra(
                feature,
                nombres,
                MUESTRA_POR_DATASET,
                rng
            )

            imprimir(
                f"Muestra obtenida: {len(X):,}",
                log
            )

            muestras.append(X)

            etiquetas.extend(
                [nombre_dataset] * len(X)
            )

        # ------------------------------------------------------
        # CONSTRUCCIÓN DE MATRIZ
        # ------------------------------------------------------

        imprimir("", log)
        imprimir("=" * 78, log)
        imprimir("CONSTRUCCIÓN DE LA MATRIZ", log)
        imprimir("=" * 78, log)

        X = np.vstack(muestras)

        y = np.array(etiquetas)

        imprimir(
            f"Objetos totales: {len(X):,}",
            log
        )

        imprimir(
            f"Features: {X.shape[1]}",
            log
        )

        imprimir("", log)

        # Comprobación de datos inválidos
        n_nan = np.isnan(X).sum()
        n_inf = np.isinf(X).sum()

        imprimir(
            f"NaN: {n_nan:,}",
            log
        )

        imprimir(
            f"INF: {n_inf:,}",
            log
        )

        if n_nan > 0 or n_inf > 0:

            imprimir(
                "",
                log
            )

            imprimir(
                "ERROR: existen NaN o INF.",
                log
            )

            imprimir(
                "No se continúa.",
                log
            )

            return

        # ------------------------------------------------------
        # TRAIN / TEST
        # ------------------------------------------------------

        imprimir("", log)
        imprimir("=" * 78, log)
        imprimir("DIVISIÓN TRAIN / TEST", log)
        imprimir("=" * 78, log)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        imprimir(
            f"Train: {len(X_train):,}",
            log
        )

        imprimir(
            f"Test : {len(X_test):,}",
            log
        )

        # ------------------------------------------------------
        # RANDOM FOREST
        # ------------------------------------------------------

        imprimir("", log)
        imprimir("=" * 78, log)
        imprimir("RANDOM FOREST", log)
        imprimir("=" * 78, log)

        inicio_rf = time.time()

        rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=RF_N_JOBS,
            max_features="sqrt"
        )

        rf.fit(
            X_train,
            y_train
        )

        y_pred_rf = rf.predict(X_test)

        tiempo_rf = time.time() - inicio_rf

        accuracy_rf = accuracy_score(
            y_test,
            y_pred_rf
        )

        imprimir(
            f"Tiempo entrenamiento: {tiempo_rf:.2f} s",
            log
        )

        imprimir(
            f"Accuracy: {accuracy_rf:.6f}",
            log
        )

        imprimir("", log)

        imprimir(
            "Classification report:",
            log
        )

        report_rf = classification_report(
            y_test,
            y_pred_rf,
            digits=5
        )

        imprimir(
            report_rf,
            log
        )

        cm_rf = confusion_matrix(
            y_test,
            y_pred_rf,
            labels=["M31", "DEEP", "DISK"]
        )

        imprimir(
            "Matriz de confusión:",
            log
        )

        imprimir(
            str(cm_rf),
            log
        )

        path_cm_rf = matriz_confusion_png(
            cm_rf,
            ["M31", "DEEP", "DISK"],
            "Random Forest - Origen del dataset",
            f"{PREFIJO}matriz_confusion_random_forest.png"
        )

        imprimir("", log)

        imprimir(
            f"Matriz guardada en: {path_cm_rf}",
            log
        )

        # ------------------------------------------------------
        # IMPORTANCIA FEATURES
        # ------------------------------------------------------

        imprimir("", log)
        imprimir("=" * 78, log)
        imprimir("IMPORTANCIA DE FEATURES - RANDOM FOREST", log)
        imprimir("=" * 78, log)

        importancias = rf.feature_importances_

        orden = np.argsort(
            importancias
        )[::-1]

        imprimir("", log)

        imprimir(
            f"{'RANGO':<8}{'FEATURE':<45}{'IMPORTANCIA':>15}",
            log
        )

        imprimir(
            "-" * 68,
            log
        )

        for rango, idx in enumerate(orden, start=1):

            imprimir(
                f"{rango:<8}"
                f"{nombres_globales[idx]:<45}"
                f"{importancias[idx]:>15.8f}",
                log
            )

        # ------------------------------------------------------
        # GUARDAR IMPORTANCIAS
        # ------------------------------------------------------

        csv_path = RESULTADOS_DIR / f"{PREFIJO}importancia_features.csv"

        with open(
            csv_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "rank,feature,importance\n"
            )

            for rango, idx in enumerate(
                orden,
                start=1
            ):

                f.write(
                    f"{rango},"
                    f"{nombres_globales[idx]},"
                    f"{importancias[idx]:.10f}\n"
                )

        imprimir("", log)

        imprimir(
            f"Importancias guardadas en: {csv_path}",
            log
        )

        # ------------------------------------------------------
        # REGRESIÓN LOGÍSTICA
        # ------------------------------------------------------

        imprimir("", log)
        imprimir("=" * 78, log)
        imprimir("REGRESIÓN LOGÍSTICA", log)
        imprimir("=" * 78, log)

        imprimir(
            "Se utiliza RobustScaler para reducir el efecto",
            log
        )

        imprimir(
            "de las diferencias de escala y valores extremos.",
            log
        )

        inicio_log = time.time()

        scaler = RobustScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )

        logistica = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs"
        )

        logistica.fit(
            X_train_scaled,
            y_train
        )

        y_pred_log = logistica.predict(
            X_test_scaled
        )

        tiempo_log = time.time() - inicio_log

        accuracy_log = accuracy_score(
            y_test,
            y_pred_log
        )

        imprimir(
            f"Tiempo entrenamiento: {tiempo_log:.2f} s",
            log
        )

        imprimir(
            f"Accuracy: {accuracy_log:.6f}",
            log
        )

        imprimir("", log)

        report_log = classification_report(
            y_test,
            y_pred_log,
            digits=5
        )

        imprimir(
            report_log,
            log
        )

        cm_log = confusion_matrix(
            y_test,
            y_pred_log,
            labels=["M31", "DEEP", "DISK"]
        )

        imprimir(
            "Matriz de confusión:",
            log
        )

        imprimir(
            str(cm_log),
            log
        )

        path_cm_log = matriz_confusion_png(
            cm_log,
            ["M31", "DEEP", "DISK"],
            "Regresión Logística - Origen del dataset",
            f"{PREFIJO}matriz_confusion_logistica.png"
        )

        imprimir("", log)

        imprimir(
            f"Matriz guardada en: {path_cm_log}",
            log
        )

        # ------------------------------------------------------
        # COEFICIENTES DE LA REGRESIÓN
        # ------------------------------------------------------

        imprimir("", log)
        imprimir("=" * 78, log)
        imprimir("IMPORTANCIA DE FEATURES - REGRESIÓN LOGÍSTICA", log)
        imprimir("=" * 78, log)

        coef_abs = np.mean(
            np.abs(logistica.coef_),
            axis=0
        )

        orden_coef = np.argsort(
            coef_abs
        )[::-1]

        imprimir("", log)

        imprimir(
            f"{'RANGO':<8}{'FEATURE':<45}{'COEF. ABS.':>15}",
            log
        )

        imprimir(
            "-" * 68,
            log
        )

        for rango, idx in enumerate(
            orden_coef,
            start=1
        ):

            imprimir(
                f"{rango:<8}"
                f"{nombres_globales[idx]:<45}"
                f"{coef_abs[idx]:>15.8f}",
                log
            )

        # ------------------------------------------------------
        # CONCLUSIÓN
        # ------------------------------------------------------

        imprimir("", log)
        imprimir("=" * 78, log)
        imprimir("INTERPRETACIÓN PRELIMINAR", log)
        imprimir("=" * 78, log)

        imprimir("", log)

        imprimir(
            f"Accuracy Random Forest : {accuracy_rf:.6f}",
            log
        )

        imprimir(
            f"Accuracy Logística    : {accuracy_log:.6f}",
            log
        )

        imprimir("", log)

        # Accuracy de referencia
        accuracy_azar = 1.0 / 3.0

        imprimir(
            f"Accuracy esperada al azar: {accuracy_azar:.6f}",
            log
        )

        imprimir("", log)

        if accuracy_rf >= 0.95:

            imprimir(
                "RESULTADO: SEPARACIÓN MUY FUERTE.",
                log
            )

            imprimir(
                "Los datasets poseen una firma estadística",
                log
            )

            imprimir(
                "muy fácilmente reconocible.",
                log
            )

            imprimir(
                "Esto constituye una ALERTA de posible sesgo.",
                log
            )

        elif accuracy_rf >= 0.75:

            imprimir(
                "RESULTADO: SEPARACIÓN MODERADA/FUERTE.",
                log
            )

            imprimir(
                "Existe información considerable que permite",
                log
            )

            imprimir(
                "identificar el dataset de procedencia.",
                log
            )

            imprimir(
                "Debe investigarse el origen de esta diferencia.",
                log
            )

        elif accuracy_rf >= 0.50:

            imprimir(
                "RESULTADO: SEPARACIÓN DÉBIL/MODERADA.",
                log
            )

            imprimir(
                "Existe cierta información que permite distinguir",
                log
            )

            imprimir(
                "los datasets, pero no de forma perfecta.",
                log
            )

        else:

            imprimir(
                "RESULTADO: SEPARACIÓN DÉBIL.",
                log
            )

            imprimir(
                "Las 42 features contienen poca información",
                log
            )

            imprimir(
                "sobre el origen del dataset.",
                log
            )

        imprimir("", log)

        imprimir(
            "IMPORTANTE:",
            log
        )

        imprimir(
            "Una clasificación alta NO demuestra que los datasets",
            log
        )

        imprimir(
            "representen poblaciones astronómicas diferentes.",
            log
        )

        imprimir(
            "Puede existir selección observacional, diferencias",
            log
        )

        imprimir(
            "instrumentales, magnitud, cadencia, campo observado",
            log
        )

        imprimir(
            "u otros efectos no astronómicos.",
            log
        )

        imprimir("", log)

        imprimir(
            "El siguiente análisis deberá determinar qué features",
            log
        )

        imprimir(
            "son responsables de la separación.",
            log
        )

        # ------------------------------------------------------
        # FIN
        # ------------------------------------------------------

        tiempo_total = time.time() - inicio

        imprimir("", log)
        imprimir("=" * 78, log)
        imprimir(
            f"Tiempo total: {tiempo_total:.2f} segundos",
            log
        )
        imprimir("=" * 78, log)

    print()
    print("=" * 78)
    print("PROCESO TERMINADO")
    print("=" * 78)
    print()
    print(f"Resultados: {salida_txt}")


# ==============================================================
# EJECUCIÓN
# ==============================================================

if __name__ == "__main__":
    main()