# -*- coding: utf-8 -*-

"""
======================================================================
SELECCION DE VARIABLES - StarEmbed / ZTF
======================================================================

Objetivo:
Determinar cuál es el conjunto MINIMO de variables de variabilidad
que conserva una capacidad discriminante significativa.

Se utilizan exclusivamente las variables de variabilidad estudiadas
en el experimento 18.

Experimentos:

  A) Variables individuales
  B) Seleccion progresiva (FORWARD SELECTION)
  C) Eliminacion progresiva (BACKWARD ELIMINATION)
  D) Control de etiquetas aleatorias

Metrica principal:
  Balanced Accuracy

Random State:
  42
"""

import os
import time
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")


# ======================================================================
# CONFIGURACION
# ======================================================================

RANDOM_STATE = 42

MAX_TRAIN = 25000
MAX_TEST = 8000

# Ruta relativa desde la ubicación del script
BASE_DIR = Path(__file__).resolve().parents[3]  # Sube 3 niveles hasta Astronomia/

# Directorio de resultados dentro de la misma carpeta que el script
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "19_"

DATA_DIR = BASE_DIR / "dataset" / "ZTF_40k_StarEmbed" / "data"

TRAIN_FILES = [
    DATA_DIR / "train-00000-of-00002.parquet",
    DATA_DIR / "train-00001-of-00002.parquet",
]

TEST_FILE = DATA_DIR / "test-00000-of-00001.parquet"

RESULTADOS = RESULTADOS_DIR / f"{PREFIJO}seleccion_variables_resultados.csv"


# ======================================================================
# VARIABLES
# ======================================================================

VARIABLES = [
    "median",
    "standard_deviation",
    "median_absolute_deviation",
    "amplitude",
    "percent_amplitude",
    "inter_percentile_range_25",
    "skew",
    "kurtosis",
    "stetson_K",
    "eta",
    "chi2",
    "maximum_slope",
]


# ======================================================================
# UTILIDADES
# ======================================================================

def linea():
    print("-" * 70)


def cargar_datos():

    print("=" * 70)
    print("CARGA DE DATOS")
    print("=" * 70)

    frames_train = []

    total = 0

    for fichero in TRAIN_FILES:

        print()
        print("Cargando TRAIN:")
        print(" ", fichero)

        if not fichero.exists():
            raise FileNotFoundError(fichero)

        df = pd.read_parquet(fichero)

        frames_train.append(df)

        total += len(df)

        print("Objetos:", f"{len(df):,}")

    train = pd.concat(
        frames_train,
        ignore_index=True
    )

    print()
    print("Cargando TEST:")
    print(" ", TEST_FILE)

    if not TEST_FILE.exists():
        raise FileNotFoundError(TEST_FILE)

    test = pd.read_parquet(TEST_FILE)

    if len(train) > MAX_TRAIN:
        train = train.iloc[:MAX_TRAIN].copy()

    if len(test) > MAX_TEST:
        test = test.iloc[:MAX_TEST].copy()

    print()
    print("TRAIN utilizado:", f"{len(train):,}")
    print("TEST utilizado :", f"{len(test):,}")

    return train, test


# ======================================================================
# EXTRACCION DE FEATURES
# ======================================================================

def obtener_banda(bands):

    if bands is None:
        return None

    try:
        if isinstance(bands, dict):
            for b in ["r", "g", "i"]:
                if b in bands and bands[b] is not None:
                    return bands[b]

        return None

    except Exception:
        return None


def extraer_features(df):

    resultados = []

    n = len(df)

    for i, (_, fila) in enumerate(df.iterrows()):

        if i % 5000 == 0:
            print(f"  {i:,}/{n:,}")

        valores = {}

        bands = fila.get("bands_data", None)

        banda = obtener_banda(bands)

        if banda is None:
            banda = {}

        target = banda.get("target", [])

        if target is None:
            target = []

        try:
            x = np.asarray(target, dtype=float)
        except Exception:
            x = np.array([], dtype=float)

        x = x[np.isfinite(x)]

        if len(x) == 0:

            for v in VARIABLES:
                valores[v] = np.nan

            resultados.append(valores)
            continue

        media = np.mean(x)
        mediana = np.median(x)

        std = np.std(x)

        mad = np.median(
            np.abs(x - mediana)
        )

        minimo = np.min(x)
        maximo = np.max(x)

        amplitud = (maximo - minimo) / 2.0

        if abs(media) > 1e-12:
            percent_amplitude = amplitud / abs(media)
        else:
            percent_amplitude = 0.0

        try:
            p25 = np.percentile(x, 12.5)
            p75 = np.percentile(x, 87.5)
            iqr25 = p75 - p25
        except Exception:
            iqr25 = np.nan

        # --------------------------------------------------------------
        # ASIMETRIA
        # --------------------------------------------------------------

        if std > 1e-12:

            z = (x - media) / std

            skew = np.mean(z ** 3)

            kurtosis = np.mean(z ** 4) - 3.0

        else:

            skew = 0.0
            kurtosis = 0.0

        # --------------------------------------------------------------
        # STETSON K
        # --------------------------------------------------------------

        if std > 1e-12 and len(x) > 1:

            delta = (x - media) / std

            stetson_k = (
                np.mean(np.abs(delta))
                /
                np.sqrt(np.mean(delta ** 2))
            )

        else:

            stetson_k = 0.0

        # --------------------------------------------------------------
        # ETA
        # --------------------------------------------------------------

        if len(x) > 1 and std > 1e-12:

            diferencias = np.diff(x)

            eta = (
                np.sum(diferencias ** 2)
                /
                ((len(x) - 1) * std ** 2)
            )

        else:

            eta = 0.0

        # --------------------------------------------------------------
        # CHI2
        # --------------------------------------------------------------

        if std > 1e-12:

            chi2 = np.sum(
                ((x - media) / std) ** 2
            ) / max(len(x) - 1, 1)

        else:

            chi2 = 0.0

        # --------------------------------------------------------------
        # MAXIMUM SLOPE
        # --------------------------------------------------------------

        try:

            mjd = banda.get("mjd", [])

            t = np.asarray(mjd, dtype=float)

            mask = np.isfinite(t) & np.isfinite(
                np.asarray(target, dtype=float)
            )

            t = t[mask]
            y = np.asarray(target, dtype=float)[mask]

            if len(t) > 1:

                dt = np.diff(t)
                dy = np.diff(y)

                valid = dt > 0

                if np.any(valid):

                    slopes = np.abs(
                        dy[valid] / dt[valid]
                    )

                    maximum_slope = np.max(slopes)

                else:

                    maximum_slope = 0.0

            else:

                maximum_slope = 0.0

        except Exception:

            maximum_slope = 0.0

        valores["median"] = mediana
        valores["standard_deviation"] = std
        valores["median_absolute_deviation"] = mad
        valores["amplitude"] = amplitud
        valores["percent_amplitude"] = percent_amplitude
        valores["inter_percentile_range_25"] = iqr25
        valores["skew"] = skew
        valores["kurtosis"] = kurtosis
        valores["stetson_K"] = stetson_k
        valores["eta"] = eta
        valores["chi2"] = chi2
        valores["maximum_slope"] = maximum_slope

        resultados.append(valores)

    return pd.DataFrame(resultados)


# ======================================================================
# LIMPIEZA
# ======================================================================

def limpiar(X_train, X_test):

    print()
    print("LIMPIEZA")

    print(
        "NaN TRAIN antes:",
        int(X_train.isna().sum().sum())
    )

    print(
        "NaN TEST antes :",
        int(X_test.isna().sum().sum())
    )

    med = X_train.median()

    X_train = X_train.fillna(med)
    X_test = X_test.fillna(med)

    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    print(
        "NaN TRAIN:",
        int(X_train.isna().sum().sum())
    )

    print(
        "NaN TEST :",
        int(X_test.isna().sum().sum())
    )

    return X_train, X_test


# ======================================================================
# EVALUACION
# ======================================================================

def evaluar(X_train, y_train, X_test, y_test, variables):

    Xtr = X_train[variables]
    Xte = X_test[variables]

    # --------------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------------

    t0 = time.time()

    rf = RandomForestClassifier(
        n_estimators=150,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_features="sqrt",
    )

    rf.fit(Xtr, y_train)

    pred_rf = rf.predict(Xte)

    rf_acc = accuracy_score(
        y_test,
        pred_rf
    )

    rf_bal = balanced_accuracy_score(
        y_test,
        pred_rf
    )

    rf_time = time.time() - t0

    # --------------------------------------------------------------
    # LOGISTICA
    # --------------------------------------------------------------

    t0 = time.time()

    log = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
    )

    log.fit(Xtr, y_train)

    pred_log = log.predict(Xte)

    log_acc = accuracy_score(
        y_test,
        pred_log
    )

    log_bal = balanced_accuracy_score(
        y_test,
        pred_log
    )

    log_time = time.time() - t0

    return {
        "RF_ACC": rf_acc,
        "RF_BAL": rf_bal,
        "LOG_ACC": log_acc,
        "LOG_BAL": log_bal,
        "RF_TIME": rf_time,
        "LOG_TIME": log_time,
    }


# ======================================================================
# RESULTADOS
# ======================================================================

def guardar_resultado(
    resultados,
    tipo,
    variables,
    metricas,
    paso
):

    resultados.append({
        "tipo": tipo,
        "paso": paso,
        "n_variables": len(variables),
        "variables": "|".join(variables),
        **metricas,
    })


# ======================================================================
# FORWARD SELECTION
# ======================================================================

def forward_selection(
    X_train,
    y_train,
    X_test,
    y_test,
    resultados
):

    print()
    print("=" * 70)
    print("SELECCION PROGRESIVA - FORWARD SELECTION")
    print("=" * 70)

    restantes = VARIABLES.copy()
    seleccionadas = []

    mejor_bal = -1.0

    paso = 0

    while restantes:

        candidatos = []

        print()
        print("VARIABLES seleccionadas:")
        print(
            "  ",
            seleccionadas if seleccionadas else "(ninguna)"
        )

        print()
        print("Probando candidatos:")

        for variable in restantes:

            conjunto = seleccionadas + [variable]

            print()
            print(
                "  +",
                variable,
                "->",
                len(conjunto),
                "variables"
            )

            metricas = evaluar(
                X_train,
                y_train,
                X_test,
                y_test,
                conjunto
            )

            candidatos.append(
                (
                    metricas["LOG_BAL"],
                    metricas["RF_BAL"],
                    variable,
                    metricas,
                )
            )

            print(
                "     RF BAL :",
                f"{metricas['RF_BAL']:.6f}"
            )

            print(
                "     LOG BAL:",
                f"{metricas['LOG_BAL']:.6f}"
            )

        candidatos.sort(
            key=lambda x: (
                x[0],
                x[1]
            ),
            reverse=True
        )

        mejor_log, mejor_rf, variable, metricas = candidatos[0]

        # --------------------------------------------------------------
        # Aceptamos la variable.
        # --------------------------------------------------------------

        seleccionadas.append(variable)
        restantes.remove(variable)

        paso += 1

        guardar_resultado(
            resultados,
            "FORWARD",
            seleccionadas.copy(),
            metricas,
            paso
        )

        print()
        print(
            ">>> AÑADIDA:",
            variable
        )

        print(
            "    LOG BAL:",
            f"{mejor_log:.6f}"
        )

        print(
            "    RF BAL :",
            f"{mejor_rf:.6f}"
        )

        # --------------------------------------------------------------
        # Parada:
        # cuando el conjunto ya alcanza prácticamente el rendimiento
        # del conjunto completo.
        # --------------------------------------------------------------

        if mejor_log > mejor_bal:
            mejor_bal = mejor_log

        if len(seleccionadas) >= len(VARIABLES):
            break

    return seleccionadas


# ======================================================================
# BACKWARD ELIMINATION
# ======================================================================

def backward_selection(
    X_train,
    y_train,
    X_test,
    y_test,
    resultados
):

    print()
    print("=" * 70)
    print("ELIMINACION PROGRESIVA - BACKWARD ELIMINATION")
    print("=" * 70)

    actuales = VARIABLES.copy()

    paso = 0

    while len(actuales) > 1:

        print()
        print(
            "Variables actuales:",
            len(actuales)
        )

        candidatos = []

        for variable in actuales:

            conjunto = [
                v for v in actuales
                if v != variable
            ]

            print()
            print(
                "  Eliminando:",
                variable
            )

            metricas = evaluar(
                X_train,
                y_train,
                X_test,
                y_test,
                conjunto
            )

            candidatos.append(
                (
                    metricas["LOG_BAL"],
                    metricas["RF_BAL"],
                    variable,
                    metricas,
                )
            )

            print(
                "     RF BAL :",
                f"{metricas['RF_BAL']:.6f}"
            )

            print(
                "     LOG BAL:",
                f"{metricas['LOG_BAL']:.6f}"
            )

        candidatos.sort(
            key=lambda x: (
                x[0],
                x[1]
            ),
            reverse=True
        )

        mejor_log, mejor_rf, eliminada, metricas = candidatos[0]

        actuales.remove(eliminada)

        paso += 1

        guardar_resultado(
            resultados,
            "BACKWARD",
            actuales.copy(),
            metricas,
            paso
        )

        print()
        print(
            ">>> ELIMINADA:",
            eliminada
        )

        print(
            "    Variables restantes:",
            len(actuales)
        )

        print(
            "    LOG BAL:",
            f"{mejor_log:.6f}"
        )

        print(
            "    RF BAL :",
            f"{mejor_rf:.6f}"
        )

    return actuales


# ======================================================================
# INDIVIDUALES
# ======================================================================

def evaluar_individuales(
    X_train,
    y_train,
    X_test,
    y_test,
    resultados
):

    print()
    print("=" * 70)
    print("EXPERIMENTO A - VARIABLES INDIVIDUALES")
    print("=" * 70)

    for variable in VARIABLES:

        print()
        print("VARIABLE:", variable)

        metricas = evaluar(
            X_train,
            y_train,
            X_test,
            y_test,
            [variable]
        )

        guardar_resultado(
            resultados,
            "INDIVIDUAL",
            [variable],
            metricas,
            1
        )

        print(
            "RF BAL :",
            f"{metricas['RF_BAL']:.6f}"
        )

        print(
            "LOG BAL:",
            f"{metricas['LOG_BAL']:.6f}"
        )


# ======================================================================
# CONTROL ALEATORIO
# ======================================================================

def control_aleatorio(
    X_train,
    y_train,
    X_test,
    y_test,
    variables,
    resultados
):

    print()
    print("=" * 70)
    print("CONTROL DE ETIQUETAS ALEATORIAS")
    print("=" * 70)

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    y_random = np.asarray(y_train).copy()

    rng.shuffle(y_random)

    metricas = evaluar(
        X_train,
        y_random,
        X_test,
        y_test,
        variables
    )

    guardar_resultado(
        resultados,
        "RANDOM_LABELS",
        variables,
        metricas,
        1
    )

    print()
    print(
        "RF ACC :",
        f"{metricas['RF_ACC']:.6f}"
    )

    print(
        "RF BAL :",
        f"{metricas['RF_BAL']:.6f}"
    )

    print(
        "LOG ACC:",
        f"{metricas['LOG_ACC']:.6f}"
    )

    print(
        "LOG BAL:",
        f"{metricas['LOG_BAL']:.6f}"
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    inicio = time.time()

    print("=" * 70)
    print("SELECCION DE VARIABLES - StarEmbed / ZTF")
    print("=" * 70)

    print()
    print("Objetivo:")
    print(
        "Determinar el conjunto minimo de variables"
    )
    print(
        "de variabilidad que conserva la señal."
    )

    print()
    print("Variables:")
    for v in VARIABLES:
        print(" ", v)

    print()
    print("Random State:", RANDOM_STATE)

    # --------------------------------------------------------------
    # CARGA
    # --------------------------------------------------------------

    train, test = cargar_datos()

    y_train = train["class_str"].astype(str)
    y_test = test["class_str"].astype(str)

    print()
    print("=" * 70)
    print("DISTRIBUCION DE CLASES")
    print("=" * 70)

    print()
    print("TRAIN:")
    print(y_train.value_counts())

    print()
    print("TEST:")
    print(y_test.value_counts())

    # --------------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("EXTRACCION DE FEATURES")
    print("=" * 70)

    print()
    print("TRAIN")

    X_train = extraer_features(train)

    print()
    print("TEST")

    X_test = extraer_features(test)

    # --------------------------------------------------------------
    # LIMPIEZA
    # --------------------------------------------------------------

    X_train, X_test = limpiar(
        X_train,
        X_test
    )

    print()
    print(
        "TRAIN shape:",
        X_train.shape
    )

    print(
        "TEST shape :",
        X_test.shape
    )

    # --------------------------------------------------------------
    # RESULTADOS
    # --------------------------------------------------------------

    resultados = []

    # --------------------------------------------------------------
    # INDIVIDUALES
    # --------------------------------------------------------------

    evaluar_individuales(
        X_train,
        y_train,
        X_test,
        y_test,
        resultados
    )

    # --------------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------------

    forward = forward_selection(
        X_train,
        y_train,
        X_test,
        y_test,
        resultados
    )

    # --------------------------------------------------------------
    # BACKWARD
    # --------------------------------------------------------------

    backward = backward_selection(
        X_train,
        y_train,
        X_test,
        y_test,
        resultados
    )

    # --------------------------------------------------------------
    # CONTROL ALEATORIO
    # --------------------------------------------------------------

    control_aleatorio(
        X_train,
        y_train,
        X_test,
        y_test,
        VARIABLES,
        resultados
    )

    # --------------------------------------------------------------
    # CSV
    # --------------------------------------------------------------

    df_resultados = pd.DataFrame(
        resultados
    )

    df_resultados.to_csv(
        RESULTADOS,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)

    print()
    print("FORWARD SELECTION:")
    print(
        "  ",
        " -> ".join(forward)
    )

    print()
    print("BACKWARD ELIMINATION:")
    print(
        "  ",
        " -> ".join(backward)
    )

    print()
    print("=" * 70)
    print("RESULTADOS GUARDADOS EN:")
    print("=" * 70)

    print(RESULTADOS)

    print()
    print("=" * 70)
    print(
        f"Tiempo total: {time.time() - inicio:.2f} segundos"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()