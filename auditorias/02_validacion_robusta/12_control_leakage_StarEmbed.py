# -*- coding: utf-8 -*-

"""
==============================================================================
CONTROL DE LEAKAGE - StarEmbed / ZTF
==============================================================================

Objetivos:

1. Comprobar que no existen sourceid duplicados entre TRAIN y TEST.
2. Comprobar duplicados de curvas de luz entre TRAIN y TEST.
3. Evaluar la clasificación usando:
       A) periodo solamente
       B) curvas de luz solamente
       C) periodo + curvas
4. Ejecutar un control correcto con etiquetas aleatorias.
5. Determinar si el resultado anterior (91.3 %) es reproducible
   sin leakage evidente.

IMPORTANTE:
    NO se utilizan:
        sourceid
        RA
        DEC

    El dataset StarEmbed contiene:
        g, r, i
        magnitudes
        errores
        tiempos
        cadencia
        periodo
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
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Ruta relativa desde la ubicación del script
BASE_DIR = Path(__file__).resolve().parents[2]  # Sube 2 niveles hasta Astronomia/

# Directorio de resultados dentro de la misma carpeta que el script
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "12_"

DATA_DIR = BASE_DIR / "dataset" / "ZTF_40k_StarEmbed" / "data"

TRAIN_FILES = [
    DATA_DIR / "train-00000-of-00002.parquet",
    DATA_DIR / "train-00001-of-00002.parquet",
]

TEST_FILE = DATA_DIR / "test-00000-of-00001.parquet"

RANDOM_STATE = 42

# Para poder hacer varias pruebas rápidamente.
# None = utilizar todo.
MAX_TRAIN = 25000
MAX_TEST = 8000


# ============================================================================
# CLASES
# ============================================================================

CLASSES = [
    "EA",
    "EW",
    "RRab",
    "RRc",
    "RRd",
    "RS CVn",
    "LPV"
]


# ============================================================================
# UTILIDADES
# ============================================================================

def banner(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def cargar_train():
    frames = []

    for fichero in TRAIN_FILES:

        print()
        print("Cargando:")
        print(" ", fichero)

        if not fichero.exists():
            raise FileNotFoundError(fichero)

        df = pd.read_parquet(fichero)

        print("Objetos:", len(df))

        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    return df


def cargar_test():

    print()
    print("Cargando:")
    print(" ", TEST_FILE)

    if not TEST_FILE.exists():
        raise FileNotFoundError(TEST_FILE)

    df = pd.read_parquet(TEST_FILE)

    print("Objetos:", len(df))

    return df


# ============================================================================
# IDENTIDAD
# ============================================================================

def comprobar_sourceid(train, test):

    banner("CONTROL 1 - SOURCEID")

    train_ids = set(train["sourceid"].astype(str))
    test_ids = set(test["sourceid"].astype(str))

    inter = train_ids.intersection(test_ids)

    print("SourceID TRAIN:", len(train_ids))
    print("SourceID TEST :", len(test_ids))
    print()
    print("SourceID comunes:", len(inter))

    if len(inter) == 0:
        print("RESULTADO: OK")
        print("No existe solapamiento de sourceid entre TRAIN y TEST.")
    else:
        print("¡¡¡ ALERTA !!!")
        print("Existen sourceid presentes en ambos conjuntos.")

        ejemplos = list(inter)[:10]

        print("Ejemplos:")
        for x in ejemplos:
            print(" ", x)

    return len(inter)


# ============================================================================
# HASH DE CURVAS
# ============================================================================

def array_bytes(arr):

    if arr is None:
        return b""

    try:
        a = np.asarray(arr, dtype=np.float64)

        if a.size == 0:
            return b""

        # Normalizamos representación
        a = np.ascontiguousarray(a)

        return a.tobytes()

    except Exception:
        return b""


def hash_curva(row):

    """
    Hash de las curvas g/r/i.

    Incluye:
        tiempo
        magnitud
        error

    No incluye:
        sourceid
        RA
        DEC
        class_str
    """

    h = hashlib.sha256()

    bands = row["bands_data"]

    for band in ["g", "r", "i"]:

        data = bands.get(band)

        if data is None:
            h.update(b"NONE")
            continue

        target = data.get("target")
        error = data.get("past_feat_dynamic_real")
        mjd = data.get("mjd")

        h.update(array_bytes(mjd))
        h.update(array_bytes(target))
        h.update(array_bytes(error))

    return h.hexdigest()


def comprobar_curvas(train, test):

    banner("CONTROL 2 - DUPLICADOS DE CURVAS")

    print("Calculando hashes TRAIN...")

    train_hashes = set()

    for i, (_, row) in enumerate(train.iterrows()):

        if i % 5000 == 0:
            print(f"  TRAIN {i:,}/{len(train):,}")

        train_hashes.add(hash_curva(row))

    print()
    print("Hashes TRAIN:", len(train_hashes))

    print()
    print("Calculando hashes TEST...")

    test_hashes = set()

    for i, (_, row) in enumerate(test.iterrows()):

        if i % 2000 == 0:
            print(f"  TEST  {i:,}/{len(test):,}")

        test_hashes.add(hash_curva(row))

    print()
    print("Hashes TEST:", len(test_hashes))

    comunes = train_hashes.intersection(test_hashes)

    print()
    print("Curvas idénticas TRAIN <-> TEST:", len(comunes))

    if len(comunes) == 0:
        print("RESULTADO: OK")
        print("No se han encontrado curvas idénticas.")
    else:
        print("¡¡¡ ALERTA !!!")
        print("Existen curvas idénticas entre TRAIN y TEST.")

    return len(comunes)


# ============================================================================
# EXTRACCIÓN DE FEATURES
# ============================================================================

def estadisticas_banda(data, prefijo):

    if data is None:
        return {
            f"{prefijo}_n": 0,
            f"{prefijo}_mean": np.nan,
            f"{prefijo}_std": np.nan,
            f"{prefijo}_median": np.nan,
            f"{prefijo}_mad": np.nan,
            f"{prefijo}_min": np.nan,
            f"{prefijo}_max": np.nan,
            f"{prefijo}_amplitude": np.nan,
            f"{prefijo}_iqr": np.nan,
            f"{prefijo}_p10": np.nan,
            f"{prefijo}_p90": np.nan,
            f"{prefijo}_slope": np.nan,
            f"{prefijo}_cadence_mean": np.nan,
            f"{prefijo}_cadence_std": np.nan,
        }

    mag = np.asarray(
        data.get("target", []),
        dtype=float
    )

    err = np.asarray(
        data.get("past_feat_dynamic_real", []),
        dtype=float
    )

    mjd = np.asarray(
        data.get("mjd", []),
        dtype=float
    )

    result = {}

    result[f"{prefijo}_n"] = len(mag)

    if len(mag) == 0:
        return {
            **result,
            f"{prefijo}_mean": np.nan,
            f"{prefijo}_std": np.nan,
            f"{prefijo}_median": np.nan,
            f"{prefijo}_mad": np.nan,
            f"{prefijo}_min": np.nan,
            f"{prefijo}_max": np.nan,
            f"{prefijo}_amplitude": np.nan,
            f"{prefijo}_iqr": np.nan,
            f"{prefijo}_p10": np.nan,
            f"{prefijo}_p90": np.nan,
            f"{prefijo}_slope": np.nan,
            f"{prefijo}_cadence_mean": np.nan,
            f"{prefijo}_cadence_std": np.nan,
        }

    mag = mag[np.isfinite(mag)]

    if len(mag) == 0:
        return {
            **result,
            f"{prefijo}_mean": np.nan,
            f"{prefijo}_std": np.nan,
            f"{prefijo}_median": np.nan,
            f"{prefijo}_mad": np.nan,
            f"{prefijo}_min": np.nan,
            f"{prefijo}_max": np.nan,
            f"{prefijo}_amplitude": np.nan,
            f"{prefijo}_iqr": np.nan,
            f"{prefijo}_p10": np.nan,
            f"{prefijo}_p90": np.nan,
            f"{prefijo}_slope": np.nan,
            f"{prefijo}_cadence_mean": np.nan,
            f"{prefijo}_cadence_std": np.nan,
        }

    median = np.median(mag)

    result[f"{prefijo}_mean"] = np.mean(mag)
    result[f"{prefijo}_std"] = np.std(mag)
    result[f"{prefijo}_median"] = median
    result[f"{prefijo}_mad"] = np.median(np.abs(mag - median))
    result[f"{prefijo}_min"] = np.min(mag)
    result[f"{prefijo}_max"] = np.max(mag)

    result[f"{prefijo}_amplitude"] = (
        np.percentile(mag, 95)
        - np.percentile(mag, 5)
    )

    result[f"{prefijo}_iqr"] = (
        np.percentile(mag, 75)
        - np.percentile(mag, 25)
    )

    result[f"{prefijo}_p10"] = np.percentile(mag, 10)
    result[f"{prefijo}_p90"] = np.percentile(mag, 90)

    # ---------------------------------------------------------------
    # Pendiente temporal
    # ---------------------------------------------------------------

    try:

        n = min(len(mjd), len(data.get("target", [])))

        x = np.asarray(
            data.get("mjd", [])[:n],
            dtype=float
        )

        y = np.asarray(
            data.get("target", [])[:n],
            dtype=float
        )

        mask = (
            np.isfinite(x)
            & np.isfinite(y)
        )

        x = x[mask]
        y = y[mask]

        if len(x) >= 3 and np.ptp(x) > 0:
            result[f"{prefijo}_slope"] = np.polyfit(
                x,
                y,
                1
            )[0]
        else:
            result[f"{prefijo}_slope"] = np.nan

    except Exception:
        result[f"{prefijo}_slope"] = np.nan

    # ---------------------------------------------------------------
    # Error fotométrico
    # ---------------------------------------------------------------

    err = err[np.isfinite(err)]

    if len(err) > 0:

        result[f"{prefijo}_err_mean"] = np.mean(err)
        result[f"{prefijo}_err_std"] = np.std(err)
        result[f"{prefijo}_err_median"] = np.median(err)

    else:

        result[f"{prefijo}_err_mean"] = np.nan
        result[f"{prefijo}_err_std"] = np.nan
        result[f"{prefijo}_err_median"] = np.nan

    # ---------------------------------------------------------------
    # Cadencia
    # ---------------------------------------------------------------

    if len(mjd) > 1:

        mjd = mjd[np.isfinite(mjd)]

        if len(mjd) > 1:

            dt = np.diff(mjd)
            dt = dt[np.isfinite(dt)]

            if len(dt) > 0:

                result[f"{prefijo}_cadence_mean"] = np.mean(dt)
                result[f"{prefijo}_cadence_std"] = np.std(dt)

            else:

                result[f"{prefijo}_cadence_mean"] = np.nan
                result[f"{prefijo}_cadence_std"] = np.nan

        else:

            result[f"{prefijo}_cadence_mean"] = np.nan
            result[f"{prefijo}_cadence_std"] = np.nan

    else:

        result[f"{prefijo}_cadence_mean"] = np.nan
        result[f"{prefijo}_cadence_std"] = np.nan

    return result


def extraer_features(df, usar_periodo=True):

    features = []

    for i, (_, row) in enumerate(df.iterrows()):

        if i % 5000 == 0:
            print(
                f"  {i:,}/{len(df):,}"
            )

        bands = row["bands_data"]

        f = {}

        for band in ["g", "r", "i"]:

            data = bands.get(band)

            f.update(
                estadisticas_banda(
                    data,
                    band
                )
            )

        if usar_periodo:
            f["period"] = row["period"]

        f["class_str"] = row["class_str"]

        features.append(f)

    return pd.DataFrame(features)


# ============================================================================
# LIMPIEZA
# ============================================================================

def limpiar(df):

    feature_cols = [
        c for c in df.columns
        if c != "class_str"
    ]

    X = df[feature_cols].copy()

    y = df["class_str"].copy()

    print("NaN antes:", X.isna().sum().sum())

    # Convertimos a numérico
    for c in X.columns:
        X[c] = pd.to_numeric(
            X[c],
            errors="coerce"
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Mediana calculada SOLO sobre TRAIN
    return X, y


def rellenar_nan_train_test(
    X_train,
    X_test
):

    medianas = X_train.median()

    X_train = X_train.fillna(medianas)
    X_test = X_test.fillna(medianas)

    return X_train, X_test


# ============================================================================
# CLASIFICADORES
# ============================================================================

def clasificar(
    X_train,
    y_train,
    X_test,
    y_test,
    nombre
):

    print()
    print("-" * 70)
    print(nombre)
    print("-" * 70)

    # ---------------------------------------------------------------
    # Random Forest
    # ---------------------------------------------------------------

    t0 = time.time()

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_features="sqrt"
    )

    rf.fit(X_train, y_train)

    pred_rf = rf.predict(X_test)

    tiempo_rf = time.time() - t0

    acc_rf = accuracy_score(
        y_test,
        pred_rf
    )

    print()
    print(
        f"Random Forest : "
        f"{acc_rf:.6f} "
        f"({tiempo_rf:.2f} s)"
    )

    # ---------------------------------------------------------------
    # Logística
    # ---------------------------------------------------------------

    t0 = time.time()

    log = Pipeline([
        (
            "scaler",
            RobustScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE,
                class_weight="balanced"
            )
        )
    ])

    log.fit(X_train, y_train)

    pred_log = log.predict(X_test)

    tiempo_log = time.time() - t0

    acc_log = accuracy_score(
        y_test,
        pred_log
    )

    print(
        f"Logística     : "
        f"{acc_log:.6f} "
        f"({tiempo_log:.2f} s)"
    )

    return {
        "RF": acc_rf,
        "LOG": acc_log,
        "rf_model": rf,
        "log_model": log
    }


# ============================================================================
# CONTROL DE ETIQUETAS ALEATORIAS
# ============================================================================

def control_etiquetas_aleatorias(
    X_train,
    y_train,
    X_test,
    y_test
):

    banner(
        "CONTROL - ETIQUETAS ALEATORIAS"
    )

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    # MUY IMPORTANTE:
    #
    # Se permutan las etiquetas dentro de TRAIN.
    #
    # Las etiquetas TEST permanecen intactas.
    #
    # De esta forma eliminamos la correspondencia
    # entre X_train e y_train.
    #
    # No se debe barajar X e y independientemente
    # ni mezclar TRAIN y TEST.

    y_train_random = np.asarray(
        y_train
    ).copy()

    rng.shuffle(
        y_train_random
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_features="sqrt"
    )

    rf.fit(
        X_train,
        y_train_random
    )

    pred = rf.predict(
        X_test
    )

    acc = accuracy_score(
        y_test,
        pred
    )

    print()
    print(
        f"Accuracy etiquetas aleatorias: "
        f"{acc:.6f}"
    )

    print(
        "Esperada aproximadamente: "
        f"{1 / len(CLASSES):.6f}"
    )

    print()
    print(
        "Classification report:"
    )

    print(
        classification_report(
            y_test,
            pred,
            labels=CLASSES,
            zero_division=0
        )
    )

    print()
    print("Matriz de confusión:")

    print(
        confusion_matrix(
            y_test,
            pred,
            labels=CLASSES
        )
    )

    return acc


# ============================================================================
# CONTROL PERIOD SOLO
# ============================================================================

def ejecutar_experimento_periodo(
    train_features,
    test_features
):

    banner(
        "EXPERIMENTO A - SOLO PERIODO"
    )

    X_train = train_features[
        ["period"]
    ].copy()

    X_test = test_features[
        ["period"]
    ].copy()

    y_train = train_features[
        "class_str"
    ]

    y_test = test_features[
        "class_str"
    ]

    X_train, X_test = rellenar_nan_train_test(
        X_train,
        X_test
    )

    return clasificar(
        X_train,
        y_train,
        X_test,
        y_test,
        "PERIODO"
    )


# ============================================================================
# CONTROL CURVAS
# ============================================================================

def ejecutar_experimento_curvas(
    train_features,
    test_features
):

    banner(
        "EXPERIMENTO B - SOLO CURVAS"
    )

    cols = [
        c
        for c in train_features.columns
        if c not in [
            "class_str",
            "period"
        ]
    ]

    X_train = train_features[
        cols
    ].copy()

    X_test = test_features[
        cols
    ].copy()

    y_train = train_features[
        "class_str"
    ]

    y_test = test_features[
        "class_str"
    ]

    X_train, X_test = rellenar_nan_train_test(
        X_train,
        X_test
    )

    return clasificar(
        X_train,
        y_train,
        X_test,
        y_test,
        "CURVAS"
    )


# ============================================================================
# CURVAS + PERIODO
# ============================================================================

def ejecutar_experimento_completo(
    train_features,
    test_features
):

    banner(
        "EXPERIMENTO C - CURVAS + PERIODO"
    )

    cols = [
        c
        for c in train_features.columns
        if c != "class_str"
    ]

    X_train = train_features[
        cols
    ].copy()

    X_test = test_features[
        cols
    ].copy()

    y_train = train_features[
        "class_str"
    ]

    y_test = test_features[
        "class_str"
    ]

    X_train, X_test = rellenar_nan_train_test(
        X_train,
        X_test
    )

    return clasificar(
        X_train,
        y_train,
        X_test,
        y_test,
        "CURVAS + PERIODO"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    inicio = time.time()

    banner(
        "CONTROL DE LEAKAGE - StarEmbed / ZTF"
    )

    print()
    print("Objetivo:")
    print(
        "Determinar si la clasificación de StarEmbed"
    )
    print(
        "es una señal física real o consecuencia de leakage."
    )

    print()
    print("Random State:", RANDOM_STATE)

    # ========================================================================
    # CARGA
    # ========================================================================

    banner("CARGA DE DATOS")

    train = cargar_train()
    test = cargar_test()

    # ------------------------------------------------------------------------
    # LIMITACIÓN OPCIONAL
    # ------------------------------------------------------------------------

    if MAX_TRAIN is not None and len(train) > MAX_TRAIN:

        train = train.sample(
            n=MAX_TRAIN,
            random_state=RANDOM_STATE
        ).reset_index(drop=True)

    if MAX_TEST is not None and len(test) > MAX_TEST:

        test = test.sample(
            n=MAX_TEST,
            random_state=RANDOM_STATE
        ).reset_index(drop=True)

    print()
    print("TRAIN utilizado:", f"{len(train):,}")
    print("TEST utilizado :", f"{len(test):,}")

    # ========================================================================
    # CLASES
    # ========================================================================

    banner("DISTRIBUCIÓN DE CLASES")

    print()
    print("TRAIN:")
    print(
        train["class_str"].value_counts()
    )

    print()
    print("TEST:")
    print(
        test["class_str"].value_counts()
    )

    # ========================================================================
    # LEAKAGE SOURCEID
    # ========================================================================

    source_overlap = comprobar_sourceid(
        train,
        test
    )

    # ========================================================================
    # LEAKAGE CURVAS
    # ========================================================================

    curve_overlap = comprobar_curvas(
        train,
        test
    )

    # ========================================================================
    # FEATURES
    # ========================================================================

    banner(
        "EXTRACCIÓN DE FEATURES"
    )

    print()
    print("TRAIN")

    train_features = extraer_features(
        train,
        usar_periodo=True
    )

    print()
    print("TEST")

    test_features = extraer_features(
        test,
        usar_periodo=True
    )

    # ========================================================================
    # LIMPIEZA
    # ========================================================================

    banner("LIMPIEZA")

    X_train_all, y_train = limpiar(
        train_features
    )

    X_test_all, y_test = limpiar(
        test_features
    )

    X_train_all, X_test_all = rellenar_nan_train_test(
        X_train_all,
        X_test_all
    )

    print(
        "NaN TRAIN:",
        X_train_all.isna().sum().sum()
    )

    print(
        "NaN TEST :",
        X_test_all.isna().sum().sum()
    )

    # Volvemos a construir los DataFrames completos
    # porque los experimentos necesitan class_str.

    train_features_clean = X_train_all.copy()
    train_features_clean["class_str"] = y_train.values

    test_features_clean = X_test_all.copy()
    test_features_clean["class_str"] = y_test.values

    # ========================================================================
    # EXPERIMENTO A
    # ========================================================================

    resultado_periodo = ejecutar_experimento_periodo(
        train_features_clean,
        test_features_clean
    )

    # ========================================================================
    # EXPERIMENTO B
    # ========================================================================

    resultado_curvas = ejecutar_experimento_curvas(
        train_features_clean,
        test_features_clean
    )

    # ========================================================================
    # EXPERIMENTO C
    # ========================================================================

    resultado_completo = ejecutar_experimento_completo(
        train_features_clean,
        test_features_clean
    )

    # ========================================================================
    # CONTROL ALEATORIO
    # ========================================================================

    banner(
        "CONTROL DE ETIQUETAS ALEATORIAS"
    )

    cols = [
        c
        for c in train_features_clean.columns
        if c != "class_str"
    ]

    X_train = train_features_clean[
        cols
    ].copy()

    X_test = test_features_clean[
        cols
    ].copy()

    y_train = train_features_clean[
        "class_str"
    ]

    y_test = test_features_clean[
        "class_str"
    ]

    random_accuracy = control_etiquetas_aleatorias(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # ========================================================================
    # RESUMEN
    # ========================================================================

    banner("RESUMEN FINAL")

    print()
    print(
        f"{'EXPERIMENTO':30s}"
        f"{'RF':>12s}"
        f"{'LOG':>12s}"
    )

    print("-" * 55)

    print(
        f"{'SOLO PERIODO':30s}"
        f"{resultado_periodo['RF']:12.6f}"
        f"{resultado_periodo['LOG']:12.6f}"
    )

    print(
        f"{'SOLO CURVAS':30s}"
        f"{resultado_curvas['RF']:12.6f}"
        f"{resultado_curvas['LOG']:12.6f}"
    )

    print(
        f"{'CURVAS + PERIODO':30s}"
        f"{resultado_completo['RF']:12.6f}"
        f"{resultado_completo['LOG']:12.6f}"
    )

    print(
        f"{'ETIQUETAS ALEATORIAS':30s}"
        f"{random_accuracy:12.6f}"
        f"{'-':>12s}"
    )

    print()
    print(
        "Accuracy esperada aleatoria:",
        f"{1 / len(CLASSES):.6f}"
    )

    # ========================================================================
    # DIAGNÓSTICO
    # ========================================================================

    banner("DIAGNÓSTICO")

    print()

    print(
        "SourceID solapados:",
        source_overlap
    )

    print(
        "Curvas idénticas:",
        curve_overlap
    )

    print()

    if random_accuracy < 0.20:

        print(
            "CONTROL ALEATORIO: OK"
        )

        print(
            "El clasificador pierde prácticamente"
        )

        print(
            "toda la capacidad predictiva al destruir"
        )

        print(
            "la correspondencia entre features y etiquetas."
        )

    else:

        print(
            "¡¡¡ ALERTA !!!"
        )

        print(
            "El control de etiquetas aleatorias"
        )

        print(
            "sigue produciendo una accuracy demasiado alta."
        )

        print(
            "No debemos considerar todavía validado"
        )

        print(
            "el experimento físico."
        )

    print()

    if source_overlap == 0 and curve_overlap == 0:

        print(
            "No se ha detectado leakage directo"
        )

        print(
            "por sourceid ni por curvas idénticas."
        )

    else:

        print(
            "Se ha detectado posible leakage."
        )

    print()

    if (
        random_accuracy < 0.20
        and source_overlap == 0
        and curve_overlap == 0
    ):

        print(
            "RESULTADO:"
        )

        print(
            "El experimento StarEmbed pasa los controles"
        )

        print(
            "básicos de leakage."
        )

        print(
            "La siguiente cuestión es separar"
        )

        print(
            "la información aportada por PERIOD"
        )

        print(
            "de la información contenida en las"
        )

        print(
            "CURVAS DE LUZ."
        )

    else:

        print(
            "RESULTADO:"
        )

        print(
            "NO SE PUEDE VALIDAR TODAVÍA."
        )

    # ========================================================================
    # CSV
    # ========================================================================

    resultados = pd.DataFrame([
        {
            "experimento": "SOLO_PERIODO",
            "RF": resultado_periodo["RF"],
            "LOG": resultado_periodo["LOG"]
        },
        {
            "experimento": "SOLO_CURVAS",
            "RF": resultado_curvas["RF"],
            "LOG": resultado_curvas["LOG"]
        },
        {
            "experimento": "CURVAS_PERIODO",
            "RF": resultado_completo["RF"],
            "LOG": resultado_completo["LOG"]
        },
        {
            "experimento": "ETIQUETAS_ALEATORIAS",
            "RF": random_accuracy,
            "LOG": np.nan
        },
        {
            "experimento": "SOURCEID_OVERLAP",
            "RF": source_overlap,
            "LOG": np.nan
        },
        {
            "experimento": "CURVE_OVERLAP",
            "RF": curve_overlap,
            "LOG": np.nan
        }
    ])

    salida = RESULTADOS_DIR / f"{PREFIJO}control_leakage_resultados.csv"

    resultados.to_csv(
        salida,
        index=False
    )

    print()
    print(
        "Resultados guardados en:"
    )
    print(
        salida
    )

    # ========================================================================
    # TIEMPO
    # ========================================================================

    tiempo = time.time() - inicio

    print()
    print("=" * 70)
    print(
        f"Tiempo total: {tiempo:.2f} segundos"
    )
    print("=" * 70)


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    main()