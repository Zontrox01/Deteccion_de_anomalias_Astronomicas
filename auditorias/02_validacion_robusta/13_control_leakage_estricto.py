# -*- coding: utf-8 -*-

"""
13_control_leakage_estricto.py

CONTROL DE LEAKAGE ESTRICTO - StarEmbed / ZTF

Objetivo:
Determinar si la capacidad de clasificación permanece después
de eliminar explícitamente las fuentes de leakage detectadas.

Controles:
    1. SOURCEID compartido
    2. Curvas idénticas TRAIN <-> TEST
    3. Test limpio
    4. Test original
    5. Etiquetas aleatorias correctamente controladas
    6. Accuracy
    7. Balanced Accuracy

IMPORTANTE:
El control de etiquetas aleatorias se realiza sobre el mismo
TEST LIMPIO utilizado en el experimento principal.
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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score

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
PREFIJO = "13_"

DATA_DIR = BASE_DIR / "dataset" / "ZTF_40k_StarEmbed" / "data"

TRAIN_FILES = [
    DATA_DIR / "train-00000-of-00002.parquet",
    DATA_DIR / "train-00001-of-00002.parquet",
]

TEST_FILE = DATA_DIR / "test-00000-of-00001.parquet"

N_TRAIN = 25000
N_TEST = 8000

RANDOM_STATE = 42

N_RANDOM_CONTROLS = 5

OUTPUT_CSV = RESULTADOS_DIR / f"{PREFIJO}control_leakage_estricto_resultados.csv"


# ============================================================================
# UTILIDADES
# ============================================================================

def cabecera(texto):
    print()
    print("=" * 70)
    print(texto)
    print("=" * 70)


def cargar_datos():
    cabecera("CARGA DE DATOS")

    train_parts = []

    for ruta in TRAIN_FILES:
        print(f"\nCargando TRAIN:")
        print(f"  {ruta}")

        if not ruta.exists():
            raise FileNotFoundError(ruta)

        df = pd.read_parquet(ruta)

        print(f"Objetos: {len(df)}")

        train_parts.append(df)

    train = pd.concat(
        train_parts,
        ignore_index=True
    )

    print(f"\nCargando TEST:")
    print(f"  {TEST_FILE}")

    if not TEST_FILE.exists():
        raise FileNotFoundError(TEST_FILE)

    test = pd.read_parquet(TEST_FILE)

    print(f"Objetos: {len(test)}")

    # Muestreo reproducible
    if len(train) > N_TRAIN:
        train = train.sample(
            n=N_TRAIN,
            random_state=RANDOM_STATE
        ).reset_index(drop=True)

    if len(test) > N_TEST:
        test = test.sample(
            n=N_TEST,
            random_state=RANDOM_STATE
        ).reset_index(drop=True)

    print(f"\nTRAIN utilizado: {len(train):,}")
    print(f"TEST utilizado : {len(test):,}")

    return train, test


# ============================================================================
# HASH DE CURVAS
# ============================================================================

def hash_array(arr):
    """
    Hash estable de una secuencia numérica.
    """

    if arr is None:
        return None

    try:
        a = np.asarray(arr, dtype=np.float64)

        if a.size == 0:
            return "EMPTY"

        # Normalizamos representación para evitar pequeñas diferencias
        a = np.round(a, 10)

        return hashlib.sha1(
            a.tobytes()
        ).hexdigest()

    except Exception:
        return None


def hash_curva(row):
    """
    Construye un hash de las curvas disponibles g/r/i.

    Incluye:
        target
        error
        tiempos

    De esta manera dos curvas idénticas producen el mismo hash.
    """

    partes = []

    bands_data = row.get("bands_data", None)

    if bands_data is None:
        return None

    for banda in ("g", "r", "i"):

        try:
            datos = bands_data.get(banda, None)
        except Exception:
            datos = None

        if datos is None:
            partes.append(f"{banda}:NONE")
            continue

        try:
            target = datos.get("target", [])
            error = datos.get("past_feat_dynamic_real", [])
            mjd = datos.get("mjd", [])

            partes.append(
                f"{banda}:"
                f"{hash_array(target)}:"
                f"{hash_array(error)}:"
                f"{hash_array(mjd)}"
            )

        except Exception:
            partes.append(f"{banda}:ERROR")

    texto = "|".join(partes)

    return hashlib.sha1(
        texto.encode("utf-8")
    ).hexdigest()


def calcular_hashes(df, nombre):
    """
    Calcula hashes de las curvas.
    """

    hashes = []

    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):

        if i % 5000 == 0:
            print(f"  {nombre} {i:,}/{total:,}")

        hashes.append(hash_curva(row))

    return hashes


# ============================================================================
# DETECCIÓN DE LEAKAGE
# ============================================================================

def detectar_leakage(train, test):

    cabecera("DETECCIÓN DE LEAKAGE")

    # ------------------------------------------------------------------
    # SOURCEID
    # ------------------------------------------------------------------

    print("\nCONTROL SOURCEID")

    train_ids = set(
        train["sourceid"].astype(str)
    )

    test_ids = set(
        test["sourceid"].astype(str)
    )

    overlap_ids = train_ids.intersection(test_ids)

    print(f"SourceID TRAIN : {len(train_ids):,}")
    print(f"SourceID TEST  : {len(test_ids):,}")
    print(f"SourceID comunes: {len(overlap_ids):,}")

    if overlap_ids:
        print("\nEjemplos:")
        for x in list(overlap_ids)[:10]:
            print(f"  {x}")

    # ------------------------------------------------------------------
    # HASH CURVAS
    # ------------------------------------------------------------------

    print("\nCalculando hashes de TRAIN...")

    train_hashes = calcular_hashes(
        train,
        "TRAIN"
    )

    train_hash_set = set(
        h for h in train_hashes
        if h is not None
    )

    print(f"\nHashes TRAIN: {len(train_hash_set):,}")

    print("\nCalculando hashes de TEST...")

    test_hashes = calcular_hashes(
        test,
        "TEST "
    )

    test_hash_set = set(
        h for h in test_hashes
        if h is not None
    )

    print(f"\nHashes TEST: {len(test_hash_set):,}")

    overlap_hashes = (
        train_hash_set.intersection(
            test_hash_set
        )
    )

    print(
        f"\nCurvas idénticas TRAIN <-> TEST: "
        f"{len(overlap_hashes):,}"
    )

    return overlap_ids, overlap_hashes, test_hashes


# ============================================================================
# ELIMINACIÓN DE LEAKAGE
# ============================================================================

def eliminar_leakage(
    train,
    test,
    overlap_ids,
    overlap_hashes,
    test_hashes
):

    cabecera("ELIMINACIÓN DE LEAKAGE")

    mask = np.ones(
        len(test),
        dtype=bool
    )

    # SOURCEID compartido
    if overlap_ids:

        ids = test["sourceid"].astype(str)

        mask &= ~ids.isin(
            overlap_ids
        ).to_numpy()

    # Curvas idénticas
    if overlap_hashes:

        for i, h in enumerate(test_hashes):

            if h in overlap_hashes:
                mask[i] = False

    test_clean = test.loc[
        mask
    ].reset_index(drop=True)

    eliminados = len(test) - len(test_clean)

    print(f"TEST original : {len(test):,}")
    print(f"Eliminados    : {eliminados:,}")
    print(f"TEST limpio   : {len(test_clean):,}")

    return test_clean


# ============================================================================
# EXTRACCIÓN DE FEATURES
# ============================================================================

def estadisticas_curva(target, error, mjd):

    target = np.asarray(
        target,
        dtype=float
    )

    error = np.asarray(
        error,
        dtype=float
    )

    mjd = np.asarray(
        mjd,
        dtype=float
    )

    if len(target) == 0:
        return [
            np.nan
        ] * 16

    mask = np.isfinite(target)

    target = target[mask]

    if len(error) == len(mask):
        error = error[mask]
    else:
        error = np.zeros(len(target))

    if len(mjd) == len(mask):
        mjd = mjd[mask]
    else:
        mjd = np.arange(len(target))

    if len(target) == 0:
        return [
            np.nan
        ] * 16

    mean = np.mean(target)
    std = np.std(target)
    median = np.median(target)

    minimum = np.min(target)
    maximum = np.max(target)

    amplitude = maximum - minimum

    q05, q25, q50, q75, q95 = np.percentile(
        target,
        [5, 25, 50, 75, 95]
    )

    iqr = q75 - q25

    mad = np.median(
        np.abs(target - median)
    )

    # Pendiente
    slope = 0.0

    if len(target) >= 2:

        try:
            slope = np.polyfit(
                mjd,
                target,
                1
            )[0]
        except Exception:
            slope = 0.0

    # Número de observaciones
    n = len(target)

    # Rango temporal
    time_span = (
        np.max(mjd) - np.min(mjd)
        if len(mjd) > 1
        else 0.0
    )

    # Error medio
    mean_error = (
        np.mean(error)
        if len(error)
        else 0.0
    )

    # S/N aproximado
    snr = (
        std / mean_error
        if mean_error > 0
        else 0.0
    )

    # Features básicas
    return [
        mean,
        std,
        median,
        amplitude,
        q05,
        q25,
        q75,
        q95,
        iqr,
        mad,
        slope,
        time_span,
        mean_error,
        snr,
        float(n),
        float(np.std(np.diff(target)))
        if len(target) > 1
        else 0.0
    ]


def extraer_features_fila(row):

    features = []

    bands_data = row.get(
        "bands_data",
        None
    )

    for banda in ("g", "r", "i"):

        if bands_data is None:
            features.extend(
                [np.nan] * 16
            )
            continue

        try:
            datos = bands_data.get(
                banda,
                None
            )
        except Exception:
            datos = None

        if datos is None:
            features.extend(
                [np.nan] * 16
            )
            continue

        try:

            target = datos.get(
                "target",
                []
            )

            error = datos.get(
                "past_feat_dynamic_real",
                []
            )

            mjd = datos.get(
                "mjd",
                []
            )

            features.extend(
                estadisticas_curva(
                    target,
                    error,
                    mjd
                )
            )

        except Exception:

            features.extend(
                [np.nan] * 16
            )

    # Periodo
    try:
        periodo = float(
            row["period"]
        )
    except Exception:
        periodo = np.nan

    features.append(periodo)

    return features


def extraer_features(df, nombre):

    print(f"\n{nombre}")

    X = []

    total = len(df)

    for i, (_, row) in enumerate(
        df.iterrows()
    ):

        if i % 5000 == 0:
            print(
                f"  {i:,}/{total:,}"
            )

        X.append(
            extraer_features_fila(
                row
            )
        )

    X = np.asarray(
        X,
        dtype=float
    )

    return X


# ============================================================================
# LIMPIEZA
# ============================================================================

def limpiar_features(X_train, X_test):

    cabecera("LIMPIEZA")

    print(
        f"NaN TRAIN antes: "
        f"{np.isnan(X_train).sum():,}"
    )

    print(
        f"NaN TEST antes : "
        f"{np.isnan(X_test).sum():,}"
    )

    # Mediana calculada SOLO sobre TRAIN
    medianas = np.nanmedian(
        X_train,
        axis=0
    )

    medianas = np.where(
        np.isfinite(medianas),
        medianas,
        0.0
    )

    X_train = np.where(
        np.isfinite(X_train),
        X_train,
        medianas
    )

    X_test = np.where(
        np.isfinite(X_test),
        X_test,
        medianas
    )

    print(
        f"NaN TRAIN: "
        f"{np.isnan(X_train).sum():,}"
    )

    print(
        f"NaN TEST : "
        f"{np.isnan(X_test).sum():,}"
    )

    return X_train, X_test


# ============================================================================
# CLASIFICADORES
# ============================================================================

def ejecutar_rf(
    X_train,
    y_train,
    X_test,
    y_test,
    nombre
):

    inicio = time.time()

    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight=None
    )

    clf.fit(
        X_train,
        y_train
    )

    pred = clf.predict(
        X_test
    )

    acc = accuracy_score(
        y_test,
        pred
    )

    balanced = balanced_accuracy_score(
        y_test,
        pred
    )

    tiempo = time.time() - inicio

    print(
        f"Random Forest : "
        f"accuracy={acc:.6f} "
        f"balanced={balanced:.6f} "
        f"({tiempo:.2f} s)"
    )

    return acc, balanced


def ejecutar_log(
    X_train,
    y_train,
    X_test,
    y_test,
    nombre
):

    inicio = time.time()

    clf = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "logistic",
            LogisticRegression(
                max_iter=3000,
                solver="lbfgs",
                random_state=RANDOM_STATE
            )
        )
    ])

    clf.fit(
        X_train,
        y_train
    )

    pred = clf.predict(
        X_test
    )

    acc = accuracy_score(
        y_test,
        pred
    )

    balanced = balanced_accuracy_score(
        y_test,
        pred
    )

    tiempo = time.time() - inicio

    print(
        f"Logística     : "
        f"accuracy={acc:.6f} "
        f"balanced={balanced:.6f} "
        f"({tiempo:.2f} s)"
    )

    return acc, balanced


# ============================================================================
# CONTROL DE ETIQUETAS ALEATORIAS
# ============================================================================

def control_etiquetas_aleatorias(
    X_train,
    y_train,
    X_test,
    y_test
):

    cabecera(
        "CONTROL - ETIQUETAS ALEATORIAS"
    )

    print(
        f"TEST utilizado: "
        f"{len(y_test):,}"
    )

    print(
        f"Repeticiones: "
        f"{N_RANDOM_CONTROLS}"
    )

    print(
        "\nIMPORTANTE:"
    )

    print(
        "Las etiquetas del TRAIN se barajan "
        "manteniendo exactamente su distribución."
    )

    print(
        "El TEST conserva sus etiquetas reales."
    )

    print(
        "Por tanto, TRAIN y TEST quedan "
        "estadísticamente desacoplados."
    )

    resultados = []

    clases = np.unique(
        y_train
    )

    for repeticion in range(
        N_RANDOM_CONTROLS
    ):

        print()
        print(
            f"CONTROL {repeticion + 1}/"
            f"{N_RANDOM_CONTROLS}"
        )

        rng = np.random.RandomState(
            RANDOM_STATE + repeticion
        )

        y_random = np.array(
            y_train,
            copy=True
        )

        rng.shuffle(
            y_random
        )

        # --------------------------------------------------------------
        # RF
        # --------------------------------------------------------------

        clf_rf = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE + repeticion
        )

        clf_rf.fit(
            X_train,
            y_random
        )

        pred_rf = clf_rf.predict(
            X_test
        )

        acc_rf = accuracy_score(
            y_test,
            pred_rf
        )

        bal_rf = balanced_accuracy_score(
            y_test,
            pred_rf
        )

        # --------------------------------------------------------------
        # LOGÍSTICA
        # --------------------------------------------------------------

        clf_log = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "logistic",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE + repeticion
                )
            )
        ])

        clf_log.fit(
            X_train,
            y_random
        )

        pred_log = clf_log.predict(
            X_test
        )

        acc_log = accuracy_score(
            y_test,
            pred_log
        )

        bal_log = balanced_accuracy_score(
            y_test,
            pred_log
        )

        print(
            f"RF  : accuracy={acc_rf:.6f} "
            f"balanced={bal_rf:.6f}"
        )

        print(
            f"LOG : accuracy={acc_log:.6f} "
            f"balanced={bal_log:.6f}"
        )

        resultados.append({
            "repeticion": repeticion + 1,
            "RF_accuracy": acc_rf,
            "RF_balanced": bal_rf,
            "LOG_accuracy": acc_log,
            "LOG_balanced": bal_log
        })

    df = pd.DataFrame(
        resultados
    )

    print()
    print(
        "------------------------------------------------------------------"
    )

    print(
        f"RF  accuracy medio   : "
        f"{df['RF_accuracy'].mean():.6f} "
        f"+/- "
        f"{df['RF_accuracy'].std():.6f}"
    )

    print(
        f"RF  balanced medio   : "
        f"{df['RF_balanced'].mean():.6f} "
        f"+/- "
        f"{df['RF_balanced'].std():.6f}"
    )

    print(
        f"LOG accuracy medio   : "
        f"{df['LOG_accuracy'].mean():.6f} "
        f"+/- "
        f"{df['LOG_accuracy'].std():.6f}"
    )

    print(
        f"LOG balanced medio   : "
        f"{df['LOG_balanced'].mean():.6f} "
        f"+/- "
        f"{df['LOG_balanced'].std():.6f}"
    )

    print(
        "------------------------------------------------------------------"
    )

    print(
        f"\nReferencia accuracy aleatoria "
        f"(aprox.): {1 / len(clases):.6f}"
    )

    print(
        "Referencia balanced accuracy "
        f"(ideal): {1 / len(clases):.6f}"
    )

    return df


# ============================================================================
# MAIN
# ============================================================================

def main():

    inicio_total = time.time()

    cabecera(
        "CONTROL DE LEAKAGE ESTRICTO - StarEmbed / ZTF"
    )

    print(
        """
Objetivo:
Determinar si la capacidad de clasificación permanece
después de eliminar explícitamente las fuentes de leakage
detectadas en el experimento 12.

Controles:
  1. SOURCEID compartido
  2. Curvas idénticas TRAIN <-> TEST
  3. Etiquetas aleatorias
  4. Balanced Accuracy
  5. Repetición del control aleatorio
"""
    )

    print(
        f"Random State: {RANDOM_STATE}"
    )

    # ==================================================================
    # CARGA
    # ==================================================================

    train, test = cargar_datos()

    # ==================================================================
    # DISTRIBUCIÓN
    # ==================================================================

    cabecera(
        "DISTRIBUCIÓN DE CLASES"
    )

    print("\nTRAIN:")
    print(
        train["class_str"].value_counts()
    )

    print("\nTEST:")
    print(
        test["class_str"].value_counts()
    )

    # ==================================================================
    # LEAKAGE
    # ==================================================================

    (
        overlap_ids,
        overlap_hashes,
        test_hashes
    ) = detectar_leakage(
        train,
        test
    )

    # ==================================================================
    # TEST LIMPIO
    # ==================================================================

    test_clean = eliminar_leakage(
        train,
        test,
        overlap_ids,
        overlap_hashes,
        test_hashes
    )

    # ==================================================================
    # FEATURES
    # ==================================================================

    cabecera(
        "EXTRACCIÓN DE FEATURES"
    )

    X_train = extraer_features(
        train,
        "TRAIN"
    )

    X_test_clean = extraer_features(
        test_clean,
        "TEST LIMPIO"
    )

    X_train, X_test_clean = limpiar_features(
        X_train,
        X_test_clean
    )

    y_train = train[
        "class_str"
    ].astype(str).to_numpy()

    y_test_clean = test_clean[
        "class_str"
    ].astype(str).to_numpy()

    print(
        f"\nTRAIN shape: "
        f"{X_train.shape}"
    )

    print(
        f"TEST shape : "
        f"{X_test_clean.shape}"
    )

    # ==================================================================
    # EXPERIMENTO A
    # ==================================================================

    cabecera(
        "EXPERIMENTO A - TEST LIMPIO"
    )

    rf_clean = ejecutar_rf(
        X_train,
        y_train,
        X_test_clean,
        y_test_clean,
        "TEST LIMPIO"
    )

    log_clean = ejecutar_log(
        X_train,
        y_train,
        X_test_clean,
        y_test_clean,
        "TEST LIMPIO"
    )

    # ==================================================================
    # EXPERIMENTO B
    # ==================================================================

    cabecera(
        "EXPERIMENTO B - TEST ORIGINAL"
    )

    X_test_original = extraer_features(
        test,
        "TEST ORIGINAL"
    )

    X_train_original, X_test_original = limpiar_features(
        X_train,
        X_test_original
    )

    y_test_original = test[
        "class_str"
    ].astype(str).to_numpy()

    rf_original = ejecutar_rf(
        X_train,
        y_train,
        X_test_original,
        y_test_original,
        "TEST ORIGINAL"
    )

    log_original = ejecutar_log(
        X_train,
        y_train,
        X_test_original,
        y_test_original,
        "TEST ORIGINAL"
    )

    # ==================================================================
    # CONTROL ALEATORIO
    # ==================================================================

    random_df = control_etiquetas_aleatorias(
        X_train,
        y_train,
        X_test_clean,
        y_test_clean
    )

    # ==================================================================
    # RESUMEN
    # ==================================================================

    cabecera(
        "RESUMEN FINAL"
    )

    print(
        f"{'EXPERIMENTO':35s}"
        f"{'RF ACC':>12s}"
        f"{'RF BAL':>12s}"
        f"{'LOG ACC':>12s}"
        f"{'LOG BAL':>12s}"
    )

    print("-" * 83)

    print(
        f"{'TEST LIMPIO':35s}"
        f"{rf_clean[0]:12.6f}"
        f"{rf_clean[1]:12.6f}"
        f"{log_clean[0]:12.6f}"
        f"{log_clean[1]:12.6f}"
    )

    print(
        f"{'TEST ORIGINAL':35s}"
        f"{rf_original[0]:12.6f}"
        f"{rf_original[1]:12.6f}"
        f"{log_original[0]:12.6f}"
        f"{log_original[1]:12.6f}"
    )

    print(
        f"{'ETIQUETAS ALEATORIAS (media)':35s}"
        f"{random_df['RF_accuracy'].mean():12.6f}"
        f"{random_df['RF_balanced'].mean():12.6f}"
        f"{random_df['LOG_accuracy'].mean():12.6f}"
        f"{random_df['LOG_balanced'].mean():12.6f}"
    )

    # ==================================================================
    # DIAGNÓSTICO
    # ==================================================================

    cabecera(
        "DIAGNÓSTICO"
    )

    print(
        f"SourceID solapados : "
        f"{len(overlap_ids):,}"
    )

    print(
        f"Curvas idénticas   : "
        f"{len(overlap_hashes):,}"
    )

    print(
        f"TEST original      : "
        f"{len(test):,}"
    )

    print(
        f"TEST limpio        : "
        f"{len(test_clean):,}"
    )

    diferencia_rf = (
        rf_original[0]
        - rf_clean[0]
    )

    diferencia_log = (
        log_original[0]
        - log_clean[0]
    )

    print(
        f"\nDiferencia RF "
        f"(original - limpio): "
        f"{diferencia_rf:+.6f}"
    )

    print(
        f"Diferencia LOG "
        f"(original - limpio): "
        f"{diferencia_log:+.6f}"
    )

    random_bal_rf = (
        random_df["RF_balanced"].mean()
    )

    random_bal_log = (
        random_df["LOG_balanced"].mean()
    )

    print(
        "\nControl de etiquetas aleatorias:"
    )

    print(
        f"RF balanced medio : "
        f"{random_bal_rf:.6f}"
    )

    print(
        f"LOG balanced medio: "
        f"{random_bal_log:.6f}"
    )

    print(
        "\nInterpretación:"
    )

    if random_bal_rf < 0.25:

        print(
            "✓ El RF cae hacia el nivel esperado "
            "cuando se destruyen las etiquetas."
        )

    else:

        print(
            "⚠ El RF conserva capacidad con "
            "etiquetas aleatorias."
        )

    if random_bal_log < 0.25:

        print(
            "✓ La logística cae hacia el nivel "
            "esperado con etiquetas aleatorias."
        )

    else:

        print(
            "⚠ La logística conserva capacidad "
            "con etiquetas aleatorias."
        )

    if abs(diferencia_rf) < 0.01:

        print(
            "\n✓ Eliminar las curvas duplicadas "
            "apenas modifica el resultado RF."
        )

    else:

        print(
            "\n⚠ La eliminación del leakage "
            "modifica apreciablemente el resultado RF."
        )

    # ==================================================================
    # GUARDAR CSV
    # ==================================================================

    filas = []

    filas.append({
        "experimento": "TEST_LIMPIO",
        "RF_accuracy": rf_clean[0],
        "RF_balanced": rf_clean[1],
        "LOG_accuracy": log_clean[0],
        "LOG_balanced": log_clean[1],
        "objetos_test": len(test_clean)
    })

    filas.append({
        "experimento": "TEST_ORIGINAL",
        "RF_accuracy": rf_original[0],
        "RF_balanced": rf_original[1],
        "LOG_accuracy": log_original[0],
        "LOG_balanced": log_original[1],
        "objetos_test": len(test)
    })

    filas.append({
        "experimento": "ETIQUETAS_ALEATORIAS_MEDIA",
        "RF_accuracy": random_df["RF_accuracy"].mean(),
        "RF_balanced": random_df["RF_balanced"].mean(),
        "LOG_accuracy": random_df["LOG_accuracy"].mean(),
        "LOG_balanced": random_df["LOG_balanced"].mean(),
        "objetos_test": len(test_clean)
    })

    filas.append({
        "experimento": "SOURCEID_OVERLAP",
        "RF_accuracy": float(len(overlap_ids)),
        "RF_balanced": np.nan,
        "LOG_accuracy": np.nan,
        "LOG_balanced": np.nan,
        "objetos_test": len(test)
    })

    filas.append({
        "experimento": "CURVE_OVERLAP",
        "RF_accuracy": float(len(overlap_hashes)),
        "RF_balanced": np.nan,
        "LOG_accuracy": np.nan,
        "LOG_balanced": np.nan,
        "objetos_test": len(test)
    })

    # Resultados individuales del control
    for _, row in random_df.iterrows():

        filas.append({
            "experimento":
                f"RANDOM_{int(row['repeticion'])}",
            "RF_accuracy":
                row["RF_accuracy"],
            "RF_balanced":
                row["RF_balanced"],
            "LOG_accuracy":
                row["LOG_accuracy"],
            "LOG_balanced":
                row["LOG_balanced"],
            "objetos_test":
                len(test_clean)
        })

    resultados = pd.DataFrame(
        filas
    )

    resultados.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nResultados guardados en:"
    )

    print(
        OUTPUT_CSV
    )

    # ==================================================================
    # TIEMPO
    # ==================================================================

    tiempo_total = (
        time.time()
        - inicio_total
    )

    print()
    print(
        "=" * 70
    )

    print(
        f"Tiempo total: "
        f"{tiempo_total:.2f} segundos"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()