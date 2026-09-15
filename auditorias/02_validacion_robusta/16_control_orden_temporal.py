# -*- coding: utf-8 -*-

"""
======================================================================
CONTROL DE ORDEN TEMPORAL - StarEmbed / ZTF
======================================================================

Objetivo:

Determinar si la señal discriminante de las clases depende de la
ESTRUCTURA TEMPORAL de las curvas o puede explicarse principalmente
por sus distribuciones estadísticas.

Experimentos:

1. ESTADISTICAS REALES
   Estadísticas normalizadas calculadas sobre las curvas originales.

2. MAGNITUDES BARajADAS
   Se mantienen los valores de magnitud, pero se destruye su orden
   temporal dentro de cada banda.

3. MAGNITUDES + ERRORES BARajados
   Magnitudes y errores se permutan independientemente.

4. TIEMPOS BARajados
   Se destruye la correspondencia temporal manteniendo las magnitudes.

La comparación fundamental es:

    REAL vs MAGNITUDES_BARAJADAS

Si la capacidad discriminante desaparece al destruir el orden temporal,
la señal depende de la estructura temporal de las curvas.

Si permanece, la señal está principalmente en las distribuciones
marginales de las observaciones.

======================================================================
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

# ======================================================================
# CONFIGURACION
# ======================================================================

RANDOM_STATE = 42

N_TRAIN = 25000
N_TEST = 8000

# Ruta relativa desde la ubicación del script
BASE_DIR = Path(__file__).resolve().parents[2]  # Sube 2 niveles hasta Astronomia/

# Directorio de resultados dentro de la misma carpeta que el script
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "16_"

DATA_DIR = BASE_DIR / "dataset" / "ZTF_40k_StarEmbed" / "data"

TRAIN_1 = DATA_DIR / "train-00000-of-00002.parquet"
TRAIN_2 = DATA_DIR / "train-00001-of-00002.parquet"
TEST_1 = DATA_DIR / "test-00000-of-00001.parquet"

OUTPUT_CSV = RESULTADOS_DIR / f"{PREFIJO}control_orden_temporal_resultados.csv"

BANDS = ["g", "r", "i"]

# ======================================================================
# CARGA
# ======================================================================

def cargar_datos():

    print("=" * 70)
    print("CARGA DE DATOS")
    print("=" * 70)

    print("\nCargando TRAIN:")
    print(f"  {TRAIN_1}")

    if not TRAIN_1.exists():
        raise FileNotFoundError(TRAIN_1)

    df1 = pd.read_parquet(TRAIN_1)

    print(f"Objetos: {len(df1):,}")

    print("\nCargando TRAIN:")
    print(f"  {TRAIN_2}")

    if not TRAIN_2.exists():
        raise FileNotFoundError(TRAIN_2)

    df2 = pd.read_parquet(TRAIN_2)

    print(f"Objetos: {len(df2):,}")

    train = pd.concat([df1, df2], ignore_index=True)

    print("\nCargando TEST:")
    print(f"  {TEST_1}")

    if not TEST_1.exists():
        raise FileNotFoundError(TEST_1)

    test = pd.read_parquet(TEST_1)

    print(f"Objetos: {len(test):,}")

    rng = np.random.RandomState(RANDOM_STATE)

    if len(train) > N_TRAIN:
        idx = rng.choice(len(train), N_TRAIN, replace=False)
        train = train.iloc[idx].reset_index(drop=True)

    if len(test) > N_TEST:
        idx = rng.choice(len(test), N_TEST, replace=False)
        test = test.iloc[idx].reset_index(drop=True)

    print(f"\nTRAIN utilizado: {len(train):,}")
    print(f"TEST utilizado : {len(test):,}")

    return train, test


# ======================================================================
# UTILIDADES
# ======================================================================

def obtener_banda(obj, banda):

    try:
        datos = obj["bands_data"]

        if datos is None:
            return None

        return datos.get(banda, None)

    except Exception:
        return None


def array_seguro(x):

    if x is None:
        return np.array([], dtype=float)

    try:
        a = np.asarray(x, dtype=float)
        return a[np.isfinite(a)]
    except Exception:
        return np.array([], dtype=float)


def estadisticas_basicas(x):

    x = array_seguro(x)

    if len(x) == 0:
        return [
            np.nan, np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan, np.nan, np.nan
        ]

    q10, q25, q50, q75, q90 = np.percentile(
        x,
        [10, 25, 50, 75, 90]
    )

    mean = np.mean(x)
    std = np.std(x)

    if std > 0:
        skew = np.mean(((x - mean) / std) ** 3)
    else:
        skew = 0.0

    mad = np.median(np.abs(x - q50))

    return [
        mean,
        std,
        q10,
        q25,
        q50,
        q75,
        q90,
        q90 - q10,
        skew,
        mad
    ]


# ======================================================================
# TRANSFORMACIONES
# ======================================================================

def barajar_magnitudes(obj, rng):

    """
    Destruye el orden temporal de target.

    Conserva:
      - número de observaciones
      - distribución de magnitudes
      - tiempos
      - errores

    Destruye:
      - correlación temporal
      - periodicidad temporal
      - tendencia temporal
      - estructura de la curva
    """

    nuevo = {}

    try:
        bands = obj["bands_data"]
    except Exception:
        return obj

    if bands is None:
        return obj

    for banda in BANDS:

        b = bands.get(banda, None)

        if b is None:
            continue

        try:
            target = np.asarray(
                b["target"],
                dtype=float
            )

            if len(target) > 1:

                perm = rng.permutation(len(target))

                nuevo_b = dict(b)
                nuevo_b["target"] = target[perm].tolist()

                # mantenemos errores y tiempos intactos
                nuevo[banda] = nuevo_b

        except Exception:
            pass

    return {
        **obj,
        "bands_data": nuevo
    }


def barajar_magnitudes_errores(obj, rng):

    """
    Destruye el orden temporal de magnitudes y errores.
    """

    nuevo = {}

    try:
        bands = obj["bands_data"]
    except Exception:
        return obj

    if bands is None:
        return obj

    for banda in BANDS:

        b = bands.get(banda, None)

        if b is None:
            continue

        try:

            target = np.asarray(
                b["target"],
                dtype=float
            )

            errores = np.asarray(
                b["past_feat_dynamic_real"],
                dtype=float
            )

            nuevo_b = dict(b)

            if len(target) > 1:
                nuevo_b["target"] = target[
                    rng.permutation(len(target))
                ].tolist()

            if len(errores) > 1:
                nuevo_b["past_feat_dynamic_real"] = errores[
                    rng.permutation(len(errores))
                ].tolist()

            nuevo[banda] = nuevo_b

        except Exception:
            pass

    return {
        **obj,
        "bands_data": nuevo
    }


def barajar_tiempos(obj, rng):

    """
    Mantiene las magnitudes pero destruye su relación temporal.
    """

    nuevo = {}

    try:
        bands = obj["bands_data"]
    except Exception:
        return obj

    if bands is None:
        return obj

    for banda in BANDS:

        b = bands.get(banda, None)

        if b is None:
            continue

        try:

            mjd = np.asarray(
                b["mjd"],
                dtype=float
            )

            nuevo_b = dict(b)

            if len(mjd) > 1:
                nuevo_b["mjd"] = mjd[
                    rng.permutation(len(mjd))
                ].tolist()

            nuevo[banda] = nuevo_b

        except Exception:
            pass

    return {
        **obj,
        "bands_data": nuevo
    }


# ======================================================================
# EXTRACCION
# ======================================================================

def extraer_features(df, transformacion=None):

    features = []

    rng = np.random.RandomState(RANDOM_STATE)

    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):

        if i % 5000 == 0:
            print(f"  {i:,}/{total:,}")

        obj = row.to_dict()

        if transformacion is not None:
            obj = transformacion(obj, rng)

        fila = []

        for banda in BANDS:

            b = obtener_banda(obj, banda)

            if b is None:
                fila.extend([np.nan] * 10)
                continue

            try:
                target = b.get("target", [])

                vals = array_seguro(target)

                # --------------------------------------------------
                # NORMALIZACION POR CURVA
                # --------------------------------------------------

                if len(vals) > 0:
                    med = np.median(vals)
                    mad = np.median(
                        np.abs(vals - med)
                    )

                    if mad > 0:
                        vals = (vals - med) / mad
                    else:
                        std = np.std(vals)

                        if std > 0:
                            vals = (vals - med) / std
                        else:
                            vals = vals - med

                fila.extend(
                    estadisticas_basicas(vals)
                )

            except Exception:
                fila.extend([np.nan] * 10)

        features.append(fila)

    columnas = []

    for banda in BANDS:

        columnas.extend([
            f"{banda}_mean",
            f"{banda}_std",
            f"{banda}_p10",
            f"{banda}_p25",
            f"{banda}_median",
            f"{banda}_p75",
            f"{banda}_p90",
            f"{banda}_range_p90_p10",
            f"{banda}_skew",
            f"{banda}_mad"
        ])

    return pd.DataFrame(
        features,
        columns=columnas
    )


# ======================================================================
# LIMPIEZA
# ======================================================================

def limpiar(X_train, X_test):

    print("\nLIMPIEZA")

    nan_train = int(X_train.isna().sum().sum())
    nan_test = int(X_test.isna().sum().sum())

    print(f"NaN TRAIN antes: {nan_train}")
    print(f"NaN TEST antes : {nan_test}")

    medianas = X_train.median()

    X_train = X_train.fillna(medianas)
    X_test = X_test.fillna(medianas)

    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

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
# CLASIFICACION
# ======================================================================

def clasificar(X_train, y_train, X_test, y_test):

    resultados = {}

    print("\n" + "-" * 70)

    # --------------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------------

    t0 = time.time()

    rf = RandomForestClassifier(
        n_estimators=250,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_features="sqrt"
    )

    rf.fit(X_train, y_train)

    pred = rf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    bal = balanced_accuracy_score(y_test, pred)

    tiempo = time.time() - t0

    print(
        f"Random Forest : "
        f"accuracy={acc:.6f} "
        f"balanced={bal:.6f} "
        f"({tiempo:.2f} s)"
    )

    resultados["RF_ACC"] = acc
    resultados["RF_BAL"] = bal

    # --------------------------------------------------------------
    # LOGISTICA
    # --------------------------------------------------------------

    t0 = time.time()

    log = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "logistic",
            LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                solver="lbfgs"
            )
        )
    ])

    log.fit(X_train, y_train)

    pred = log.predict(X_test)

    acc = accuracy_score(y_test, pred)
    bal = balanced_accuracy_score(y_test, pred)

    tiempo = time.time() - t0

    print(
        f"Logística     : "
        f"accuracy={acc:.6f} "
        f"balanced={bal:.6f} "
        f"({tiempo:.2f} s)"
    )

    resultados["LOG_ACC"] = acc
    resultados["LOG_BAL"] = bal

    return resultados


# ======================================================================
# EXPERIMENTO
# ======================================================================

def ejecutar_experimento(
    nombre,
    train,
    test,
    y_train,
    y_test,
    transformacion
):

    print("\n")
    print("=" * 70)
    print(f"EXPERIMENTO: {nombre}")
    print("=" * 70)

    print("\nExtrayendo TRAIN...")

    X_train = extraer_features(
        train,
        transformacion
    )

    print("\nExtrayendo TEST...")

    X_test = extraer_features(
        test,
        transformacion
    )

    print(
        f"\nTRAIN shape: {X_train.shape}"
    )

    print(
        f"TEST shape : {X_test.shape}"
    )

    X_train, X_test = limpiar(
        X_train,
        X_test
    )

    print(
        f"\n{'-' * 70}\n"
        f"{nombre}\n"
        f"{'-' * 70}"
    )

    resultados = clasificar(
        X_train,
        y_train,
        X_test,
        y_test
    )

    resultados["experimento"] = nombre
    resultados["features"] = X_train.shape[1]
    resultados["objetos_train"] = len(X_train)
    resultados["objetos_test"] = len(X_test)

    return resultados


# ======================================================================
# MAIN
# ======================================================================

def main():

    inicio = time.time()

    print("=" * 70)
    print("CONTROL DE ORDEN TEMPORAL - StarEmbed / ZTF")
    print("=" * 70)

    print(
        """
Objetivo:
Determinar si la señal discriminante depende del
ORDEN TEMPORAL de las observaciones o principalmente
de sus distribuciones estadísticas.

Experimentos:

  1. CURVAS REALES
  2. MAGNITUDES BARAJADAS
  3. MAGNITUDES + ERRORES BARAJADOS
  4. TIEMPOS BARAJADOS
"""
    )

    print(
        f"Random State: {RANDOM_STATE}"
    )

    # --------------------------------------------------------------
    # CARGA
    # --------------------------------------------------------------

    train, test = cargar_datos()

    y_train = train["class_str"].astype(str)
    y_test = test["class_str"].astype(str)

    print("\n" + "=" * 70)
    print("DISTRIBUCION DE CLASES")
    print("=" * 70)

    print("\nTRAIN:")
    print(y_train.value_counts())

    print("\nTEST:")
    print(y_test.value_counts())

    resultados = []

    # --------------------------------------------------------------
    # 1. REAL
    # --------------------------------------------------------------

    r = ejecutar_experimento(
        "CURVAS_REALES",
        train,
        test,
        y_train,
        y_test,
        None
    )

    resultados.append(r)

    # --------------------------------------------------------------
    # 2. MAGNITUDES BARAJADAS
    # --------------------------------------------------------------

    r = ejecutar_experimento(
        "MAGNITUDES_BARAJADAS",
        train,
        test,
        y_train,
        y_test,
        barajar_magnitudes
    )

    resultados.append(r)

    # --------------------------------------------------------------
    # 3. MAGNITUDES + ERRORES BARAJADOS
    # --------------------------------------------------------------

    r = ejecutar_experimento(
        "MAGNITUDES_ERRORES_BARAJADOS",
        train,
        test,
        y_train,
        y_test,
        barajar_magnitudes_errores
    )

    resultados.append(r)

    # --------------------------------------------------------------
    # 4. TIEMPOS BARAJADOS
    # --------------------------------------------------------------

    r = ejecutar_experimento(
        "TIEMPOS_BARAJADOS",
        train,
        test,
        y_train,
        y_test,
        barajar_tiempos
    )

    resultados.append(r)

    # --------------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------------

    df_resultados = pd.DataFrame(resultados)

    print("\n")
    print("=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)

    print()

    print(
        df_resultados[
            [
                "experimento",
                "features",
                "objetos_train",
                "objetos_test",
                "RF_ACC",
                "RF_BAL",
                "LOG_ACC",
                "LOG_BAL"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------------
    # DIAGNOSTICO
    # --------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("DIAGNOSTICO")
    print("=" * 70)

    real = df_resultados[
        df_resultados["experimento"]
        == "CURVAS_REALES"
    ].iloc[0]

    barajado = df_resultados[
        df_resultados["experimento"]
        == "MAGNITUDES_BARAJADAS"
    ].iloc[0]

    delta_rf = (
        real["RF_BAL"] -
        barajado["RF_BAL"]
    )

    delta_log = (
        real["LOG_BAL"] -
        barajado["LOG_BAL"]
    )

    print(
        f"""
Balanced accuracy REAL:
  RF  : {real["RF_BAL"]:.6f}
  LOG : {real["LOG_BAL"]:.6f}

Balanced accuracy MAGNITUDES BARAJADAS:
  RF  : {barajado["RF_BAL"]:.6f}
  LOG : {barajado["LOG_BAL"]:.6f}

Caída al destruir el orden temporal:
  RF  : {delta_rf:+.6f}
  LOG : {delta_log:+.6f}
"""
    )

    if delta_rf > 0.15 and delta_log > 0.15:

        print(
            """
RESULTADO:

La capacidad discriminante cae claramente al destruir
el orden temporal.

Esto indica que una parte importante de la señal
está asociada a la estructura temporal de las curvas,
y no únicamente a sus distribuciones estadísticas.
"""
        )

    elif delta_rf < 0.05 and delta_log < 0.05:

        print(
            """
RESULTADO:

La capacidad discriminante apenas cambia al destruir
el orden temporal.

La señal parece estar principalmente contenida
en las distribuciones estadísticas marginales.
"""
        )

    else:

        print(
            """
RESULTADO:

Existe una caída intermedia.

La señal parece contener una combinación de:

  - propiedades estadísticas marginales
  - estructura temporal

Será necesario un experimento adicional para
separar ambas contribuciones.
"""
        )

    # --------------------------------------------------------------
    # CSV
    # --------------------------------------------------------------

    df_resultados.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(
        "\nResultados guardados en:"
    )

    print(
        OUTPUT_CSV
    )

    print(
        f"\nTiempo total: "
        f"{time.time() - inicio:.2f} segundos"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()