# -*- coding: utf-8 -*-

"""
======================================================================
15 - CONTROL DE ESTADISTICAS - StarEmbed / ZTF
======================================================================

Objetivo:
Determinar qué familias de estadísticas contienen la información
discriminante entre las clases de StarEmbed.

Separamos:

  A) NIVEL FOTOMETRICO
  B) DISPERSION
  C) ASIMETRIA / FORMA ESTADISTICA
  D) VARIABILIDAD
  E) TODAS LAS ESTADISTICAS

Además:
  - Random Forest
  - Regresión Logística
  - Accuracy
  - Balanced Accuracy
  - Control de etiquetas aleatorias
  - Normalización por objeto de la curva

IMPORTANTE:
No se utilizan:
  - period
  - RA / DEC
  - sourceid
  - información explícita de observación
  - morfología normalizada

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
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score

warnings.filterwarnings("ignore")


# =====================================================================
# CONFIGURACION
# =====================================================================

RANDOM_STATE = 42

# Ruta relativa desde la ubicación del script
BASE_DIR = Path(__file__).resolve().parents[3]  # Sube 3 niveles hasta Astronomia/

# Directorio de resultados dentro de la misma carpeta que el script
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "15_"

DATA_DIR = BASE_DIR / "dataset" / "ZTF_40k_StarEmbed" / "data"

TRAIN_FILES = [
    DATA_DIR / "train-00000-of-00002.parquet",
    DATA_DIR / "train-00001-of-00002.parquet",
]

TEST_FILE = DATA_DIR / "test-00000-of-00001.parquet"

MAX_TRAIN = 25000
MAX_TEST = 8000

OUTPUT_CSV = RESULTADOS_DIR / f"{PREFIJO}control_estadisticas_resultados.csv"


# =====================================================================
# UTILIDADES
# =====================================================================

def separador():
    print("-" * 70)


def cargar_datos():
    print("\n" + "=" * 70)
    print("CARGA DE DATOS")
    print("=" * 70)

    trains = []

    for fichero in TRAIN_FILES:
        print("\nCargando TRAIN:")
        print(" ", fichero)

        if not fichero.exists():
            raise FileNotFoundError(fichero)

        df = pd.read_parquet(fichero)

        print("Objetos:", f"{len(df):,}")

        trains.append(df)

    train = pd.concat(trains, ignore_index=True)

    print("\nCargando TEST:")
    print(" ", TEST_FILE)

    if not TEST_FILE.exists():
        raise FileNotFoundError(TEST_FILE)

    test = pd.read_parquet(TEST_FILE)

    # Selección reproducible
    if len(train) > MAX_TRAIN:
        train = train.sample(
            MAX_TRAIN,
            random_state=RANDOM_STATE
        ).reset_index(drop=True)

    if len(test) > MAX_TEST:
        test = test.sample(
            MAX_TEST,
            random_state=RANDOM_STATE
        ).reset_index(drop=True)

    print("\nTRAIN utilizado:", f"{len(train):,}")
    print("TEST utilizado :", f"{len(test):,}")

    print("\n" + "=" * 70)
    print("DISTRIBUCION DE CLASES")
    print("=" * 70)

    print("\nTRAIN:")
    print(train["class_str"].value_counts())

    print("\nTEST:")
    print(test["class_str"].value_counts())

    return train, test


# =====================================================================
# EXTRACCION DE CURVA
# =====================================================================

def obtener_curva(row):
    """
    Combina las bandas g, r, i.

    Devuelve:
        magnitudes
        errores
        tiempos
    """

    magnitudes = []
    errores = []
    tiempos = []

    bands = row["bands_data"]

    if bands is None:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    for banda in ("g", "r", "i"):

        try:
            datos = bands.get(banda)

            if datos is None:
                continue

            target = datos.get("target")
            error = datos.get("past_feat_dynamic_real")
            mjd = datos.get("mjd")

            if target is None:
                continue

            target = np.asarray(target, dtype=float)

            if error is None:
                error = np.full(len(target), np.nan)
            else:
                error = np.asarray(error, dtype=float)

            if mjd is None:
                mjd = np.arange(len(target), dtype=float)
            else:
                mjd = np.asarray(mjd, dtype=float)

            n = min(
                len(target),
                len(error),
                len(mjd)
            )

            target = target[:n]
            error = error[:n]
            mjd = mjd[:n]

            mask = (
                np.isfinite(target)
                & np.isfinite(error)
                & np.isfinite(mjd)
            )

            magnitudes.extend(target[mask])
            errores.extend(error[mask])
            tiempos.extend(mjd[mask])

        except Exception:
            continue

    if len(magnitudes) == 0:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    return (
        np.asarray(magnitudes, dtype=float),
        np.asarray(errores, dtype=float),
        np.asarray(tiempos, dtype=float),
    )


# =====================================================================
# ESTADISTICAS
# =====================================================================

def estadisticas_curva(row):
    """
    Extrae únicamente estadísticas de la curva.

    No utiliza:
        period
        RA
        DEC
        sourceid
    """

    mag, err, t = obtener_curva(row)

    f = {}

    # ---------------------------------------------------------------
    # NIVEL FOTOMETRICO
    # ---------------------------------------------------------------

    if len(mag):

        f["mean"] = np.mean(mag)
        f["median"] = np.median(mag)

        # Media ponderada por incertidumbre
        sigma = np.maximum(err, 1e-6)

        pesos = 1.0 / (sigma ** 2)

        if np.sum(pesos) > 0:
            f["weighted_mean"] = np.sum(
                mag * pesos
            ) / np.sum(pesos)
        else:
            f["weighted_mean"] = np.mean(mag)

        f["min_mag"] = np.min(mag)
        f["max_mag"] = np.max(mag)

    else:
        f["mean"] = np.nan
        f["median"] = np.nan
        f["weighted_mean"] = np.nan
        f["min_mag"] = np.nan
        f["max_mag"] = np.nan

    # ---------------------------------------------------------------
    # DISPERSION
    # ---------------------------------------------------------------

    if len(mag) > 1:

        f["standard_deviation"] = np.std(mag)

        med = np.median(mag)

        f["mad"] = np.median(
            np.abs(mag - med)
        )

        f["q25"] = np.percentile(mag, 25)
        f["q75"] = np.percentile(mag, 75)

        f["iqr"] = f["q75"] - f["q25"]

        f["range"] = np.max(mag) - np.min(mag)

    else:

        f["standard_deviation"] = np.nan
        f["mad"] = np.nan
        f["q25"] = np.nan
        f["q75"] = np.nan
        f["iqr"] = np.nan
        f["range"] = np.nan

    # ---------------------------------------------------------------
    # ASIMETRIA / FORMA
    # ---------------------------------------------------------------

    if len(mag) > 2:

        media = np.mean(mag)
        std = np.std(mag)

        if std > 0:

            z = (mag - media) / std

            f["skew"] = np.mean(z ** 3)
            f["kurtosis"] = np.mean(z ** 4) - 3.0

        else:

            f["skew"] = 0.0
            f["kurtosis"] = 0.0

    else:

        f["skew"] = np.nan
        f["kurtosis"] = np.nan

    # ---------------------------------------------------------------
    # VARIABILIDAD
    # ---------------------------------------------------------------

    if len(mag) > 1:

        dif = np.diff(mag)

        f["mean_abs_change"] = np.mean(
            np.abs(dif)
        )

        f["std_change"] = np.std(dif)

        f["max_change"] = np.max(
            np.abs(dif)
        )

        f["amplitude"] = (
            np.max(mag) - np.min(mag)
        ) / 2.0

        f["percent_amplitude"] = (
            f["amplitude"]
            / max(abs(np.median(mag)), 1e-6)
        )

        # CUSUM simple
        media = np.mean(mag)

        desv = np.std(mag)

        if desv > 0:

            z = (mag - media) / desv

            cusum = np.cumsum(z)

            f["cusum"] = (
                np.max(cusum)
                - np.min(cusum)
            )

        else:

            f["cusum"] = 0.0

    else:

        f["mean_abs_change"] = np.nan
        f["std_change"] = np.nan
        f["max_change"] = np.nan
        f["amplitude"] = np.nan
        f["percent_amplitude"] = np.nan
        f["cusum"] = np.nan

    # ---------------------------------------------------------------
    # INCERTIDUMBRE FOTOMETRICA
    # ---------------------------------------------------------------

    if len(err):

        f["mean_error"] = np.mean(err)
        f["median_error"] = np.median(err)
        f["std_error"] = np.std(err)

    else:

        f["mean_error"] = np.nan
        f["median_error"] = np.nan
        f["std_error"] = np.nan

    # ---------------------------------------------------------------
    # NUMERO DE OBSERVACIONES
    # ---------------------------------------------------------------

    f["n_observations"] = len(mag)

    # ---------------------------------------------------------------
    # CADENCIA BASICA
    # ---------------------------------------------------------------

    if len(t) > 1:

        dt = np.diff(np.sort(t))

        dt = dt[np.isfinite(dt)]

        if len(dt):

            f["mean_dt"] = np.mean(dt)
            f["median_dt"] = np.median(dt)
            f["std_dt"] = np.std(dt)

        else:

            f["mean_dt"] = np.nan
            f["median_dt"] = np.nan
            f["std_dt"] = np.nan

    else:

        f["mean_dt"] = np.nan
        f["median_dt"] = np.nan
        f["std_dt"] = np.nan

    return f


# =====================================================================
# EXTRACCION DEL DATASET
# =====================================================================

def extraer_features(df):

    registros = []

    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):

        if i % 5000 == 0:
            print(f"  {i:,}/{total:,}")

        registros.append(
            estadisticas_curva(row)
        )

    X = pd.DataFrame(registros)

    y = df.iloc[:len(X)]["class_str"].reset_index(drop=True)

    return X, y


# =====================================================================
# NORMALIZACION POR OBJETO
# =====================================================================

def normalizar_por_objeto(X):

    """
    Elimina explícitamente el nivel fotométrico absoluto.

    Se conserva:
      - dispersión relativa
      - forma estadística
      - variabilidad relativa

    """

    X = X.copy()

    # Estas variables contienen escala absoluta
    absolutos = [
        "mean",
        "median",
        "weighted_mean",
        "min_mag",
        "max_mag",
        "q25",
        "q75",
        "mean_error",
        "median_error",
    ]

    for c in absolutos:
        if c in X.columns:
            X[c] = np.nan

    # Normalizaciones relativas
    if (
        "standard_deviation" in X.columns
        and "amplitude" in X.columns
    ):
        X["std_relative"] = (
            X["standard_deviation"]
            / X["amplitude"].abs().clip(lower=1e-6)
        )

    if (
        "iqr" in X.columns
        and "amplitude" in X.columns
    ):
        X["iqr_relative"] = (
            X["iqr"]
            / X["amplitude"].abs().clip(lower=1e-6)
        )

    if (
        "mean_abs_change" in X.columns
        and "amplitude" in X.columns
    ):
        X["change_relative"] = (
            X["mean_abs_change"]
            / X["amplitude"].abs().clip(lower=1e-6)
        )

    if (
        "std_error" in X.columns
        and "standard_deviation" in X.columns
    ):
        X["error_relative"] = (
            X["std_error"]
            / X["standard_deviation"].abs().clip(lower=1e-6)
        )

    return X


# =====================================================================
# LIMPIEZA
# =====================================================================

def limpiar(X_train, X_test):

    print("\nLIMPIEZA")

    print(
        "NaN TRAIN antes:",
        int(X_train.isna().sum().sum())
    )

    print(
        "NaN TEST antes :",
        int(X_test.isna().sum().sum())
    )

    # Inf -> NaN
    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Mediana calculada SOLO en TRAIN
    medianas = X_train.median()

    X_train = X_train.fillna(medianas)
    X_test = X_test.fillna(medianas)

    # Columnas que siguen completamente vacías
    columnas_validas = [
        c for c in X_train.columns
        if np.isfinite(
            X_train[c].to_numpy()
        ).all()
    ]

    X_train = X_train[columnas_validas]
    X_test = X_test[columnas_validas]

    print(
        "NaN TRAIN:",
        int(X_train.isna().sum().sum())
    )

    print(
        "NaN TEST :",
        int(X_test.isna().sum().sum())
    )

    return X_train, X_test


# =====================================================================
# CLASIFICACION
# =====================================================================

def clasificar(X_train, y_train, X_test, y_test, nombre):

    print("\n")
    separador()
    print(nombre)
    separador()

    # ---------------------------------------------------------------
    # RANDOM FOREST
    # ---------------------------------------------------------------

    t0 = time.time()

    rf = RandomForestClassifier(
        n_estimators=250,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
        max_features="sqrt"
    )

    rf.fit(X_train, y_train)

    pred_rf = rf.predict(X_test)

    rf_acc = accuracy_score(
        y_test,
        pred_rf
    )

    rf_bal = balanced_accuracy_score(
        y_test,
        pred_rf
    )

    tiempo_rf = time.time() - t0

    print(
        f"Random Forest : "
        f"accuracy={rf_acc:.6f} "
        f"balanced={rf_bal:.6f} "
        f"({tiempo_rf:.2f} s)"
    )

    # ---------------------------------------------------------------
    # LOGISTICA
    # ---------------------------------------------------------------

    t0 = time.time()

    log = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        )
    )

    log.fit(X_train, y_train)

    pred_log = log.predict(X_test)

    log_acc = accuracy_score(
        y_test,
        pred_log
    )

    log_bal = balanced_accuracy_score(
        y_test,
        pred_log
    )

    tiempo_log = time.time() - t0

    print(
        f"Logística     : "
        f"accuracy={log_acc:.6f} "
        f"balanced={log_bal:.6f} "
        f"({tiempo_log:.2f} s)"
    )

    return {
        "RF_ACC": rf_acc,
        "RF_BAL": rf_bal,
        "LOG_ACC": log_acc,
        "LOG_BAL": log_bal
    }


# =====================================================================
# CONTROL ETIQUETAS ALEATORIAS
# =====================================================================

def control_aleatorio(
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n")
    print("=" * 70)
    print("CONTROL DE ETIQUETAS ALEATORIAS")
    print("=" * 70)

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    y_random = y_train.to_numpy().copy()

    rng.shuffle(y_random)

    return clasificar(
        X_train,
        y_random,
        X_test,
        y_test,
        "ETIQUETAS ALEATORIAS"
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    inicio = time.time()

    print("=" * 70)
    print("CONTROL DE ESTADISTICAS - StarEmbed / ZTF")
    print("=" * 70)

    print("""
Objetivo:
Determinar qué familias de estadísticas contienen
la señal discriminante entre las clases.

Se eliminan explícitamente:
  - periodo
  - RA
  - DEC
  - sourceid
  - morfología de la curva
""")

    print("Random State:", RANDOM_STATE)

    # ---------------------------------------------------------------
    # CARGA
    # ---------------------------------------------------------------

    train, test = cargar_datos()

    # ---------------------------------------------------------------
    # EXTRACCION
    # ---------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("EXTRACCION DE ESTADISTICAS")
    print("=" * 70)

    print("\nTRAIN")

    X_train_full, y_train = extraer_features(
        train
    )

    print("\nTEST")

    X_test_full, y_test = extraer_features(
        test
    )

    print(
        "\nTRAIN shape:",
        X_train_full.shape
    )

    print(
        "TEST shape :",
        X_test_full.shape
    )

    # ---------------------------------------------------------------
    # DEFINICION DE GRUPOS
    # ---------------------------------------------------------------

    grupos = {

        "NIVEL FOTOMETRICO": [
            "mean",
            "median",
            "weighted_mean",
            "min_mag",
            "max_mag",
        ],

        "DISPERSION": [
            "standard_deviation",
            "mad",
            "q25",
            "q75",
            "iqr",
            "range",
        ],

        "ASIMETRIA_FORMA": [
            "skew",
            "kurtosis",
        ],

        "VARIABILIDAD": [
            "mean_abs_change",
            "std_change",
            "max_change",
            "amplitude",
            "percent_amplitude",
            "cusum",
        ],

        "INCERTIDUMBRE": [
            "mean_error",
            "median_error",
            "std_error",
        ],

        "OBSERVACIONES_CADENCIA": [
            "n_observations",
            "mean_dt",
            "median_dt",
            "std_dt",
        ],
    }

    # ---------------------------------------------------------------
    # EXPERIMENTOS
    # ---------------------------------------------------------------

    resultados = []

    for nombre, columnas in grupos.items():

        print("\n")
        print("=" * 70)
        print("EXPERIMENTO:", nombre)
        print("=" * 70)

        columnas = [
            c for c in columnas
            if c in X_train_full.columns
        ]

        print(
            "Features:",
            len(columnas)
        )

        Xtr = X_train_full[columnas].copy()
        Xte = X_test_full[columnas].copy()

        Xtr, Xte = limpiar(
            Xtr,
            Xte
        )

        res = clasificar(
            Xtr,
            y_train,
            Xte,
            y_test,
            nombre
        )

        resultados.append({
            "experimento": nombre,
            "features": len(Xtr.columns),
            "objetos_train": len(Xtr),
            "objetos_test": len(Xte),
            **res
        })

    # ---------------------------------------------------------------
    # TODAS LAS ESTADISTICAS
    # ---------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("EXPERIMENTO: TODAS LAS ESTADISTICAS")
    print("=" * 70)

    Xtr, Xte = limpiar(
        X_train_full.copy(),
        X_test_full.copy()
    )

    res = clasificar(
        Xtr,
        y_train,
        Xte,
        y_test,
        "TODAS_ESTADISTICAS"
    )

    resultados.append({
        "experimento": "TODAS_ESTADISTICAS",
        "features": len(Xtr.columns),
        "objetos_train": len(Xtr),
        "objetos_test": len(Xte),
        **res
    })

    # ---------------------------------------------------------------
    # NORMALIZACION POR OBJETO
    # ---------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("EXPERIMENTO: ESTADISTICAS NORMALIZADAS")
    print("=" * 70)

    Xtr = normalizar_por_objeto(
        X_train_full.copy()
    )

    Xte = normalizar_por_objeto(
        X_test_full.copy()
    )

    Xtr, Xte = limpiar(
        Xtr,
        Xte
    )

    res = clasificar(
        Xtr,
        y_train,
        Xte,
        y_test,
        "ESTADISTICAS_NORMALIZADAS"
    )

    resultados.append({
        "experimento": "ESTADISTICAS_NORMALIZADAS",
        "features": len(Xtr.columns),
        "objetos_train": len(Xtr),
        "objetos_test": len(Xte),
        **res
    })

    # ---------------------------------------------------------------
    # CONTROL ALEATORIO
    # ---------------------------------------------------------------

    res = control_aleatorio(
        Xtr,
        y_train,
        Xte,
        y_test
    )

    resultados.append({
        "experimento": "ETIQUETAS_ALEATORIAS",
        "features": len(Xtr.columns),
        "objetos_train": len(Xtr),
        "objetos_test": len(Xte),
        **res
    })

    # ---------------------------------------------------------------
    # RESUMEN
    # ---------------------------------------------------------------

    resultados_df = pd.DataFrame(
        resultados
    )

    print("\n")
    print("=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)

    print(
        resultados_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # ---------------------------------------------------------------
    # DIAGNOSTICO
    # ---------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("DIAGNOSTICO")
    print("=" * 70)

    aleatorio = resultados_df[
        resultados_df["experimento"]
        == "ETIQUETAS_ALEATORIAS"
    ]

    normalizado = resultados_df[
        resultados_df["experimento"]
        == "ESTADISTICAS_NORMALIZADAS"
    ]

    if len(normalizado) and len(aleatorio):

        bal_norm = normalizado.iloc[0]["LOG_BAL"]
        bal_rand = aleatorio.iloc[0]["LOG_BAL"]

        print(
            f"\nLogística normalizada : {bal_norm:.6f}"
        )

        print(
            f"Logística aleatoria  : {bal_rand:.6f}"
        )

        print(
            f"Diferencia           : "
            f"{bal_norm - bal_rand:.6f}"
        )

        if bal_norm > bal_rand + 0.10:

            print("""
RESULTADO:

La señal discriminante sobrevive en buena medida
después de eliminar el nivel fotométrico absoluto.

Esto justifica investigar propiedades intrínsecas
de variabilidad y forma estadística.
""")

        else:

            print("""
RESULTADO:

La señal cae considerablemente al eliminar
la escala fotométrica absoluta.

La diferencia entre clases puede estar relacionada
con propiedades observacionales o fotométricas.
""")

    # ---------------------------------------------------------------
    # CSV
    # ---------------------------------------------------------------

    resultados_df.to_csv(
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
        f"{time.time() - inicio:.2f} s"
    )

    print("=" * 70)


# =====================================================================
# EJECUCION
# =====================================================================

if __name__ == "__main__":
    main()