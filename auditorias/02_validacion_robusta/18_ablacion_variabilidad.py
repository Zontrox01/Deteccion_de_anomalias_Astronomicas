# -*- coding: utf-8 -*-

"""
18_ablacion_variabilidad.py

StarEmbed / ZTF

Objetivo:
Determinar qué variables del bloque de VARIABILIDAD
aportan realmente capacidad discriminante.

Método:
    1. Modelo completo con TODAS_VARIABLES.
    2. Eliminar una variable de variabilidad cada vez.
    3. Comparar RF y Regresión Logística.
    4. Calcular la pérdida de balanced accuracy.
    5. Repetir con etiquetas aleatorias como control.

IMPORTANTE:
No se utiliza:
    - sourceid
    - RA
    - DEC
    - morfología
    - magnitud media

Se utilizan las variables estadísticas disponibles en
StarEmbed y el periodo.

Random State: 42
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

# Ruta relativa desde la ubicación del script
BASE_DIR = Path(__file__).resolve().parents[2]  # Sube 2 niveles hasta Astronomia/

# Directorio de resultados dentro de la misma carpeta que el script
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "18_"

DATA_DIR = BASE_DIR / "dataset" / "ZTF_40k_StarEmbed" / "data"

TRAIN_FILES = [
    DATA_DIR / "train-00000-of-00002.parquet",
    DATA_DIR / "train-00001-of-00002.parquet",
]

TEST_FILE = DATA_DIR / "test-00000-of-00001.parquet"

MAX_TRAIN = 25000
MAX_TEST = 8000

OUTPUT_CSV = RESULTADOS_DIR / f"{PREFIJO}ablacion_variabilidad_resultados.csv"

# ======================================================================
# VARIABLES
# ======================================================================

# Variables estadísticas disponibles en StarEmbed.
#
# NOTA:
# Estas variables se calculan directamente sobre las curvas.
# No incluyen la magnitud media.

VARIABLES = [
    "mean",
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

# Bloque de variabilidad identificado en el experimento 17.
#
# Se mantiene separado para poder hacer la ablación.

VARIABILIDAD = [
    "standard_deviation",
    "median_absolute_deviation",
    "amplitude",
    "percent_amplitude",
    "inter_percentile_range_25",
    "stetson_K",
    "eta",
    "chi2",
    "maximum_slope",
    "skew",
    "kurtosis",
]

# Periodo
PERIODO = [
    "period"
]

# Variables que NO queremos utilizar.
EXCLUIDAS = {
    "sourceid",
    "ra",
    "dec",
    "mean",
}


# ======================================================================
# UTILIDADES
# ======================================================================

def titulo(texto):
    print()
    print("=" * 70)
    print(texto)
    print("=" * 70)


def limpiar_dataframe(X):
    """
    Limpieza robusta de NaN e infinitos.
    """

    X = X.copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    nan_antes = int(X.isna().sum().sum())

    # Mediana de cada columna
    for col in X.columns:

        if X[col].isna().any():

            mediana = X[col].median()

            if pd.isna(mediana):
                mediana = 0.0

            X[col] = X[col].fillna(mediana)

    nan_despues = int(X.isna().sum().sum())

    return X, nan_antes, nan_despues


# ======================================================================
# CARGA
# ======================================================================

def cargar_datos():

    titulo("CARGA DE DATOS")

    partes_train = []

    total = 0

    for fichero in TRAIN_FILES:

        print()
        print("Cargando TRAIN:")
        print(" ", fichero)

        if not fichero.exists():
            raise FileNotFoundError(fichero)

        df = pd.read_parquet(fichero)

        print("Objetos:", len(df))

        partes_train.append(df)

        total += len(df)

    train = pd.concat(
        partes_train,
        ignore_index=True
    )

    if len(train) > MAX_TRAIN:
        train = train.iloc[:MAX_TRAIN].copy()

    print()
    print("Cargando TEST:")
    print(" ", TEST_FILE)

    if not TEST_FILE.exists():
        raise FileNotFoundError(TEST_FILE)

    test = pd.read_parquet(TEST_FILE)

    if len(test) > MAX_TEST:
        test = test.iloc[:MAX_TEST].copy()

    print()
    print("TRAIN utilizado:", f"{len(train):,}")
    print("TEST utilizado :", f"{len(test):,}")

    return train, test


# ======================================================================
# EXTRACCION DE UNA CURVA
# ======================================================================

def obtener_curva(objeto):
    """
    Combina las bandas disponibles.

    Devuelve magnitudes, errores y tiempos.
    """

    mags = []
    errs = []
    tiempos = []

    bands = objeto.get("bands_data")

    if bands is None:
        return (
            np.array([]),
            np.array([]),
            np.array([])
        )

    for banda in ["g", "r", "i"]:

        datos = bands.get(banda)

        if datos is None:
            continue

        try:
            target = datos["target"]
            error = datos["past_feat_dynamic_real"]
            mjd = datos["mjd"]

        except Exception:
            continue

        if target is None:
            continue

        try:
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

        except Exception:
            continue

        n = min(
            len(target),
            len(error),
            len(mjd)
        )

        if n == 0:
            continue

        mags.extend(target[:n])
        errs.extend(error[:n])
        tiempos.extend(mjd[:n])

    return (
        np.asarray(mags, dtype=float),
        np.asarray(errs, dtype=float),
        np.asarray(tiempos, dtype=float)
    )


# ======================================================================
# ESTADISTICAS
# ======================================================================

def calcular_estadisticas(mags, errs, tiempos):

    resultado = {}

    if len(mags) == 0:

        for v in VARIABLES:
            resultado[v] = np.nan

        return resultado

    # --------------------------------------------------------------
    # NIVEL
    # --------------------------------------------------------------

    resultado["mean"] = np.mean(mags)

    resultado["median"] = np.median(mags)

    # --------------------------------------------------------------
    # DISPERSION
    # --------------------------------------------------------------

    if len(mags) > 1:
        resultado["standard_deviation"] = np.std(
            mags,
            ddof=1
        )
    else:
        resultado["standard_deviation"] = 0.0

    mediana = resultado["median"]

    resultado["median_absolute_deviation"] = np.median(
        np.abs(mags - mediana)
    )

    # --------------------------------------------------------------
    # AMPLITUD
    # --------------------------------------------------------------

    if len(mags) > 0:

        resultado["amplitude"] = (
            np.max(mags) -
            np.min(mags)
        )

        resultado["percent_amplitude"] = (
            resultado["amplitude"] /
            max(abs(mediana), 1e-12)
        )

    else:

        resultado["amplitude"] = 0.0
        resultado["percent_amplitude"] = 0.0

    # --------------------------------------------------------------
    # RANGO PERCENTIL
    # --------------------------------------------------------------

    if len(mags) >= 2:

        p25 = np.percentile(
            mags,
            25
        )

        p75 = np.percentile(
            mags,
            75
        )

        resultado["inter_percentile_range_25"] = (
            p75 - p25
        )

    else:

        resultado["inter_percentile_range_25"] = 0.0

    # --------------------------------------------------------------
    # ASIMETRIA
    # --------------------------------------------------------------

    if len(mags) > 2:

        media = np.mean(mags)
        std = np.std(mags)

        if std > 0:

            resultado["skew"] = np.mean(
                ((mags - media) / std) ** 3
            )

        else:
            resultado["skew"] = 0.0

    else:

        resultado["skew"] = 0.0

    # --------------------------------------------------------------
    # KURTOSIS
    # --------------------------------------------------------------

    if len(mags) > 3:

        media = np.mean(mags)
        std = np.std(mags)

        if std > 0:

            resultado["kurtosis"] = (
                np.mean(
                    ((mags - media) / std) ** 4
                ) - 3.0
            )

        else:
            resultado["kurtosis"] = 0.0

    else:

        resultado["kurtosis"] = 0.0

    # --------------------------------------------------------------
    # STETSON K
    # --------------------------------------------------------------

    if (
        len(mags) > 2 and
        len(errs) == len(mags)
    ):

        sigma = np.asarray(
            errs,
            dtype=float
        )

        sigma[sigma <= 0] = np.nan

        media = np.nanmean(mags)

        delta = (
            np.sqrt(len(mags) / (len(mags) - 1))
            * (mags - media)
            / sigma
        )

        delta = delta[np.isfinite(delta)]

        if len(delta) > 0:

            resultado["stetson_K"] = (
                np.mean(np.abs(delta))
                /
                np.sqrt(
                    np.mean(delta ** 2)
                )
            )

        else:

            resultado["stetson_K"] = 0.0

    else:

        resultado["stetson_K"] = 0.0

    # --------------------------------------------------------------
    # ETA
    # --------------------------------------------------------------

    if len(mags) > 1:

        diferencias = np.diff(mags)

        denominador = np.sum(
            (mags - np.mean(mags)) ** 2
        )

        if denominador > 0:

            resultado["eta"] = (
                np.sum(
                    diferencias ** 2
                )
                /
                denominador
            )

        else:

            resultado["eta"] = 0.0

    else:

        resultado["eta"] = 0.0

    # --------------------------------------------------------------
    # CHI2
    # --------------------------------------------------------------

    if (
        len(mags) > 1 and
        len(errs) == len(mags)
    ):

        sigma = np.asarray(
            errs,
            dtype=float
        )

        sigma[sigma <= 0] = np.nan

        valido = np.isfinite(sigma)

        if np.any(valido):

            resultado["chi2"] = np.mean(
                (
                    (mags[valido] - np.mean(mags[valido]))
                    /
                    sigma[valido]
                ) ** 2
            )

        else:

            resultado["chi2"] = 0.0

    else:

        resultado["chi2"] = 0.0

    # --------------------------------------------------------------
    # MAXIMUM SLOPE
    # --------------------------------------------------------------

    if len(mags) > 1:

        orden = np.argsort(tiempos)

        t = tiempos[orden]
        m = mags[orden]

        dt = np.diff(t)
        dm = np.diff(m)

        valido = dt != 0

        if np.any(valido):

            slopes = np.abs(
                dm[valido] /
                dt[valido]
            )

            slopes = slopes[
                np.isfinite(slopes)
            ]

            if len(slopes):

                resultado["maximum_slope"] = np.max(
                    slopes
                )

            else:

                resultado["maximum_slope"] = 0.0

        else:

            resultado["maximum_slope"] = 0.0

    else:

        resultado["maximum_slope"] = 0.0

    return resultado


# ======================================================================
# EXTRACCION DATAFRAME
# ======================================================================

def extraer_features(df, nombre):

    print()
    print(nombre)

    resultados = []

    n = len(df)

    for i in range(n):

        if i % 5000 == 0:
            print(f"  {i:,}/{n:,}")

        fila = df.iloc[i]

        mags, errs, tiempos = obtener_curva(
            fila
        )

        stats = calcular_estadisticas(
            mags,
            errs,
            tiempos
        )

        # Periodo
        try:
            stats["period"] = float(
                fila["period"]
            )
        except Exception:
            stats["period"] = np.nan

        resultados.append(stats)

    return pd.DataFrame(
        resultados
    )


# ======================================================================
# MODELOS
# ======================================================================

def evaluar(
    X_train,
    y_train,
    X_test,
    y_test,
    nombre
):

    titulo(nombre)

    # --------------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------------

    inicio = time.time()

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=None
    )

    rf.fit(
        X_train,
        y_train
    )

    pred_rf = rf.predict(
        X_test
    )

    rf_time = time.time() - inicio

    rf_acc = accuracy_score(
        y_test,
        pred_rf
    )

    rf_bal = balanced_accuracy_score(
        y_test,
        pred_rf
    )

    # --------------------------------------------------------------
    # LOGISTICA
    # --------------------------------------------------------------

    inicio = time.time()

    log = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
                C=1.0
            )
        )
    ])

    log.fit(
        X_train,
        y_train
    )

    pred_log = log.predict(
        X_test
    )

    log_time = time.time() - inicio

    log_acc = accuracy_score(
        y_test,
        pred_log
    )

    log_bal = balanced_accuracy_score(
        y_test,
        pred_log
    )

    print(
        f"Random Forest : "
        f"accuracy={rf_acc:.6f} "
        f"balanced={rf_bal:.6f} "
        f"({rf_time:.2f} s)"
    )

    print(
        f"Logística     : "
        f"accuracy={log_acc:.6f} "
        f"balanced={log_bal:.6f} "
        f"({log_time:.2f} s)"
    )

    return {
        "RF_ACC": rf_acc,
        "RF_BAL": rf_bal,
        "LOG_ACC": log_acc,
        "LOG_BAL": log_bal,
        "RF_TIME": rf_time,
        "LOG_TIME": log_time
    }


# ======================================================================
# MAIN
# ======================================================================

def main():

    inicio_total = time.time()

    titulo(
        "ABLACION DE VARIABILIDAD - StarEmbed / ZTF"
    )

    print("""
Objetivo:
Determinar qué variables del bloque de VARIABILIDAD
aportan realmente la capacidad discriminante.

Método:
Se parte del modelo TODAS_VARIABLES y se elimina
UNA variable cada vez.

Una caída importante de balanced accuracy indica
que la variable eliminada contiene información
discriminante no totalmente redundante.
""")

    print(
        "Random State:",
        RANDOM_STATE
    )

    # --------------------------------------------------------------
    # CARGA
    # --------------------------------------------------------------

    train, test = cargar_datos()

    y_train = train["class_str"].astype(str)
    y_test = test["class_str"].astype(str)

    titulo("DISTRIBUCION DE CLASES")

    print()
    print("TRAIN:")
    print(y_train.value_counts())

    print()
    print("TEST:")
    print(y_test.value_counts())

    # --------------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------------

    titulo(
        "EXTRACCION DE FEATURES"
    )

    X_train_full = extraer_features(
        train,
        "TRAIN"
    )

    X_test_full = extraer_features(
        test,
        "TEST"
    )

    # --------------------------------------------------------------
    # LIMPIEZA
    # --------------------------------------------------------------

    titulo("LIMPIEZA")

    X_train_full, nan_tr_1, nan_tr_2 = limpiar_dataframe(
        X_train_full
    )

    X_test_full, nan_te_1, nan_te_2 = limpiar_dataframe(
        X_test_full
    )

    print(
        "NaN TRAIN antes:",
        nan_tr_1
    )

    print(
        "NaN TEST antes :",
        nan_te_1
    )

    print(
        "NaN TRAIN:",
        nan_tr_2
    )

    print(
        "NaN TEST :",
        nan_te_2
    )

    # --------------------------------------------------------------
    # ALINEACION
    # --------------------------------------------------------------

    n_train = min(
        len(X_train_full),
        len(y_train)
    )

    n_test = min(
        len(X_test_full),
        len(y_test)
    )

    X_train_full = X_train_full.iloc[
        :n_train
    ].reset_index(drop=True)

    X_test_full = X_test_full.iloc[
        :n_test
    ].reset_index(drop=True)

    y_train = y_train.iloc[
        :n_train
    ].reset_index(drop=True)

    y_test = y_test.iloc[
        :n_test
    ].reset_index(drop=True)

    # --------------------------------------------------------------
    # VARIABLES DISPONIBLES
    # --------------------------------------------------------------

    variables_disponibles = [
        v for v in VARIABLES
        if v in X_train_full.columns
    ]

    print()
    print(
        "Variables disponibles:"
    )

    for v in variables_disponibles:
        print(" ", v)

    # ==============================================================
    # MODELO BASE
    # ==============================================================

    titulo(
        "MODELO BASE - TODAS VARIABLES"
    )

    X_base_cols = [
        v for v in variables_disponibles
        if v not in EXCLUIDAS
    ]

    print()
    print(
        "Variables utilizadas:",
        len(X_base_cols)
    )

    for v in X_base_cols:
        print(" ", v)

    Xtr = X_train_full[
        X_base_cols
    ]

    Xte = X_test_full[
        X_base_cols
    ]

    base = evaluar(
        Xtr,
        y_train,
        Xte,
        y_test,
        "MODELO BASE"
    )

    # ==============================================================
    # ABLACION
    # ==============================================================

    titulo(
        "ABLACION UNA VARIABLE CADA VEZ"
    )

    resultados = []

    # Resultado base
    resultados.append({
        "experimento": "BASE",
        "variable_eliminada": "",
        "features": len(X_base_cols),
        "RF_ACC": base["RF_ACC"],
        "RF_BAL": base["RF_BAL"],
        "LOG_ACC": base["LOG_ACC"],
        "LOG_BAL": base["LOG_BAL"],
        "RF_TIME": base["RF_TIME"],
        "LOG_TIME": base["LOG_TIME"],
        "CAIDA_RF_BAL": 0.0,
        "CAIDA_LOG_BAL": 0.0
    })

    for variable in VARIABILIDAD:

        if variable not in X_base_cols:
            continue

        print()
        print("=" * 70)
        print(
            "ELIMINANDO:",
            variable
        )
        print("=" * 70)

        columnas = [
            c for c in X_base_cols
            if c != variable
        ]

        print(
            "Features restantes:",
            len(columnas)
        )

        Xtr = X_train_full[
            columnas
        ]

        Xte = X_test_full[
            columnas
        ]

        resultado = evaluar(
            Xtr,
            y_train,
            Xte,
            y_test,
            f"SIN {variable}"
        )

        caida_rf = (
            base["RF_BAL"]
            -
            resultado["RF_BAL"]
        )

        caida_log = (
            base["LOG_BAL"]
            -
            resultado["LOG_BAL"]
        )

        resultados.append({
            "experimento": "ABLACION",
            "variable_eliminada": variable,
            "features": len(columnas),
            "RF_ACC": resultado["RF_ACC"],
            "RF_BAL": resultado["RF_BAL"],
            "LOG_ACC": resultado["LOG_ACC"],
            "LOG_BAL": resultado["LOG_BAL"],
            "RF_TIME": resultado["RF_TIME"],
            "LOG_TIME": resultado["LOG_TIME"],
            "CAIDA_RF_BAL": caida_rf,
            "CAIDA_LOG_BAL": caida_log
        })

        print()
        print(
            f"Caída RF balanced : "
            f"{caida_rf:+.6f}"
        )

        print(
            f"Caída LOG balanced: "
            f"{caida_log:+.6f}"
        )

    # ==============================================================
    # CONTROL ALEATORIO
    # ==============================================================

    titulo(
        "CONTROL DE ETIQUETAS ALEATORIAS"
    )

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    y_random = y_train.to_numpy().copy()

    rng.shuffle(
        y_random
    )

    control = evaluar(
        X_train_full[X_base_cols],
        y_random,
        X_test_full[X_base_cols],
        y_test,
        "ETIQUETAS ALEATORIAS"
    )

    resultados.append({
        "experimento": "ETIQUETAS_ALEATORIAS",
        "variable_eliminada": "",
        "features": len(X_base_cols),
        "RF_ACC": control["RF_ACC"],
        "RF_BAL": control["RF_BAL"],
        "LOG_ACC": control["LOG_ACC"],
        "LOG_BAL": control["LOG_BAL"],
        "RF_TIME": control["RF_TIME"],
        "LOG_TIME": control["LOG_TIME"],
        "CAIDA_RF_BAL": np.nan,
        "CAIDA_LOG_BAL": np.nan
    })

    # ==============================================================
    # ORDENAR ABLACIONES
    # ==============================================================

    df_resultados = pd.DataFrame(
        resultados
    )

    df_ablacion = df_resultados[
        df_resultados["experimento"]
        == "ABLACION"
    ].copy()

    df_ablacion = df_ablacion.sort_values(
        "CAIDA_LOG_BAL",
        ascending=False
    )

    # ==============================================================
    # RESUMEN
    # ==============================================================

    titulo(
        "RESUMEN DE ABLACION"
    )

    print()

    print(
        f"{'VARIABLE':30s}"
        f"{'RF BAL':>12s}"
        f"{'LOG BAL':>12s}"
        f"{'CAIDA RF':>12s}"
        f"{'CAIDA LOG':>12s}"
    )

    print("-" * 78)

    print(
        f"{'BASE':30s}"
        f"{base['RF_BAL']:12.6f}"
        f"{base['LOG_BAL']:12.6f}"
        f"{0:12.6f}"
        f"{0:12.6f}"
    )

    for _, fila in df_ablacion.iterrows():

        print(
            f"{fila['variable_eliminada']:30s}"
            f"{fila['RF_BAL']:12.6f}"
            f"{fila['LOG_BAL']:12.6f}"
            f"{fila['CAIDA_RF_BAL']:12.6f}"
            f"{fila['CAIDA_LOG_BAL']:12.6f}"
        )

    # ==============================================================
    # INTERPRETACION
    # ==============================================================

    titulo(
        "DIAGNOSTICO"
    )

    if len(df_ablacion) > 0:

        mejor_log = df_ablacion.iloc[
            df_ablacion["CAIDA_LOG_BAL"].argmax()
        ]

        mejor_rf = df_ablacion.iloc[
            df_ablacion["CAIDA_RF_BAL"].argmax()
        ]

        print()
        print(
            "Variable cuya eliminación más perjudica "
            "a la LOGISTICA:"
        )

        print(
            " ",
            mejor_log["variable_eliminada"]
        )

        print(
            " Caída LOG balanced:",
            f"{mejor_log['CAIDA_LOG_BAL']:.6f}"
        )

        print()
        print(
            "Variable cuya eliminación más perjudica "
            "al RANDOM FOREST:"
        )

        print(
            " ",
            mejor_rf["variable_eliminada"]
        )

        print(
            " Caída RF balanced:",
            f"{mejor_rf['CAIDA_RF_BAL']:.6f}"
        )

        print()

        if (
            mejor_log["CAIDA_LOG_BAL"] > 0.02
        ):
            print(
                "RESULTADO:"
            )
            print(
                "Existe al menos una variable de "
                "variabilidad con contribución "
                "discriminante apreciable."
            )
        else:
            print(
                "RESULTADO:"
            )
            print(
                "La información parece estar "
                "fuertemente distribuida entre "
                "variables redundantes."
            )

    # ==============================================================
    # GUARDAR
    # ==============================================================

    df_resultados.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8"
    )

    print()
    print(
        "Resultados guardados en:"
    )

    print(
        OUTPUT_CSV
    )

    print()
    print(
        f"Tiempo total: "
        f"{time.time() - inicio_total:.2f} segundos"
    )

    print("=" * 70)


# ======================================================================
# EJECUCION
# ======================================================================

if __name__ == "__main__":
    main()