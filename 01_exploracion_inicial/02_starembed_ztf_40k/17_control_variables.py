# -*- coding: utf-8 -*-

"""
17_control_variables.py

CONTROL DE VARIABLES - StarEmbed / ZTF

Objetivo:
Determinar qué variables estadísticas individuales contienen
la mayor parte de la señal discriminante entre las clases.

Se estudian individualmente:

    - magnitud media
    - mediana
    - desviación estándar
    - MAD
    - amplitud
    - percent amplitude
    - percent difference
    - skewness
    - kurtosis
    - Stetson K
    - eta
    - eta_e
    - chi2
    - maximum slope
    - número de observaciones
    - cadencia
    - error fotométrico medio
    - periodo

Además:
    - ranking univariable
    - ranking multivariable
    - control con etiquetas aleatorias

IMPORTANTE:
Las variables se calculan únicamente a partir de las curvas
ZTF contenidas en StarEmbed.
"""

import os
import time
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
BASE_DIR = Path(__file__).resolve().parents[3]  # Sube 3 niveles hasta Astronomia/

# Directorio de resultados dentro de la misma carpeta que el script
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "17_"

DATA_DIR = BASE_DIR / "dataset" / "ZTF_40k_StarEmbed" / "data"

TRAIN_1 = DATA_DIR / "train-00000-of-00002.parquet"
TRAIN_2 = DATA_DIR / "train-00001-of-00002.parquet"
TEST_FILE = DATA_DIR / "test-00000-of-00001.parquet"

N_TRAIN = 25000
N_TEST = 8000

OUTPUT_CSV = RESULTADOS_DIR / f"{PREFIJO}control_variables_resultados.csv"


# ======================================================================
# UTILIDADES
# ======================================================================

def imprimir_titulo(texto):
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


def limpiar_dataframe(X):
    """
    Limpieza robusta:
    - inf -> NaN
    - mediana de cada variable
    """

    X = X.copy()

    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    n_nan = int(X.isna().sum().sum())

    if n_nan > 0:
        medianas = X.median(numeric_only=True)

        for col in X.columns:
            if pd.isna(medianas[col]):
                medianas[col] = 0.0

        X = X.fillna(medianas)

    return X, n_nan


def evaluar(X_train, y_train, X_test, y_test):

    X_train, nan_train = limpiar_dataframe(X_train)
    X_test, nan_test = limpiar_dataframe(X_test)

    # Aseguramos que ambos tengan exactamente las mismas columnas
    X_test = X_test[X_train.columns]

    resultados = {}

    # --------------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------------

    inicio = time.time()

    rf = RandomForestClassifier(
        n_estimators=250,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_features="sqrt"
    )

    rf.fit(X_train, y_train)

    pred = rf.predict(X_test)

    rf_acc = accuracy_score(y_test, pred)
    rf_bal = balanced_accuracy_score(y_test, pred)

    tiempo_rf = time.time() - inicio

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
                class_weight="balanced",
                C=1.0
            )
        )
    ])

    log.fit(X_train, y_train)

    pred = log.predict(X_test)

    log_acc = accuracy_score(y_test, pred)
    log_bal = balanced_accuracy_score(y_test, pred)

    tiempo_log = time.time() - inicio

    resultados["RF_ACC"] = rf_acc
    resultados["RF_BAL"] = rf_bal
    resultados["LOG_ACC"] = log_acc
    resultados["LOG_BAL"] = log_bal

    resultados["RF_TIME"] = tiempo_rf
    resultados["LOG_TIME"] = tiempo_log

    resultados["NAN_TRAIN"] = nan_train
    resultados["NAN_TEST"] = nan_test

    return resultados


# ======================================================================
# EXTRACCION DE CURVAS
# ======================================================================

def extraer_banda(bands_data, banda):

    if bands_data is None:
        return None

    try:
        datos = bands_data.get(banda)
    except Exception:
        return None

    if datos is None:
        return None

    try:
        target = np.asarray(
            datos.get("target", []),
            dtype=float
        )

        error = np.asarray(
            datos.get("past_feat_dynamic_real", []),
            dtype=float
        )

        delta_t = np.asarray(
            datos.get("feat_dynamic_real", []),
            dtype=float
        )

        mjd = np.asarray(
            datos.get("mjd", []),
            dtype=float
        )

    except Exception:
        return None

    if len(target) == 0:
        return None

    return {
        "target": target,
        "error": error,
        "delta_t": delta_t,
        "mjd": mjd
    }


# ======================================================================
# ESTADISTICAS DE UNA CURVA
# ======================================================================

def estadisticas_curva(datos):

    if datos is None:
        return {}

    y = np.asarray(datos["target"], dtype=float)
    err = np.asarray(datos["error"], dtype=float)
    dt = np.asarray(datos["delta_t"], dtype=float)

    y = y[np.isfinite(y)]

    if len(y) == 0:
        return {}

    resultado = {}

    # --------------------------------------------------------------
    # NIVEL FOTOMETRICO
    # --------------------------------------------------------------

    resultado["mean"] = np.mean(y)
    resultado["median"] = np.median(y)

    # --------------------------------------------------------------
    # DISPERSION
    # --------------------------------------------------------------

    if len(y) > 1:
        resultado["std"] = np.std(y, ddof=1)
    else:
        resultado["std"] = 0.0

    mediana = np.median(y)

    resultado["mad"] = np.median(
        np.abs(y - mediana)
    )

    # --------------------------------------------------------------
    # AMPLITUD
    # --------------------------------------------------------------

    resultado["amplitude"] = (
        np.max(y) - np.min(y)
    )

    resultado["percent_amplitude"] = (
        100.0 *
        (np.max(y) - np.min(y))
        /
        max(abs(np.mean(y)), 1e-12)
    )

    # --------------------------------------------------------------
    # PERCENTILES
    # --------------------------------------------------------------

    p5 = np.percentile(y, 5)
    p20 = np.percentile(y, 20)
    p80 = np.percentile(y, 80)
    p95 = np.percentile(y, 95)

    resultado["percent_diff_5"] = (
        p95 - p5
    )

    resultado["percent_diff_20"] = (
        p80 - p20
    )

    # --------------------------------------------------------------
    # ASIMETRIA
    # --------------------------------------------------------------

    if len(y) > 2:

        mu = np.mean(y)
        sigma = np.std(y)

        if sigma > 0:

            z = (y - mu) / sigma

            resultado["skew"] = np.mean(z ** 3)
            resultado["kurtosis"] = np.mean(z ** 4) - 3.0

        else:
            resultado["skew"] = 0.0
            resultado["kurtosis"] = 0.0

    else:
        resultado["skew"] = 0.0
        resultado["kurtosis"] = 0.0

    # --------------------------------------------------------------
    # STETSON K APROXIMADO
    # --------------------------------------------------------------

    if len(y) > 1:

        sigma = np.std(y)

        if sigma > 0:

            delta = np.abs(
                (y - np.mean(y)) / sigma
            )

            resultado["stetson_k"] = (
                np.mean(delta)
                /
                np.sqrt(np.mean(delta ** 2))
            )

        else:
            resultado["stetson_k"] = 0.0

    else:
        resultado["stetson_k"] = 0.0

    # --------------------------------------------------------------
    # ETA
    # --------------------------------------------------------------

    if len(y) > 1:

        var = np.var(y)

        if var > 0:

            resultado["eta"] = (
                np.sum(np.diff(y) ** 2)
                /
                ((len(y) - 1) * var)
            )

        else:
            resultado["eta"] = 0.0

    else:
        resultado["eta"] = 0.0

    # --------------------------------------------------------------
    # CHI2
    # --------------------------------------------------------------

    if len(y) > 1:

        if len(err) == len(y):

            valid_err = np.isfinite(err) & (err > 0)

            if np.sum(valid_err) > 1:

                yy = y[valid_err]
                ee = err[valid_err]

                media = np.average(
                    yy,
                    weights=1.0 / (ee ** 2)
                )

                resultado["chi2"] = np.mean(
                    ((yy - media) / ee) ** 2
                )

            else:
                resultado["chi2"] = 0.0

        else:
            resultado["chi2"] = 0.0

    else:
        resultado["chi2"] = 0.0

    # --------------------------------------------------------------
    # MAXIMUM SLOPE
    # --------------------------------------------------------------

    if len(y) > 1 and len(dt) == len(y):

        valid = (
            np.isfinite(y)
            &
            np.isfinite(dt)
            &
            (dt > 0)
        )

        if np.sum(valid) > 1:

            yy = y[valid]
            dd = dt[valid]

            slopes = np.abs(
                np.diff(yy) / dd[1:]
            )

            if len(slopes):
                resultado["maximum_slope"] = np.max(slopes)
            else:
                resultado["maximum_slope"] = 0.0

        else:
            resultado["maximum_slope"] = 0.0

    else:
        resultado["maximum_slope"] = 0.0

    # --------------------------------------------------------------
    # ERROR FOTOMETRICO
    # --------------------------------------------------------------

    err_valid = err[np.isfinite(err)]

    if len(err_valid):

        resultado["mean_error"] = np.mean(err_valid)
        resultado["median_error"] = np.median(err_valid)
        resultado["std_error"] = np.std(err_valid)

    else:

        resultado["mean_error"] = 0.0
        resultado["median_error"] = 0.0
        resultado["std_error"] = 0.0

    # --------------------------------------------------------------
    # OBSERVACIONES
    # --------------------------------------------------------------

    resultado["n_obs"] = len(y)

    # --------------------------------------------------------------
    # CADENCIA
    # --------------------------------------------------------------

    dt_valid = dt[
        np.isfinite(dt) & (dt > 0)
    ]

    if len(dt_valid):

        resultado["mean_dt"] = np.mean(dt_valid)
        resultado["median_dt"] = np.median(dt_valid)
        resultado["std_dt"] = np.std(dt_valid)

    else:

        resultado["mean_dt"] = 0.0
        resultado["median_dt"] = 0.0
        resultado["std_dt"] = 0.0

    return resultado


# ======================================================================
# EXTRACCION DE OBJETO
# ======================================================================

def extraer_objeto(row):

    bandas = row["bands_data"]

    resultados = {}

    curvas = []

    for banda in ["g", "r", "i"]:

        datos = extraer_banda(
            bandas,
            banda
        )

        if datos is not None:

            curvas.append(datos)

    if not curvas:

        return resultados

    # --------------------------------------------------------------
    # ESTADISTICAS POR BANDA
    # --------------------------------------------------------------

    todas = []

    for curva in curvas:

        stats = estadisticas_curva(curva)

        if stats:
            todas.append(stats)

    if not todas:
        return resultados

    # --------------------------------------------------------------
    # MEDIAMOS LAS BANDAS
    # --------------------------------------------------------------

    claves = set()

    for stats in todas:
        claves.update(stats.keys())

    for clave in claves:

        valores = [
            s[clave]
            for s in todas
            if clave in s
            and np.isfinite(s[clave])
        ]

        if valores:
            resultados[clave] = np.mean(valores)
        else:
            resultados[clave] = np.nan

    return resultados


# ======================================================================
# DATAFRAME COMPLETO
# ======================================================================

def extraer_features(df, nombre):

    print(f"\nExtrayendo {nombre}...")

    registros = []

    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):

        if i % 5000 == 0:
            print(f"  {i:,}/{total:,}")

        registros.append(
            extraer_objeto(row)
        )

    X = pd.DataFrame(registros)

    return X


# ======================================================================
# CARGA
# ======================================================================

def cargar_datos():

    imprimir_titulo("CARGA DE DATOS")

    print("Cargando TRAIN:")
    print(" ", TRAIN_1)

    if not TRAIN_1.exists():
        raise FileNotFoundError(TRAIN_1)

    train1 = pd.read_parquet(TRAIN_1)

    print("Objetos:", f"{len(train1):,}")

    print("\nCargando TRAIN:")
    print(" ", TRAIN_2)

    if not TRAIN_2.exists():
        raise FileNotFoundError(TRAIN_2)

    train2 = pd.read_parquet(TRAIN_2)

    print("Objetos:", f"{len(train2):,}")

    print("\nCargando TEST:")
    print(" ", TEST_FILE)

    if not TEST_FILE.exists():
        raise FileNotFoundError(TEST_FILE)

    test = pd.read_parquet(TEST_FILE)

    train = pd.concat(
        [train1, train2],
        ignore_index=True
    )

    train = train.iloc[:N_TRAIN].copy()
    test = test.iloc[:N_TEST].copy()

    print(
        f"\nTRAIN utilizado: {len(train):,}"
    )

    print(
        f"TEST utilizado : {len(test):,}"
    )

    return train, test


# ======================================================================
# EXPERIMENTO
# ======================================================================

def ejecutar_experimento(
    nombre,
    columnas,
    X_train,
    y_train,
    X_test,
    y_test
):

    imprimir_titulo(
        f"EXPERIMENTO: {nombre}"
    )

    columnas_validas = [
        c for c in columnas
        if c in X_train.columns
    ]

    print(
        "Variables:",
        len(columnas_validas)
    )

    print(
        "  " + "\n  ".join(columnas_validas)
    )

    Xtr = X_train[columnas_validas].copy()
    Xte = X_test[columnas_validas].copy()

    print("\nLIMPIEZA")

    resultados = evaluar(
        Xtr,
        y_train,
        Xte,
        y_test
    )

    print(
        f"Random Forest : "
        f"accuracy={resultados['RF_ACC']:.6f} "
        f"balanced={resultados['RF_BAL']:.6f} "
        f"({resultados['RF_TIME']:.2f} s)"
    )

    print(
        f"Logística     : "
        f"accuracy={resultados['LOG_ACC']:.6f} "
        f"balanced={resultados['LOG_BAL']:.6f} "
        f"({resultados['LOG_TIME']:.2f} s)"
    )

    resultados["experimento"] = nombre
    resultados["features"] = len(columnas_validas)

    return resultados


# ======================================================================
# CONTROL ALEATORIO
# ======================================================================

def experimento_aleatorio(
    X_train,
    y_train,
    X_test,
    y_test,
    columnas
):

    imprimir_titulo(
        "CONTROL DE ETIQUETAS ALEATORIAS"
    )

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    y_random = y_train.copy()

    rng.shuffle(y_random)

    return ejecutar_experimento(
        "ETIQUETAS_ALEATORIAS",
        columnas,
        X_train,
        y_random,
        X_test,
        y_test
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    inicio_total = time.time()

    imprimir_titulo(
        "CONTROL DE VARIABLES - StarEmbed / ZTF"
    )

    print("""
Objetivo:
Determinar qué variables estadísticas individuales
contienen la señal discriminante entre las clases.

Se busca identificar las variables responsables de
la capacidad de clasificación observada anteriormente.
""")

    print(
        f"Random State: {RANDOM_STATE}"
    )

    # --------------------------------------------------------------
    # CARGA
    # --------------------------------------------------------------

    train, test = cargar_datos()

    imprimir_titulo(
        "DISTRIBUCION DE CLASES"
    )

    print("\nTRAIN:")
    print(train["class_str"].value_counts())

    print("\nTEST:")
    print(test["class_str"].value_counts())

    y_train = train["class_str"].copy()
    y_test = test["class_str"].copy()

    # --------------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------------

    X_train = extraer_features(
        train,
        "TRAIN"
    )

    X_test = extraer_features(
        test,
        "TEST"
    )

    # Periodo
    X_train["period"] = train["period"].values
    X_test["period"] = test["period"].values

    print("\nTRAIN shape:", X_train.shape)
    print("TEST shape :", X_test.shape)

    # --------------------------------------------------------------
    # DEFINICION DE VARIABLES
    # --------------------------------------------------------------

    grupos = {

        "MAGNITUD_MEDIA": [
            "mean"
        ],

        "MEDIANA": [
            "median"
        ],

        "DISPERSION": [
            "std"
        ],

        "MAD": [
            "mad"
        ],

        "AMPLITUD": [
            "amplitude"
        ],

        "PERCENT_AMPLITUDE": [
            "percent_amplitude"
        ],

        "RANGO_PERCENTIL": [
            "percent_diff_5",
            "percent_diff_20"
        ],

        "ASIMETRIA": [
            "skew"
        ],

        "KURTOSIS": [
            "kurtosis"
        ],

        "STETSON_K": [
            "stetson_k"
        ],

        "ETA": [
            "eta"
        ],

        "CHI2": [
            "chi2"
        ],

        "MAXIMUM_SLOPE": [
            "maximum_slope"
        ],

        "ERROR_FOTOMETRICO": [
            "mean_error",
            "median_error",
            "std_error"
        ],

        "N_OBSERVACIONES": [
            "n_obs"
        ],

        "CADENCIA": [
            "mean_dt",
            "median_dt",
            "std_dt"
        ],

        "PERIODO": [
            "period"
        ]
    }

    resultados = []

    # --------------------------------------------------------------
    # INDIVIDUALES
    # --------------------------------------------------------------

    for nombre, columnas in grupos.items():

        resultado = ejecutar_experimento(
            nombre,
            columnas,
            X_train,
            y_train,
            X_test,
            y_test
        )

        resultados.append(resultado)

    # --------------------------------------------------------------
    # VARIABLES DE VARIABILIDAD
    # --------------------------------------------------------------

    imprimir_titulo(
        "EXPERIMENTO: VARIABILIDAD"
    )

    columnas = [
        "std",
        "mad",
        "amplitude",
        "percent_amplitude",
        "percent_diff_5",
        "percent_diff_20",
        "skew",
        "kurtosis",
        "stetson_k",
        "eta",
        "chi2",
        "maximum_slope"
    ]

    resultado = ejecutar_experimento(
        "VARIABILIDAD",
        columnas,
        X_train,
        y_train,
        X_test,
        y_test
    )

    resultados.append(resultado)

    # --------------------------------------------------------------
    # FOTOMETRIA SIN NIVEL ABSOLUTO
    # --------------------------------------------------------------

    columnas = [
        "std",
        "mad",
        "amplitude",
        "percent_amplitude",
        "percent_diff_5",
        "percent_diff_20",
        "skew",
        "kurtosis",
        "stetson_k",
        "eta",
        "chi2",
        "maximum_slope",
        "mean_error",
        "median_error",
        "std_error",
        "n_obs",
        "mean_dt",
        "median_dt",
        "std_dt"
    ]

    resultado = ejecutar_experimento(
        "TODO_SIN_MAGNITUD_MEDIA",
        columnas,
        X_train,
        y_train,
        X_test,
        y_test
    )

    resultados.append(resultado)

    # --------------------------------------------------------------
    # TODO
    # --------------------------------------------------------------

    columnas_todas = list(X_train.columns)

    resultado = ejecutar_experimento(
        "TODAS_VARIABLES",
        columnas_todas,
        X_train,
        y_train,
        X_test,
        y_test
    )

    resultados.append(resultado)

    # --------------------------------------------------------------
    # CONTROL ALEATORIO
    # --------------------------------------------------------------

    resultado = experimento_aleatorio(
        X_train,
        y_train,
        X_test,
        y_test,
        columnas_todas
    )

    resultados.append(resultado)

    # --------------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------------

    imprimir_titulo(
        "RESUMEN FINAL"
    )

    df_resultados = pd.DataFrame(
        resultados
    )

    columnas_salida = [
        "experimento",
        "features",
        "RF_ACC",
        "RF_BAL",
        "LOG_ACC",
        "LOG_BAL"
    ]

    print(
        df_resultados[
            columnas_salida
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------------
    # RANKING
    # --------------------------------------------------------------

    imprimir_titulo(
        "RANKING POR BALANCED ACCURACY"
    )

    ranking = df_resultados[
        df_resultados["experimento"]
        != "ETIQUETAS_ALEATORIAS"
    ].copy()

    ranking = ranking.sort_values(
        "LOG_BAL",
        ascending=False
    )

    print(
        ranking[
            [
                "experimento",
                "RF_BAL",
                "LOG_BAL"
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------------
    # DIAGNOSTICO
    # --------------------------------------------------------------

    imprimir_titulo(
        "DIAGNOSTICO"
    )

    no_aleatorio = df_resultados[
        df_resultados["experimento"]
        != "ETIQUETAS_ALEATORIAS"
    ]

    mejor = no_aleatorio.loc[
        no_aleatorio["LOG_BAL"].idxmax()
    ]

    aleatorio = df_resultados[
        df_resultados["experimento"]
        == "ETIQUETAS_ALEATORIAS"
    ].iloc[0]

    diferencia = (
        mejor["LOG_BAL"]
        -
        aleatorio["LOG_BAL"]
    )

    print(
        f"Mejor experimento LOG balanced : "
        f"{mejor['LOG_BAL']:.6f}"
    )

    print(
        f"Referencia aleatoria           : "
        f"{aleatorio['LOG_BAL']:.6f}"
    )

    print(
        f"Diferencia                      : "
        f"{diferencia:.6f}"
    )

    print("\nRESULTADO:")

    if diferencia > 0.30:

        print(
            "La señal discriminante es claramente "
            "superior al nivel aleatorio."
        )

        print(
            "El ranking permite identificar qué "
            "variables estadísticas son las "
            "principales responsables."
        )

    else:

        print(
            "La señal discriminante es débil."
        )

    # --------------------------------------------------------------
    # GUARDAR
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
        f"{time.time() - inicio_total:.2f} segundos"
    )


# ======================================================================
# EJECUCION
# ======================================================================

if __name__ == "__main__":
    main()