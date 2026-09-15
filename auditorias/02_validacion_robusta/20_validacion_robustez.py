# -*- coding: utf-8 -*-

"""
======================================================================
VALIDACION DE ROBUSTEZ - StarEmbed / ZTF
======================================================================

Objetivo:
Comprobar si la señal encontrada es estable
frente a diferentes semillas aleatorias.

Modelos:

  MINIMO:
      median_absolute_deviation
      skew
      percent_amplitude
      amplitude
      eta

  COMPLETO:
      12 variables de variabilidad

Semillas:
    [7, 21, 42, 84, 123]

La extracción de variables es IDENTICA a la utilizada
en 19_seleccion_variables.py.

Metrica principal:
    Balanced Accuracy
"""

import os
import time
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

MAX_TRAIN = 25000
MAX_TEST = 8000

SEMILLAS = [7, 21, 42, 84, 123]

# Ruta relativa desde la ubicación del script
BASE_DIR = Path(__file__).resolve().parents[2]  # Sube 2 niveles hasta Astronomia/

# Directorio de resultados dentro de la misma carpeta que el script
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "20_"

DATA_DIR = BASE_DIR / "dataset" / "ZTF_40k_StarEmbed" / "data"

TRAIN_FILES = [
    DATA_DIR / "train-00000-of-00002.parquet",
    DATA_DIR / "train-00001-of-00002.parquet",
]

TEST_FILE = DATA_DIR / "test-00000-of-00001.parquet"

RESULTADOS = RESULTADOS_DIR / f"{PREFIJO}validacion_robustez_resultados.csv"


# ======================================================================
# VARIABLES
# ======================================================================

VARIABLES_COMPLETO = [
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


# Las primeras cinco corresponden al conjunto mínimo
# obtenido en la selección progresiva del experimento 19.
VARIABLES_MINIMO = [
    "median_absolute_deviation",
    "skew",
    "percent_amplitude",
    "amplitude",
    "eta",
]


# ======================================================================
# UTILIDADES
# ======================================================================

def linea():
    print("-" * 70)


# ======================================================================
# CARGA DE DATOS
# ======================================================================

def cargar_datos():

    print("=" * 70)
    print("CARGA DE DATOS")
    print("=" * 70)

    frames_train = []

    for fichero in TRAIN_FILES:

        print()
        print("Cargando TRAIN:")
        print(" ", fichero)

        if not fichero.exists():
            raise FileNotFoundError(fichero)

        df = pd.read_parquet(fichero)

        frames_train.append(df)

        print(
            "Objetos:",
            f"{len(df):,}"
        )

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
    print(
        "TRAIN utilizado:",
        f"{len(train):,}"
    )

    print(
        "TEST utilizado :",
        f"{len(test):,}"
    )

    return train, test


# ======================================================================
# OBTENER BANDA
# ======================================================================

def obtener_banda(bands):

    if bands is None:
        return None

    try:

        if isinstance(bands, dict):

            for b in ["r", "g", "i"]:

                if (
                    b in bands
                    and bands[b] is not None
                ):
                    return bands[b]

        return None

    except Exception:

        return None


# ======================================================================
# EXTRACCION DE FEATURES
# ======================================================================

def extraer_features(df):

    resultados = []

    n = len(df)

    for i, (_, fila) in enumerate(df.iterrows()):

        if i % 5000 == 0:

            print(
                f"  {i:,}/{n:,}"
            )

        valores = {}

        bands = fila.get(
            "bands_data",
            None
        )

        banda = obtener_banda(bands)

        if banda is None:
            banda = {}

        target = banda.get(
            "target",
            []
        )

        if target is None:
            target = []

        try:

            x = np.asarray(
                target,
                dtype=float
            )

        except Exception:

            x = np.array(
                [],
                dtype=float
            )

        x = x[np.isfinite(x)]

        # --------------------------------------------------------------
        # SIN DATOS
        # --------------------------------------------------------------

        if len(x) == 0:

            for variable in VARIABLES_COMPLETO:
                valores[variable] = np.nan

            resultados.append(valores)

            continue

        # --------------------------------------------------------------
        # ESTADISTICAS BASICAS
        # --------------------------------------------------------------

        media = np.mean(x)

        mediana = np.median(x)

        std = np.std(x)

        mad = np.median(
            np.abs(
                x - mediana
            )
        )

        minimo = np.min(x)

        maximo = np.max(x)

        amplitud = (
            maximo - minimo
        ) / 2.0

        if abs(media) > 1e-12:

            percent_amplitude = (
                amplitud
                / abs(media)
            )

        else:

            percent_amplitude = 0.0

        # --------------------------------------------------------------
        # RANGO PERCENTIL
        # --------------------------------------------------------------

        try:

            p25 = np.percentile(
                x,
                12.5
            )

            p75 = np.percentile(
                x,
                87.5
            )

            iqr25 = p75 - p25

        except Exception:

            iqr25 = np.nan

        # --------------------------------------------------------------
        # ASIMETRIA Y KURTOSIS
        # --------------------------------------------------------------

        if std > 1e-12:

            z = (
                x - media
            ) / std

            skew = np.mean(
                z ** 3
            )

            kurtosis = (
                np.mean(
                    z ** 4
                ) - 3.0
            )

        else:

            skew = 0.0
            kurtosis = 0.0

        # --------------------------------------------------------------
        # STETSON K
        # --------------------------------------------------------------

        if (
            std > 1e-12
            and len(x) > 1
        ):

            delta = (
                x - media
            ) / std

            denominador = np.sqrt(
                np.mean(
                    delta ** 2
                )
            )

            if denominador > 1e-12:

                stetson_k = (
                    np.mean(
                        np.abs(delta)
                    )
                    / denominador
                )

            else:

                stetson_k = 0.0

        else:

            stetson_k = 0.0

        # --------------------------------------------------------------
        # ETA
        # --------------------------------------------------------------

        if (
            len(x) > 1
            and std > 1e-12
        ):

            diferencias = np.diff(x)

            eta = (
                np.sum(
                    diferencias ** 2
                )
                /
                (
                    (len(x) - 1)
                    * std ** 2
                )
            )

        else:

            eta = 0.0

        # --------------------------------------------------------------
        # CHI2
        # --------------------------------------------------------------

        if std > 1e-12:

            chi2 = (
                np.sum(
                    (
                        (x - media)
                        / std
                    ) ** 2
                )
                /
                max(
                    len(x) - 1,
                    1
                )
            )

        else:

            chi2 = 0.0

        # --------------------------------------------------------------
        # MAXIMUM SLOPE
        # --------------------------------------------------------------

        try:

            mjd = banda.get(
                "mjd",
                []
            )

            t = np.asarray(
                mjd,
                dtype=float
            )

            target_array = np.asarray(
                target,
                dtype=float
            )

            mask = (
                np.isfinite(t)
                &
                np.isfinite(
                    target_array
                )
            )

            t = t[mask]

            y = target_array[mask]

            if len(t) > 1:

                dt = np.diff(t)

                dy = np.diff(y)

                valid = dt > 0

                if np.any(valid):

                    slopes = np.abs(
                        dy[valid]
                        / dt[valid]
                    )

                    maximum_slope = np.max(
                        slopes
                    )

                else:

                    maximum_slope = 0.0

            else:

                maximum_slope = 0.0

        except Exception:

            maximum_slope = 0.0

        # --------------------------------------------------------------
        # GUARDAR
        # --------------------------------------------------------------

        valores[
            "median"
        ] = mediana

        valores[
            "standard_deviation"
        ] = std

        valores[
            "median_absolute_deviation"
        ] = mad

        valores[
            "amplitude"
        ] = amplitud

        valores[
            "percent_amplitude"
        ] = percent_amplitude

        valores[
            "inter_percentile_range_25"
        ] = iqr25

        valores[
            "skew"
        ] = skew

        valores[
            "kurtosis"
        ] = kurtosis

        valores[
            "stetson_K"
        ] = stetson_k

        valores[
            "eta"
        ] = eta

        valores[
            "chi2"
        ] = chi2

        valores[
            "maximum_slope"
        ] = maximum_slope

        resultados.append(valores)

    return pd.DataFrame(
        resultados
    )


# ======================================================================
# LIMPIEZA
# ======================================================================

def limpiar(
    X_train,
    X_test
):

    print()
    print("=" * 70)
    print("LIMPIEZA")
    print("=" * 70)

    print(
        "NaN TRAIN antes:",
        int(
            X_train.isna()
            .sum()
            .sum()
        )
    )

    print(
        "NaN TEST antes :",
        int(
            X_test.isna()
            .sum()
            .sum()
        )
    )

    # Estadisticas calculadas SOLO sobre TRAIN
    med = X_train.median()

    X_train = X_train.fillna(
        med
    )

    X_test = X_test.fillna(
        med
    )

    X_train = (
        X_train
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    X_test = (
        X_test
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    print(
        "NaN TRAIN:",
        int(
            X_train.isna()
            .sum()
            .sum()
        )
    )

    print(
        "NaN TEST :",
        int(
            X_test.isna()
            .sum()
            .sum()
        )
    )

    return X_train, X_test


# ======================================================================
# EVALUACION
# ======================================================================

def evaluar(
    X_train,
    y_train,
    X_test,
    y_test,
    variables,
    random_state
):

    Xtr = X_train[
        variables
    ]

    Xte = X_test[
        variables
    ]

    # ==============================================================
    # RANDOM FOREST
    # ==============================================================

    t0 = time.time()

    rf = RandomForestClassifier(
        n_estimators=150,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_features="sqrt"
    )

    rf.fit(
        Xtr,
        y_train
    )

    pred_rf = rf.predict(
        Xte
    )

    rf_acc = accuracy_score(
        y_test,
        pred_rf
    )

    rf_bal = balanced_accuracy_score(
        y_test,
        pred_rf
    )

    rf_time = (
        time.time() - t0
    )

    # ==============================================================
    # LOGISTICA
    # ==============================================================

    t0 = time.time()

    log = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced"
        )
    )

    log.fit(
        Xtr,
        y_train
    )

    pred_log = log.predict(
        Xte
    )

    log_acc = accuracy_score(
        y_test,
        pred_log
    )

    log_bal = balanced_accuracy_score(
        y_test,
        pred_log
    )

    log_time = (
        time.time() - t0
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
# MOSTRAR RESULTADO
# ======================================================================

def mostrar_resultado(
    nombre,
    metricas
):

    print()
    linea()

    print(nombre)

    print(
        "Random Forest : "
        f"accuracy={metricas['RF_ACC']:.6f} "
        f"balanced={metricas['RF_BAL']:.6f} "
        f"({metricas['RF_TIME']:.2f} s)"
    )

    print(
        "Logística     : "
        f"accuracy={metricas['LOG_ACC']:.6f} "
        f"balanced={metricas['LOG_BAL']:.6f} "
        f"({metricas['LOG_TIME']:.2f} s)"
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    inicio = time.time()

    print()
    print("=" * 70)
    print("VALIDACION DE ROBUSTEZ - StarEmbed / ZTF")
    print("=" * 70)

    print()
    print("Objetivo:")
    print(
        "Comprobar si la señal encontrada es estable"
    )
    print(
        "frente a diferentes semillas aleatorias."
    )

    print()
    print("Modelos:")

    print()
    print("  MINIMO:")
    for v in VARIABLES_MINIMO:
        print("    ", v)

    print()
    print("  COMPLETO:")
    for v in VARIABLES_COMPLETO:
        print("    ", v)

    print()
    print(
        "Semillas:",
        SEMILLAS
    )

    # ==============================================================
    # CARGA
    # ==============================================================

    train, test = cargar_datos()

    y_train = (
        train["class_str"]
        .astype(str)
    )

    y_test = (
        test["class_str"]
        .astype(str)
    )

    # ==============================================================
    # DISTRIBUCION
    # ==============================================================

    print()
    print("=" * 70)
    print("DISTRIBUCION DE CLASES")
    print("=" * 70)

    print()
    print("TRAIN:")
    print(
        y_train.value_counts()
    )

    print()
    print("TEST:")
    print(
        y_test.value_counts()
    )

    # ==============================================================
    # EXTRACCION
    # ==============================================================

    print()
    print("=" * 70)
    print("EXTRACCION DE FEATURES")
    print("=" * 70)

    print()
    print("TRAIN")

    X_train = extraer_features(
        train
    )

    print()
    print("TEST")

    X_test = extraer_features(
        test
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

    # ==============================================================
    # COMPROBAR VARIABLES
    # ==============================================================

    faltan_train = [
        v for v in VARIABLES_COMPLETO
        if v not in X_train.columns
    ]

    faltan_test = [
        v for v in VARIABLES_COMPLETO
        if v not in X_test.columns
    ]

    if faltan_train or faltan_test:

        print()
        print(
            "ERROR: faltan variables."
        )

        if faltan_train:
            print(
                "TRAIN:",
                faltan_train
            )

        if faltan_test:
            print(
                "TEST:",
                faltan_test
            )

        raise RuntimeError(
            "No se han podido generar "
            "todas las variables."
        )

    # ==============================================================
    # LIMPIEZA
    # ==============================================================

    X_train, X_test = limpiar(
        X_train,
        X_test
    )

    # ==============================================================
    # RESULTADOS
    # ==============================================================

    resultados = []

    # ==============================================================
    # EXPERIMENTOS
    # ==============================================================

    for semilla in SEMILLAS:

        print()
        print()
        print("=" * 70)
        print(
            f"SEMILLA: {semilla}"
        )
        print("=" * 70)

        # ----------------------------------------------------------
        # MINIMO
        # ----------------------------------------------------------

        metricas_minimo = evaluar(
            X_train,
            y_train,
            X_test,
            y_test,
            VARIABLES_MINIMO,
            semilla
        )

        mostrar_resultado(
            "MODELO MINIMO",
            metricas_minimo
        )

        resultados.append({
            "semilla": semilla,
            "modelo": "MINIMO",
            "n_variables": len(
                VARIABLES_MINIMO
            ),
            "variables": "|".join(
                VARIABLES_MINIMO
            ),
            **metricas_minimo
        })

        # ----------------------------------------------------------
        # COMPLETO
        # ----------------------------------------------------------

        metricas_completo = evaluar(
            X_train,
            y_train,
            X_test,
            y_test,
            VARIABLES_COMPLETO,
            semilla
        )

        mostrar_resultado(
            "MODELO COMPLETO",
            metricas_completo
        )

        resultados.append({
            "semilla": semilla,
            "modelo": "COMPLETO",
            "n_variables": len(
                VARIABLES_COMPLETO
            ),
            "variables": "|".join(
                VARIABLES_COMPLETO
            ),
            **metricas_completo
        })

    # ==============================================================
    # CONTROL DE ETIQUETAS ALEATORIAS
    # ==============================================================

    print()
    print()
    print("=" * 70)
    print("CONTROL DE ETIQUETAS ALEATORIAS")
    print("=" * 70)

    rng = np.random.RandomState(42)

    y_random = np.asarray(
        y_train
    ).copy()

    rng.shuffle(
        y_random
    )

    metricas_random = evaluar(
        X_train,
        y_random,
        X_test,
        y_test,
        VARIABLES_COMPLETO,
        42
    )

    mostrar_resultado(
        "ETIQUETAS ALEATORIAS",
        metricas_random
    )

    resultados.append({
        "semilla": 42,
        "modelo": "ETIQUETAS_ALEATORIAS",
        "n_variables": len(
            VARIABLES_COMPLETO
        ),
        "variables": "|".join(
            VARIABLES_COMPLETO
        ),
        **metricas_random
    })

    # ==============================================================
    # DATAFRAME
    # ==============================================================

    resultados_df = pd.DataFrame(
        resultados
    )

    resultados_df.to_csv(
        RESULTADOS,
        index=False
    )

    # ==============================================================
    # RESUMEN
    # ==============================================================

    print()
    print()
    print("=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)

    resumen = (
        resultados_df[
            resultados_df["modelo"]
            .isin(
                ["MINIMO", "COMPLETO"]
            )
        ]
        .groupby("modelo")
        [["RF_BAL", "LOG_BAL"]]
        .agg(
            ["mean", "std", "min", "max"]
        )
    )

    print(
        resumen.to_string()
    )

    # ==============================================================
    # DIAGNOSTICO
    # ==============================================================

    print()
    print()
    print("=" * 70)
    print("DIAGNOSTICO")
    print("=" * 70)

    for modelo in [
        "MINIMO",
        "COMPLETO"
    ]:

        datos = resultados_df[
            resultados_df["modelo"]
            == modelo
        ]

        rf_mean = datos[
            "RF_BAL"
        ].mean()

        rf_std = datos[
            "RF_BAL"
        ].std()

        log_mean = datos[
            "LOG_BAL"
        ].mean()

        log_std = datos[
            "LOG_BAL"
        ].std()

        print()
        print(
            modelo
        )

        print(
            f"  RF balanced : "
            f"{rf_mean:.6f} "
            f"+/- {rf_std:.6f}"
        )

        print(
            f"  LOG balanced: "
            f"{log_mean:.6f} "
            f"+/- {log_std:.6f}"
        )

    random_rf = metricas_random[
        "RF_BAL"
    ]

    random_log = metricas_random[
        "LOG_BAL"
    ]

    print()
    print(
        "Referencia etiquetas aleatorias:"
    )

    print(
        f"  RF  : {random_rf:.6f}"
    )

    print(
        f"  LOG : {random_log:.6f}"
    )

    print()
    print(
        "Resultados guardados en:"
    )

    print(
        RESULTADOS
    )

    print()
    print(
        f"Tiempo total: "
        f"{time.time() - inicio:.2f} segundos"
    )


# ======================================================================
# EJECUCION
# ======================================================================

if __name__ == "__main__":
    main()