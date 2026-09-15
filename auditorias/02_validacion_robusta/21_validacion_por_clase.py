# ======================================================================
# 21_validacion_por_clase.py
# ======================================================================

import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

# ======================================================================
# CONFIGURACION
# ======================================================================

# Ruta relativa desde la ubicación del script
BASE_DIR = Path(__file__).resolve().parents[2]  # Sube 2 niveles hasta Astronomia/

# Directorio de resultados dentro de la misma carpeta que el script
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "21_"

DATA_DIR = BASE_DIR / "dataset" / "ZTF_40k_StarEmbed" / "data"

TRAIN_FILES = [
    DATA_DIR / "train-00000-of-00002.parquet",
    DATA_DIR / "train-00001-of-00002.parquet",
]

TEST_FILE = DATA_DIR / "test-00000-of-00001.parquet"

MAX_TRAIN = 25000
MAX_TEST = 8000

RANDOM_STATE = 42

RESULTADOS_RF = RESULTADOS_DIR / f"{PREFIJO}resultados_por_clase_RF.csv"
RESULTADOS_LOG = RESULTADOS_DIR / f"{PREFIJO}resultados_por_clase_LOG.csv"
IMPORTANCIA = RESULTADOS_DIR / f"{PREFIJO}importancia_variables.csv"
N_OBS_TRAIN = RESULTADOS_DIR / f"{PREFIJO}n_observaciones_train.csv"
N_OBS_TEST = RESULTADOS_DIR / f"{PREFIJO}n_observaciones_test.csv"
CONTROL_SIN_ETA_CHI2 = RESULTADOS_DIR / f"{PREFIJO}control_sin_eta_chi2.csv"

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

VARIABLES_SIN_ETA_CHI2 = [
    "median",
    "standard_deviation",
    "median_absolute_deviation",
    "amplitude",
    "percent_amplitude",
    "inter_percentile_range_25",
    "skew",
    "kurtosis",
    "stetson_K",
    "maximum_slope",
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

    test = pd.read_parquet(
        TEST_FILE
    )

    if len(train) > MAX_TRAIN:
        train = train.iloc[
            :MAX_TRAIN
        ].copy()

    if len(test) > MAX_TEST:
        test = test.iloc[
            :MAX_TEST
        ].copy()

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
# EXTRAER CURVA
# ======================================================================

def obtener_curva(fila):

    bands = fila.get(
        "bands_data",
        None
    )

    banda = obtener_banda(
        bands
    )

    if banda is None:
        return np.array([])

    target = banda.get(
        "target",
        []
    )

    if target is None:
        return np.array([])

    try:

        x = np.asarray(
            target,
            dtype=float
        )

    except Exception:

        return np.array([])

    x = x[
        np.isfinite(x)
    ]

    return x


# ======================================================================
# EXTRAER FEATURES
# ======================================================================

def calcular_features(fila):

    valores = {}

    x = obtener_curva(
        fila
    )

    if len(x) == 0:

        for variable in VARIABLES_COMPLETO:
            valores[variable] = np.nan

        return valores

    # --------------------------------------------------------------
    # BASICAS
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
            amplitud /
            abs(media)
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

        iqr25 = (
            p75 - p25
        )

    except Exception:

        iqr25 = np.nan

    # --------------------------------------------------------------
    # SKEW / KURTOSIS
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
            )
            - 3.0
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

    maximum_slope = 0.0

    try:

        bands = fila.get(
            "bands_data",
            None
        )

        banda = obtener_banda(
            bands
        )

        if banda is not None:

            mjd = banda.get(
                "mjd",
                []
            )

            target = banda.get(
                "target",
                []
            )

            t = np.asarray(
                mjd,
                dtype=float
            )

            y = np.asarray(
                target,
                dtype=float
            )

            mask = (
                np.isfinite(t)
                &
                np.isfinite(y)
            )

            t = t[mask]
            y = y[mask]

            if len(t) > 1:

                dt = np.diff(t)
                dy = np.diff(y)

                valid = dt > 0

                if np.any(valid):

                    slopes = np.abs(
                        dy[valid]
                        /
                        dt[valid]
                    )

                    if len(slopes) > 0:

                        maximum_slope = np.max(
                            slopes
                        )

    except Exception:

        maximum_slope = 0.0

    # --------------------------------------------------------------
    # GUARDAR
    # --------------------------------------------------------------

    valores["median"] = mediana

    valores["standard_deviation"] = std

    valores[
        "median_absolute_deviation"
    ] = mad

    valores["amplitude"] = amplitud

    valores[
        "percent_amplitude"
    ] = percent_amplitude

    valores[
        "inter_percentile_range_25"
    ] = iqr25

    valores["skew"] = skew

    valores["kurtosis"] = kurtosis

    valores["stetson_K"] = stetson_k

    valores["eta"] = eta

    valores["chi2"] = chi2

    valores[
        "maximum_slope"
    ] = maximum_slope

    return valores


# ======================================================================
# EXTRAER DATAFRAME DE FEATURES
# ======================================================================

def extraer_features(df):

    resultados = []

    n = len(df)

    for i, (_, fila) in enumerate(
        df.iterrows()
    ):

        if i % 5000 == 0:

            print(
                f"  {i:,}/{n:,}"
            )

        resultados.append(
            calcular_features(
                fila
            )
        )

    return pd.DataFrame(
        resultados
    )


# ======================================================================
# NUMERO DE OBSERVACIONES
# ======================================================================

def extraer_n_observaciones(df):

    resultados = []

    n = len(df)

    for i, (_, fila) in enumerate(
        df.iterrows()
    ):

        if i % 5000 == 0:

            print(
                f"  {i:,}/{n:,}"
            )

        x = obtener_curva(
            fila
        )

        resultados.append(
            len(x)
        )

    return np.asarray(
        resultados,
        dtype=int
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
# NUMERO DE OBSERVACIONES POR CLASE
# ======================================================================

def resumen_observaciones(
    n_obs,
    clases
):

    df = pd.DataFrame({
        "clase": clases.values,
        "n_observaciones": n_obs
    })

    resumen = (
        df.groupby("clase")
        ["n_observaciones"]
        .agg([
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max"
        ])
        .reset_index()
    )

    return resumen


# ======================================================================
# MODELO
# ======================================================================

def entrenar_modelos(
    X_train,
    y_train,
    X_test,
    y_test,
    variables
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

    inicio = time.time()

    rf = RandomForestClassifier(
        n_estimators=150,
        random_state=RANDOM_STATE,
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
        time.time() - inicio
    )

    # ==============================================================
    # LOGISTICA
    # ==============================================================

    inicio = time.time()

    log = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
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
        time.time() - inicio
    )

    return (
        rf,
        log,
        pred_rf,
        pred_log,
        {
            "RF_ACC": rf_acc,
            "RF_BAL": rf_bal,
            "LOG_ACC": log_acc,
            "LOG_BAL": log_bal,
            "RF_TIME": rf_time,
            "LOG_TIME": log_time
        }
    )


# ======================================================================
# MOSTRAR MATRIZ
# ======================================================================

def mostrar_matriz(
    matriz,
    clases
):

    df = pd.DataFrame(
        matriz,
        index=clases,
        columns=clases
    )

    print(
        df.to_string()
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    inicio_total = time.time()

    print()
    print("=" * 70)
    print(
        "VALIDACION POR CLASE - StarEmbed / ZTF"
    )
    print("=" * 70)

    print()
    print("Objetivo:")
    print(
        "Determinar si la señal discriminante se mantiene"
    )
    print(
        "de forma homogénea entre las diferentes clases."
    )

    print()
    print("Controles:")
    print("  1. Rendimiento por clase")
    print("  2. Matriz de confusion")
    print("  3. Numero de observaciones por clase")
    print("  4. Importancia de variables")
    print("  5. Control sin ETA + CHI2")

    print()
    print(
        "Random State:",
        RANDOM_STATE
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
    # OBSERVACIONES
    # ==============================================================

    print()
    print("=" * 70)
    print(
        "CONTROL: NUMERO DE OBSERVACIONES"
    )
    print("=" * 70)

    print()
    print("Extrayendo observaciones TRAIN...")

    n_obs_train = (
        extraer_n_observaciones(
            train
        )
    )

    print()
    print("Extrayendo observaciones TEST...")

    n_obs_test = (
        extraer_n_observaciones(
            test
        )
    )

    resumen_train = resumen_observaciones(
        n_obs_train,
        y_train
    )

    resumen_test = resumen_observaciones(
        n_obs_test,
        y_test
    )

    print()
    print("TRAIN")

    print(
        resumen_train.to_string(
            index=False
        )
    )

    print()
    print("TEST")

    print(
        resumen_test.to_string(
            index=False
        )
    )

    resumen_train.to_csv(
        N_OBS_TRAIN,
        index=False
    )

    resumen_test.to_csv(
        N_OBS_TEST,
        index=False
    )

    # ==============================================================
    # LIMPIEZA
    # ==============================================================

    X_train, X_test = limpiar(
        X_train,
        X_test
    )

    # ==============================================================
    # MODELO COMPLETO
    # ==============================================================

    print()
    print("=" * 70)
    print("MODELO COMPLETO")
    print("=" * 70)

    (
        rf,
        log,
        pred_rf,
        pred_log,
        metricas
    ) = entrenar_modelos(
        X_train,
        y_train,
        X_test,
        y_test,
        VARIABLES_COMPLETO
    )

    # ==============================================================
    # RESULTADOS GENERALES
    # ==============================================================

    print()
    print("=" * 70)
    print("RESULTADOS GENERALES")
    print("=" * 70)

    print()
    print(
        "RF              "
        f"accuracy={metricas['RF_ACC']:.6f} "
        f"balanced={metricas['RF_BAL']:.6f}"
    )

    print()
    print(
        "LOG             "
        f"accuracy={metricas['LOG_ACC']:.6f} "
        f"balanced={metricas['LOG_BAL']:.6f}"
    )

    # ==============================================================
    # CLASES
    # ==============================================================

    clases = sorted(
        y_test.unique()
    )

    # ==============================================================
    # RESULTADOS POR CLASE RF
    # ==============================================================

    print()
    print("=" * 70)
    print(
        "RESULTADOS POR CLASE - RANDOM FOREST"
    )
    print("=" * 70)

    reporte_rf = classification_report(
        y_test,
        pred_rf,
        labels=clases,
        output_dict=True,
        zero_division=0
    )

    filas_rf = []

    for clase in clases:

        fila = reporte_rf.get(
            clase,
            {}
        )

        filas_rf.append({
            "clase": clase,
            "precision": fila.get(
                "precision",
                0.0
            ),
            "recall": fila.get(
                "recall",
                0.0
            ),
            "f1": fila.get(
                "f1-score",
                0.0
            ),
            "N": int(
                (y_test == clase).sum()
            )
        })

        print(
            f"{clase:<8} "
            f"precision={fila.get('precision', 0.0):.4f} "
            f"recall={fila.get('recall', 0.0):.4f} "
            f"F1={fila.get('f1-score', 0.0):.4f} "
            f"N={int((y_test == clase).sum()):5d}"
        )

    pd.DataFrame(
        filas_rf
    ).to_csv(
        RESULTADOS_RF,
        index=False
    )

    # ==============================================================
    # RESULTADOS POR CLASE LOG
    # ==============================================================

    print()
    print("=" * 70)
    print(
        "RESULTADOS POR CLASE - LOGISTICA"
    )
    print("=" * 70)

    reporte_log = classification_report(
        y_test,
        pred_log,
        labels=clases,
        output_dict=True,
        zero_division=0
    )

    filas_log = []

    for clase in clases:

        fila = reporte_log.get(
            clase,
            {}
        )

        filas_log.append({
            "clase": clase,
            "precision": fila.get(
                "precision",
                0.0
            ),
            "recall": fila.get(
                "recall",
                0.0
            ),
            "f1": fila.get(
                "f1-score",
                0.0
            ),
            "N": int(
                (y_test == clase).sum()
            )
        })

        print(
            f"{clase:<8} "
            f"precision={fila.get('precision', 0.0):.4f} "
            f"recall={fila.get('recall', 0.0):.4f} "
            f"F1={fila.get('f1-score', 0.0):.4f} "
            f"N={int((y_test == clase).sum()):5d}"
        )

    pd.DataFrame(
        filas_log
    ).to_csv(
        RESULTADOS_LOG,
        index=False
    )

    # ==============================================================
    # MATRIZ RF
    # ==============================================================

    print()
    print("=" * 70)
    print(
        "MATRIZ DE CONFUSION - RANDOM FOREST"
    )
    print("=" * 70)

    matriz_rf = confusion_matrix(
        y_test,
        pred_rf,
        labels=clases
    )

    mostrar_matriz(
        matriz_rf,
        clases
    )

    # ==============================================================
    # MATRIZ LOG
    # ==============================================================

    print()
    print("=" * 70)
    print(
        "MATRIZ DE CONFUSION - LOGISTICA"
    )
    print("=" * 70)

    matriz_log = confusion_matrix(
        y_test,
        pred_log,
        labels=clases
    )

    mostrar_matriz(
        matriz_log,
        clases
    )

    # ==============================================================
    # IMPORTANCIA RF
    # ==============================================================

    print()
    print("=" * 70)
    print(
        "IMPORTANCIA DE VARIABLES - RANDOM FOREST"
    )
    print("=" * 70)

    importancia = pd.DataFrame({
        "variable": VARIABLES_COMPLETO,
        "importance": rf.feature_importances_
    })

    importancia = (
        importancia
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    print(
        importancia.to_string(
            index=False
        )
    )

    importancia.to_csv(
        IMPORTANCIA,
        index=False
    )

    # ==============================================================
    # CONTROL SIN ETA + CHI2
    # ==============================================================

    print()
    print("=" * 70)
    print(
        "CONTROL SIN ETA + CHI2"
    )
    print("=" * 70)

    print()
    print("Variables utilizadas:")

    for variable in VARIABLES_SIN_ETA_CHI2:
        print(
            " ",
            variable
        )

    (
        rf2,
        log2,
        pred_rf2,
        pred_log2,
        metricas2
    ) = entrenar_modelos(
        X_train,
        y_train,
        X_test,
        y_test,
        VARIABLES_SIN_ETA_CHI2
    )

    print()
    print(
        "Random Forest : "
        f"accuracy={metricas2['RF_ACC']:.6f} "
        f"balanced={metricas2['RF_BAL']:.6f} "
        f"({metricas2['RF_TIME']:.2f} s)"
    )

    print(
        "Logística     : "
        f"accuracy={metricas2['LOG_ACC']:.6f} "
        f"balanced={metricas2['LOG_BAL']:.6f} "
        f"({metricas2['LOG_TIME']:.2f} s)"
    )

    control = pd.DataFrame([
        {
            "modelo": "COMPLETO",
            "variables": "|".join(
                VARIABLES_COMPLETO
            ),
            **metricas
        },
        {
            "modelo": "SIN_ETA_CHI2",
            "variables": "|".join(
                VARIABLES_SIN_ETA_CHI2
            ),
            **metricas2
        }
    ])

    control.to_csv(
        CONTROL_SIN_ETA_CHI2,
        index=False
    )

    # ==============================================================
    # DIAGNOSTICO
    # ==============================================================

    print()
    print("=" * 70)
    print("DIAGNOSTICO")
    print("=" * 70)

    print()
    print(
        f"RF balanced accuracy: "
        f"{metricas['RF_BAL']:.6f}"
    )

    print(
        f"LOG balanced accuracy: "
        f"{metricas['LOG_BAL']:.6f}"
    )

    print()
    print(
        f"RF sin ETA + CHI2: "
        f"{metricas2['RF_BAL']:.6f}"
    )

    print(
        f"LOG sin ETA + CHI2: "
        f"{metricas2['LOG_BAL']:.6f}"
    )

    caida_rf = (
        metricas["RF_BAL"]
        -
        metricas2["RF_BAL"]
    )

    caida_log = (
        metricas["LOG_BAL"]
        -
        metricas2["LOG_BAL"]
    )

    print()
    print(
        f"Caída RF al eliminar ETA + CHI2: "
        f"{caida_rf:.6f}"
    )

    print(
        f"Caída LOG al eliminar ETA + CHI2: "
        f"{caida_log:.6f}"
    )

    # ==============================================================
    # ARCHIVOS
    # ==============================================================

    print()
    print("Archivos generados:")

    print(
        " ",
        RESULTADOS_RF
    )

    print(
        " ",
        RESULTADOS_LOG
    )

    print(
        " ",
        IMPORTANCIA
    )

    print(
        " ",
        N_OBS_TRAIN
    )

    print(
        " ",
        N_OBS_TEST
    )

    print(
        " ",
        CONTROL_SIN_ETA_CHI2
    )

    print()
    print("=" * 70)
    print("FIN")
    print("=" * 70)

    print()
    print(
        f"Tiempo total: "
        f"{time.time() - inicio_total:.2f} segundos"
    )


# ======================================================================
# EJECUCION
# ======================================================================

if __name__ == "__main__":
    main()