# ======================================================================
# 22_analisis_dependencia_observaciones.py
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
PREFIJO = "22_"

DATA_DIR = BASE_DIR / "dataset" / "ZTF_40k_StarEmbed" / "data"

TRAIN_FILES = [
    DATA_DIR / "train-00000-of-00002.parquet",
    DATA_DIR / "train-00001-of-00002.parquet",
]

TEST_FILE = DATA_DIR / "test-00000-of-00001.parquet"

MAX_TRAIN = 25000
MAX_TEST = 8000
RANDOM_STATE = 42

RESULTADOS_OBS = RESULTADOS_DIR / f"{PREFIJO}resultados_por_observaciones.csv"
RESULTADOS_CLASE_OBS = RESULTADOS_DIR / f"{PREFIJO}resultados_por_clase_observaciones.csv"
CORRELACION = RESULTADOS_DIR / f"{PREFIJO}correlacion_observaciones_acierto.csv"

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
# GRUPOS DE OBSERVACIONES
# ======================================================================

GRUPOS = [
    ("<100", 0, 99),
    ("100-249", 100, 249),
    ("250-499", 250, 499),
    ("500-999", 500, 999),
    (">=1000", 1000, np.inf),
]

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

    x = obtener_curva(fila)

    if len(x) == 0:

        for variable in VARIABLES:
            valores[variable] = np.nan

        return valores

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

        iqr25 = p75 - p25

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

            stetson_K = (
                np.mean(
                    np.abs(delta)
                )
                / denominador
            )

        else:

            stetson_K = 0.0

    else:

        stetson_K = 0.0

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

    valores[
        "standard_deviation"
    ] = std

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
    valores["stetson_K"] = stetson_K
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
                f"  {i:,}"
            )

        resultados.append(
            calcular_features(fila)
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
                f"  {i:,}"
            )

        curva = obtener_curva(
            fila
        )

        resultados.append(
            len(curva)
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
# ENTRENAR MODELOS
# ======================================================================

def entrenar_modelos(
    X_train,
    y_train,
    X_test,
    y_test
):

    # --------------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------------

    inicio = time.time()

    rf = RandomForestClassifier(
        n_estimators=150,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_features="sqrt"
    )

    rf.fit(
        X_train[VARIABLES],
        y_train
    )

    pred_rf = rf.predict(
        X_test[VARIABLES]
    )

    rf_acc = accuracy_score(
        y_test,
        pred_rf
    )

    rf_bal = balanced_accuracy_score(
        y_test,
        pred_rf
    )

    rf_time = time.time() - inicio

    # --------------------------------------------------------------
    # LOGISTICA
    # --------------------------------------------------------------

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
        X_train[VARIABLES],
        y_train
    )

    pred_log = log.predict(
        X_test[VARIABLES]
    )

    log_acc = accuracy_score(
        y_test,
        pred_log
    )

    log_bal = balanced_accuracy_score(
        y_test,
        pred_log
    )

    log_time = time.time() - inicio

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
# ASIGNAR GRUPO
# ======================================================================

def asignar_grupo(n):

    if n < 100:
        return "<100"

    if n < 250:
        return "100-249"

    if n < 500:
        return "250-499"

    if n < 1000:
        return "500-999"

    return ">=1000"


# ======================================================================
# RESULTADOS POR GRUPO
# ======================================================================

def resultados_por_grupo(
    n_obs,
    y_real,
    pred_rf,
    pred_log
):

    df = pd.DataFrame({

        "n_observaciones": n_obs,

        "clase": y_real.values,

        "correcto_rf": (
            y_real.values
            == pred_rf
        ),

        "correcto_log": (
            y_real.values
            == pred_log
        )

    })

    df["grupo"] = (
        df["n_observaciones"]
        .apply(asignar_grupo)
    )

    filas = []

    for nombre, minimo, maximo in GRUPOS:

        sub = df[
            df["grupo"] == nombre
        ]

        if len(sub) == 0:

            filas.append({
                "grupo": nombre,
                "N_objetos": 0,
                "observaciones_media": np.nan,
                "observaciones_mediana": np.nan,
                "RF_accuracy": np.nan,
                "RF_balanced": np.nan,
                "LOG_accuracy": np.nan,
                "LOG_balanced": np.nan
            })

            continue

        rf_acc = (
            sub["correcto_rf"]
            .mean()
        )

        log_acc = (
            sub["correcto_log"]
            .mean()
        )

        # Balanced accuracy dentro del grupo
        rf_bal = balanced_accuracy_score(
            sub["clase"],
            pred_rf[sub.index]
        )

        log_bal = balanced_accuracy_score(
            sub["clase"],
            pred_log[sub.index]
        )

        filas.append({

            "grupo": nombre,

            "N_objetos": len(sub),

            "observaciones_media":
                sub[
                    "n_observaciones"
                ].mean(),

            "observaciones_mediana":
                sub[
                    "n_observaciones"
                ].median(),

            "RF_accuracy":
                rf_acc,

            "RF_balanced":
                rf_bal,

            "LOG_accuracy":
                log_acc,

            "LOG_balanced":
                log_bal
        })

    return pd.DataFrame(filas), df


# ======================================================================
# RESULTADOS POR CLASE Y GRUPO
# ======================================================================

def resultados_por_clase_grupo(df):

    filas = []

    for (grupo, clase), sub in (
        df.groupby(
            ["grupo", "clase"]
        )
    ):

        if len(sub) == 0:
            continue

        filas.append({

            "grupo": grupo,

            "clase": clase,

            "N_objetos": len(sub),

            "observaciones_media":
                sub[
                    "n_observaciones"
                ].mean(),

            "observaciones_mediana":
                sub[
                    "n_observaciones"
                ].median(),

            "RF_accuracy":
                sub[
                    "correcto_rf"
                ].mean(),

            "LOG_accuracy":
                sub[
                    "correcto_log"
                ].mean()
        })

    return pd.DataFrame(filas)


# ======================================================================
# CORRELACION
# ======================================================================

def calcular_correlacion(
    n_obs,
    pred_rf,
    pred_log,
    y_test
):

    acierto_rf = (
        pred_rf == y_test.values
    ).astype(int)

    acierto_log = (
        pred_log == y_test.values
    ).astype(int)

    corr_rf = np.corrcoef(
        n_obs,
        acierto_rf
    )[0, 1]

    corr_log = np.corrcoef(
        n_obs,
        acierto_log
    )[0, 1]

    return corr_rf, corr_log


# ======================================================================
# MAIN
# ======================================================================

def main():

    inicio_total = time.time()

    print()
    print("=" * 70)
    print(
        "ANALISIS DE DEPENDENCIA DEL NUMERO DE OBSERVACIONES"
    )
    print(
        "StarEmbed / ZTF"
    )
    print("=" * 70)

    print()
    print("Objetivo:")
    print(
        "Comprobar si el rendimiento de clasificación"
    )
    print(
        "depende del número de observaciones de cada curva."
    )

    print()
    print("Grupos:")

    for nombre, minimo, maximo in GRUPOS:
        print(
            f"  {nombre}"
        )

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

    # FEATURES

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

    # NUMERO DE OBSERVACIONES

    # ==============================================================

    print()
    print("=" * 70)
    print(
        "EXTRACCION DEL NUMERO DE OBSERVACIONES"
    )
    print("=" * 70)

    print()
    print("TRAIN")

    n_obs_train = (
        extraer_n_observaciones(
            train
        )
    )

    print()
    print("TEST")

    n_obs_test = (
        extraer_n_observaciones(
            test
        )
    )

    # ==============================================================

    # CONTROL DE DATOS

    # ==============================================================

    print()
    print("=" * 70)
    print(
        "CONTROL DEL NUMERO DE OBSERVACIONES"
    )
    print("=" * 70)

    print()
    print(
        "TRAIN:"
    )

    print(
        pd.Series(
            n_obs_train
        ).describe()
    )

    print()
    print(
        "TEST:"
    )

    print(
        pd.Series(
            n_obs_test
        ).describe()
    )

    # ==============================================================

    # DISTRIBUCION POR GRUPOS

    # ==============================================================

    print()
    print("=" * 70)
    print(
        "DISTRIBUCION POR NUMERO DE OBSERVACIONES"
    )
    print("=" * 70)

    grupos_test = pd.Series(
        n_obs_test
    ).apply(
        asignar_grupo
    )

    for nombre, minimo, maximo in GRUPOS:

        cantidad = (
            grupos_test == nombre
        ).sum()

        porcentaje = (
            cantidad /
            len(n_obs_test)
            * 100
        )

        print(
            f"{nombre:>10}: "
            f"{cantidad:5d} "
            f"({porcentaje:6.2f} %)"
        )

    # ==============================================================

    # LIMPIEZA

    # ==============================================================

    X_train, X_test = limpiar(
        X_train,
        X_test
    )

    # ==============================================================

    # ENTRENAMIENTO

    # ==============================================================

    print()
    print("=" * 70)
    print(
        "ENTRENAMIENTO DE MODELOS"
    )
    print("=" * 70)

    print()
    print(
        "Random Forest..."
    )

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
        y_test
    )

    print(
        f"RF terminado: "
        f"{metricas['RF_TIME']:.2f} s"
    )

    print()
    print(
        "Logística..."
    )

    print(
        f"LOG terminado: "
        f"{metricas['LOG_TIME']:.2f} s"
    )

    # ==============================================================

    # RESULTADOS GENERALES

    # ==============================================================

    print()
    print("=" * 70)
    print(
        "RESULTADOS GENERALES"
    )
    print("=" * 70)

    print()
    print(
        "RF  "
        f"accuracy={metricas['RF_ACC']:.6f} "
        f"balanced={metricas['RF_BAL']:.6f}"
    )

    print(
        "LOG "
        f"accuracy={metricas['LOG_ACC']:.6f} "
        f"balanced={metricas['LOG_BAL']:.6f}"
    )

    # ==============================================================

    # RESULTADOS POR GRUPO

    # ==============================================================

    resultados_grupo, df_detalle = (
        resultados_por_grupo(
            n_obs_test,
            y_test,
            pred_rf,
            pred_log
        )
    )

    print()
    print("=" * 70)
    print(
        "RESULTADOS POR NUMERO DE OBSERVACIONES"
    )
    print("=" * 70)

    for _, fila in resultados_grupo.iterrows():

        print()
        print(
            fila["grupo"]
        )

        print(
            f"  N objetos       : "
            f"{int(fila['N_objetos']):,}"
        )

        print(
            f"  Observaciones   : "
            f"media={fila['observaciones_media']:.1f} "
            f"mediana={fila['observaciones_mediana']:.1f}"
        )

        if not np.isnan(
            fila["RF_accuracy"]
        ):

            print(
                f"  RF              : "
                f"acc={fila['RF_accuracy']:.6f} "
                f"balanced={fila['RF_balanced']:.6f}"
            )

            print(
                f"  LOG             : "
                f"acc={fila['LOG_accuracy']:.6f} "
                f"balanced={fila['LOG_balanced']:.6f}"
            )

        else:

            print(
                "  Sin objetos"
            )

    resultados_grupo.to_csv(
        RESULTADOS_OBS,
        index=False
    )

    # ==============================================================

    # RESULTADOS POR CLASE

    # ==============================================================

    print()
    print("=" * 70)
    print(
        "RESULTADOS POR CLASE Y NUMERO DE OBSERVACIONES"
    )
    print("=" * 70)

    resultados_clase = (
        resultados_por_clase_grupo(
            df_detalle
        )
    )

    for _, fila in resultados_clase.iterrows():

        print(
            f"{fila['grupo']:<10} "
            f"{fila['clase']:<8} "
            f"N={int(fila['N_objetos']):5d} "
            f"obs_mediana={fila['observaciones_mediana']:7.1f} "
            f"RF={fila['RF_accuracy']:.4f} "
            f"LOG={fila['LOG_accuracy']:.4f}"
        )

    resultados_clase.to_csv(
        RESULTADOS_CLASE_OBS,
        index=False
    )

    # ==============================================================

    # CORRELACION

    # ==============================================================

    print()
    print("=" * 70)
    print(
        "CORRELACION ENTRE OBSERVACIONES Y ACIERTO"
    )
    print("=" * 70)

    corr_rf, corr_log = (
        calcular_correlacion(
            n_obs_test,
            pred_rf,
            pred_log,
            y_test
        )
    )

    print()
    print(
        f"Correlación RF  : "
        f"{corr_rf:.6f}"
    )

    print(
        f"Correlación LOG : "
        f"{corr_log:.6f}"
    )

    pd.DataFrame([
        {
            "modelo": "RF",
            "correlacion_observaciones_acierto":
                corr_rf
        },
        {
            "modelo": "LOG",
            "correlacion_observaciones_acierto":
                corr_log
        }
    ]).to_csv(
        CORRELACION,
        index=False
    )

    # ==============================================================

    # DIAGNOSTICO

    # ==============================================================

    print()
    print("=" * 70)
    print(
        "DIAGNOSTICO"
    )
    print("=" * 70)

    print()
    print(
        f"RF balanced global  : "
        f"{metricas['RF_BAL']:.6f}"
    )

    print(
        f"LOG balanced global : "
        f"{metricas['LOG_BAL']:.6f}"
    )

    print()

    print(
        "Interpretación de la correlación:"
    )

    print(
        "  Cerca de 0  -> poca relación lineal"
    )

    print(
        "  Positiva    -> más observaciones tienden a mejorar"
    )

    print(
        "  Negativa    -> más observaciones tienden a empeorar"
    )

    # ==============================================================

    # ARCHIVOS

    # ==============================================================

    print()
    print(
        "Archivos generados:"
    )

    print(
        " ",
        RESULTADOS_OBS
    )

    print(
        " ",
        RESULTADOS_CLASE_OBS
    )

    print(
        " ",
        CORRELACION
    )

    print()
    print("=" * 70)
    print(
        "FIN"
    )
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