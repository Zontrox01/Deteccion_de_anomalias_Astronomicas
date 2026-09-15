# ======================================================================
# 23_estabilidad_features_por_clase.py
# ======================================================================

import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

# ======================================================================
# CONFIGURACION
# ======================================================================

# Ruta relativa desde la ubicación del script
BASE_DIR = Path(__file__).resolve().parents[2]  # Sube 2 niveles hasta Astronomia/

# Directorio de resultados dentro de la misma carpeta que el script
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "23_"

DATA_DIR = BASE_DIR / "dataset" / "ZTF_40k_StarEmbed" / "data"

TRAIN_FILES = [
    DATA_DIR / "train-00000-of-00002.parquet",
    DATA_DIR / "train-00001-of-00002.parquet",
]

TEST_FILE = DATA_DIR / "test-00000-of-00001.parquet"

MAX_TRAIN = 25000
MAX_TEST = 8000

RESULTADO_CLASE = RESULTADOS_DIR / f"{PREFIJO}estabilidad_por_clase.csv"
RESULTADO_OBS = RESULTADOS_DIR / f"{PREFIJO}estabilidad_por_observaciones.csv"
RESULTADO_RESUMEN = RESULTADOS_DIR / f"{PREFIJO}resumen_estabilidad.csv"

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

VARIABLES_IMPORTANTES = [
    "skew",
    "eta",
    "inter_percentile_range_25",
    "standard_deviation",
    "median_absolute_deviation",
    "stetson_K",
]

GRUPOS_OBS = [
    "<100",
    "100-249",
    "250-499",
    "500-999",
    ">=1000",
]


# ======================================================================
# UTILIDADES
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

    except Exception:
        pass

    return None


# ======================================================================
# OBTENER CURVA
# ======================================================================

def obtener_curva(fila):

    bands = fila.get(
        "bands_data",
        None
    )

    banda = obtener_banda(bands)

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
# CALCULO DE FEATURES
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
    # IPR
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

            stetson_K = (
                np.mean(
                    np.abs(delta)
                )
                /
                denominador
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
    valores["standard_deviation"] = std
    valores["median_absolute_deviation"] = mad
    valores["amplitude"] = amplitud
    valores["percent_amplitude"] = percent_amplitude
    valores["inter_percentile_range_25"] = iqr25
    valores["skew"] = skew
    valores["kurtosis"] = kurtosis
    valores["stetson_K"] = stetson_K
    valores["eta"] = eta
    valores["chi2"] = chi2
    valores["maximum_slope"] = maximum_slope

    return valores


# ======================================================================
# EXTRAER FEATURES
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
                f"  {i:,}"
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
# GRUPO DE OBSERVACIONES
# ======================================================================

def grupo_observaciones(n):

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
# CARGA
# ======================================================================

def cargar_datos():

    print("=" * 70)
    print("CARGA DE DATOS")
    print("=" * 70)

    frames = []

    for fichero in TRAIN_FILES:

        print()
        print("Cargando TRAIN:")
        print(" ", fichero)

        if not fichero.exists():
            raise FileNotFoundError(fichero)

        df = pd.read_parquet(
            fichero
        )

        frames.append(df)

        print(
            "Objetos:",
            f"{len(df):,}"
        )

    train = pd.concat(
        frames,
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

    train = train.iloc[
        :MAX_TRAIN
    ].copy()

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
# LIMPIEZA
# ======================================================================

def limpiar(X):

    X = (
        X
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    return X


# ======================================================================
# RESUMEN ESTADISTICO
# ======================================================================

def calcular_resumen(
    X,
    clases,
    n_obs
):

    datos = X.copy()

    datos["clase"] = (
        clases.values
    )

    datos["n_observaciones"] = (
        n_obs
    )

    datos["grupo_observaciones"] = [
        grupo_observaciones(n)
        for n in n_obs
    ]

    filas = []

    # ==============================================================
    # POR CLASE
    # ==============================================================

    for clase in sorted(
        datos["clase"].unique()
    ):

        subset = datos[
            datos["clase"] == clase
        ]

        for variable in VARIABLES_IMPORTANTES:

            valores = pd.to_numeric(
                subset[variable],
                errors="coerce"
            ).dropna()

            if len(valores) == 0:
                continue

            filas.append({

                "nivel": "clase",

                "grupo": clase,

                "variable": variable,

                "N": len(valores),

                "media": valores.mean(),

                "mediana": valores.median(),

                "std": valores.std(),

                "min": valores.min(),

                "max": valores.max(),

                "q25": valores.quantile(
                    0.25
                ),

                "q75": valores.quantile(
                    0.75
                )
            })

    # ==============================================================
    # POR GRUPO DE OBSERVACIONES
    # ==============================================================

    for grupo in GRUPOS_OBS:

        subset = datos[
            datos["grupo_observaciones"]
            == grupo
        ]

        for variable in VARIABLES_IMPORTANTES:

            valores = pd.to_numeric(
                subset[variable],
                errors="coerce"
            ).dropna()

            if len(valores) == 0:
                continue

            filas.append({

                "nivel": "observaciones",

                "grupo": grupo,

                "variable": variable,

                "N": len(valores),

                "media": valores.mean(),

                "mediana": valores.median(),

                "std": valores.std(),

                "min": valores.min(),

                "max": valores.max(),

                "q25": valores.quantile(
                    0.25
                ),

                "q75": valores.quantile(
                    0.75
                )
            })

    return pd.DataFrame(
        filas
    )


# ======================================================================
# COMPARACION DE ESTABILIDAD
# ======================================================================

def calcular_estabilidad(
    X,
    clases,
    n_obs
):

    datos = X.copy()

    datos["clase"] = (
        clases.values
    )

    datos["grupo_observaciones"] = [
        grupo_observaciones(n)
        for n in n_obs
    ]

    filas = []

    for clase in sorted(
        datos["clase"].unique()
    ):

        subset_clase = datos[
            datos["clase"] == clase
        ]

        for variable in VARIABLES_IMPORTANTES:

            valores = pd.to_numeric(
                subset_clase[variable],
                errors="coerce"
            ).dropna()

            if len(valores) < 2:
                continue

            media = valores.mean()
            std = valores.std()

            cv = np.nan

            if abs(media) > 1e-12:

                cv = abs(
                    std / media
                )

            filas.append({

                "clase": clase,

                "variable": variable,

                "N": len(valores),

                "media": media,

                "std": std,

                "CV": cv
            })

    return pd.DataFrame(
        filas
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    inicio = time.time()

    print()
    print("=" * 70)
    print(
        "ANALISIS DE ESTABILIDAD DE FEATURES"
    )
    print(
        "StarEmbed / ZTF"
    )
    print("=" * 70)

    print()
    print("Objetivo:")
    print(
        "Comprobar si las features discriminantes"
    )
    print(
        "mantienen un comportamiento estable"
    )
    print(
        "entre clases y cantidades de observaciones."
    )

    print()
    print("Variables analizadas:")

    for variable in VARIABLES_IMPORTANTES:
        print(
            " ",
            variable
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
    # OBSERVACIONES
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
    # LIMPIEZA
    # ==============================================================

    print()
    print("=" * 70)
    print("LIMPIEZA")
    print("=" * 70)

    X_train = limpiar(
        X_train
    )

    X_test = limpiar(
        X_test
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

    # ==============================================================
    # ANALISIS TEST
    # ==============================================================

    print()
    print("=" * 70)
    print(
        "ANALISIS DE ESTABILIDAD - TEST"
    )
    print("=" * 70)

    resumen = calcular_resumen(
        X_test,
        y_test,
        n_obs_test
    )

    estabilidad = calcular_estabilidad(
        X_test,
        y_test,
        n_obs_test
    )

    # ==============================================================
    # MOSTRAR POR CLASE
    # ==============================================================

    print()
    print("=" * 70)
    print(
        "ESTABILIDAD POR CLASE"
    )
    print("=" * 70)

    for clase in sorted(
        y_test.unique()
    ):

        print()
        print(
            f"CLASE: {clase}"
        )

        datos = estabilidad[
            estabilidad["clase"]
            == clase
        ]

        print(
            datos.to_string(
                index=False
            )
        )

    # ==============================================================
    # MOSTRAR POR OBSERVACIONES
    # ==============================================================

    print()
    print("=" * 70)
    print(
        "ESTABILIDAD POR NUMERO DE OBSERVACIONES"
    )
    print("=" * 70)

    datos_obs = resumen[
        resumen["nivel"]
        == "observaciones"
    ]

    for grupo in GRUPOS_OBS:

        print()
        print(
            f"GRUPO: {grupo}"
        )

        datos = datos_obs[
            datos_obs["grupo"]
            == grupo
        ]

        print(
            datos.to_string(
                index=False
            )
        )

    # ==============================================================
    # GUARDAR
    # ==============================================================

    resumen_clase = resumen[
        resumen["nivel"]
        == "clase"
    ]

    resumen_clase.to_csv(
        RESULTADO_CLASE,
        index=False
    )

    datos_obs.to_csv(
        RESULTADO_OBS,
        index=False
    )

    estabilidad.to_csv(
        RESULTADO_RESUMEN,
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
        "El 23 no entrena un nuevo clasificador."
    )

    print(
        "Analiza la estabilidad estadistica"
    )

    print(
        "de las variables que mostraron mayor"
    )

    print(
        "importancia en el experimento 21."
    )

    print()
    print("Archivos generados:")

    print(
        " ",
        RESULTADO_CLASE
    )

    print(
        " ",
        RESULTADO_OBS
    )

    print(
        " ",
        RESULTADO_RESUMEN
    )

    print()
    print("=" * 70)
    print("FIN")
    print("=" * 70)

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