# -*- coding: utf-8 -*-

"""
======================================================================
16c - DIAGNOSTICO SISTEMATICO DE PADDING / TIMESTAMPS DUPLICADOS
======================================================================

Motivacion
----------
El 16b confirmo que las curvas ya estan ordenadas cronologicamente
(0 inversiones en 8000/8000 objetos), asi que ETA es de fiar tal
como esta calculada en 20-23. Pero el mismo 16b encontro un patron
inesperado: 7.870/8.000 objetos (98.4%) tienen puntos con mjd
EXACTAMENTE repetido dentro del mismo objeto, con una media de ~99
repeticiones por objeto y un maximo de 2.429.

Como las curvas ya estan ordenadas de forma no decreciente, cualquier
grupo de mjd identicos es necesariamente CONTIGUO en el array. Este
script comprueba, para cada objeto:

  1. Si existe una "cola" final de mjd repetido (los ultimos k puntos
     comparten el mismo mjd).
  2. Si dentro de esa cola el valor de 'target' (magnitud) es tambien
     constante. Si target NO varia mientras mjd tampoco varia, es la
     firma tipica de padding: el pipeline que genero el parquet relleno
     la secuencia repitiendo el ultimo punto real hasta una longitud
     fija.
  3. Cuenta ademas los empates que NO estan al final (empates
     intermedios), que si existen en volumen relevante indicarian otra
     causa distinta al padding por relleno de cola.

A partir de ahi:

  - Calcula n_observaciones_efectivas = longitud real sin la cola de
    relleno.
  - Recalcula las 12 variables SOLO sobre la parte real de la curva
    (sin la cola), y compara contra las variables calculadas sobre el
    array completo (que es lo que hace el pipeline actual, 20-23).
  - Repite el analisis de dependencia con el numero de observaciones
    (como en 22/23) pero usando n_observaciones_efectivas en vez de
    len(target), para ver si las conclusiones cambian.
  - Reentrena RF y logistica con las features "sin padding" y compara
    balanced accuracy contra las features originales.

Notas
-----
- Sigue usando train.iloc[:MAX_TRAIN] / test.iloc[:MAX_TEST] para ser
  comparable con el resto del pipeline (20-23) y con el propio 16b.
- Solo se analiza la banda prioritaria (misma logica que 20-23:
  intenta 'r', si no existe usa 'g', si no 'i').
- Si un objeto no tiene cola de relleno detectable, se queda igual
  que estaba: n_observaciones_efectivas = n_observaciones_original.
======================================================================
"""

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score

warnings.filterwarnings("ignore")


# ======================================================================
# CONFIGURACION
# ======================================================================

# Ruta relativa desde la ubicación del script
BASE_DIR = Path(__file__).resolve().parents[2]  # Sube 2 niveles hasta Astronomia/

# Directorio de resultados dentro de la misma carpeta que el script
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "16c_"

DATA_DIR = BASE_DIR / "dataset" / "ZTF_40k_StarEmbed" / "data"

TRAIN_FILES = [
    DATA_DIR / "train-00000-of-00002.parquet",
    DATA_DIR / "train-00001-of-00002.parquet",
]

TEST_FILE = DATA_DIR / "test-00000-of-00001.parquet"

MAX_TRAIN = 25000
MAX_TEST = 8000

RANDOM_STATE = 42

BANDS_PRIORIDAD = ["r", "g", "i"]

# Tolerancia para considerar dos magnitudes "iguales" dentro de una
# cola de mjd repetido. 0.0 = exigir igualdad exacta (bits identicos
# tras pasar por float64). Si el padding se genera con algun redondeo
# o casteo, sube esto a algo como 1e-9.
TOLERANCIA_TARGET_IGUAL = 0.0

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

VARIABLES_IMPORTANTES_23 = [
    "skew",
    "eta",
    "inter_percentile_range_25",
    "standard_deviation",
    "median_absolute_deviation",
    "stetson_K",
]

GRUPOS_OBS = [
    ("<100", 0, 99),
    ("100-249", 100, 249),
    ("250-499", 250, 499),
    ("500-999", 500, 999),
    (">=1000", 1000, np.inf),
]

SALIDA_DIAGNOSTICO_PADDING = RESULTADOS_DIR / f"{PREFIJO}diagnostico_padding.csv"
SALIDA_COMPARACION_FEATURES = RESULTADOS_DIR / f"{PREFIJO}comparacion_features.csv"
SALIDA_GRUPOS_RECALCULADOS_ORIGINAL = RESULTADOS_DIR / f"{PREFIJO}grupos_nobs_recalculados_original.csv"
SALIDA_GRUPOS_RECALCULADOS_EFECTIVO = RESULTADOS_DIR / f"{PREFIJO}grupos_nobs_recalculados_efectivo.csv"
SALIDA_RESULTADOS_CLASIFICACION = RESULTADOS_DIR / f"{PREFIJO}resultados_clasificacion.csv"
SALIDA_CAMBIO_DE_GRUPO = RESULTADOS_DIR / f"{PREFIJO}cambio_de_grupo_observaciones.csv"


def banner(texto):
    print()
    print("=" * 70)
    print(texto)
    print("=" * 70)


# ======================================================================
# CARGA DE DATOS
# ======================================================================

def cargar_datos():

    banner("CARGA DE DATOS")

    frames = []

    for fichero in TRAIN_FILES:
        print("Cargando TRAIN:", fichero)

        if not fichero.exists():
            raise FileNotFoundError(fichero)

        df = pd.read_parquet(fichero)
        frames.append(df)
        print("  Objetos:", f"{len(df):,}")

    train = pd.concat(frames, ignore_index=True)

    print()
    print("Cargando TEST:", TEST_FILE)

    if not TEST_FILE.exists():
        raise FileNotFoundError(TEST_FILE)

    test = pd.read_parquet(TEST_FILE)

    train = train.iloc[:MAX_TRAIN].copy()
    test = test.iloc[:MAX_TEST].copy()

    print()
    print("TRAIN utilizado:", f"{len(train):,}")
    print("TEST utilizado :", f"{len(test):,}")

    return train, test


# ======================================================================
# ACCESO A LA BANDA / CURVA
# ======================================================================

def obtener_banda(bands):

    if bands is None:
        return None

    try:
        if isinstance(bands, dict):
            for b in BANDS_PRIORIDAD:
                if b in bands and bands[b] is not None:
                    return bands[b]
        return None
    except Exception:
        return None


def obtener_curva_cruda(fila):

    bands = fila.get("bands_data", None)
    banda = obtener_banda(bands)

    if banda is None:
        return np.array([]), np.array([])

    target = banda.get("target", [])
    mjd = banda.get("mjd", [])

    if target is None:
        target = []
    if mjd is None:
        mjd = []

    try:
        x = np.asarray(target, dtype=float)
    except Exception:
        x = np.array([], dtype=float)

    try:
        t = np.asarray(mjd, dtype=float)
    except Exception:
        t = np.array([], dtype=float)

    n = min(len(x), len(t))
    x = x[:n]
    t = t[:n]

    mask = np.isfinite(x) & np.isfinite(t)
    x = x[mask]
    t = t[mask]

    return x, t


# ======================================================================
# DETECCION DE PADDING EN LA COLA
# ======================================================================

def detectar_padding_cola(x, t):
    """
    Como la curva ya esta ordenada de forma no decreciente por mjd
    (confirmado en 16b), cualquier tramo de mjd repetido es contiguo.
    Esta funcion mira el tramo final: si los ultimos k puntos
    comparten mjd Y target es constante en ese tramo, se considera
    cola de relleno (padding).

    Devuelve un diccionario con:
        n_original            -> longitud total del array
        n_relleno_cola        -> cuantos puntos finales son relleno
        n_efectivo            -> n_original - n_relleno_cola
        cola_target_constante -> True/False
        empates_intermedios   -> empates de mjd que NO forman parte
                                  de la cola final (posible otra causa)
    """

    n_original = len(t)

    resultado = {
        "n_original": n_original,
        "n_relleno_cola": 0,
        "n_efectivo": n_original,
        "cola_target_constante": False,
        "empates_intermedios": 0,
    }

    if n_original < 2:
        return resultado

    # ------------------------------------------------------------------
    # Tamano del tramo final con mjd == mjd[-1]
    # ------------------------------------------------------------------

    ultimo_mjd = t[-1]

    i = n_original - 1
    while i > 0 and t[i - 1] == ultimo_mjd:
        i -= 1

    longitud_cola = n_original - i  # incluye el ultimo punto

    if longitud_cola > 1:

        objetivo = x[i]
        cola_target = x[i:n_original]

        constante = np.all(
            np.abs(cola_target - objetivo) <= TOLERANCIA_TARGET_IGUAL
        )

        if constante:
            resultado["n_relleno_cola"] = longitud_cola - 1
            resultado["n_efectivo"] = n_original - (longitud_cola - 1)
            resultado["cola_target_constante"] = True

    # ------------------------------------------------------------------
    # Empates que no forman parte de la cola de relleno detectada
    # ------------------------------------------------------------------

    limite_no_cola = n_original - longitud_cola if longitud_cola > 1 else n_original

    if limite_no_cola > 1:
        dt_resto = np.diff(t[:limite_no_cola])
        resultado["empates_intermedios"] = int(np.sum(dt_resto == 0))

    return resultado


def diagnosticar_padding(df):

    banner("DIAGNOSTICO DE PADDING (COLA DE MJD REPETIDO)")

    filas = []
    n = len(df)

    for i, (_, fila) in enumerate(df.iterrows()):

        if i % 5000 == 0:
            print(f"  {i:,}/{n:,}")

        x, t = obtener_curva_cruda(fila)
        diag = detectar_padding_cola(x, t)
        filas.append(diag)

    resumen = pd.DataFrame(filas)

    n_con_padding = int((resumen["n_relleno_cola"] > 0).sum())
    n_objetos = len(resumen)

    print()
    print(f"Objetos analizados: {n_objetos:,}")
    print(
        f"Objetos con cola de relleno detectada (target constante): "
        f"{n_con_padding:,} ({100.0 * n_con_padding / n_objetos:.2f}%)"
    )

    if n_con_padding > 0:
        afectados = resumen[resumen["n_relleno_cola"] > 0]
        print()
        print("Puntos de relleno por objeto afectado:")
        print(afectados["n_relleno_cola"].describe().to_string())

        print()
        print(
            "Fraccion de la curva que es relleno (por objeto afectado):"
        )
        frac = afectados["n_relleno_cola"] / afectados["n_original"]
        print(frac.describe().to_string())

    print()
    print("Empates intermedios (fuera de la cola) — estadisticas globales:")
    print(resumen["empates_intermedios"].describe().to_string())

    n_con_empates_intermedios = int(
        (resumen["empates_intermedios"] > 0).sum()
    )
    print()
    print(
        f"Objetos con empates intermedios (no explicados por la cola): "
        f"{n_con_empates_intermedios:,} "
        f"({100.0 * n_con_empates_intermedios / n_objetos:.2f}%)"
    )

    resumen.to_csv(SALIDA_DIAGNOSTICO_PADDING, index=False)
    print()
    print("Diagnostico guardado en:", SALIDA_DIAGNOSTICO_PADDING)

    return resumen


# ======================================================================
# CALCULO DE FEATURES (12 VARIABLES, IDENTICO A 20/21/22/23)
# ======================================================================

def _stats_basicas(x):

    media = np.mean(x)
    mediana = np.median(x)
    std = np.std(x)
    mad = np.median(np.abs(x - mediana))

    minimo = np.min(x)
    maximo = np.max(x)
    amplitud = (maximo - minimo) / 2.0

    if abs(media) > 1e-12:
        percent_amplitude = amplitud / abs(media)
    else:
        percent_amplitude = 0.0

    try:
        p25 = np.percentile(x, 12.5)
        p75 = np.percentile(x, 87.5)
        iqr25 = p75 - p25
    except Exception:
        iqr25 = np.nan

    if std > 1e-12:
        z = (x - media) / std
        skew = np.mean(z ** 3)
        kurtosis = np.mean(z ** 4) - 3.0
    else:
        skew = 0.0
        kurtosis = 0.0

    if std > 1e-12 and len(x) > 1:
        delta = (x - media) / std
        denominador = np.sqrt(np.mean(delta ** 2))
        if denominador > 1e-12:
            stetson_K = np.mean(np.abs(delta)) / denominador
        else:
            stetson_K = 0.0
    else:
        stetson_K = 0.0

    if std > 1e-12:
        chi2 = np.sum(((x - media) / std) ** 2) / max(len(x) - 1, 1)
    else:
        chi2 = 0.0

    return {
        "median": mediana,
        "standard_deviation": std,
        "median_absolute_deviation": mad,
        "amplitude": amplitud,
        "percent_amplitude": percent_amplitude,
        "inter_percentile_range_25": iqr25,
        "skew": skew,
        "kurtosis": kurtosis,
        "stetson_K": stetson_K,
        "chi2": chi2,
    }


def _eta(x):

    if len(x) > 1 and np.std(x) > 1e-12:
        diferencias = np.diff(x)
        std = np.std(x)
        return np.sum(diferencias ** 2) / ((len(x) - 1) * std ** 2)

    return 0.0


def _maximum_slope(x, t):

    if len(t) > 1:
        dt = np.diff(t)
        dy = np.diff(x)
        valid = dt > 0
        if np.any(valid):
            slopes = np.abs(dy[valid] / dt[valid])
            return float(np.max(slopes))
        return 0.0

    return 0.0


def calcular_features_desde_arrays(x, t):

    if len(x) == 0:
        return {v: np.nan for v in VARIABLES}

    valores = _stats_basicas(x)
    valores["eta"] = _eta(x)
    valores["maximum_slope"] = _maximum_slope(x, t)

    return valores


def extraer_features_y_diagnostico(df, recorte_relleno):
    """
    recorte_relleno: dict indexado por posicion (0..len(df)-1) con la
    salida de detectar_padding_cola para cada fila, ya calculada
    previamente (para no repetir el trabajo).

    Devuelve dos DataFrames de features: uno "original" (curva
    completa, igual que hace el pipeline actual) y otro "sin_relleno"
    (recortando la cola de padding detectada).
    """

    filas_original = []
    filas_sin_relleno = []
    n_obs_original = []
    n_obs_efectivas = []

    n = len(df)

    for i, (_, fila) in enumerate(df.iterrows()):

        if i % 5000 == 0:
            print(f"  {i:,}/{n:,}")

        x, t = obtener_curva_cruda(fila)

        diag = recorte_relleno[i]
        n_efectivo = diag["n_efectivo"]

        filas_original.append(
            calcular_features_desde_arrays(x, t)
        )

        x_rec = x[:n_efectivo]
        t_rec = t[:n_efectivo]

        filas_sin_relleno.append(
            calcular_features_desde_arrays(x_rec, t_rec)
        )

        n_obs_original.append(len(x))
        n_obs_efectivas.append(n_efectivo)

    X_original = pd.DataFrame(filas_original)
    X_sin_relleno = pd.DataFrame(filas_sin_relleno)

    return (
        X_original,
        X_sin_relleno,
        np.array(n_obs_original),
        np.array(n_obs_efectivas),
    )


# ======================================================================
# LIMPIEZA
# ======================================================================

def limpiar(X_train, X_test):

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    medianas = X_train.median()

    X_train = X_train.fillna(medianas).fillna(0.0)
    X_test = X_test.fillna(medianas).fillna(0.0)

    return X_train, X_test


# ======================================================================
# CLASIFICACION
# ======================================================================

def evaluar(X_train, y_train, X_test, y_test):

    resultados = {}

    inicio = time.time()

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_features="sqrt",
    )

    rf.fit(X_train[VARIABLES], y_train)
    pred_rf = rf.predict(X_test[VARIABLES])

    resultados["RF_ACC"] = accuracy_score(y_test, pred_rf)
    resultados["RF_BAL"] = balanced_accuracy_score(y_test, pred_rf)
    resultados["RF_TIME"] = time.time() - inicio

    inicio = time.time()

    log = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
    )

    log.fit(X_train[VARIABLES], y_train)
    pred_log = log.predict(X_test[VARIABLES])

    resultados["LOG_ACC"] = accuracy_score(y_test, pred_log)
    resultados["LOG_BAL"] = balanced_accuracy_score(y_test, pred_log)
    resultados["LOG_TIME"] = time.time() - inicio

    return resultados


# ======================================================================
# GRUPOS DE OBSERVACIONES (RECALCULADOS)
# ======================================================================

def asignar_grupo(n):

    for nombre, minimo, maximo in GRUPOS_OBS:
        if minimo <= n <= maximo:
            return nombre

    return None


def resumen_por_grupo(n_obs, X, variables):

    grupo = np.array([asignar_grupo(n) for n in n_obs])

    filas = []

    for nombre, _, _ in GRUPOS_OBS:

        mask = grupo == nombre

        if mask.sum() == 0:
            continue

        fila = {"grupo": nombre, "N": int(mask.sum())}

        for var in variables:
            fila[f"{var}_media"] = X.loc[mask, var].mean()

        filas.append(fila)

    return pd.DataFrame(filas)


# ======================================================================
# MAIN
# ======================================================================

def main():

    inicio_total = time.time()

    banner("16c - DIAGNOSTICO DE PADDING Y RECALCULO SIN RELLENO")

    train, test = cargar_datos()

    if "class_str" not in train.columns or "class_str" not in test.columns:
        raise RuntimeError("Falta la columna class_str.")

    y_train = train["class_str"].astype(str)
    y_test = test["class_str"].astype(str)

    # ------------------------------------------------------------------
    # PASO 1: DIAGNOSTICO DE PADDING SOBRE TEST Y TRAIN
    # ------------------------------------------------------------------

    print()
    print(">>> Diagnostico sobre TEST")
    diag_test = diagnosticar_padding(test)

    print()
    print(">>> Diagnostico sobre TRAIN")
    diag_train = diagnosticar_padding(train)

    # ------------------------------------------------------------------
    # PASO 2: RECALCULAR FEATURES CON Y SIN RELLENO
    # ------------------------------------------------------------------

    banner("RECALCULO DE FEATURES: ORIGINAL vs SIN_RELLENO")

    diag_test_dict = diag_test.to_dict("records")
    diag_train_dict = diag_train.to_dict("records")

    print()
    print("TRAIN...")
    (
        X_train_original,
        X_train_sin_relleno,
        n_obs_train_original,
        n_obs_train_efectivo,
    ) = extraer_features_y_diagnostico(train, diag_train_dict)

    print()
    print("TEST...")
    (
        X_test_original,
        X_test_sin_relleno,
        n_obs_test_original,
        n_obs_test_efectivo,
    ) = extraer_features_y_diagnostico(test, diag_test_dict)

    X_train_original, X_test_original = limpiar(
        X_train_original, X_test_original
    )
    X_train_sin_relleno, X_test_sin_relleno = limpiar(
        X_train_sin_relleno, X_test_sin_relleno
    )

    # ------------------------------------------------------------------
    # PASO 3: COMPARAR LAS VARIABLES IMPORTANTES DEL 23
    # ------------------------------------------------------------------

    banner("COMPARACION DE FEATURES IMPORTANTES: ORIGINAL vs SIN_RELLENO")

    filas_comp = []

    for var in VARIABLES_IMPORTANTES_23:

        original = X_test_original[var].values
        sin_relleno = X_test_sin_relleno[var].values

        diff_abs = np.abs(original - sin_relleno)

        filas_comp.append({
            "variable": var,
            "diff_media": np.nanmean(diff_abs),
            "diff_mediana": np.nanmedian(diff_abs),
            "diff_maxima": np.nanmax(diff_abs),
            "objetos_con_diferencia": int(
                np.sum(diff_abs > 1e-9)
            ),
            "pct_objetos_con_diferencia": round(
                100.0 * np.sum(diff_abs > 1e-9) / len(diff_abs), 2
            ),
        })

    tabla_comp = pd.DataFrame(filas_comp)
    print(tabla_comp.to_string(index=False))

    tabla_comp.to_csv(SALIDA_COMPARACION_FEATURES, index=False)

    # ------------------------------------------------------------------
    # PASO 4: GRUPOS DE OBSERVACIONES RECALCULADOS (COMO EN 22/23)
    # ------------------------------------------------------------------

    banner("MEDIAS POR GRUPO DE OBSERVACIONES: ORIGINAL vs EFECTIVO")

    print()
    print("Usando n_observaciones ORIGINAL (como hoy en 22/23):")
    resumen_original = resumen_por_grupo(
        n_obs_test_original, X_test_original, VARIABLES_IMPORTANTES_23
    )
    print(resumen_original.to_string(index=False))

    print()
    print("Usando n_observaciones EFECTIVAS (sin la cola de relleno):")
    resumen_efectivo = resumen_por_grupo(
        n_obs_test_efectivo, X_test_sin_relleno, VARIABLES_IMPORTANTES_23
    )
    print(resumen_efectivo.to_string(index=False))

    resumen_original.to_csv(
        SALIDA_GRUPOS_RECALCULADOS_ORIGINAL,
        index=False,
    )
    resumen_efectivo.to_csv(
        SALIDA_GRUPOS_RECALCULADOS_EFECTIVO,
        index=False,
    )

    # ------------------------------------------------------------------
    # PASO 5: ¿CAMBIA UN OBJETO DE GRUPO AL QUITAR EL RELLENO?
    # ------------------------------------------------------------------

    banner("OBJETOS QUE CAMBIAN DE GRUPO DE OBSERVACIONES")

    grupo_original = np.array(
        [asignar_grupo(n) for n in n_obs_test_original]
    )
    grupo_efectivo = np.array(
        [asignar_grupo(n) for n in n_obs_test_efectivo]
    )

    cambia = grupo_original != grupo_efectivo

    tabla_cambio = pd.DataFrame({
        "n_obs_original": n_obs_test_original,
        "n_obs_efectivo": n_obs_test_efectivo,
        "grupo_original": grupo_original,
        "grupo_efectivo": grupo_efectivo,
        "cambia_de_grupo": cambia,
    })

    print(
        f"Objetos que cambian de grupo al quitar el relleno: "
        f"{int(cambia.sum()):,} / {len(cambia):,} "
        f"({100.0 * cambia.sum() / len(cambia):.2f}%)"
    )

    tabla_cambio.to_csv(SALIDA_CAMBIO_DE_GRUPO, index=False)

    # ------------------------------------------------------------------
    # PASO 6: RE-ENTRENAR Y COMPARAR CLASIFICACION
    # ------------------------------------------------------------------

    banner("CLASIFICACION: ORIGINAL vs SIN_RELLENO")

    print()
    print("Entrenando con features ORIGINALES (curva completa)...")
    resultados_original = evaluar(
        X_train_original, y_train, X_test_original, y_test
    )
    print(
        f"RF  : acc={resultados_original['RF_ACC']:.6f}  "
        f"balanced={resultados_original['RF_BAL']:.6f}"
    )
    print(
        f"LOG : acc={resultados_original['LOG_ACC']:.6f}  "
        f"balanced={resultados_original['LOG_BAL']:.6f}"
    )

    print()
    print("Entrenando con features SIN RELLENO (cola de padding recortada)...")
    resultados_sin_relleno = evaluar(
        X_train_sin_relleno, y_train, X_test_sin_relleno, y_test
    )
    print(
        f"RF  : acc={resultados_sin_relleno['RF_ACC']:.6f}  "
        f"balanced={resultados_sin_relleno['RF_BAL']:.6f}"
    )
    print(
        f"LOG : acc={resultados_sin_relleno['LOG_ACC']:.6f}  "
        f"balanced={resultados_sin_relleno['LOG_BAL']:.6f}"
    )

    tabla_resultados = pd.DataFrame({
        "original": resultados_original,
        "sin_relleno": resultados_sin_relleno,
    }).T
    tabla_resultados.index.name = "modo"

    tabla_resultados.to_csv(SALIDA_RESULTADOS_CLASIFICACION)

    # ------------------------------------------------------------------
    # RESUMEN FINAL
    # ------------------------------------------------------------------

    banner("RESUMEN E INTERPRETACION")

    n_con_padding_test = int((diag_test["n_relleno_cola"] > 0).sum())

    print()
    print(
        f"Objetos de TEST con cola de relleno confirmada "
        f"(mjd repetido + target constante): "
        f"{n_con_padding_test:,} / {len(diag_test):,}"
    )

    print()
    print("Como leer los resultados:")
    print()
    print(
        "- Si 'objetos_con_diferencia' es bajo (%) para las variables "
        "importantes del 23 y las balanced accuracy ORIGINAL/SIN_RELLENO "
        "son casi identicas: el padding existe pero no esta distorsionando "
        "de forma relevante ni las features ni el clasificador. Podeis "
        "seguir usando el pipeline actual sin cambios."
    )
    print(
        "- Si las diferencias son notables (features que cambian en un "
        "porcentaje alto de objetos, o balanced accuracy claramente "
        "distinta): el padding SI esta afectando a los resultados de "
        "21/22/23, y conviene adoptar 'n_observaciones_efectivas' y las "
        "features SIN_RELLENO como version de referencia de aqui en "
        "adelante, recalculando retroactivamente esos experimentos."
    )
    print(
        "- Revisa tambien 'objetos que cambian de grupo de observaciones': "
        "si un porcentaje relevante de objetos migra de bucket "
        "(por ejemplo de '>=1000' a '250-499' al quitar el relleno), la "
        "tabla de dependencia con el numero de observaciones del "
        "experimento 22 dejaria de ser fiable tal como esta."
    )
    print(
        "- Si 'empates_intermedios' es alto en muchos objetos (no solo "
        "la cola final), hay otra causa ademas del padding que conviene "
        "investigar aparte (por ejemplo, fusion de multiples exposiciones "
        "bajo el mismo mjd, o precision limitada del propio mjd)."
    )

    print()
    print("Archivos guardados:")
    print(" ", SALIDA_DIAGNOSTICO_PADDING)
    print(" ", SALIDA_COMPARACION_FEATURES)
    print(" ", SALIDA_GRUPOS_RECALCULADOS_ORIGINAL)
    print(" ", SALIDA_GRUPOS_RECALCULADOS_EFECTIVO)
    print(" ", SALIDA_CAMBIO_DE_GRUPO)
    print(" ", SALIDA_RESULTADOS_CLASIFICACION)

    print()
    print(f"Tiempo total: {time.time() - inicio_total:.2f} segundos")


if __name__ == "__main__":
    main()