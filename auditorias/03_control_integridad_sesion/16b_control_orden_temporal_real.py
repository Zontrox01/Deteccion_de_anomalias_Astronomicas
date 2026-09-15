# -*- coding: utf-8 -*-

"""
======================================================================
16b - CONTROL DE ORDEN TEMPORAL SOBRE EL ESPACIO REAL DE 12 VARIABLES
======================================================================

Motivacion
----------
El experimento 16 comprobo si barajar el tiempo afectaba a un banco
de 30 estadisticas puramente distribucionales (media, std, percentiles,
skew, MAD por banda). Esas estadisticas son invariantes a la
permutacion del array, asi que ese experimento NO podia detectar
ningun efecto de orden temporal, tuviera este o no.

De las 12 variables usadas en 20/21/22/23, solo dos dependen
realmente del orden en el que llegan los puntos:

    - eta            (usa np.diff(x) sobre el array 'target' tal cual
                       esta almacenado, SIN mirar 'mjd')
    - maximum_slope   (si usa 'mjd', pero sin ordenar 't' antes de
                       diferenciar; el filtro 'dt > 0' sugiere que el
                       orden de almacenamiento no siempre es creciente)

Y eta es, con diferencia, la variable mas importante del modelo
(ablacion 18: quitarla cuesta 3-4 veces mas balanced accuracy que
la siguiente variable en importancia).

Este script responde tres preguntas concretas:

  1. ¿El array 'target' esta ya almacenado en orden cronologico
     (segun 'mjd'), objeto a objeto? Si lo esta, no hay problema.

  2. Si NO lo esta: ¿cuanto cambia eta (y maximum_slope) al
     recalcularlas ordenando explicitamente por mjd, frente a
     calcularlas sobre el orden de almacenamiento actual (que es
     lo que hacen 20/21/22/23)?

  3. ¿Cambia el rendimiento del clasificador (RF y logistica, 12
     variables) si comparamos:
        A) ORDEN_ALMACENAMIENTO -> exactamente lo que hace el pipeline
           actual (20-23).
        B) ORDEN_TEMPORAL_REAL  -> mismas 12 variables, pero eta y
           maximum_slope recalculadas ordenando por mjd primero.
        C) TIEMPOS_BARAJADOS    -> se destruye la relacion real
           target<->mjd barajando mjd de forma aleatoria, y se
           recalculan eta/maximum_slope sobre esa relacion falsa.
           Si el modelo pierde señal aqui, es la prueba de que SI
           existe informacion temporal real siendo aprovechada.

Nota importante
----------------
La carga de datos sigue usando train.iloc[:MAX_TRAIN] / test.iloc[:MAX_TEST]
para ser comparable con 20/21/22/23. Esto NO resuelve la duda pendiente
sobre si conviene barajar antes de truncar (ver auditoria); es un tema
independiente y deliberadamente no se toca aqui para no mezclar dos
controles en un mismo experimento.
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
PREFIJO = "16b_"

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

# Variables cuyo valor SI depende del orden en el que llegan los puntos.
# El resto (median, std, mad, amplitude, percent_amplitude, iqr25,
# skew, kurtosis, stetson_K, chi2) son invariantes a la permutacion
# del array y no hace falta recalcularlas.
VARIABLES_ORDEN_DEPENDIENTE = ["eta", "maximum_slope"]

SALIDA_DIAGNOSTICO = RESULTADOS_DIR / f"{PREFIJO}diagnostico_orden.csv"
SALIDA_COMPARACION_ETA = RESULTADOS_DIR / f"{PREFIJO}comparacion_eta.csv"
SALIDA_RESULTADOS = RESULTADOS_DIR / f"{PREFIJO}resultados_clasificacion.csv"
SALIDA_CORRELACION_ETA_NOBS = RESULTADOS_DIR / f"{PREFIJO}correlacion_eta_nobs.csv"


def linea():
    print("-" * 70)


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
    """
    Devuelve target y mjd EXACTAMENTE como estan almacenados,
    sin ordenar ni filtrar mas alla de quitar NaN/inf de forma
    conjunta (para no descuadrar la correspondencia target<->mjd).
    """

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
# DIAGNOSTICO: ¿ESTA EL ARRAY YA ORDENADO POR TIEMPO?
# ======================================================================

def diagnosticar_orden(df):
    """
    Para cada objeto, comprueba si mjd (tal como esta almacenado)
    es ya no-decreciente. Si lo es, x=target esta en orden
    cronologico y no hay ningun problema. Si no lo es, cuenta
    cuantas inversiones hay (pares consecutivos con dt <= 0).
    """

    banner("DIAGNOSTICO: ¿EL ORDEN DE ALMACENAMIENTO ES CRONOLOGICO?")

    filas = []

    n = len(df)

    for i, (_, fila) in enumerate(df.iterrows()):

        if i % 5000 == 0:
            print(f"  {i:,}/{n:,}")

        x, t = obtener_curva_cruda(fila)

        if len(t) < 2:
            filas.append({
                "n_puntos": len(t),
                "ya_ordenado": True,
                "n_inversiones": 0,
                "frac_inversiones": 0.0,
            })
            continue

        dt = np.diff(t)
        inversiones = int(np.sum(dt < 0))
        empates = int(np.sum(dt == 0))

        filas.append({
            "n_puntos": len(t),
            "ya_ordenado": inversiones == 0,
            "n_inversiones": inversiones,
            "n_empates_temporales": empates,
            "frac_inversiones": inversiones / max(len(dt), 1),
        })

    resumen = pd.DataFrame(filas)

    n_objetos_con_datos = int((resumen["n_puntos"] >= 2).sum())
    n_ya_ordenados = int(resumen["ya_ordenado"].sum())
    n_desordenados = n_objetos_con_datos - n_ya_ordenados

    print()
    print("Objetos con >=2 puntos       :", n_objetos_con_datos)
    print("Objetos ya en orden cronologico:", n_ya_ordenados)
    print("Objetos CON inversiones de tiempo:", n_desordenados)

    if n_objetos_con_datos > 0:
        pct = 100.0 * n_desordenados / n_objetos_con_datos
        print(f"Porcentaje de objetos con desorden temporal: {pct:.2f}%")

    if n_desordenados > 0:
        print()
        print("Estadisticas de la fraccion de inversiones (solo objetos afectados):")
        afectados = resumen[~resumen["ya_ordenado"]]
        print(afectados["frac_inversiones"].describe().to_string())

    resumen.to_csv(SALIDA_DIAGNOSTICO, index=False)
    print()
    print("Diagnostico guardado en:", SALIDA_DIAGNOSTICO)

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


def calcular_features_objeto(fila, modo, rng=None):
    """
    modo:
      'almacenamiento' -> exactamente lo que hace 20/21/22/23 hoy:
                           eta se calcula sobre 'target' tal cual
                           esta guardado; maximum_slope usa mjd sin
                           reordenar (filtrando dt>0).
      'temporal_real'  -> se ordena explicitamente x por mjd antes
                           de calcular eta y maximum_slope. El resto
                           de variables no cambia (son invariantes
                           al orden).
      'barajado'       -> se rompe la relacion real target<->mjd
                           barajando mjd de forma aleatoria antes de
                           calcular eta y maximum_slope.
    """

    x, t = obtener_curva_cruda(fila)

    if len(x) == 0:
        return {v: np.nan for v in VARIABLES}

    valores = _stats_basicas(x)

    if modo == "almacenamiento":
        x_eta = x
        t_slope = t
        x_slope = x

    elif modo == "temporal_real":
        if len(t) == len(x) and len(t) > 1:
            orden = np.argsort(t, kind="mergesort")
            x_eta = x[orden]
            t_slope = t[orden]
            x_slope = x[orden]
        else:
            x_eta = x
            t_slope = t
            x_slope = x

    elif modo == "barajado":
        if rng is None:
            rng = np.random.RandomState(RANDOM_STATE)
        if len(t) > 1:
            perm = rng.permutation(len(t))
            t_bar = t[perm]
        else:
            t_bar = t
        # eta se sigue calculando sobre el array target almacenado
        # (igual que hace el pipeline actual); lo que cambia es la
        # pareja mjd usada por maximum_slope.
        x_eta = x
        t_slope = t_bar
        x_slope = x

    else:
        raise ValueError(f"modo desconocido: {modo}")

    valores["eta"] = _eta(x_eta)
    valores["maximum_slope"] = _maximum_slope(x_slope, t_slope)

    return valores


def extraer_features(df, modo, seed_base=0):

    filas = []
    n = len(df)

    rng = np.random.RandomState(RANDOM_STATE + seed_base)

    for i, (_, fila) in enumerate(df.iterrows()):

        if i % 5000 == 0:
            print(f"  [{modo}] {i:,}/{n:,}")

        filas.append(
            calcular_features_objeto(fila, modo=modo, rng=rng)
        )

    return pd.DataFrame(filas)


def extraer_n_observaciones(df):

    n_obs = []

    for _, fila in df.iterrows():
        x, t = obtener_curva_cruda(fila)
        n_obs.append(len(x))

    return np.array(n_obs)


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

    return resultados, pred_rf, pred_log


# ======================================================================
# MAIN
# ======================================================================

def main():

    inicio_total = time.time()

    banner("16b - CONTROL DE ORDEN TEMPORAL REAL (12 VARIABLES)")

    train, test = cargar_datos()

    if "class_str" not in train.columns or "class_str" not in test.columns:
        raise RuntimeError("Falta la columna class_str.")

    y_train = train["class_str"].astype(str)
    y_test = test["class_str"].astype(str)

    # ------------------------------------------------------------------
    # PASO 1: DIAGNOSTICO DE ORDEN (solo sobre TEST, es mas rapido y
    # ya nos dice si el problema existe; si quieres, cambia a train)
    # ------------------------------------------------------------------

    diagnostico_test = diagnosticar_orden(test)

    # ------------------------------------------------------------------
    # PASO 2: EXTRAER FEATURES EN LOS TRES MODOS
    # ------------------------------------------------------------------

    resultados_por_modo = {}
    predicciones_por_modo = {}
    features_por_modo = {}

    for modo in ["almacenamiento", "temporal_real", "barajado"]:

        banner(f"EXTRACCION DE FEATURES - MODO: {modo.upper()}")

        X_train = extraer_features(train, modo=modo, seed_base=0)
        X_test = extraer_features(test, modo=modo, seed_base=1)

        X_train, X_test = limpiar(X_train, X_test)

        features_por_modo[modo] = (X_train, X_test)

        print()
        print(f"Entrenando clasificadores (modo={modo})...")

        r, pred_rf, pred_log = evaluar(X_train, y_train, X_test, y_test)

        print(
            f"RF  : acc={r['RF_ACC']:.6f}  balanced={r['RF_BAL']:.6f}  "
            f"({r['RF_TIME']:.1f}s)"
        )
        print(
            f"LOG : acc={r['LOG_ACC']:.6f}  balanced={r['LOG_BAL']:.6f}  "
            f"({r['LOG_TIME']:.1f}s)"
        )

        resultados_por_modo[modo] = r
        predicciones_por_modo[modo] = {"rf": pred_rf, "log": pred_log}

    # ------------------------------------------------------------------
    # PASO 3: COMPARAR ETA ENTRE ALMACENAMIENTO Y TEMPORAL_REAL
    # ------------------------------------------------------------------

    banner("COMPARACION DE ETA: ALMACENAMIENTO vs TEMPORAL_REAL")

    eta_almacen = features_por_modo["almacenamiento"][1]["eta"].values
    eta_temporal = features_por_modo["temporal_real"][1]["eta"].values

    diff_abs = np.abs(eta_almacen - eta_temporal)

    comparacion = pd.DataFrame({
        "eta_almacenamiento": eta_almacen,
        "eta_temporal_real": eta_temporal,
        "diferencia_absoluta": diff_abs,
    })

    comparacion.to_csv(SALIDA_COMPARACION_ETA, index=False)

    print("Diferencia absoluta media  :", np.nanmean(diff_abs))
    print("Diferencia absoluta mediana:", np.nanmedian(diff_abs))
    print("Diferencia absoluta maxima :", np.nanmax(diff_abs))

    identicos = np.sum(diff_abs < 1e-9)
    print()
    print(
        f"Objetos con eta IDENTICA entre ambos modos: "
        f"{identicos:,} / {len(diff_abs):,} "
        f"({100.0 * identicos / len(diff_abs):.2f}%)"
    )

    if len(eta_almacen) > 1:
        corr = np.corrcoef(eta_almacen, eta_temporal)[0, 1]
        print(f"Correlacion eta_almacenamiento vs eta_temporal_real: {corr:.6f}")

    # ------------------------------------------------------------------
    # PASO 4: ETA vs N_OBSERVACIONES, RECALCULADO CON ORDEN TEMPORAL REAL
    # ------------------------------------------------------------------

    banner("ETA vs N_OBSERVACIONES (RECALCULADO CON ORDEN TEMPORAL REAL)")

    n_obs_test = extraer_n_observaciones(test)

    grupos = [
        ("<100", 0, 99),
        ("100-249", 100, 249),
        ("250-499", 250, 499),
        ("500-999", 500, 999),
        (">=1000", 1000, np.inf),
    ]

    filas_corr = []

    for nombre, minimo, maximo in grupos:

        mask = (n_obs_test >= minimo) & (n_obs_test <= maximo)

        if mask.sum() == 0:
            continue

        eta_grupo_almacen = eta_almacen[mask]
        eta_grupo_temporal = eta_temporal[mask]

        filas_corr.append({
            "grupo": nombre,
            "N": int(mask.sum()),
            "eta_media_almacenamiento": np.nanmean(eta_grupo_almacen),
            "eta_media_temporal_real": np.nanmean(eta_grupo_temporal),
        })

    tabla_corr = pd.DataFrame(filas_corr)
    print(tabla_corr.to_string(index=False))

    tabla_corr.to_csv(SALIDA_CORRELACION_ETA_NOBS, index=False)

    # ------------------------------------------------------------------
    # PASO 5: RESUMEN Y GUARDADO
    # ------------------------------------------------------------------

    banner("RESUMEN FINAL")

    tabla_resultados = pd.DataFrame(resultados_por_modo).T
    tabla_resultados.index.name = "modo"
    print(tabla_resultados.to_string())

    tabla_resultados.to_csv(SALIDA_RESULTADOS)

    print()
    print("INTERPRETACION:")
    print()
    print(
        "- Si 'ya_ordenado' en el diagnostico es ~100%, el array ya "
        "estaba en orden cronologico y ETA en 20-23 es de fiar tal "
        "cual esta calculada hoy."
    )
    print(
        "- Si ALMACENAMIENTO y TEMPORAL_REAL dan balanced accuracy "
        "y valores de eta practicamente identicos, el orden de "
        "almacenamiento coincide de facto con el orden temporal: "
        "no hay problema real, aunque el diagnostico muestre algunas "
        "inversiones menores."
    )
    print(
        "- Si BARAJADO pierde balanced accuracy de forma clara "
        "respecto a ALMACENAMIENTO/TEMPORAL_REAL, es la prueba de "
        "que SI existe señal temporal real siendo aprovechada por "
        "eta/maximum_slope (a diferencia de lo que sugeria, sin "
        "querer, el experimento 16)."
    )
    print(
        "- Si ALMACENAMIENTO y TEMPORAL_REAL difieren claramente "
        "entre si, hay que decidir cual usar de aqui en adelante: "
        "la version correcta es TEMPORAL_REAL, y habria que "
        "recalcular retroactivamente eta en 21/22/23 antes de dar "
        "esos resultados por definitivos."
    )

    print()
    print("Archivos guardados:")
    print(" ", SALIDA_DIAGNOSTICO)
    print(" ", SALIDA_COMPARACION_ETA)
    print(" ", SALIDA_CORRELACION_ETA_NOBS)
    print(" ", SALIDA_RESULTADOS)

    print()
    print(f"Tiempo total: {time.time() - inicio_total:.2f} segundos")


if __name__ == "__main__":
    main()