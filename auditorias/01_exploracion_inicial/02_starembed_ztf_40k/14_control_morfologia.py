# -*- coding: utf-8 -*-

"""
14_control_morfologia.py

CONTROL DE MORFOLOGIA - StarEmbed / ZTF

Objetivo:
Determinar qué parte de la capacidad de clasificación procede de:

    1. PERIODO
    2. ESTRUCTURA OBSERVACIONAL
       - número de observaciones
       - duración
       - cadencia
       - errores fotométricos
    3. ESTADISTICOS DE MAGNITUD
    4. MORFOLOGIA DE LA CURVA
       - magnitud normalizada
       - tiempo normalizado

La prueba clave es:

    MORFOLOGIA PURA

donde se elimina:
    - nivel absoluto de magnitud
    - escala temporal absoluta

Si la clasificación permanece alta, existe evidencia
de que la forma de la variabilidad contiene información
específica de la clase.

IMPORTANTE:
Esto no demuestra por sí solo que la señal sea "física".
Solo separa progresivamente propiedades observacionales
de la morfología de la curva.
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ======================================================================
# CONFIGURACIÓN
# ======================================================================

RANDOM_STATE = 42

# Ruta relativa desde la ubicación del script
BASE_DIR = Path(__file__).resolve().parents[3]  # Sube 3 niveles hasta Astronomia/

# Directorio de resultados dentro de la misma carpeta que el script
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "14_"

DATA_DIR = BASE_DIR / "dataset" / "ZTF_40k_StarEmbed" / "data"

TRAIN_FILES = [
    DATA_DIR / "train-00000-of-00002.parquet",
    DATA_DIR / "train-00001-of-00002.parquet",
]

TEST_FILE = DATA_DIR / "test-00000-of-00001.parquet"

N_TRAIN = 25000
N_TEST = 7945

MIN_POINTS = 10

N_BINS = 64

RF_TREES = 250

OUTPUT_CSV = RESULTADOS_DIR / f"{PREFIJO}control_morfologia_resultados.csv"


# ======================================================================
# UTILIDADES
# ======================================================================

def separador():
    print("=" * 70)


def cargar_datos():

    print()
    separador()
    print("CARGA DE DATOS")
    separador()

    trains = []

    for path in TRAIN_FILES:

        print()
        print("Cargando TRAIN:")
        print(" ", path)

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_parquet(path)

        print("Objetos:", f"{len(df):,}")

        trains.append(df)

    train = pd.concat(
        trains,
        ignore_index=True
    )

    print()
    print("Cargando TEST:")
    print(" ", TEST_FILE)

    if not TEST_FILE.exists():
        raise FileNotFoundError(TEST_FILE)

    test = pd.read_parquet(TEST_FILE)

    # Muestreo reproducible
    if len(train) > N_TRAIN:
        train = train.sample(
            N_TRAIN,
            random_state=RANDOM_STATE
        ).reset_index(drop=True)

    if len(test) > N_TEST:
        test = test.sample(
            N_TEST,
            random_state=RANDOM_STATE
        ).reset_index(drop=True)

    print()
    print("TRAIN utilizado:", f"{len(train):,}")
    print("TEST utilizado :", f"{len(test):,}")

    return train, test


# ======================================================================
# EXTRACCIÓN DE CURVAS
# ======================================================================

def obtener_banda(row):

    """
    Selecciona la banda con mayor número de observaciones.
    """

    bandas = row["bands_data"]

    if bandas is None:
        return None

    mejor = None
    mejor_n = -1

    for banda in ["g", "r", "i"]:

        try:
            datos = bandas[banda]
        except Exception:
            continue

        if datos is None:
            continue

        try:
            n = int(datos["length"])
        except Exception:
            continue

        if n > mejor_n:
            mejor = datos
            mejor_n = n

    return mejor


def extraer_curva(row):

    datos = obtener_banda(row)

    if datos is None:
        return None

    try:

        y = np.asarray(
            datos["target"],
            dtype=np.float64
        )

        err = np.asarray(
            datos["past_feat_dynamic_real"],
            dtype=np.float64
        )

        dt = np.asarray(
            datos["feat_dynamic_real"],
            dtype=np.float64
        )

        mjd = np.asarray(
            datos["mjd"],
            dtype=np.float64
        )

    except Exception:
        return None

    n = min(
        len(y),
        len(err),
        len(dt),
        len(mjd)
    )

    if n < MIN_POINTS:
        return None

    y = y[:n]
    err = err[:n]
    dt = dt[:n]
    mjd = mjd[:n]

    mask = (
        np.isfinite(y)
        & np.isfinite(err)
        & np.isfinite(dt)
        & np.isfinite(mjd)
    )

    y = y[mask]
    err = err[mask]
    dt = dt[mask]
    mjd = mjd[mask]

    if len(y) < MIN_POINTS:
        return None

    order = np.argsort(mjd)

    y = y[order]
    err = err[order]
    dt = dt[order]
    mjd = mjd[order]

    return {
        "y": y,
        "err": err,
        "dt": dt,
        "mjd": mjd
    }


# ======================================================================
# FEATURES
# ======================================================================

def features_periodo(row):

    p = row["period"]

    if not np.isfinite(p):
        p = 0.0

    return [
        float(p)
    ]


def features_observacion(curva):

    y = curva["y"]
    err = curva["err"]
    mjd = curva["mjd"]

    n = len(y)

    duracion = mjd[-1] - mjd[0]

    if duracion <= 0:
        duracion = 0.0

    if n > 1:
        cadencia_media = np.mean(
            np.diff(mjd)
        )

        cadencia_mediana = np.median(
            np.diff(mjd)
        )

        cadencia_std = np.std(
            np.diff(mjd)
        )
    else:
        cadencia_media = 0.0
        cadencia_mediana = 0.0
        cadencia_std = 0.0

    return [
        float(n),
        float(duracion),
        float(cadencia_media),
        float(cadencia_mediana),
        float(cadencia_std),
        float(np.mean(err)),
        float(np.median(err)),
        float(np.std(err)),
    ]


def features_estadisticas(curva):

    y = curva["y"]

    q05, q25, q50, q75, q95 = np.percentile(
        y,
        [5, 25, 50, 75, 95]
    )

    media = np.mean(y)
    std = np.std(y)

    amplitud = np.max(y) - np.min(y)

    mad = np.median(
        np.abs(y - q50)
    )

    return [
        float(media),
        float(std),
        float(amplitud),
        float(mad),
        float(q05),
        float(q25),
        float(q50),
        float(q75),
        float(q95),
        float(q95 - q05),
        float(q75 - q25),
    ]


# ======================================================================
# MORFOLOGIA NORMALIZADA
# ======================================================================

def morfologia_normalizada(curva):

    """
    Elimina el nivel absoluto de magnitud y la escala temporal.

    La curva se transforma a:

        tiempo -> [0,1]
        magnitud -> media 0 / desviación 1

    Después se interpola a N_BINS puntos.

    El resultado describe solamente la forma relativa
    de la curva.
    """

    y = curva["y"]
    t = curva["mjd"]

    if len(y) < MIN_POINTS:
        return None

    # --------------------------------------------------------------
    # Normalización temporal
    # --------------------------------------------------------------

    t0 = t[0]
    t1 = t[-1]

    duracion = t1 - t0

    if duracion <= 0:
        return None

    tn = (t - t0) / duracion

    # Eliminar posibles tiempos repetidos
    unique_t, idx = np.unique(
        tn,
        return_index=True
    )

    y = y[idx]
    tn = unique_t

    if len(y) < MIN_POINTS:
        return None

    # --------------------------------------------------------------
    # Normalización de magnitud
    # --------------------------------------------------------------

    media = np.mean(y)
    std = np.std(y)

    if not np.isfinite(std) or std <= 1e-12:
        return None

    yn = (y - media) / std

    # --------------------------------------------------------------
    # Interpolación
    # --------------------------------------------------------------

    grid = np.linspace(
        0.0,
        1.0,
        N_BINS
    )

    valores = np.interp(
        grid,
        tn,
        yn
    )

    valores = np.asarray(
        valores,
        dtype=np.float64
    )

    valores[~np.isfinite(valores)] = 0.0

    return valores


# ======================================================================
# EXTRACCIÓN COMPLETA
# ======================================================================

def construir_features(df, modo):

    X = []
    y = []

    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):

        if i % 5000 == 0:
            print(
                f"  {i:,}/{total:,}"
            )

        curva = extraer_curva(row)

        if curva is None:
            continue

        clase = row["class_str"]

        # ----------------------------------------------------------
        # PERIODO
        # ----------------------------------------------------------

        if modo == "PERIODO":

            f = features_periodo(row)

        # ----------------------------------------------------------
        # OBSERVACIONAL
        # ----------------------------------------------------------

        elif modo == "OBSERVACION":

            f = features_observacion(curva)

        # ----------------------------------------------------------
        # ESTADISTICAS
        # ----------------------------------------------------------

        elif modo == "ESTADISTICAS":

            f = features_estadisticas(curva)

        # ----------------------------------------------------------
        # MORFOLOGIA PURA
        # ----------------------------------------------------------

        elif modo == "MORFOLOGIA":

            f = morfologia_normalizada(curva)

            if f is None:
                continue

        # ----------------------------------------------------------
        # MORFOLOGIA + PERIODO
        # ----------------------------------------------------------

        elif modo == "MORFOLOGIA_PERIODO":

            morfo = morfologia_normalizada(curva)

            if morfo is None:
                continue

            f = np.concatenate([
                morfo,
                features_periodo(row)
            ])

        # ----------------------------------------------------------
        # TODO
        # ----------------------------------------------------------

        elif modo == "TODO":

            morfo = morfologia_normalizada(curva)

            if morfo is None:
                continue

            f = np.concatenate([
                features_periodo(row),
                features_observacion(curva),
                features_estadisticas(curva),
                morfo
            ])

        else:

            raise ValueError(
                f"Modo desconocido: {modo}"
            )

        X.append(f)
        y.append(clase)

    return np.asarray(X), np.asarray(y)


# ======================================================================
# LIMPIEZA
# ======================================================================

def limpiar(X_train, X_test):

    print()
    print("LIMPIEZA")

    print(
        "NaN TRAIN antes:",
        int(np.isnan(X_train).sum())
    )

    print(
        "NaN TEST antes :",
        int(np.isnan(X_test).sum())
    )

    X_train = np.nan_to_num(
        X_train,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    X_test = np.nan_to_num(
        X_test,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    print(
        "NaN TRAIN:",
        int(np.isnan(X_train).sum())
    )

    print(
        "NaN TEST :",
        int(np.isnan(X_test).sum())
    )

    return X_train, X_test


# ======================================================================
# CLASIFICACIÓN
# ======================================================================

def clasificar(
    nombre,
    X_train,
    y_train,
    X_test,
    y_test
):

    print()
    print("-" * 70)
    print(nombre)
    print("-" * 70)

    resultados = {}

    # --------------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------------

    inicio = time.time()

    rf = RandomForestClassifier(
        n_estimators=RF_TREES,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )

    rf.fit(
        X_train,
        y_train
    )

    pred = rf.predict(X_test)

    acc = accuracy_score(
        y_test,
        pred
    )

    bal = balanced_accuracy_score(
        y_test,
        pred
    )

    tiempo = time.time() - inicio

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

    inicio = time.time()

    log = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        )
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        log.fit(
            X_train,
            y_train
        )

    pred = log.predict(X_test)

    acc = accuracy_score(
        y_test,
        pred
    )

    bal = balanced_accuracy_score(
        y_test,
        pred
    )

    tiempo = time.time() - inicio

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
# CONTROL DE ETIQUETAS
# ======================================================================

def control_etiquetas(
    X_train,
    y_train,
    X_test,
    y_test
):

    print()
    separador()
    print("CONTROL DE ETIQUETAS ALEATORIAS")
    separador()

    print(
        "Las etiquetas TRAIN se barajan manteniendo "
        "su distribución."
    )

    y_random = y_train.copy()

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    rng.shuffle(y_random)

    resultados = clasificar(
        "ETIQUETAS ALEATORIAS",
        X_train,
        y_random,
        X_test,
        y_test
    )

    print()
    print(
        "Referencia balanced accuracy:",
        f"{1 / len(np.unique(y_test)):.6f}"
    )

    return resultados


# ======================================================================
# MAIN
# ======================================================================

def main():

    inicio_total = time.time()

    separador()
    print("CONTROL DE MORFOLOGIA - StarEmbed / ZTF")
    separador()

    print()
    print("Objetivo:")
    print(
        "Separar propiedades observacionales, estadisticas"
    )
    print(
        "y periodo de la MORFOLOGIA de las curvas."
    )

    print()
    print("Random State:", RANDOM_STATE)
    print("Bins morfologia:", N_BINS)

    # --------------------------------------------------------------
    # CARGA
    # --------------------------------------------------------------

    train, test = cargar_datos()

    print()
    separador()
    print("DISTRIBUCIÓN DE CLASES")
    separador()

    print()
    print("TRAIN:")
    print(train["class_str"].value_counts())

    print()
    print("TEST:")
    print(test["class_str"].value_counts())

    # --------------------------------------------------------------
    # EXPERIMENTOS
    # --------------------------------------------------------------

    experimentos = [
        ("PERIODO", "SOLO PERIODO"),
        ("OBSERVACION", "SOLO OBSERVACION"),
        ("ESTADISTICAS", "SOLO ESTADISTICAS"),
        ("MORFOLOGIA", "MORFOLOGIA PURA"),
        ("MORFOLOGIA_PERIODO", "MORFOLOGIA + PERIODO"),
        ("TODO", "TODO"),
    ]

    resultados_finales = []

    X_cache = {}

    # --------------------------------------------------------------
    # EJECUTAR
    # --------------------------------------------------------------

    for modo, nombre in experimentos:

        print()
        separador()
        print("EXPERIMENTO:", nombre)
        separador()

        print()
        print("Extrayendo TRAIN...")

        X_train, y_train = construir_features(
            train,
            modo
        )

        print()
        print("Extrayendo TEST...")

        X_test, y_test = construir_features(
            test,
            modo
        )

        print()
        print("TRAIN shape:", X_train.shape)
        print("TEST shape :", X_test.shape)

        X_train, X_test = limpiar(
            X_train,
            X_test
        )

        resultado = clasificar(
            nombre,
            X_train,
            y_train,
            X_test,
            y_test
        )

        resultados_finales.append({
            "experimento": nombre,
            "features": X_train.shape[1],
            "objetos_train": len(y_train),
            "objetos_test": len(y_test),
            **resultado
        })

        # Guardamos la morfología para el control aleatorio
        if modo == "MORFOLOGIA":

            X_cache["X_train"] = X_train
            X_cache["y_train"] = y_train
            X_cache["X_test"] = X_test
            X_cache["y_test"] = y_test

    # --------------------------------------------------------------
    # CONTROL ALEATORIO
    # --------------------------------------------------------------

    if X_cache:

        resultado_random = control_etiquetas(
            X_cache["X_train"],
            X_cache["y_train"],
            X_cache["X_test"],
            X_cache["y_test"]
        )

        resultados_finales.append({
            "experimento": "ETIQUETAS_ALEATORIAS",
            "features": X_cache["X_train"].shape[1],
            "objetos_train": len(X_cache["y_train"]),
            "objetos_test": len(X_cache["y_test"]),
            **resultado_random
        })

    # --------------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------------

    df_resultados = pd.DataFrame(
        resultados_finales
    )

    print()
    separador()
    print("RESUMEN FINAL")
    separador()

    print()

    print(
        f"{'EXPERIMENTO':30s}"
        f"{'FEATURES':>10s}"
        f"{'RF ACC':>12s}"
        f"{'RF BAL':>12s}"
        f"{'LOG ACC':>12s}"
        f"{'LOG BAL':>12s}"
    )

    print("-" * 90)

    for _, r in df_resultados.iterrows():

        print(
            f"{r['experimento']:30s}"
            f"{int(r['features']):10d}"
            f"{r['RF_ACC']:12.6f}"
            f"{r['RF_BAL']:12.6f}"
            f"{r['LOG_ACC']:12.6f}"
            f"{r['LOG_BAL']:12.6f}"
        )

    # --------------------------------------------------------------
    # DIAGNÓSTICO
    # --------------------------------------------------------------

    print()
    separador()
    print("DIAGNÓSTICO")
    separador()

    morfo = df_resultados[
        df_resultados["experimento"]
        == "MORFOLOGIA PURA"
    ]

    periodo = df_resultados[
        df_resultados["experimento"]
        == "SOLO PERIODO"
    ]

    observ = df_resultados[
        df_resultados["experimento"]
        == "SOLO OBSERVACION"
    ]

    if len(morfo):

        rf_m = float(
            morfo.iloc[0]["RF_BAL"]
        )

        log_m = float(
            morfo.iloc[0]["LOG_BAL"]
        )

        aleatorio = 1 / 7

        print()
        print(
            "MORFOLOGIA PURA:"
        )

        print(
            f"  RF balanced : {rf_m:.6f}"
        )

        print(
            f"  LOG balanced: {log_m:.6f}"
        )

        print(
            f"  Aleatorio   : {aleatorio:.6f}"
        )

        print()

        if rf_m > aleatorio + 0.10:

            print(
                "RESULTADO:"
            )

            print(
                "La morfologia normalizada conserva "
                "capacidad discriminante importante."
            )

            print(
                "Esto indica que la clasificación no depende "
                "exclusivamente de magnitud absoluta, periodo "
                "o escala temporal."
            )

        else:

            print(
                "RESULTADO:"
            )

            print(
                "La morfologia normalizada pierde "
                "gran parte de la capacidad discriminante."
            )

            print(
                "La clasificación parece depender en buena "
                "medida de propiedades observacionales "
                "o escala temporal."
            )

    # --------------------------------------------------------------
    # GUARDAR
    # --------------------------------------------------------------

    df_resultados.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print()
    print("Resultados guardados en:")
    print(OUTPUT_CSV)

    # --------------------------------------------------------------
    # FIN
    # --------------------------------------------------------------

    tiempo_total = time.time() - inicio_total

    print()
    separador()
    print(
        f"Tiempo total: {tiempo_total:.2f} segundos"
    )
    separador()


# ======================================================================
# EJECUCIÓN
# ======================================================================

if __name__ == "__main__":
    main()