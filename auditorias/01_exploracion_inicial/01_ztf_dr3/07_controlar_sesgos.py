# -*- coding: utf-8 -*-
"""
07_controlar_sesgos.py

CONTROL DE SESGOS ENTRE DATASETS

Objetivo
--------
Determinar si la separación estadística observada entre M31, DEEP y DISK
continúa existiendo después de eliminar progresivamente features que pueden
actuar como proxies de:

    - magnitud
    - escala
    - eta_e
    - tendencias
    - información temporal

No pretende demostrar diferencias astronómicas.
Pretende determinar si la clasificación puede explicarse por
características observacionales/estadísticas evidentes.

Dataset:
    ZTF DR3
    M31
    DEEP
    DISK

Muestra:
    50.000 objetos por dataset

Random State:
    42
"""

import os
import time
from pathlib import Path
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Ruta relativa desde la ubicación del script
BASE_DIR = Path(__file__).resolve().parents[3]  # Sube 3 niveles hasta Astronomia/

# Directorio de resultados dentro de la misma carpeta que el script
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "07_"

SAMPLE_SIZE = 50000
RANDOM_STATE = 42

DATASETS = ["M31", "DEEP", "DISK"]

N_ESTIMATORS = 150
N_JOBS = -1


# ============================================================================
# FEATURES
# ============================================================================

FEATURES = [
    "amplitude",
    "beyond_1_std",
    "beyond_2_std",
    "cusum",
    "eta",
    "eta_e",
    "inter_percentile_range_25",
    "inter_percentile_range_10",
    "kurtosis",
    "linear_fit_slope",
    "linear_fit_slope_sigma",
    "linear_fit_reduced_chi2",
    "linear_trend",
    "linear_trend_sigma",
    "magnitude_percentage_ratio_40_5",
    "magnitude_percentage_ratio_20_10",
    "maximum_slope",
    "mean",
    "median_absolute_deviation",
    "median_buffer_range_percentage_5",
    "percent_amplitude",
    "percent_difference_magnitude_percentile_5",
    "percent_difference_magnitude_percentile_20",
    "period_0",
    "period_s_to_n_0",
    "period_1",
    "period_s_to_n_1",
    "period_2",
    "period_s_to_n_2",
    "periodogram_amplitude",
    "periodogram_beyond_1_std",
    "periodogram_beyond_2_std",
    "periodogram_cusum",
    "periodogram_eta",
    "periodogram_inter_percentile_range_25",
    "periodogram_standard_deviation",
    "periodogram_percent_amplitude",
    "chi2",
    "skew",
    "standard_deviation",
    "stetson_K",
    "weighted_mean",
]


# ============================================================================
# GRUPOS DE FEATURES
# ============================================================================

MAGNITUDE_FEATURES = {
    "mean",
    "weighted_mean",
}

ETA_FEATURES = {
    "eta_e",
}

TREND_FEATURES = {
    "linear_fit_slope",
    "linear_fit_slope_sigma",
    "linear_fit_reduced_chi2",
    "linear_trend",
    "linear_trend_sigma",
}

TEMPORAL_FEATURES = {
    "period_0",
    "period_s_to_n_0",
    "period_1",
    "period_s_to_n_1",
    "period_2",
    "period_s_to_n_2",
}

SCALE_FEATURES = {
    "amplitude",
    "inter_percentile_range_25",
    "inter_percentile_range_10",
    "maximum_slope",
    "median_absolute_deviation",
    "percent_amplitude",
    "standard_deviation",
    "percent_difference_magnitude_percentile_5",
    "percent_difference_magnitude_percentile_20",
}


# ============================================================================
# CARGA DATASET
# ============================================================================

def cargar_dataset(nombre):
    """
    Carga un dataset utilizando memmap.
    """

    feature_path = BASE_DIR / "dataset" / "ZTF_DR3" / f"feature_{nombre.lower()}.dat"
    name_path = BASE_DIR / "dataset" / "ZTF_DR3" / f"feature_{nombre.lower()}.name"
    oid_path = BASE_DIR / "dataset" / "ZTF_DR3" / f"oid_{nombre.lower()}.dat"

    with open(name_path, "r", encoding="utf-8") as f:
        names = f.read().split()

    dtype = [(name, np.float32) for name in names]

    oid = np.memmap(
        oid_path,
        mode="r",
        dtype=np.uint64
    )

    feature = np.memmap(
        feature_path,
        mode="r",
        dtype=dtype,
        shape=oid.shape
    )

    n = len(oid)

    rng = np.random.default_rng(RANDOM_STATE)

    if n > SAMPLE_SIZE:
        indices = rng.choice(
            n,
            size=SAMPLE_SIZE,
            replace=False
        )
    else:
        indices = np.arange(n)

    indices.sort()

    X = np.empty(
        (len(indices), len(names)),
        dtype=np.float32
    )

    for i, name in enumerate(names):
        X[:, i] = feature[name][indices]

    return X, names


# ============================================================================
# CONSTRUCCIÓN DE MATRIZ
# ============================================================================

def cargar_datos():

    print("=" * 78)
    print("CARGA DE DATOS")
    print("=" * 78)

    matrices = []
    etiquetas = []

    feature_names = None

    for clase, dataset in enumerate(DATASETS):

        print()
        print(f"Dataset: {dataset}")

        X, names = cargar_dataset(dataset)

        if feature_names is None:
            feature_names = names
        else:
            if names != feature_names:
                raise RuntimeError(
                    "Los datasets no tienen las mismas features."
                )

        print(f"Objetos: {len(X):,}")
        print(f"Features: {X.shape[1]}")

        matrices.append(X)
        etiquetas.append(
            np.full(len(X), clase, dtype=np.int8)
        )

    X = np.vstack(matrices)
    y = np.concatenate(etiquetas)

    print()
    print("-" * 78)
    print(f"Objetos totales : {len(X):,}")
    print(f"Features         : {X.shape[1]}")
    print(f"NaN              : {np.isnan(X).sum():,}")
    print(f"INF              : {np.isinf(X).sum():,}")

    return X, y, feature_names


# ============================================================================
# SELECCIÓN DE FEATURES
# ============================================================================

def seleccionar_features(feature_names, excluidas):

    indices = [
        i for i, name in enumerate(feature_names)
        if name not in excluidas
    ]

    return indices


# ============================================================================
# EVALUACIÓN
# ============================================================================

def evaluar(X, y, nombre):

    print()
    print("-" * 78)
    print(nombre)
    print("-" * 78)

    print(f"Features utilizadas: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # ------------------------------------------------------------------------
    # RANDOM FOREST
    # ------------------------------------------------------------------------

    inicio = time.time()

    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        max_features="sqrt",
        class_weight=None
    )

    rf.fit(X_train, y_train)

    pred = rf.predict(X_test)

    tiempo_rf = time.time() - inicio

    accuracy_rf = accuracy_score(y_test, pred)
    balanced_rf = balanced_accuracy_score(y_test, pred)

    print(
        f"Random Forest : {accuracy_rf:.6f} "
        f"(balanced={balanced_rf:.6f}) "
        f"[{tiempo_rf:.2f} s]"
    )

    # ------------------------------------------------------------------------
    # REGRESIÓN LOGÍSTICA
    # ------------------------------------------------------------------------

    inicio = time.time()

    modelo_log = Pipeline([
        (
            "scaler",
            RobustScaler()
        ),
        (
            "logistic",
            LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                random_state=RANDOM_STATE
            )
        )
    ])

    modelo_log.fit(X_train, y_train)

    pred_log = modelo_log.predict(X_test)

    tiempo_log = time.time() - inicio

    accuracy_log = accuracy_score(y_test, pred_log)
    balanced_log = balanced_accuracy_score(y_test, pred_log)

    print(
        f"Logística     : {accuracy_log:.6f} "
        f"(balanced={balanced_log:.6f}) "
        f"[{tiempo_log:.2f} s]"
    )

    return {
        "rf": accuracy_rf,
        "rf_balanced": balanced_rf,
        "log": accuracy_log,
        "log_balanced": balanced_log,
        "features": X.shape[1]
    }


# ============================================================================
# EXPERIMENTOS
# ============================================================================

def construir_experimentos():

    experimentos = []

    # ------------------------------------------------------------------------
    # 1. BASELINE
    # ------------------------------------------------------------------------

    experimentos.append(
        (
            "BASELINE - 42 FEATURES",
            set()
        )
    )

    # ------------------------------------------------------------------------
    # 2. SIN ETA_E
    # ------------------------------------------------------------------------

    experimentos.append(
        (
            "SIN eta_e",
            ETA_FEATURES
        )
    )

    # ------------------------------------------------------------------------
    # 3. SIN MAGNITUD
    # ------------------------------------------------------------------------

    experimentos.append(
        (
            "SIN MAGNITUD - mean + weighted_mean",
            MAGNITUDE_FEATURES
        )
    )

    # ------------------------------------------------------------------------
    # 4. SIN ETA_E + MAGNITUD
    # ------------------------------------------------------------------------

    experimentos.append(
        (
            "SIN eta_e + MAGNITUD",
            ETA_FEATURES | MAGNITUDE_FEATURES
        )
    )

    # ------------------------------------------------------------------------
    # 5. SIN TENDENCIAS
    # ------------------------------------------------------------------------

    experimentos.append(
        (
            "SIN TENDENCIAS",
            TREND_FEATURES
        )
    )

    # ------------------------------------------------------------------------
    # 6. SIN ETA_E + MAGNITUD + TENDENCIAS
    # ------------------------------------------------------------------------

    experimentos.append(
        (
            "SIN eta_e + MAGNITUD + TENDENCIAS",
            ETA_FEATURES |
            MAGNITUDE_FEATURES |
            TREND_FEATURES
        )
    )

    # ------------------------------------------------------------------------
    # 7. SIN VARIABLES TEMPORALES
    # ------------------------------------------------------------------------

    experimentos.append(
        (
            "SIN VARIABLES TEMPORALES",
            TEMPORAL_FEATURES
        )
    )

    # ------------------------------------------------------------------------
    # 8. SIN ESCALA
    # ------------------------------------------------------------------------

    experimentos.append(
        (
            "SIN VARIABLES DE ESCALA",
            SCALE_FEATURES
        )
    )

    # ------------------------------------------------------------------------
    # 9. CONTROL FUERTE
    #
    # Elimina simultáneamente:
    #   eta_e
    #   magnitud
    #   tendencias
    #   temporales
    #   escala
    #
    # Es deliberadamente agresivo.
    # ------------------------------------------------------------------------

    experimentos.append(
        (
            "CONTROL FUERTE",
            ETA_FEATURES |
            MAGNITUDE_FEATURES |
            TREND_FEATURES |
            TEMPORAL_FEATURES |
            SCALE_FEATURES
        )
    )

    return experimentos


# ============================================================================
# ETIQUETAS ALEATORIAS
# ============================================================================

def prueba_aleatoria(X, y):

    print()
    print("=" * 78)
    print("CONTROL - ETIQUETAS ALEATORIAS")
    print("=" * 78)

    rng = np.random.default_rng(RANDOM_STATE)

    y_random = y.copy()
    rng.shuffle(y_random)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_random,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_random
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        max_features="sqrt"
    )

    rf.fit(X_train, y_train)

    pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    print(f"Accuracy: {accuracy:.6f}")
    print("Esperada aproximadamente: 0.333333")

    return accuracy


# ============================================================================
# MAIN
# ============================================================================

def main():

    inicio_total = time.time()

    print("=" * 78)
    print("CONTROL DE SESGOS ENTRE DATASETS")
    print("=" * 78)

    print()
    print("Objetivo:")
    print("Determinar si la separación entre M31, DEEP y DISK")
    print("continúa después de eliminar posibles fuentes de sesgo.")

    print()
    print(f"Muestra por dataset : {SAMPLE_SIZE:,}")
    print(f"Random State        : {RANDOM_STATE}")

    # ------------------------------------------------------------------------
    # CARGAR
    # ------------------------------------------------------------------------

    X, y, feature_names = cargar_datos()

    # ------------------------------------------------------------------------
    # EXPERIMENTOS
    # ------------------------------------------------------------------------

    experimentos = construir_experimentos()

    resultados = []

    print()
    print("=" * 78)
    print("EXPERIMENTOS")
    print("=" * 78)

    for numero, (nombre, excluidas) in enumerate(
        experimentos,
        start=1
    ):

        indices = seleccionar_features(
            feature_names,
            excluidas
        )

        X_exp = X[:, indices]

        print()
        print()
        print(f"EXPERIMENTO {numero}/{len(experimentos)}")
        print(nombre)

        resultado = evaluar(
            X_exp,
            y,
            nombre
        )

        resultado["nombre"] = nombre
        resultados.append(resultado)

        del X_exp

    # ------------------------------------------------------------------------
    # ETIQUETAS ALEATORIAS
    # ------------------------------------------------------------------------

    prueba_aleatoria(X, y)

    # ------------------------------------------------------------------------
    # RESUMEN
    # ------------------------------------------------------------------------

    print()
    print()
    print("=" * 78)
    print("RESUMEN FINAL")
    print("=" * 78)

    print()
    print(
        f"{'EXPERIMENTO':48s}"
        f"{'FEATURES':>10s}"
        f"{'RF':>12s}"
        f"{'LOG':>12s}"
    )

    print("-" * 78)

    for r in resultados:

        nombre = r["nombre"]

        if len(nombre) > 48:
            nombre = nombre[:48]

        print(
            f"{nombre:48s}"
            f"{r['features']:10d}"
            f"{r['rf']:12.6f}"
            f"{r['log']:12.6f}"
        )

    # ------------------------------------------------------------------------
    # INTERPRETACIÓN AUTOMÁTICA
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("INTERPRETACIÓN")
    print("=" * 78)

    baseline = resultados[0]["rf"]
    control = resultados[-1]["rf"]

    diferencia = baseline - control

    print()
    print(f"RF baseline       : {baseline:.6f}")
    print(f"RF control fuerte : {control:.6f}")
    print(f"Diferencia        : {diferencia:.6f}")

    print()

    if control > 0.80:
        print(
            "RESULTADO:"
        )
        print(
            "La separación entre datasets permanece FUERTE"
        )
        print(
            "incluso después de eliminar numerosas features."
        )
        print()
        print(
            "Esto justifica investigar si existe una estructura"
        )
        print(
            "no explicada por los sesgos básicos analizados."
        )

    elif control > 0.60:
        print(
            "RESULTADO:"
        )
        print(
            "La separación disminuye considerablemente,"
        )
        print(
            "pero todavía permanece señal clasificable."
        )
        print()
        print(
            "Será necesario realizar controles adicionales."
        )

    else:
        print(
            "RESULTADO:"
        )
        print(
            "La mayor parte de la separación desaparece"
        )
        print(
            "cuando se eliminan las features potencialmente sesgadas."
        )
        print()
        print(
            "Esto apunta a un fuerte efecto de selección/"
        )
        print(
            "instrumentación o construcción del dataset."
        )

    # ------------------------------------------------------------------------
    # GUARDAR RESULTADOS
    # ------------------------------------------------------------------------

    csv_path = RESULTADOS_DIR / f"{PREFIJO}control_sesgos_resultados.csv"

    with open(csv_path, "w", encoding="utf-8") as f:

        f.write(
            "experimento,features,rf_accuracy,"
            "rf_balanced_accuracy,log_accuracy,"
            "log_balanced_accuracy\n"
        )

        for r in resultados:

            f.write(
                f'"{r["nombre"]}",'
                f'{r["features"]},'
                f'{r["rf"]:.8f},'
                f'{r["rf_balanced"]:.8f},'
                f'{r["log"]:.8f},'
                f'{r["log_balanced"]:.8f}\n'
            )

    print()
    print(f"Resultados guardados en:")
    print(csv_path)

    # ------------------------------------------------------------------------
    # TIEMPO
    # ------------------------------------------------------------------------

    tiempo_total = time.time() - inicio_total

    print()
    print("=" * 78)
    print(f"Tiempo total: {tiempo_total:.2f} s")
    print("=" * 78)


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    main()