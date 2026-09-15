import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Ruta relativa desde la ubicación del script
BASE_DIR = Path(__file__).resolve().parents[3]  # Sube 3 niveles hasta Astronomia/

# Directorio de resultados dentro de la misma carpeta que el script
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "11_"

DATA_DIR = BASE_DIR / "dataset" / "ZTF_40k_StarEmbed" / "data"

RANDOM_STATE = 42

# Para que el primer experimento sea manejable.
# None = utilizar todos los objetos disponibles.
MAX_TRAIN = 25000
MAX_TEST = 8000

CLASSES = [
    "EW",
    "EA",
    "RRab",
    "RRc",
    "RRd",
    "RS CVn",
    "LPV"
]

BANDS = ["g", "r", "i"]


# ============================================================================
# UTILIDADES
# ============================================================================

def percentiles(x):
    """Devuelve P10, P25, P50, P75, P90."""
    if len(x) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    p = np.percentile(x, [10, 25, 50, 75, 90])
    return p[0], p[1], p[2], p[3], p[4]


def safe_std(x):
    if len(x) < 2:
        return 0.0
    return float(np.std(x))


def safe_mean(x):
    if len(x) == 0:
        return np.nan
    return float(np.mean(x))


def safe_median(x):
    if len(x) == 0:
        return np.nan
    return float(np.median(x))


def safe_slope(t, y):
    """
    Pendiente lineal de magnitud frente a tiempo.
    """
    if len(t) < 3:
        return 0.0

    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        mask = np.isfinite(t) & np.isfinite(y)
        t = t[mask]
        y = y[mask]

    if len(t) < 3:
        return 0.0

    if np.ptp(t) == 0:
        return 0.0

    try:
        return float(np.polyfit(t, y, 1)[0])
    except Exception:
        return 0.0


def extraer_features_banda(banda, prefijo):
    """
    Extrae características estadísticas de una banda.
    """

    if banda is None:
        return {}

    try:
        mag = np.asarray(banda["target"], dtype=float)
        err = np.asarray(
            banda["past_feat_dynamic_real"],
            dtype=float
        )
        dt = np.asarray(
            banda["feat_dynamic_real"],
            dtype=float
        )
        mjd = np.asarray(
            banda["mjd"],
            dtype=float
        )
    except Exception:
        return {}

    # ------------------------------------------------------------------
    # Limpiar
    # ------------------------------------------------------------------

    n = min(len(mag), len(err), len(mjd))

    if n == 0:
        return {}

    mag = mag[:n]
    err = err[:n]
    mjd = mjd[:n]

    mask = (
        np.isfinite(mag)
        & np.isfinite(err)
        & np.isfinite(mjd)
    )

    mag = mag[mask]
    err = err[mask]
    mjd = mjd[mask]

    if len(mag) == 0:
        return {}

    resultado = {}

    # ------------------------------------------------------------------
    # Número de observaciones
    # ------------------------------------------------------------------

    resultado[f"{prefijo}_n"] = len(mag)

    # ------------------------------------------------------------------
    # Estadística básica de magnitud
    # ------------------------------------------------------------------

    resultado[f"{prefijo}_mean"] = safe_mean(mag)
    resultado[f"{prefijo}_median"] = safe_median(mag)
    resultado[f"{prefijo}_std"] = safe_std(mag)

    p10, p25, p50, p75, p90 = percentiles(mag)

    resultado[f"{prefijo}_p10"] = p10
    resultado[f"{prefijo}_p25"] = p25
    resultado[f"{prefijo}_p75"] = p75
    resultado[f"{prefijo}_p90"] = p90

    resultado[f"{prefijo}_iqr"] = p75 - p25
    resultado[f"{prefijo}_range_10_90"] = p90 - p10

    # ------------------------------------------------------------------
    # Amplitud
    # ------------------------------------------------------------------

    resultado[f"{prefijo}_amplitude"] = (
        np.max(mag) - np.min(mag)
    )

    med = np.median(mag)

    resultado[f"{prefijo}_mad"] = (
        np.median(np.abs(mag - med))
    )

    # ------------------------------------------------------------------
    # Asimetría sencilla
    # ------------------------------------------------------------------

    std = np.std(mag)

    if std > 0:
        resultado[f"{prefijo}_skew"] = float(
            np.mean(((mag - med) / std) ** 3)
        )
    else:
        resultado[f"{prefijo}_skew"] = 0.0

    # ------------------------------------------------------------------
    # Errores fotométricos
    # ------------------------------------------------------------------

    resultado[f"{prefijo}_err_mean"] = safe_mean(err)
    resultado[f"{prefijo}_err_median"] = safe_median(err)
    resultado[f"{prefijo}_err_std"] = safe_std(err)

    # ------------------------------------------------------------------
    # Cadencia
    # ------------------------------------------------------------------

    if len(mjd) >= 2:
        diferencias = np.diff(np.sort(mjd))

        diferencias = diferencias[
            np.isfinite(diferencias)
            & (diferencias >= 0)
        ]

        if len(diferencias) > 0:
            resultado[f"{prefijo}_cadence_mean"] = float(
                np.mean(diferencias)
            )
            resultado[f"{prefijo}_cadence_median"] = float(
                np.median(diferencias)
            )
            resultado[f"{prefijo}_cadence_std"] = safe_std(
                diferencias
            )
            resultado[f"{prefijo}_baseline"] = float(
                np.max(mjd) - np.min(mjd)
            )
        else:
            resultado[f"{prefijo}_cadence_mean"] = 0.0
            resultado[f"{prefijo}_cadence_median"] = 0.0
            resultado[f"{prefijo}_cadence_std"] = 0.0
            resultado[f"{prefijo}_baseline"] = 0.0
    else:
        resultado[f"{prefijo}_cadence_mean"] = 0.0
        resultado[f"{prefijo}_cadence_median"] = 0.0
        resultado[f"{prefijo}_cadence_std"] = 0.0
        resultado[f"{prefijo}_baseline"] = 0.0

    # ------------------------------------------------------------------
    # Tendencia
    # ------------------------------------------------------------------

    resultado[f"{prefijo}_slope"] = safe_slope(mjd, mag)

    return resultado


# ============================================================================
# EXTRACCIÓN DE FEATURES DE UN OBJETO
# ============================================================================

def extraer_features_objeto(row):

    features = {}

    # ------------------------------------------------------------------
    # Periodo proporcionado por StarEmbed
    # ------------------------------------------------------------------

    periodo = row.get("period", np.nan)

    if periodo is None:
        periodo = np.nan

    try:
        periodo = float(periodo)
    except Exception:
        periodo = np.nan

    features["period"] = periodo

    # ------------------------------------------------------------------
    # Bandas
    # ------------------------------------------------------------------

    bandas = row.get("bands_data")

    if bandas is None:
        bandas = {}

    for banda in BANDS:

        datos_banda = None

        try:
            datos_banda = bandas.get(banda)
        except Exception:
            pass

        features.update(
            extraer_features_banda(
                datos_banda,
                banda
            )
        )

    return features


# ============================================================================
# CARGA PARQUET
# ============================================================================

def cargar_parquet(ruta, max_objetos=None):

    print(f"\nCargando:")
    print(f"  {ruta}")

    tabla = pq.read_table(ruta)

    df = tabla.to_pandas()

    print(f"Objetos encontrados: {len(df):,}")

    if max_objetos is not None and len(df) > max_objetos:
        df = df.sample(
            n=max_objetos,
            random_state=RANDOM_STATE
        ).reset_index(drop=True)

        print(
            f"Muestra utilizada  : {len(df):,}"
        )

    return df


# ============================================================================
# CONVERTIR DATAFRAME A MATRIZ
# ============================================================================

def construir_matriz(df, nombre):

    print(
        f"\nExtrayendo features: {nombre}"
    )

    registros = []

    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):

        if i % 1000 == 0 or i == total - 1:
            print(
                f"\r  {i + 1:,}/{total:,}",
                end=""
            )

        try:
            f = extraer_features_objeto(row)
            f["class_str"] = row["class_str"]
            registros.append(f)

        except Exception:
            continue

    print()

    resultado = pd.DataFrame(registros)

    return resultado


# ============================================================================
# LIMPIEZA
# ============================================================================

def limpiar_datos(df):

    print("\n" + "=" * 70)
    print("LIMPIEZA")
    print("=" * 70)

    print(
        f"Objetos antes : {len(df):,}"
    )

    # Solo clases conocidas
    df = df[
        df["class_str"].isin(CLASSES)
    ].copy()

    print(
        f"Después clases: {len(df):,}"
    )

    X = df.drop(columns=["class_str"])

    # Sustituir infinitos
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Estadísticas
    print(
        f"NaN antes     : {X.isna().sum().sum():,}"
    )

    # Mediana por feature
    X = X.fillna(
        X.median(numeric_only=True)
    )

    # Por si alguna columna queda completamente vacía
    X = X.fillna(0.0)

    print(
        f"NaN después   : {X.isna().sum().sum():,}"
    )

    return X, df["class_str"].values


# ============================================================================
# RANDOM FOREST
# ============================================================================

def entrenar_random_forest(X_train, y_train, X_test, y_test):

    print("\n" + "=" * 70)
    print("RANDOM FOREST")
    print("=" * 70)

    inicio = time.time()

    modelo = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
        max_features="sqrt"
    )

    modelo.fit(
        X_train,
        y_train
    )

    pred = modelo.predict(X_test)

    tiempo = time.time() - inicio

    accuracy = accuracy_score(
        y_test,
        pred
    )

    print(
        f"Tiempo entrenamiento: {tiempo:.2f} s"
    )

    print(
        f"Accuracy: {accuracy:.6f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            pred,
            labels=CLASSES,
            zero_division=0
        )
    )

    matriz = confusion_matrix(
        y_test,
        pred,
        labels=CLASSES
    )

    print("Matriz de confusión:")
    print(matriz)

    return modelo, pred, accuracy, matriz


# ============================================================================
# REGRESIÓN LOGÍSTICA
# ============================================================================

def entrenar_logistica(
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n" + "=" * 70)
    print("REGRESIÓN LOGÍSTICA")
    print("=" * 70)

    inicio = time.time()

    modelo = Pipeline(
        [
            (
                "scaler",
                RobustScaler()
            ),
            (
                "logistic",
                LogisticRegression(
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced"
                )
            )
        ]
    )

    modelo.fit(
        X_train,
        y_train
    )

    pred = modelo.predict(X_test)

    tiempo = time.time() - inicio

    accuracy = accuracy_score(
        y_test,
        pred
    )

    print(
        f"Tiempo entrenamiento: {tiempo:.2f} s"
    )

    print(
        f"Accuracy: {accuracy:.6f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            pred,
            labels=CLASSES,
            zero_division=0
        )
    )

    matriz = confusion_matrix(
        y_test,
        pred,
        labels=CLASSES
    )

    print("Matriz de confusión:")
    print(matriz)

    return modelo, pred, accuracy, matriz


# ============================================================================
# IMPORTANCIA RANDOM FOREST
# ============================================================================

def mostrar_importancias(modelo, columnas):

    print("\n" + "=" * 70)
    print("IMPORTANCIA DE FEATURES - RANDOM FOREST")
    print("=" * 70)

    importancia = pd.DataFrame(
        {
            "feature": columnas,
            "importance": modelo.feature_importances_
        }
    )

    importancia = importancia.sort_values(
        "importance",
        ascending=False
    )

    print(
        f"\n{'RANGO':<8}"
        f"{'FEATURE':<45}"
        f"IMPORTANCIA"
    )

    print("-" * 70)

    for i, fila in enumerate(
        importancia.itertuples(index=False),
        1
    ):

        print(
            f"{i:<8}"
            f"{fila.feature:<45}"
            f"{fila.importance:.8f}"
        )

    ruta = RESULTADOS_DIR / f"{PREFIJO}importancia_features.csv"

    importancia.to_csv(
        ruta,
        index=False
    )

    print(
        f"\nGuardado en: {ruta}"
    )

    return importancia


# ============================================================================
# CONTROL ETIQUETAS ALEATORIAS
# ============================================================================

def control_etiquetas_aleatorias(
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n" + "=" * 70)
    print("CONTROL - ETIQUETAS ALEATORIAS")
    print("=" * 70)

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    y_train_random = rng.permutation(
        y_train
    )

    y_test_random = rng.permutation(
        y_test
    )

    modelo = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt"
    )

    modelo.fit(
        X_train,
        y_train_random
    )

    pred = modelo.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test_random,
        pred
    )

    print(
        f"Accuracy: {accuracy:.6f}"
    )

    print(
        "Esperada aproximadamente: "
        f"{1 / len(CLASSES):.6f}"
    )

    return accuracy


# ============================================================================
# MAIN
# ============================================================================

def main():

    inicio_total = time.time()

    print("=" * 70)
    print("CLASIFICADOR FÍSICO - StarEmbed / ZTF")
    print("=" * 70)

    print(
        """
Objetivo:
Comprobar si características extraídas directamente
de las curvas de luz ZTF permiten identificar las
clases físicas de StarEmbed.

NO se utilizan:
  - sourceid
  - RA
  - DEC

Sí se utilizan:
  - magnitudes
  - errores fotométricos
  - tiempos de observación
  - cadencia
  - periodo catalogado
  - información g/r/i
"""
    )

    print(
        f"Random State: {RANDOM_STATE}"
    )

    # ==================================================================
    # CARGA
    # ==================================================================

    print("\n" + "=" * 70)
    print("CARGA DE DATOS")
    print("=" * 70)

    train_paths = [
        DATA_DIR / "train-00000-of-00002.parquet",
        DATA_DIR / "train-00001-of-00002.parquet"
    ]

    test_path = DATA_DIR / "test-00000-of-00001.parquet"

    for ruta in train_paths + [test_path]:

        if not ruta.exists():

            raise FileNotFoundError(
                f"\nNo existe:\n{ruta}\n\n"
                "Modifica DATA_DIR al principio "
                "del programa."
            )

    train_dfs = []

    for ruta in train_paths:

        train_dfs.append(
            cargar_parquet(
                ruta,
                MAX_TRAIN
            )
        )

    train_df = pd.concat(
        train_dfs,
        ignore_index=True
    )

    # Volver a limitar después de juntar
    if (
        MAX_TRAIN is not None
        and len(train_df) > MAX_TRAIN
    ):

        train_df = train_df.sample(
            n=MAX_TRAIN,
            random_state=RANDOM_STATE
        ).reset_index(drop=True)

    test_df = cargar_parquet(
        test_path,
        MAX_TEST
    )

    # ==================================================================
    # DISTRIBUCIÓN DE CLASES
    # ==================================================================

    print("\n" + "=" * 70)
    print("CLASES")
    print("=" * 70)

    print("\nTRAIN:")

    print(
        train_df["class_str"]
        .value_counts()
        .sort_index()
    )

    print("\nTEST:")

    print(
        test_df["class_str"]
        .value_counts()
        .sort_index()
    )

    # ==================================================================
    # EXTRACCIÓN
    # ==================================================================

    train_features = construir_matriz(
        train_df,
        "TRAIN"
    )

    test_features = construir_matriz(
        test_df,
        "TEST"
    )

    # ==================================================================
    # MATRICES
    # ==================================================================

    X_train, y_train = limpiar_datos(
        train_features
    )

    X_test, y_test = limpiar_datos(
        test_features
    )

    # Asegurar mismas columnas
    columnas = list(X_train.columns)

    X_test = X_test.reindex(
        columns=columnas
    )

    print("\n" + "=" * 70)
    print("MATRIZ FINAL")
    print("=" * 70)

    print(
        f"Train : {X_train.shape[0]:,}"
    )

    print(
        f"Test  : {X_test.shape[0]:,}"
    )

    print(
        f"Features: {X_train.shape[1]}"
    )

    print(
        f"NaN: {X_train.isna().sum().sum()}"
    )

    print(
        f"INF: {np.isinf(X_train.values).sum()}"
    )

    # ==================================================================
    # RANDOM FOREST
    # ==================================================================

    (
        rf,
        rf_pred,
        rf_accuracy,
        rf_matrix
    ) = entrenar_random_forest(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # ==================================================================
    # IMPORTANCIAS
    # ==================================================================

    importancia = mostrar_importancias(
        rf,
        columnas
    )

    # ==================================================================
    # LOGÍSTICA
    # ==================================================================

    (
        logistica,
        log_pred,
        log_accuracy,
        log_matrix
    ) = entrenar_logistica(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # ==================================================================
    # CONTROL
    # ==================================================================

    random_accuracy = control_etiquetas_aleatorias(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # ==================================================================
    # RESUMEN
    # ==================================================================

    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)

    print(
        f"\nRandom Forest : {rf_accuracy:.6f}"
    )

    print(
        f"Logística     : {log_accuracy:.6f}"
    )

    print(
        f"Etiquetas aleatorias: "
        f"{random_accuracy:.6f}"
    )

    print(
        f"\nAccuracy esperada al azar: "
        f"{1 / len(CLASSES):.6f}"
    )

    print("\n" + "=" * 70)
    print("INTERPRETACIÓN PRELIMINAR")
    print("=" * 70)

    if rf_accuracy > 0.8:

        print(
            """
La información contenida en las curvas de luz
permite recuperar con bastante eficacia las clases
de StarEmbed.

Esto demuestra que las características fotométricas
y temporales contienen información relacionada con
la clase física.
"""
        )

    elif rf_accuracy > 0.5:

        print(
            """
Existe señal clasificatoria por encima del azar,
pero la separación no es extremadamente fuerte.
"""
        )

    else:

        print(
            """
La separación obtenida es relativamente débil.
No podemos concluir todavía que las características
extraídas sean suficientes para identificar las
clases físicas.
"""
        )

    print(
        """
IMPORTANTE:

Este experimento NO demuestra que M31, DEEP y DISK
sean poblaciones físicas diferentes.

Su objetivo es distinto:

comprobar que, en un dataset donde las etiquetas
sí representan clases astronómicas conocidas,
las curvas de luz contienen señal suficiente para
recuperar esas clases.

Si este experimento funciona, tendremos una referencia
física independiente para continuar la investigación.
"""
    )

    # ==================================================================
    # GUARDAR RESUMEN
    # ==================================================================

    resumen = pd.DataFrame(
        [
            {
                "experimento": "StarEmbed",
                "RF": rf_accuracy,
                "LOG": log_accuracy,
                "CONTROL_RANDOM": random_accuracy,
                "features": X_train.shape[1],
                "train": len(X_train),
                "test": len(X_test)
            }
        ]
    )

    ruta_resumen = RESULTADOS_DIR / f"{PREFIJO}StarEmbed_resultados.csv"

    resumen.to_csv(
        ruta_resumen,
        index=False
    )

    print(
        f"\nResultados guardados en:"
        f"\n{ruta_resumen}"
    )

    tiempo_total = time.time() - inicio_total

    print(
        f"\nTiempo total: "
        f"{tiempo_total:.2f} segundos"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()