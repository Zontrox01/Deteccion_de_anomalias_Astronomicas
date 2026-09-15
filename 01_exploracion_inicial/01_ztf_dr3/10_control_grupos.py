# -*- coding: utf-8 -*-

"""
10_control_grupos.py

CONTROL DE FUGA POR GRUPOS

Objetivo:
Comprobar si la elevada capacidad para identificar M31, DEEP y DISK
se debe parcialmente a que objetos relacionados con un mismo campo,
observación, campaña, fuente o estructura de adquisición terminan
simultáneamente en TRAIN y TEST.

Compara:

1. Split aleatorio
2. GroupShuffleSplit
3. GroupKFold (si existen suficientes grupos)

IMPORTANTE:
No se inventa ninguna variable de agrupación.
El programa intenta localizar automáticamente columnas candidatas.
"""

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit,
    GroupKFold
)
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

RANDOM_STATE = 42
N_MUESTRA = 50_000

# Ruta relativa desde la ubicación del script
BASE_DIR = Path(__file__).resolve().parents[3]  # Sube 3 niveles hasta Astronomia/

# Directorio de resultados dentro de la misma carpeta que el script
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "10_"

# ---------------------------------------------------------------------------
# AJUSTA ESTAS RUTAS SI EN TUS SCRIPTS ANTERIORES SON DIFERENTES
# ---------------------------------------------------------------------------

DATASETS = {
    "M31": BASE_DIR / "dataset" / "ZTF_DR3" / "M31.csv",
    "DEEP": BASE_DIR / "dataset" / "ZTF_DR3" / "DEEP.csv",
    "DISK": BASE_DIR / "dataset" / "ZTF_DR3" / "DISK.csv",
}

# Features utilizadas en los análisis anteriores
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
# POSIBLES COLUMNAS DE GRUPO
# ============================================================================

GROUP_CANDIDATES = [
    "field",
    "field_id",
    "fieldid",
    "fieldId",

    "observation",
    "observation_id",
    "observationid",
    "obs_id",
    "obsid",

    "visit",
    "visit_id",
    "visitid",

    "exposure",
    "exposure_id",
    "exposureid",

    "campaign",
    "campaign_id",
    "campaignid",

    "pointing",
    "pointing_id",
    "pointingid",

    "tile",
    "tile_id",
    "tileid",

    "plate",
    "plate_id",
    "plateid",

    "run",
    "run_id",
    "runid",

    "frame",
    "frame_id",
    "frameid",

    "image",
    "image_id",
    "imageid",

    "source",
    "source_id",
    "sourceid",

    "object",
    "object_id",
    "objectid",

    "parent",
    "parent_id",
    "parentid",

    "region",
    "region_id",
    "regionid",
]


# ============================================================================
# UTILIDADES
# ============================================================================

def cargar_dataset(nombre, ruta):

    print(f"\nDataset: {nombre}")
    print(f"Ruta   : {ruta}")

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el archivo:\n{ruta}\n\n"
            f"Modifica DATASETS al principio del programa."
        )

    df = pd.read_csv(ruta)

    print(f"Objetos disponibles: {len(df):,}")
    print(f"Columnas            : {len(df.columns)}")

    if len(df) > N_MUESTRA:
        df = df.sample(
            N_MUESTRA,
            random_state=RANDOM_STATE
        ).copy()
    else:
        df = df.copy()

    print(f"Muestra utilizada   : {len(df):,}")

    return df


def buscar_columnas_grupo(df):

    encontradas = []

    columnas_lower = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidato in GROUP_CANDIDATES:

        if candidato.lower() in columnas_lower:

            original = columnas_lower[candidato.lower()]

            if original not in encontradas:
                encontradas.append(original)

    return encontradas


def analizar_columna_grupo(df, columna):

    s = df[columna]

    n = len(s)
    n_unique = s.nunique(dropna=False)

    counts = s.value_counts(dropna=False)

    grupos_con_multiples = int((counts > 1).sum())

    max_grupo = int(counts.max())

    return {
        "columna": columna,
        "filas": n,
        "grupos": n_unique,
        "grupos_multiples": grupos_con_multiples,
        "max_objetos_grupo": max_grupo,
        "ratio_grupos_filas": n_unique / n,
    }


def preparar_X(df):

    faltantes = [
        f for f in FEATURES
        if f not in df.columns
    ]

    if faltantes:
        raise RuntimeError(
            "Faltan features:\n" +
            "\n".join(faltantes)
        )

    X = df[FEATURES].copy()

    X = X.replace([np.inf, -np.inf], np.nan)

    # Eliminamos filas con NaN de forma conjunta
    valid = ~X.isna().any(axis=1)

    return X.loc[valid], valid


# ============================================================================
# MODELOS
# ============================================================================

def entrenar_random_forest(X_train, X_test, y_train, y_test):

    modelo = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_features="sqrt",
    )

    t0 = time.time()

    modelo.fit(X_train, y_train)

    pred = modelo.predict(X_test)

    tiempo = time.time() - t0

    acc = accuracy_score(y_test, pred)

    return acc, tiempo


def entrenar_logistica(X_train, X_test, y_train, y_test):

    modelo = Pipeline([
        (
            "scaler",
            RobustScaler()
        ),
        (
            "logistic",
            LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE,
                solver="lbfgs",
                multi_class="auto"
            )
        )
    ])

    t0 = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            ConvergenceWarning
        )

        modelo.fit(X_train, y_train)

    pred = modelo.predict(X_test)

    tiempo = time.time() - t0

    acc = accuracy_score(y_test, pred)

    return acc, tiempo


# ============================================================================
# SPLIT ALEATORIO
# ============================================================================

def experimento_random(X, y):

    print("\n" + "-" * 78)
    print("SPLIT ALEATORIO")
    print("-" * 78)

    indices = np.arange(len(y))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print(f"Train: {len(train_idx):,}")
    print(f"Test : {len(test_idx):,}")

    rf, t_rf = entrenar_random_forest(
        X_train,
        X_test,
        y_train,
        y_test
    )

    log, t_log = entrenar_logistica(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print(f"\nRandom Forest : {rf:.6f} ({t_rf:.2f} s)")
    print(f"Logística     : {log:.6f} ({t_log:.2f} s)")

    return rf, log


# ============================================================================
# GROUP SHUFFLE SPLIT
# ============================================================================

def experimento_group_shuffle(X, y, groups):

    print("\n" + "-" * 78)
    print("GROUP SHUFFLE SPLIT")
    print("-" * 78)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups)
    )

    grupos_train = set(groups.iloc[train_idx])
    grupos_test = set(groups.iloc[test_idx])

    solapamiento = grupos_train.intersection(grupos_test)

    print(f"Train: {len(train_idx):,}")
    print(f"Test : {len(test_idx):,}")

    print(
        f"Grupos train: {len(grupos_train):,}"
    )

    print(
        f"Grupos test : {len(grupos_test):,}"
    )

    print(
        f"Grupos compartidos: {len(solapamiento)}"
    )

    if len(solapamiento) != 0:
        raise RuntimeError(
            "ERROR: existen grupos compartidos entre TRAIN y TEST."
        )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    rf, t_rf = entrenar_random_forest(
        X_train,
        X_test,
        y_train,
        y_test
    )

    log, t_log = entrenar_logistica(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print(f"\nRandom Forest : {rf:.6f} ({t_rf:.2f} s)")
    print(f"Logística     : {log:.6f} ({t_log:.2f} s)")

    return rf, log


# ============================================================================
# GROUP K-FOLD
# ============================================================================

def experimento_group_kfold(X, y, groups):

    print("\n" + "-" * 78)
    print("GROUP K-FOLD")
    print("-" * 78)

    n_groups = groups.nunique()

    if n_groups < 5:

        print(
            f"Solo existen {n_groups} grupos."
        )

        print(
            "No se puede realizar GroupKFold con 5 folds."
        )

        return np.nan, np.nan

    n_splits = min(5, n_groups)

    splitter = GroupKFold(
        n_splits=n_splits
    )

    rf_scores = []
    log_scores = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups),
        start=1
    ):

        print(f"\nFold {fold}/{n_splits}")

        grupos_train = set(groups.iloc[train_idx])
        grupos_test = set(groups.iloc[test_idx])

        overlap = grupos_train.intersection(
            grupos_test
        )

        if overlap:
            raise RuntimeError(
                "Fuga de grupos detectada."
            )

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        rf, _ = entrenar_random_forest(
            X_train,
            X_test,
            y_train,
            y_test
        )

        log, _ = entrenar_logistica(
            X_train,
            X_test,
            y_train,
            y_test
        )

        rf_scores.append(rf)
        log_scores.append(log)

        print(f"RF  : {rf:.6f}")
        print(f"LOG : {log:.6f}")

    rf_mean = np.mean(rf_scores)
    log_mean = np.mean(log_scores)

    print("\nResultados GroupKFold:")
    print(
        f"RF  media: {rf_mean:.6f} "
        f"+/- {np.std(rf_scores):.6f}"
    )

    print(
        f"LOG media: {log_mean:.6f} "
        f"+/- {np.std(log_scores):.6f}"
    )

    return rf_mean, log_mean


# ============================================================================
# MAIN
# ============================================================================

def main():

    inicio = time.time()

    print("=" * 78)
    print("CONTROL DE FUGA POR GRUPOS")
    print("=" * 78)

    print(
        """
Objetivo:
Comprobar si la clasificación M31 / DEEP / DISK
se mantiene cuando objetos pertenecientes al mismo
grupo de observación no pueden aparecer simultáneamente
en TRAIN y TEST.
"""
    )

    # ------------------------------------------------------------------------
    # CARGA
    # ------------------------------------------------------------------------

    dfs = {}

    for nombre, ruta in DATASETS.items():
        dfs[nombre] = cargar_dataset(
            nombre,
            ruta
        )

    # ------------------------------------------------------------------------
    # BUSCAR COLUMNAS DE GRUPO
    # ------------------------------------------------------------------------

    print("\n" + "=" * 78)
    print("BÚSQUEDA DE VARIABLES DE AGRUPACIÓN")
    print("=" * 78)

    candidatos_globales = set()

    for nombre, df in dfs.items():

        encontrados = buscar_columnas_grupo(df)

        print(f"\n{nombre}")

        if encontrados:

            for c in encontrados:
                print(f"  ✓ {c}")

                candidatos_globales.add(c)

        else:

            print(
                "  No se encontró ninguna columna candidata."
            )

    # ------------------------------------------------------------------------
    # SI NO HAY GRUPO
    # ------------------------------------------------------------------------

    if not candidatos_globales:

        print("\n" + "=" * 78)
        print("NO SE HA ENCONTRADO UNA VARIABLE DE GRUPO")
        print("=" * 78)

        print(
            """
El programa NO puede realizar un control GroupShuffleSplit
sin conocer qué objetos pertenecen al mismo campo,
observación, exposición, campaña, etc.

Esto es importante:
NO debemos utilizar object_id como grupo simplemente porque
exista, ya que normalmente cada objeto tendría un grupo único
y el control perdería su significado.

Necesitamos una variable que relacione varios objetos.

Columnas disponibles en M31:
"""
        )

        print(
            list(dfs["M31"].columns)
        )

        print("\nPrograma terminado sin inventar un grupo.")

        return

    # ------------------------------------------------------------------------
    # MATRIZ
    # ------------------------------------------------------------------------

    Xs = []
    ys = []
    grupos = {}

    print("\n" + "=" * 78)
    print("CONSTRUCCIÓN DE MATRIZ")
    print("=" * 78)

    for nombre, df in dfs.items():

        X, valid = preparar_X(df)

        df_valid = df.loc[valid].copy()

        Xs.append(X)

        ys.append(
            pd.Series(
                nombre,
                index=X.index
            )
        )

        grupos[nombre] = df_valid

    X = pd.concat(
        Xs,
        axis=0,
        ignore_index=True
    )

    y = pd.concat(
        ys,
        axis=0,
        ignore_index=True
    )

    print(f"\nObjetos totales: {len(X):,}")
    print(f"Features       : {X.shape[1]}")

    # ------------------------------------------------------------------------
    # PROBAR CADA COLUMNA CANDIDATA
    # ------------------------------------------------------------------------

    resultados = []

    for columna in sorted(candidatos_globales):

        print("\n" + "=" * 78)
        print(f"ANÁLISIS DE GRUPO: {columna}")
        print("=" * 78)

        group_series = []

        valido = True

        for nombre in dfs:

            df = dfs[nombre]

            if columna not in df.columns:

                print(
                    f"{nombre}: no contiene '{columna}'"
                )

                valido = False
                break

            X_tmp, valid = preparar_X(df)

            g = df.loc[valid, columna]

            group_series.append(
                g.reset_index(drop=True)
            )

        if not valido:
            continue

        groups = pd.concat(
            group_series,
            ignore_index=True
        )

        # Convertimos NaN a identificador explícito
        groups = groups.astype(str)

        info = analizar_columna_grupo(
            pd.DataFrame({columna: groups}),
            columna
        )

        print(
            f"Filas             : {info['filas']:,}"
        )

        print(
            f"Grupos            : {info['grupos']:,}"
        )

        print(
            f"Grupos >1 objeto  : "
            f"{info['grupos_multiples']:,}"
        )

        print(
            f"Máximo por grupo  : "
            f"{info['max_objetos_grupo']:,}"
        )

        # --------------------------------------------------------------------
        # SPLIT ALEATORIO
        # --------------------------------------------------------------------

        rf_random, log_random = experimento_random(
            X,
            y
        )

        # --------------------------------------------------------------------
        # GROUP SHUFFLE
        # --------------------------------------------------------------------

        rf_group, log_group = experimento_group_shuffle(
            X,
            y,
            groups
        )

        # --------------------------------------------------------------------
        # GROUP K-FOLD
        # --------------------------------------------------------------------

        rf_kfold, log_kfold = experimento_group_kfold(
            X,
            y,
            groups
        )

        resultados.append({
            "grupo": columna,
            "objetos": len(X),
            "grupos": info["grupos"],
            "max_objetos_grupo":
                info["max_objetos_grupo"],
            "RF_random": rf_random,
            "LOG_random": log_random,
            "RF_group": rf_group,
            "LOG_group": log_group,
            "RF_kfold": rf_kfold,
            "LOG_kfold": log_kfold,
        })

    # ------------------------------------------------------------------------
    # RESULTADOS
    # ------------------------------------------------------------------------

    print("\n" + "=" * 78)
    print("RESUMEN FINAL")
    print("=" * 78)

    if not resultados:

        print(
            "No se pudo analizar ninguna columna."
        )

        return

    resultados_df = pd.DataFrame(
        resultados
    )

    print()

    print(
        resultados_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # ------------------------------------------------------------------------
    # GUARDAR
    # ------------------------------------------------------------------------

    salida = RESULTADOS_DIR / f"{PREFIJO}control_grupos_resultados.csv"

    resultados_df.to_csv(
        salida,
        index=False
    )

    print("\n" + "=" * 78)
    print("INTERPRETACIÓN")
    print("=" * 78)

    for _, r in resultados_df.iterrows():

        print(
            f"""
Grupo: {r['grupo']}

RF aleatorio : {r['RF_random']:.6f}
RF por grupo : {r['RF_group']:.6f}

LOG aleatorio: {r['LOG_random']:.6f}
LOG por grupo: {r['LOG_group']:.6f}
"""
        )

        delta_rf = (
            r["RF_random"] -
            r["RF_group"]
        )

        delta_log = (
            r["LOG_random"] -
            r["LOG_group"]
        )

        print(
            f"Caída RF  : {delta_rf:.6f}"
        )

        print(
            f"Caída LOG : {delta_log:.6f}"
        )

        if delta_rf > 0.10:

            print(
                "⚠️ FUERTE indicio de fuga/correlación por grupos."
            )

        elif delta_rf > 0.03:

            print(
                "⚠️ Existe un efecto apreciable de agrupación."
            )

        else:

            print(
                "La agrupación apenas modifica la clasificación."
            )

    print("\nResultado guardado en:")
    print(salida)

    print(
        f"\nTiempo total: {time.time() - inicio:.2f} s"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()