# -*- coding: utf-8 -*-

"""
======================================================================
24 - DETECCION DE ANOMALIAS (ISOLATION FOREST + LOF)
======================================================================

Punto de partida
-----------------
Toda la validacion previa (16, 16b, 16c, 16d, 20b, 21, 22, 23) ha
confirmado que:

  - Las curvas estan correctamente ordenadas en el tiempo -> ETA es
    de fiar.
  - El padding de cola y los empates por cuantizacion de mjd no
    distorsionan las features de forma relevante.
  - No hay leakage relevante entre TRAIN y TEST.
  - iloc[:25000]/iloc[:8000] es representativo de la poblacion
    completa (chi-cuadrado + deciles).
  - ETA y stetson_K son las variables mas importantes y mas estables
    del espacio de 12 variables; skew es matematicamente inestable
    cerca de cero; maximum_slope pierde ~16% de los pares por la
    cuantizacion de mjd (caveat conocido, impacto menor).
  - RRd y RS CVn son dificiles de clasificar en TODOS los regimenes
    de numero de observaciones -> es un problema real de solapamiento
    de clases, no un artefacto de muestreo.

Este script YA NO valida el clasificador supervisado. Da un paso de
fase: usa el espacio de 12 variables (ya verificado) para entrenar
detectores NO supervisados de anomalias sobre las 7 clases conocidas,
y evalua que objetos de TEST se apartan de ese espacio.

Metodo
------
Dos detectores independientes, entrenados solo sobre TRAIN (las 7
clases conocidas):

  - Isolation Forest: aísla puntos con pocos cortes en arboles
    aleatorios. Cuantos menos cortes hacen falta para aislar un
    punto, mas anomalo es.
  - Local Outlier Factor (modo 'novelty', fit en TRAIN, score en
    TEST): compara la densidad local de un punto con la de sus
    vecinos. Es sensible a anomalias locales que Isolation Forest
    puede pasar por alto (por ejemplo, un objeto que cae en una
    zona de baja densidad DENTRO de una clase, no fuera de todas).

Ambos dan una puntuacion de anomalia por objeto de TEST. Un candidato
"fuerte" es el que ambos metodos coinciden en senalar como anomalo
(interseccion del top N%, no solo uno de los dos).

El control critico
-------------------
Ya sabemos por el 21 que RF confunde sistematicamente RRd y RS CVn
con otras clases, en todos los regimenes de muestreo. Cualquier
candidato a anomalia que resulte ser simplemente "otro RRd mal
clasificado" no es un descubrimiento -- es el sesgo conocido del
clasificador disfrazado de anomalia. Por eso cada candidato se marca
con dos flags:

  - es_clase_dificil: pertenece a RRd o RS CVn.
  - mal_clasificado_rf: el RF (reentrenado aqui, mismo criterio que
    el 21) no acierta su clase.

Los candidatos interesantes de verdad son los que tienen score alto
de anomalia SIN estar explicados por ninguno de esos dos flags.

Robustez frente al periodo
---------------------------
El 12 mostro que 'period' (precomputado por StarEmbed, no por
nosotros) es una variable muy informativa para clasificacion. Pero
no sabemos aun si su comportamiento como feature de anomalias es
igual de fiable que las 12 variables ya auditadas. Por eso el script
corre el pipeline completo DOS veces -- sin periodo y con periodo -- y
compara cuanto se solapan los candidatos fuertes entre ambas
versiones. Si el solapamiento es alto, el resultado es robusto a esa
decision. Si es bajo, hay que decidir con mas cuidado si confiar en
'period' para esto.
======================================================================
"""

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

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
PREFIJO = "24_"

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

VARIABLES_BASE = [
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

CLASES_DIFICILES = ["RRd", "RS CVn"]

# Porcentaje superior de cada metodo que se considera "candidato".
# 0.02 = top 2%. Sobre 8000 objetos de TEST, ~160 candidatos por
# metodo antes de cruzar con el otro.
TOP_PERCENTIL = 0.02

# Columna de identificador del objeto, si existe. Si no esta en el
# parquet, se usa el indice de fila como identificador de respaldo.
POSIBLES_COLUMNAS_ID = ["object_id", "source_id", "sourceid", "id"]


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

    train = pd.concat(frames, ignore_index=True)

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


def obtener_columna_id(df):

    for col in POSIBLES_COLUMNAS_ID:
        if col in df.columns:
            return col

    return None


# ======================================================================
# ACCESO A LA BANDA / CURVA Y CALCULO DE FEATURES (12 VARIABLES)
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


def calcular_features_objeto(fila, usar_periodo):

    x, t = obtener_curva_cruda(fila)

    if len(x) == 0:
        vals = {v: np.nan for v in VARIABLES_BASE}
        if usar_periodo:
            vals["period"] = np.nan
        return vals

    media = np.mean(x)
    mediana = np.median(x)
    std = np.std(x)
    mad = np.median(np.abs(x - mediana))

    minimo = np.min(x)
    maximo = np.max(x)
    amplitud = (maximo - minimo) / 2.0

    percent_amplitude = amplitud / abs(media) if abs(media) > 1e-12 else 0.0

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
        denom = np.sqrt(np.mean(z ** 2))
        stetson_K = np.mean(np.abs(z)) / denom if denom > 1e-12 else 0.0
        chi2 = np.sum(z ** 2) / max(len(x) - 1, 1)
    else:
        skew = 0.0
        kurtosis = 0.0
        stetson_K = 0.0
        chi2 = 0.0

    if len(x) > 1 and std > 1e-12:
        diferencias = np.diff(x)
        eta = np.sum(diferencias ** 2) / ((len(x) - 1) * std ** 2)
    else:
        eta = 0.0

    if len(t) > 1:
        dt = np.diff(t)
        dy = np.diff(x)
        valid = dt > 0
        max_slope = float(np.max(np.abs(dy[valid] / dt[valid]))) if np.any(valid) else 0.0
    else:
        max_slope = 0.0

    vals = {
        "median": mediana,
        "standard_deviation": std,
        "median_absolute_deviation": mad,
        "amplitude": amplitud,
        "percent_amplitude": percent_amplitude,
        "inter_percentile_range_25": iqr25,
        "skew": skew,
        "kurtosis": kurtosis,
        "stetson_K": stetson_K,
        "eta": eta,
        "chi2": chi2,
        "maximum_slope": max_slope,
    }

    if usar_periodo:
        vals["period"] = fila.get("period", np.nan)

    return vals


def extraer_features(df, usar_periodo, etiqueta):

    filas = []
    n = len(df)

    for i, (_, fila) in enumerate(df.iterrows()):
        if i % 5000 == 0:
            print(f"  [{etiqueta}] {i:,}/{n:,}")
        filas.append(calcular_features_objeto(fila, usar_periodo))

    return pd.DataFrame(filas)


def limpiar(X_train, X_test):

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    medianas = X_train.median()

    X_train = X_train.fillna(medianas).fillna(0.0)
    X_test = X_test.fillna(medianas).fillna(0.0)

    return X_train, X_test


# ======================================================================
# CLASIFICADOR SUPERVISADO DE CONTROL (MISMO CRITERIO QUE EL 21)
# ======================================================================

def entrenar_rf_control(X_train, y_train, X_test, y_test, variables):

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_features="sqrt",
    )

    rf.fit(X_train[variables], y_train)
    pred = rf.predict(X_test[variables])

    balanced = balanced_accuracy_score(y_test, pred)
    print(f"RF de control -> balanced accuracy: {balanced:.4f}")

    mal_clasificado = (pred != y_test.values)

    return mal_clasificado, pred


# ======================================================================
# DETECTORES DE ANOMALIAS
# ======================================================================

def entrenar_detectores(X_train_scaled, X_test_scaled):

    print()
    print("Entrenando Isolation Forest...")

    iso = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X_train_scaled)

    # score_samples: mayor = mas normal. Invertimos el signo para que
    # mayor = mas anomalo, mas intuitivo para ordenar candidatos.
    score_if = -iso.score_samples(X_test_scaled)

    print("Entrenando Local Outlier Factor (modo novelty)...")

    lof = LocalOutlierFactor(
        n_neighbors=20,
        novelty=True,
        contamination="auto",
        n_jobs=-1,
    )
    lof.fit(X_train_scaled)

    # score_samples de LOF: mayor = mas normal (igual criterio que IF).
    # Invertimos igual que arriba.
    score_lof = -lof.score_samples(X_test_scaled)

    return score_if, score_lof


# ======================================================================
# PIPELINE COMPLETO PARA UNA CONFIGURACION (CON O SIN PERIODO)
# ======================================================================

def ejecutar_pipeline(train, test, usar_periodo, etiqueta):

    banner(f"PIPELINE DE ANOMALIAS - {etiqueta}")

    variables = list(VARIABLES_BASE)
    if usar_periodo:
        variables = variables + ["period"]

    print()
    print("Extrayendo features de TRAIN...")
    X_train = extraer_features(train, usar_periodo, f"{etiqueta}_TRAIN")

    print()
    print("Extrayendo features de TEST...")
    X_test = extraer_features(test, usar_periodo, f"{etiqueta}_TEST")

    X_train, X_test = limpiar(X_train, X_test)

    y_train = train["class_str"].astype(str)
    y_test = test["class_str"].astype(str)

    # ------------------------------------------------------------------
    # Clasificador de control (para el flag mal_clasificado_rf)
    # ------------------------------------------------------------------

    print()
    mal_clasificado, pred_rf = entrenar_rf_control(
        X_train, y_train, X_test, y_test, variables
    )

    # ------------------------------------------------------------------
    # Escalado y deteccion de anomalias
    # ------------------------------------------------------------------

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[variables])
    X_test_scaled = scaler.transform(X_test[variables])

    score_if, score_lof = entrenar_detectores(X_train_scaled, X_test_scaled)

    # ------------------------------------------------------------------
    # Construir tabla de resultados
    # ------------------------------------------------------------------

    col_id = obtener_columna_id(test)

    tabla = pd.DataFrame({
        "fila_test": np.arange(len(test)),
        "id_objeto": test[col_id].values if col_id else np.arange(len(test)),
        "clase": y_test.values,
        "prediccion_rf": pred_rf,
        "mal_clasificado_rf": mal_clasificado,
        "es_clase_dificil": y_test.isin(CLASES_DIFICILES).values,
        "score_isolation_forest": score_if,
        "score_lof": score_lof,
    })

    tabla["rank_if"] = tabla["score_isolation_forest"].rank(
        ascending=False, method="min"
    ).astype(int)
    tabla["rank_lof"] = tabla["score_lof"].rank(
        ascending=False, method="min"
    ).astype(int)

    umbral_n = max(int(len(tabla) * TOP_PERCENTIL), 1)

    tabla["top_if"] = tabla["rank_if"] <= umbral_n
    tabla["top_lof"] = tabla["rank_lof"] <= umbral_n

    tabla["candidato_fuerte"] = tabla["top_if"] & tabla["top_lof"]

    tabla["candidato_interesante"] = (
        tabla["candidato_fuerte"]
        & (~tabla["es_clase_dificil"])
        & (~tabla["mal_clasificado_rf"])
    )

    # ------------------------------------------------------------------
    # Resumenes
    # ------------------------------------------------------------------

    n_fuertes = int(tabla["candidato_fuerte"].sum())
    n_interesantes = int(tabla["candidato_interesante"].sum())

    print()
    print(f"Umbral top {TOP_PERCENTIL * 100:.0f}%: {umbral_n} objetos por metodo")
    print(f"Candidatos fuertes (IF y LOF coinciden): {n_fuertes}")
    print(
        f"  De ellos, 'interesantes' (no son clase dificil NI mal "
        f"clasificados por RF): {n_interesantes}"
    )

    resumen_clase = tabla.groupby("clase").agg(
        n=("clase", "size"),
        score_if_medio=("score_isolation_forest", "mean"),
        score_lof_medio=("score_lof", "mean"),
        candidatos_fuertes=("candidato_fuerte", "sum"),
    ).reset_index()

    print()
    print("Resumen de anomalia por clase:")
    print(resumen_clase.to_string(index=False))

    return tabla, resumen_clase


# ======================================================================
# MAIN
# ======================================================================

def main():

    inicio_total = time.time()

    banner("24 - DETECCION DE ANOMALIAS (ISOLATION FOREST + LOF)")

    train, test = cargar_datos()

    if "class_str" not in train.columns or "class_str" not in test.columns:
        raise RuntimeError("Falta la columna class_str.")

    tiene_periodo = "period" in test.columns

    if not tiene_periodo:
        print()
        print(
            "AVISO: no se encontro la columna 'period' en el parquet. "
            "Solo se ejecutara la version SIN periodo."
        )

    # ------------------------------------------------------------------
    # PIPELINE SIN PERIODO
    # ------------------------------------------------------------------

    tabla_sin, resumen_sin = ejecutar_pipeline(
        train, test, usar_periodo=False, etiqueta="SIN_PERIODO"
    )

    tabla_sin.to_csv(
        RESULTADOS_DIR / f"{PREFIJO}scores_anomalia_sin_periodo.csv",
        index=False,
    )
    resumen_sin.to_csv(
        RESULTADOS_DIR / f"{PREFIJO}resumen_por_clase_sin_periodo.csv",
        index=False,
    )

    tabla_con = None

    # ------------------------------------------------------------------
    # PIPELINE CON PERIODO (SI EXISTE LA COLUMNA)
    # ------------------------------------------------------------------

    if tiene_periodo:

        tabla_con, resumen_con = ejecutar_pipeline(
            train, test, usar_periodo=True, etiqueta="CON_PERIODO"
        )

        tabla_con.to_csv(
            RESULTADOS_DIR / f"{PREFIJO}scores_anomalia_con_periodo.csv",
            index=False,
        )
        resumen_con.to_csv(
            RESULTADOS_DIR / f"{PREFIJO}resumen_por_clase_con_periodo.csv",
            index=False,
        )

        # --------------------------------------------------------------
        # COMPARAR ROBUSTEZ: ¿coinciden los candidatos con y sin periodo?
        # --------------------------------------------------------------

        banner("ROBUSTEZ: CANDIDATOS SIN PERIODO vs CON PERIODO")

        ids_sin = set(
            tabla_sin.loc[tabla_sin["candidato_fuerte"], "fila_test"]
        )
        ids_con = set(
            tabla_con.loc[tabla_con["candidato_fuerte"], "fila_test"]
        )

        interseccion = ids_sin & ids_con
        union = ids_sin | ids_con

        jaccard = len(interseccion) / len(union) if union else np.nan

        print()
        print(f"Candidatos fuertes SIN periodo: {len(ids_sin)}")
        print(f"Candidatos fuertes CON periodo: {len(ids_con)}")
        print(f"En comun: {len(interseccion)}")
        print(f"Indice de Jaccard (solapamiento): {jaccard:.3f}")

        if jaccard >= 0.5:
            print(
                "  -> Solapamiento razonable: los candidatos son "
                "relativamente estables independientemente de si se "
                "incluye 'period'."
            )
        else:
            print(
                "  -> Solapamiento bajo: la lista de candidatos depende "
                "mucho de si se incluye 'period'. Conviene decidir con "
                "cuidado (o inspeccionar ambas listas) antes de dar "
                "cualquier candidato por bueno."
            )

    # ------------------------------------------------------------------
    # CANDIDATOS FINALES PARA INSPECCION MANUAL
    # ------------------------------------------------------------------

    banner("CANDIDATOS PARA INSPECCION MANUAL")

    tabla_final = tabla_con if tabla_con is not None else tabla_sin

    top_candidatos = tabla_final[
        tabla_final["candidato_interesante"]
    ].sort_values(
        by=["score_isolation_forest", "score_lof"], ascending=False
    )

    print()
    print(
        f"{len(top_candidatos)} candidatos 'interesantes' "
        f"(candidato fuerte, no es clase dificil, RF los clasifica bien)."
    )
    print()
    if len(top_candidatos) > 0:
        print(
            top_candidatos[
                ["id_objeto", "clase", "prediccion_rf",
                 "score_isolation_forest", "score_lof"]
            ].head(20).to_string(index=False)
        )

    top_candidatos.to_csv(
        RESULTADOS_DIR / f"{PREFIJO}candidatos_top.csv", index=False
    )

    banner("RESUMEN E INTERPRETACION")

    print(
        "Siguiente paso recomendado (fuera de este script): tomar la "
        "lista de '24_candidatos_top.csv' y, para cada objeto, mirar "
        "la curva de luz real e intentar un cruce con catalogos "
        "externos (VSX, SIMBAD, Gaia DR3 variability) usando "
        "'id_objeto'. Un candidato solo pasa a ser una anomalia "
        "cientifica cuando esa inspeccion no lo explica como ninguna "
        "clase conocida ni como artefacto."
    )
    print()
    print(
        "Recuerda: 'candidato_fuerte' incluye objetos de clases "
        "dificiles y mal clasificados por el RF -- esos SIGUEN estando "
        "en los CSV completos ('24_scores_anomalia_*.csv'), por si "
        "quieres revisarlos aparte, pero no estan en "
        "'24_candidatos_top.csv' precisamente para no confundir sesgo "
        "de clasificador con anomalia real."
    )

    print()
    print(f"Tiempo total: {time.time() - inicio_total:.2f} segundos")


if __name__ == "__main__":
    main()