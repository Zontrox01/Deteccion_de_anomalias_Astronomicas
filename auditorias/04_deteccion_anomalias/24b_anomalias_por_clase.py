# -*- coding: utf-8 -*-

"""
======================================================================
24b - ANOMALIAS NORMALIZADAS POR CLASE (CONTROL DE TAMANO DE MUESTRA)
======================================================================

Motivacion
----------
El 24 encontro que LPV esta sobrerrepresentada 15-18 veces su tasa
base entre los candidatos fuertes, de forma consistente con y sin
'period'. Pero LPV es tambien la clase mas pequena de TRAIN (~1% de
25.000 objetos). Con tan poca muestra, tanto Isolation Forest como
LOF tienen mucha menos densidad de referencia en esa zona del
espacio de features -- asi que CUALQUIER objeto LPV tiene mas
papeletas de parecer anomalo simplemente por falta de ejemplos
"normales" con los que compararlo, no porque sea realmente raro
dentro de su propia clase.

Este script resuelve la ambiguedad entrenando un detector de
anomalias INDEPENDIENTE para cada clase, usando solo los objetos de
esa clase en TRAIN como referencia. Un objeto ya no se compara con
"todo TRAIN" sino con "los demas objetos de su misma clase predicha".
Esto responde a la pregunta correcta: ¿es este LPV raro para ser un
LPV? -- no "¿es este LPV raro comparado con un EW o una RRc?", que es
una comparacion que no tiene mucho sentido fisico de entrada.

Con esto, el efecto de "clase pequena = parece mas rara por defecto"
queda neutralizado: cada clase se juzga contra su propio tamano de
muestra, en igualdad de condiciones.

Metodo
------
Para cada clase con al menos MIN_OBJETOS_POR_CLASE ejemplos en TRAIN:

  1. Se entrena un Isolation Forest y un LOF (novelty=True) SOLO con
     los objetos de esa clase en TRAIN.
  2. Se punkua a los objetos de TEST de esa misma clase (segun la
     clase real, no la predicha, para no mezclar el control con los
     errores del clasificador supervisado).
  3. El score se normaliza a un percentil DENTRO de esa clase (0-100),
     para que sea directamente comparable entre clases de tamanos muy
     distintos.

Al final se comparan los candidatos fuertes de este metodo (normalizado
por clase) contra los del 24 (global), y se mide cuantos LPV
sobreviven al control -- esa es la respuesta a la pregunta que abrio
este script.

Clases con menos de MIN_OBJETOS_POR_CLASE ejemplos en TRAIN se
excluyen de este analisis (no hay base suficiente para entrenar un
detector fiable solo con esa clase) y se marcan como tal en la salida.
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
PREFIJO = "24b_"

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
    "period",
]

CLASES_DIFICILES = ["RRd", "RS CVn"]

# Minimo de objetos de TRAIN necesarios para entrenar un detector solo
# con esa clase. Por debajo de esto, el resultado seria demasiado
# ruidoso para confiar en el.
MIN_OBJETOS_POR_CLASE = 50

# Percentil (dentro de la propia clase) a partir del cual un objeto se
# considera candidato. 98 = top 2% de su clase, igual criterio que el
# TOP_PERCENTIL del 24 para poder comparar directamente.
PERCENTIL_CANDIDATO = 98

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
# ACCESO A LA BANDA / CURVA Y CALCULO DE FEATURES (12 VARIABLES + PERIOD)
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


def calcular_features_objeto(fila):

    x, t = obtener_curva_cruda(fila)

    if len(x) == 0:
        vals = {v: np.nan for v in VARIABLES}
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
        "eta": eta,
        "chi2": chi2,
        "maximum_slope": max_slope,
        "period": fila.get("period", np.nan),
    }


def extraer_features(df, etiqueta):

    filas = []
    n = len(df)

    for i, (_, fila) in enumerate(df.iterrows()):
        if i % 5000 == 0:
            print(f"  [{etiqueta}] {i:,}/{n:,}")
        filas.append(calcular_features_objeto(fila))

    return pd.DataFrame(filas)


def limpiar(X_train, X_test):

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    medianas = X_train.median()

    X_train = X_train.fillna(medianas).fillna(0.0)
    X_test = X_test.fillna(medianas).fillna(0.0)

    return X_train, X_test


# ======================================================================
# CLASIFICADOR SUPERVISADO DE CONTROL (MISMO CRITERIO QUE 21/24)
# ======================================================================

def entrenar_rf_control(X_train, y_train, X_test, y_test):

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_features="sqrt",
    )

    rf.fit(X_train[VARIABLES], y_train)
    pred = rf.predict(X_test[VARIABLES])

    balanced = balanced_accuracy_score(y_test, pred)
    print(f"RF de control -> balanced accuracy: {balanced:.4f}")

    mal_clasificado = (pred != y_test.values)

    return mal_clasificado, pred


# ======================================================================
# DETECCION DE ANOMALIAS POR CLASE
# ======================================================================

def detectar_anomalias_por_clase(X_train, y_train, X_test, y_test):

    banner("ENTRENANDO UN DETECTOR INDEPENDIENTE POR CLASE")

    clases = sorted(y_train.unique())

    resultados = []

    for clase in clases:

        mask_train = (y_train.values == clase)
        n_train_clase = int(mask_train.sum())

        mask_test = (y_test.values == clase)
        n_test_clase = int(mask_test.sum())

        print()
        print(f"--- Clase: {clase} ({n_train_clase} en TRAIN, {n_test_clase} en TEST) ---")

        if n_train_clase < MIN_OBJETOS_POR_CLASE:
            print(
                f"  Omitida: menos de {MIN_OBJETOS_POR_CLASE} objetos "
                f"en TRAIN, no hay base suficiente para un detector "
                f"fiable solo con esta clase."
            )

            for idx in np.where(mask_test)[0]:
                resultados.append({
                    "fila_test": idx,
                    "clase": clase,
                    "score_if_percentil_clase": np.nan,
                    "score_lof_percentil_clase": np.nan,
                    "n_train_clase": n_train_clase,
                    "clase_con_base_suficiente": False,
                })
            continue

        scaler = StandardScaler()
        X_train_clase = scaler.fit_transform(X_train.loc[mask_train, VARIABLES])
        X_test_clase = scaler.transform(X_test.loc[mask_test, VARIABLES])

        iso = IsolationForest(
            n_estimators=300,
            contamination="auto",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        iso.fit(X_train_clase)
        score_if = -iso.score_samples(X_test_clase)

        # LOF necesita mas vecinos disponibles que n_neighbors; se
        # ajusta automaticamente si la clase es pequena.
        n_vecinos = min(20, max(n_train_clase - 1, 1))

        lof = LocalOutlierFactor(
            n_neighbors=n_vecinos,
            novelty=True,
            contamination="auto",
            n_jobs=-1,
        )
        lof.fit(X_train_clase)
        score_lof = -lof.score_samples(X_test_clase)

        # percentil DENTRO de la clase (0-100), para poder comparar
        # clases de tamanos muy distintos en igualdad de condiciones
        percentil_if = pd.Series(score_if).rank(pct=True).values * 100
        percentil_lof = pd.Series(score_lof).rank(pct=True).values * 100

        indices_test_clase = np.where(mask_test)[0]

        for j, idx in enumerate(indices_test_clase):
            resultados.append({
                "fila_test": idx,
                "clase": clase,
                "score_if_percentil_clase": percentil_if[j],
                "score_lof_percentil_clase": percentil_lof[j],
                "n_train_clase": n_train_clase,
                "clase_con_base_suficiente": True,
            })

    return pd.DataFrame(resultados)


# ======================================================================
# MAIN
# ======================================================================

def main():

    inicio_total = time.time()

    banner("24b - ANOMALIAS NORMALIZADAS POR CLASE")

    train, test = cargar_datos()

    if "class_str" not in train.columns or "class_str" not in test.columns:
        raise RuntimeError("Falta la columna class_str.")

    if "period" not in test.columns:
        raise RuntimeError(
            "Este script asume que existe la columna 'period' "
            "(el 24 confirmo que si esta disponible). Si no la "
            "tienes, quita 'period' de VARIABLES arriba."
        )

    print()
    print("Extrayendo features de TRAIN...")
    X_train = extraer_features(train, "TRAIN")

    print()
    print("Extrayendo features de TEST...")
    X_test = extraer_features(test, "TEST")

    X_train, X_test = limpiar(X_train, X_test)

    y_train = train["class_str"].astype(str)
    y_test = test["class_str"].astype(str)

    # ------------------------------------------------------------------
    # Clasificador de control (para poder cruzar con mal_clasificado_rf)
    # ------------------------------------------------------------------

    banner("CLASIFICADOR DE CONTROL")

    mal_clasificado, pred_rf = entrenar_rf_control(
        X_train, y_train, X_test, y_test
    )

    # ------------------------------------------------------------------
    # Deteccion de anomalias POR CLASE
    # ------------------------------------------------------------------

    resultados_clase = detectar_anomalias_por_clase(
        X_train, y_train, X_test, y_test
    )

    # ------------------------------------------------------------------
    # CONSTRUIR TABLA FINAL
    # ------------------------------------------------------------------

    col_id = obtener_columna_id(test)

    tabla = pd.DataFrame({
        "fila_test": np.arange(len(test)),
        "id_objeto": test[col_id].values if col_id else np.arange(len(test)),
        "clase": y_test.values,
        "prediccion_rf": pred_rf,
        "mal_clasificado_rf": mal_clasificado,
        "es_clase_dificil": y_test.isin(CLASES_DIFICILES).values,
    })

    tabla = tabla.merge(
        resultados_clase[[
            "fila_test", "score_if_percentil_clase",
            "score_lof_percentil_clase", "n_train_clase",
            "clase_con_base_suficiente",
        ]],
        on="fila_test", how="left",
    )

    tabla["candidato_fuerte_por_clase"] = (
        (tabla["score_if_percentil_clase"] >= PERCENTIL_CANDIDATO)
        & (tabla["score_lof_percentil_clase"] >= PERCENTIL_CANDIDATO)
        & (tabla["clase_con_base_suficiente"])
    )

    tabla["candidato_interesante_por_clase"] = (
        tabla["candidato_fuerte_por_clase"]
        & (~tabla["es_clase_dificil"])
        & (~tabla["mal_clasificado_rf"])
    )

    tabla.to_csv(
        RESULTADOS_DIR / f"{PREFIJO}scores_anomalia_por_clase.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # RESUMEN POR CLASE
    # ------------------------------------------------------------------

    banner("RESUMEN DE CANDIDATOS POR CLASE (NORMALIZADO)")

    resumen = tabla.groupby("clase").agg(
        n=("clase", "size"),
        n_train_clase=("n_train_clase", "first"),
        candidatos_fuertes=("candidato_fuerte_por_clase", "sum"),
    ).reset_index()

    resumen["pct_poblacion"] = 100 * resumen["n"] / resumen["n"].sum()
    total_candidatos = resumen["candidatos_fuertes"].sum()
    resumen["pct_candidatos"] = (
        100 * resumen["candidatos_fuertes"] / total_candidatos
        if total_candidatos > 0 else 0.0
    )
    resumen["ratio_sobrerrepresentacion"] = (
        resumen["pct_candidatos"] / resumen["pct_poblacion"]
    ).replace([np.inf, -np.inf], np.nan)

    print(resumen.round(2).to_string(index=False))

    resumen.to_csv(
        RESULTADOS_DIR / f"{PREFIJO}resumen_por_clase.csv", index=False
    )

    # ------------------------------------------------------------------
    # COMPARAR CON EL 24 (GLOBAL): ¿SOBREVIVE LPV AL CONTROL?
    # ------------------------------------------------------------------

    banner("COMPARACION: ¿SOBREVIVEN LOS CANDIDATOS LPV DEL 24 AL CONTROL?")

    ruta_24 = RESULTADOS_DIR / "24_scores_anomalia_con_periodo.csv"

    if ruta_24.exists():

        tabla_24 = pd.read_csv(ruta_24)

        candidatos_24 = set(
            tabla_24.loc[tabla_24["candidato_fuerte"], "fila_test"]
        )
        candidatos_24b = set(
            tabla.loc[tabla["candidato_fuerte_por_clase"], "fila_test"]
        )

        interseccion = candidatos_24 & candidatos_24b

        print()
        print(f"Candidatos fuertes en el 24 (global)      : {len(candidatos_24)}")
        print(f"Candidatos fuertes en el 24b (por clase)  : {len(candidatos_24b)}")
        print(f"En comun (sobreviven a ambos criterios)   : {len(interseccion)}")

        candidatos_24_lpv = set(
            tabla_24.loc[
                (tabla_24["candidato_fuerte"]) & (tabla_24["clase"] == "LPV"),
                "fila_test",
            ]
        )
        lpv_que_sobreviven = candidatos_24_lpv & candidatos_24b

        print()
        print(f"LPV candidatos en el 24                   : {len(candidatos_24_lpv)}")
        print(
            f"De esos, tambien candidatos en el 24b "
            f"(es decir, LPV raros DENTRO de su propia clase): "
            f"{len(lpv_que_sobreviven)}"
        )

        if len(candidatos_24_lpv) > 0:
            pct = 100.0 * len(lpv_que_sobreviven) / len(candidatos_24_lpv)
            print(f"  -> {pct:.1f}% de los LPV candidatos del 24 sobreviven al control.")

            if pct < 30:
                print(
                    "  -> Mayoritariamente NO sobreviven: la sobrerrepresentacion "
                    "de LPV en el 24 era, en buena parte, un efecto de poca "
                    "muestra de entrenamiento, no anomalia real."
                )
            elif pct > 70:
                print(
                    "  -> La mayoria SI sobreviven: los LPV candidatos del 24 "
                    "parecen ser realmente atipicos dentro de su propia clase, "
                    "no solo un artefacto de tamano de muestra."
                )
            else:
                print(
                    "  -> Resultado mixto: unos sobreviven y otros no. Revisa "
                    "'24b_scores_anomalia_por_clase.csv' para ver cuales."
                )
    else:
        print(
            f"No se encontro {ruta_24} en el directorio. Ejecuta el 24 "
            f"primero (con periodo) si quieres esta comparacion directa."
        )

    # ------------------------------------------------------------------
    # CANDIDATOS FINALES (INTERESANTES, NORMALIZADOS POR CLASE)
    # ------------------------------------------------------------------

    banner("CANDIDATOS INTERESANTES (NORMALIZADOS POR CLASE)")

    top = tabla[tabla["candidato_interesante_por_clase"]].sort_values(
        by=["score_if_percentil_clase", "score_lof_percentil_clase"],
        ascending=False,
    )

    print()
    print(f"{len(top)} candidatos interesantes tras el control por clase.")
    if len(top) > 0:
        print(
            top[[
                "id_objeto", "clase", "prediccion_rf",
                "score_if_percentil_clase", "score_lof_percentil_clase",
            ]].to_string(index=False)
        )

    top.to_csv(
        RESULTADOS_DIR / f"{PREFIJO}candidatos_top.csv", index=False
    )

    print()
    print(f"Tiempo total: {time.time() - inicio_total:.2f} segundos")


if __name__ == "__main__":
    main()