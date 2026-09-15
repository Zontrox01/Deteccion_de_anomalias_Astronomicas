# -*- coding: utf-8 -*-

"""
======================================================================
20b - CONTROL DE REPRESENTATIVIDAD DEL TRUNCADO iloc[:MAX_TRAIN/TEST]
======================================================================

Motivacion
----------
Desde el 20 en adelante (20, 21, 22, 23, y los propios 16b/16c/16d),
todo el pipeline carga los datos asi:

    train = train.iloc[:MAX_TRAIN]   # primeros 25.000
    test  = test.iloc[:MAX_TEST]     # primeros 8.000

Esto es bueno para la reproducibilidad entre experimentos (todos usan
exactamente el mismo subconjunto), pero deja una pregunta sin
responder: ¿el parquet original de StarEmbed ya viene barajado, o
coger "los primeros N" introduce un sesgo porque el orden de
almacenamiento correlaciona con la clase, el numero de observaciones,
el periodo, o cualquier otra variable relevante?

Este script responde eso de forma directa, sin dar nada por supuesto:

  1. Carga el parquet COMPLETO (sin truncar) para train y test.
  2. Compara la composicion del subconjunto actual (iloc[:MAX]) contra
     la poblacion completa: distribucion de clases, numero de
     observaciones y periodo (si existe la columna).
  3. Divide el dataset completo en 10 bloques segun su posicion
     original en el parquet (deciles de indice de fila) y mira si
     esas variables cambian de forma sistematica segun la posicion.
     Si el parquet estuviera ordenado por algun criterio (campo,
     fecha de ingesta, clase...), esto lo pondria de manifiesto de
     forma muy visible.
  4. Test estadistico: chi-cuadrado comparando la distribucion de
     clases del subconjunto iloc[:MAX] contra la distribucion de
     clases de la poblacion completa.
  5. Impacto practico: entrena el clasificador de 12 variables sobre
     el subconjunto ACTUAL (iloc[:MAX], sin barajar) y sobre un
     subconjunto ALEATORIO del mismo tamano, evaluando ambos contra
     el resto de la poblacion (lo que no entra en ninguno de los dos
     subconjuntos de entrenamiento) para ver si hay diferencia real
     de rendimiento en un conjunto de validacion mas amplio.

Con esto se cierra la duda con evidencia, no con suposiciones.
======================================================================
"""

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chisquare

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
PREFIJO = "20b_"

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

N_DECILES = 10

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

SALIDA_DECILES_TRAIN = RESULTADOS_DIR / f"{PREFIJO}deciles_train.csv"
SALIDA_DECILES_TEST = RESULTADOS_DIR / f"{PREFIJO}deciles_test.csv"
SALIDA_COMPARACION_CLASES = RESULTADOS_DIR / f"{PREFIJO}comparacion_clases.csv"
SALIDA_COMPARACION_CLASES_TRAIN = RESULTADOS_DIR / f"{PREFIJO}comparacion_clases_train.csv"
SALIDA_COMPARACION_CLASES_TEST = RESULTADOS_DIR / f"{PREFIJO}comparacion_clases_test.csv"
SALIDA_RESULTADOS_IMPACTO = RESULTADOS_DIR / f"{PREFIJO}resultados_impacto.csv"


def banner(texto):
    print()
    print("=" * 70)
    print(texto)
    print("=" * 70)


# ======================================================================
# CARGA COMPLETA (SIN TRUNCAR)
# ======================================================================

def cargar_datos_completos():

    banner("CARGA DE DATOS COMPLETOS (SIN TRUNCAR)")

    frames = []

    for fichero in TRAIN_FILES:
        print("Cargando TRAIN:", fichero)

        if not fichero.exists():
            raise FileNotFoundError(fichero)

        df = pd.read_parquet(fichero)
        frames.append(df)
        print("  Objetos:", f"{len(df):,}")

    train_completo = pd.concat(frames, ignore_index=True)

    print()
    print("Cargando TEST:", TEST_FILE)

    if not TEST_FILE.exists():
        raise FileNotFoundError(TEST_FILE)

    test_completo = pd.read_parquet(TEST_FILE)

    print()
    print("TRAIN completo:", f"{len(train_completo):,}")
    print("TEST completo :", f"{len(test_completo):,}")
    print()
    print(f"Subconjunto actual usado en 20-23: {MAX_TRAIN:,} TRAIN / {MAX_TEST:,} TEST")

    if len(train_completo) <= MAX_TRAIN:
        print(
            "AVISO: el TRAIN completo no es mayor que MAX_TRAIN, "
            "no hay nada que truncar. Este control no aplica."
        )

    if len(test_completo) <= MAX_TEST:
        print(
            "AVISO: el TEST completo no es mayor que MAX_TEST, "
            "no hay nada que truncar. Este control no aplica."
        )

    return train_completo, test_completo


# ======================================================================
# ACCESO A LA BANDA / CURVA (PARA n_observaciones)
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


def contar_observaciones(fila):

    bands = fila.get("bands_data", None)
    banda = obtener_banda(bands)

    if banda is None:
        return 0

    target = banda.get("target", [])

    if target is None:
        return 0

    try:
        x = np.asarray(target, dtype=float)
    except Exception:
        return 0

    return int(np.sum(np.isfinite(x)))


# ======================================================================
# PASO 1: COMPARAR SUBCONJUNTO ACTUAL vs POBLACION COMPLETA
# ======================================================================

def comparar_subconjunto_vs_poblacion(df_completo, max_n, nombre):

    banner(f"SUBCONJUNTO ACTUAL (iloc[:{max_n}]) vs POBLACION COMPLETA - {nombre}")

    if len(df_completo) <= max_n:
        print("No aplica (poblacion no mayor que el subconjunto usado).")
        return None

    subconjunto = df_completo.iloc[:max_n]
    resto = df_completo.iloc[max_n:]

    print()
    print("--- Distribucion de clases ---")

    prop_completa = df_completo["class_str"].value_counts(normalize=True)
    prop_subconjunto = subconjunto["class_str"].value_counts(normalize=True)
    prop_resto = resto["class_str"].value_counts(normalize=True)

    tabla = pd.DataFrame({
        "poblacion_completa": prop_completa,
        "subconjunto_actual": prop_subconjunto,
        "resto_no_usado": prop_resto,
    }).fillna(0.0)

    print(tabla.round(4).to_string())

    # ------------------------------------------------------------------
    # Chi-cuadrado: ¿la composicion del subconjunto es compatible con
    # un muestreo aleatorio de la poblacion completa?
    # ------------------------------------------------------------------

    clases = prop_completa.index.tolist()

    conteos_subconjunto = subconjunto["class_str"].value_counts().reindex(
        clases, fill_value=0
    )

    esperado = prop_completa.reindex(clases, fill_value=0) * len(subconjunto)

    # evitar divisiones raras si alguna clase esperada es 0
    mask_valida = esperado > 0

    stat, p_valor = chisquare(
        f_obs=conteos_subconjunto[mask_valida],
        f_exp=esperado[mask_valida],
    )

    print()
    print(f"Chi-cuadrado (subconjunto vs poblacion completa): stat={stat:.4f}  p={p_valor:.6f}")

    if p_valor < 0.05:
        print(
            "  -> p < 0.05: la composicion del subconjunto actual es "
            "ESTADISTICAMENTE DIFERENTE de la poblacion completa. "
            "El truncado SIN barajar SI introduce sesgo de clase."
        )
    else:
        print(
            "  -> p >= 0.05: no hay evidencia de que la composicion del "
            "subconjunto difiera de la poblacion completa. El truncado "
            "actual parece representativo, al menos en clase."
        )

    # Guardar en archivo específico por conjunto
    if nombre.upper() == "TRAIN":
        tabla.to_csv(SALIDA_COMPARACION_CLASES_TRAIN)
    else:
        tabla.to_csv(SALIDA_COMPARACION_CLASES_TEST)

    return {
        "conjunto": nombre,
        "chi2_stat": stat,
        "chi2_p_valor": p_valor,
        "n_poblacion": len(df_completo),
        "n_subconjunto": len(subconjunto),
    }


# ======================================================================
# PASO 2: DECILES POR POSICION EN EL PARQUET
# ======================================================================

def analizar_deciles(df_completo, nombre, salida_csv):

    banner(f"ANALISIS POR DECILES DE POSICION - {nombre}")

    n = len(df_completo)
    limites = np.linspace(0, n, N_DECILES + 1).astype(int)

    print()
    print("Calculando n_observaciones para cada objeto (puede tardar)...")

    n_obs = []
    for i, (_, fila) in enumerate(df_completo.iterrows()):
        if i % 5000 == 0:
            print(f"  {i:,}/{n:,}")
        n_obs.append(contar_observaciones(fila))

    df_completo = df_completo.copy()
    df_completo["_n_obs_tmp"] = n_obs

    tiene_periodo = "period" in df_completo.columns

    filas = []

    for d in range(N_DECILES):

        inicio = limites[d]
        fin = limites[d + 1]

        bloque = df_completo.iloc[inicio:fin]

        fila_resumen = {
            "decil": d + 1,
            "fila_inicio": inicio,
            "fila_fin": fin,
            "n_objetos": len(bloque),
            "n_obs_media": bloque["_n_obs_tmp"].mean(),
            "n_obs_mediana": bloque["_n_obs_tmp"].median(),
        }

        if tiene_periodo:
            fila_resumen["periodo_media"] = bloque["period"].mean()
            fila_resumen["periodo_mediana"] = bloque["period"].median()

        # proporcion de cada clase en este decil
        props = bloque["class_str"].value_counts(normalize=True)
        for clase, valor in props.items():
            fila_resumen[f"clase_{clase}"] = round(valor, 4)

        filas.append(fila_resumen)

    tabla_deciles = pd.DataFrame(filas)

    print()
    print(tabla_deciles.to_string(index=False))

    tabla_deciles.to_csv(salida_csv, index=False)
    print()
    print("Guardado en:", salida_csv)

    print()
    print("Como leerlo:")
    print(
        "  - Si 'n_obs_media' o 'periodo_media' cambian de forma "
        "marcada y sistematica de decil 1 a decil 10 (una tendencia "
        "clara, no ruido), el parquet esta ordenado por algo "
        "relacionado con esas variables."
    )
    print(
        "  - Si las proporciones 'clase_X' varian mucho entre deciles "
        "(por ejemplo, una clase que solo aparece en los primeros "
        "deciles), el parquet esta ordenado por clase o por algo "
        "correlacionado con ella."
    )
    print(
        "  - Si todo se ve parecido entre deciles (dentro de la "
        "fluctuacion normal de muestreo), el parquet ya esta "
        "efectivamente barajado y iloc[:N] es seguro."
    )

    return tabla_deciles


# ======================================================================
# CALCULO DE FEATURES (12 VARIABLES, IDENTICO A 20/21/22/23)
# ======================================================================

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
        return {v: np.nan for v in VARIABLES}

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
        delta = z
        denom = np.sqrt(np.mean(delta ** 2))
        stetson_K = np.mean(np.abs(delta)) / denom if denom > 1e-12 else 0.0
        chi2 = np.sum(delta ** 2) / max(len(x) - 1, 1)
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


def evaluar(X_train, y_train, X_test, y_test):

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_features="sqrt",
    )
    rf.fit(X_train[VARIABLES], y_train)
    pred_rf = rf.predict(X_test[VARIABLES])

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

    return {
        "RF_ACC": accuracy_score(y_test, pred_rf),
        "RF_BAL": balanced_accuracy_score(y_test, pred_rf),
        "LOG_ACC": accuracy_score(y_test, pred_log),
        "LOG_BAL": balanced_accuracy_score(y_test, pred_log),
    }


# ======================================================================
# PASO 3: IMPACTO PRACTICO - ACTUAL (iloc) vs ALEATORIO, MISMO TAMANO
# ======================================================================

def comparar_impacto_practico(train_completo):

    banner("IMPACTO PRACTICO: TRAIN ACTUAL (iloc) vs TRAIN ALEATORIO")

    if len(train_completo) <= MAX_TRAIN:
        print("No aplica: no hay poblacion de TRAIN mayor que MAX_TRAIN.")
        return None

    # El conjunto de validacion es lo que NUNCA entra en ninguno de los
    # dos TRAIN que vamos a comparar: el resto tras quitar los primeros
    # MAX_TRAIN (usado por el actual) y una muestra aleatoria del mismo
    # tamano (usada por la alternativa). Para que la comparacion sea
    # limpia, evaluamos ambos modelos contra el MISMO conjunto de
    # validacion: los objetos que quedan fuera del subconjunto actual
    # (train_completo.iloc[MAX_TRAIN:]).

    train_actual = train_completo.iloc[:MAX_TRAIN].copy()
    validacion = train_completo.iloc[MAX_TRAIN:].copy()

    if len(validacion) < 500:
        print(
            "Aviso: el conjunto de validacion disponible es pequeno "
            f"({len(validacion)} objetos), los resultados seran ruidosos."
        )

    train_aleatorio = train_completo.sample(
        n=MAX_TRAIN, random_state=RANDOM_STATE
    ).copy()

    print()
    print("Extrayendo features de VALIDACION (comun a ambas comparaciones)...")
    X_val = extraer_features(validacion, "VALIDACION")
    y_val = validacion["class_str"].astype(str)

    print()
    print("Extrayendo features de TRAIN ACTUAL (iloc[:MAX_TRAIN])...")
    X_train_actual = extraer_features(train_actual, "TRAIN_ACTUAL")
    y_train_actual = train_actual["class_str"].astype(str)

    print()
    print("Extrayendo features de TRAIN ALEATORIO (sample aleatorio)...")
    X_train_aleatorio = extraer_features(train_aleatorio, "TRAIN_ALEATORIO")
    y_train_aleatorio = train_aleatorio["class_str"].astype(str)

    X_train_actual, X_val_a = limpiar(X_train_actual, X_val.copy())
    X_train_aleatorio, X_val_b = limpiar(X_train_aleatorio, X_val.copy())

    print()
    print("Entrenando con TRAIN ACTUAL, evaluando contra VALIDACION...")
    r_actual = evaluar(X_train_actual, y_train_actual, X_val_a, y_val)
    print(r_actual)

    print()
    print("Entrenando con TRAIN ALEATORIO, evaluando contra VALIDACION...")
    r_aleatorio = evaluar(X_train_aleatorio, y_train_aleatorio, X_val_b, y_val)
    print(r_aleatorio)

    tabla = pd.DataFrame({
        "actual_iloc": r_actual,
        "aleatorio_sample": r_aleatorio,
    }).T
    tabla.index.name = "estrategia_train"

    print()
    print(tabla.to_string())

    tabla.to_csv(SALIDA_RESULTADOS_IMPACTO)

    return tabla


# ======================================================================
# MAIN
# ======================================================================

def main():

    inicio_total = time.time()

    banner("20b - CONTROL DE REPRESENTATIVIDAD DEL TRUNCADO")

    train_completo, test_completo = cargar_datos_completos()

    resultados_chi2 = []

    r_train = comparar_subconjunto_vs_poblacion(
        train_completo, MAX_TRAIN, "TRAIN"
    )
    if r_train:
        resultados_chi2.append(r_train)

    r_test = comparar_subconjunto_vs_poblacion(
        test_completo, MAX_TEST, "TEST"
    )
    if r_test:
        resultados_chi2.append(r_test)

    if resultados_chi2:
        pd.DataFrame(resultados_chi2).to_csv(
            SALIDA_COMPARACION_CLASES, index=False
        )

    analizar_deciles(train_completo, "TRAIN", SALIDA_DECILES_TRAIN)
    analizar_deciles(test_completo, "TEST", SALIDA_DECILES_TEST)

    comparar_impacto_practico(train_completo)

    banner("RESUMEN E INTERPRETACION")

    print(
        "Revisa, en este orden:\n"
        "  1) El p-valor del chi-cuadrado (arriba): si es < 0.05, hay "
        "sesgo de clase demostrado en el truncado actual.\n"
        "  2) Las tablas de deciles: busca tendencias sistematicas, no "
        "ruido aleatorio, en n_obs_media, periodo_media o las "
        "proporciones de clase.\n"
        "  3) La tabla de impacto practico: si TRAIN_ACTUAL y "
        "TRAIN_ALEATORIO dan balanced accuracy parecida contra el mismo "
        "conjunto de validacion, el sesgo (si existe) no esta afectando "
        "de forma relevante a los resultados ya obtenidos en 21/22/23.\n"
    )

    print(f"Tiempo total: {time.time() - inicio_total:.2f} segundos")


if __name__ == "__main__":
    main()