# -*- coding: utf-8 -*-

"""
======================================================================
16d - CARACTERIZACION DE LOS EMPATES INTERMEDIOS DE MJD
======================================================================

Motivacion
----------
16c descarto la hipotesis de padding en cola como problema relevante
(solo 1.1% de objetos, impacto practicamente nulo). Pero el propio
16c confirmo que el fenomeno grande sigue sin explicar: ~98% de los
objetos tienen de media ~100 puntos con mjd EXACTAMENTE repetido en
mitad de la curva (no al final).

Este script responde dos preguntas sobre esos empates intermedios:

  1. Cuando mjd se repite, ¿se repite tambien 'target'?
       - Si SI se repite -> son registros duplicados de verdad
         (mismo punto fotometrico guardado mas de una vez). Hay que
         deduplicar antes de calcular ninguna feature.
       - Si NO se repite -> son mediciones distintas que comparten
         el mismo mjd, probablemente porque mjd esta redondeado a
         una precision limitada (varias exposiciones de la misma
         noche cayendo en el mismo valor). No es un error de datos,
         pero tiene consecuencias para 'maximum_slope'.

  2. ¿Que fraccion de los pares consecutivos de cada curva quedan
     excluidos de 'maximum_slope' por el filtro 'dt > 0'? Si esa
     fraccion es grande, 'maximum_slope' se esta calculando sobre
     una minoria de los pares disponibles, y probablemente explica
     por que es una variable poco importante en la ablacion del 18:
     no es que la pendiente maxima no importe, es que el calculo
     esta tirando la mayoria de la informacion.

Ademas, estima la precision efectiva de mjd (el hueco minimo no nulo
entre valores distintos de mjd, por objeto), para tener una cifra
concreta de a que escala temporal se estan fusionando observaciones.

No se toca el pipeline de clasificacion en este script: es puramente
diagnostico. Los cambios al pipeline (si hacen falta) se deciden
despues de ver estos resultados.
======================================================================
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

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
PREFIJO = "16d_"

DATA_DIR = BASE_DIR / "dataset" / "ZTF_40k_StarEmbed" / "data"

TRAIN_FILES = [
    DATA_DIR / "train-00000-of-00002.parquet",
    DATA_DIR / "train-00001-of-00002.parquet",
]

TEST_FILE = DATA_DIR / "test-00000-of-00001.parquet"

MAX_TRAIN = 25000
MAX_TEST = 8000

BANDS_PRIORIDAD = ["r", "g", "i"]

# Tolerancia para considerar dos valores de target "iguales" dentro
# de un grupo de mjd repetido. 0.0 = igualdad exacta.
TOLERANCIA_TARGET_IGUAL = 0.0

SALIDA_RESUMEN_OBJETO = RESULTADOS_DIR / f"{PREFIJO}resumen_empates_por_objeto.csv"
SALIDA_RESUMEN_GLOBAL = RESULTADOS_DIR / f"{PREFIJO}resumen_global.csv"
SALIDA_EJEMPLOS = RESULTADOS_DIR / f"{PREFIJO}ejemplos_grupos_target_distinto.csv"

N_EJEMPLOS_A_GUARDAR = 30


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
# DETECCION DE GRUPOS DE MJD REPETIDO (RUN-LENGTH ENCODING)
# ======================================================================

def grupos_de_mjd_repetido(t):
    """
    Como la curva ya esta ordenada de forma no decreciente, cualquier
    tramo de mjd identico es contiguo. Devuelve una lista de tuplas
    (indice_inicio, longitud) para cada grupo de longitud >= 2.
    """

    n = len(t)

    if n < 2:
        return []

    grupos = []
    inicio = 0

    for i in range(1, n):
        if t[i] != t[inicio]:
            if i - inicio >= 2:
                grupos.append((inicio, i - inicio))
            inicio = i

    if n - inicio >= 2:
        grupos.append((inicio, n - inicio))

    return grupos


def analizar_objeto(x, t):

    n_original = len(t)

    resultado = {
        "n_original": n_original,
        "n_grupos_intermedios": 0,
        "n_puntos_en_grupos": 0,
        "n_grupos_target_constante": 0,
        "n_grupos_target_distinto": 0,
        "n_pares_excluidos_max_slope": 0,
        "n_pares_totales": max(n_original - 1, 0),
        "gap_mjd_minimo_no_nulo": np.nan,
    }

    if n_original < 2:
        return resultado, []

    grupos = grupos_de_mjd_repetido(t)

    # excluir el ultimo grupo si toca el final del array (eso ya lo
    # cubre 16c como "cola"; aqui nos interesan los intermedios)
    if grupos and (grupos[-1][0] + grupos[-1][1] == n_original):
        grupos_intermedios = grupos[:-1]
    else:
        grupos_intermedios = grupos

    ejemplos_target_distinto = []

    for inicio, longitud in grupos_intermedios:

        objetivo = x[inicio]
        valores_grupo = x[inicio:inicio + longitud]

        constante = np.all(
            np.abs(valores_grupo - objetivo) <= TOLERANCIA_TARGET_IGUAL
        )

        resultado["n_grupos_intermedios"] += 1
        resultado["n_puntos_en_grupos"] += longitud

        if constante:
            resultado["n_grupos_target_constante"] += 1
        else:
            resultado["n_grupos_target_distinto"] += 1
            ejemplos_target_distinto.append({
                "mjd_repetido": t[inicio],
                "n_puntos_en_grupo": longitud,
                "target_min": float(np.min(valores_grupo)),
                "target_max": float(np.max(valores_grupo)),
                "target_rango": float(
                    np.max(valores_grupo) - np.min(valores_grupo)
                ),
            })

    # pares excluidos de maximum_slope por dt == 0
    dt = np.diff(t)
    resultado["n_pares_excluidos_max_slope"] = int(np.sum(dt == 0))

    # precision efectiva de mjd: menor salto no nulo entre valores
    # distintos consecutivos
    dt_no_nulo = dt[dt > 0]
    if len(dt_no_nulo) > 0:
        resultado["gap_mjd_minimo_no_nulo"] = float(np.min(dt_no_nulo))

    return resultado, ejemplos_target_distinto


# ======================================================================
# MAIN
# ======================================================================

def procesar_conjunto(df, nombre):

    banner(f"ANALIZANDO EMPATES INTERMEDIOS - {nombre}")

    filas = []
    todos_ejemplos = []

    n = len(df)

    for i, (_, fila) in enumerate(df.iterrows()):

        if i % 5000 == 0:
            print(f"  {i:,}/{n:,}")

        x, t = obtener_curva_cruda(fila)
        resultado, ejemplos = analizar_objeto(x, t)
        filas.append(resultado)

        for ej in ejemplos:
            ej["conjunto"] = nombre
            ej["fila"] = i
            todos_ejemplos.append(ej)

    resumen = pd.DataFrame(filas)

    return resumen, todos_ejemplos


def main():

    train, test = cargar_datos()

    resumen_train, ejemplos_train = procesar_conjunto(train, "TRAIN")
    resumen_test, ejemplos_test = procesar_conjunto(test, "TEST")

    # ------------------------------------------------------------------
    # PREGUNTA 1: ¿el target se repite tambien, o varia?
    # ------------------------------------------------------------------

    banner("PREGUNTA 1: ¿SE REPITE TAMBIEN EL TARGET DENTRO DE LOS EMPATES?")

    for nombre, resumen in [("TRAIN", resumen_train), ("TEST", resumen_test)]:

        total_grupos = resumen["n_grupos_intermedios"].sum()
        grupos_constantes = resumen["n_grupos_target_constante"].sum()
        grupos_distintos = resumen["n_grupos_target_distinto"].sum()

        print()
        print(f"--- {nombre} ---")
        print(f"Total de grupos de mjd repetido (intermedios): {total_grupos:,}")

        if total_grupos > 0:
            pct_constante = 100.0 * grupos_constantes / total_grupos
            pct_distinto = 100.0 * grupos_distintos / total_grupos

            print(
                f"  Grupos con target CONSTANTE (duplicado real): "
                f"{grupos_constantes:,} ({pct_constante:.2f}%)"
            )
            print(
                f"  Grupos con target DISTINTO (medicion real, mjd "
                f"redondeado): {grupos_distintos:,} ({pct_distinto:.2f}%)"
            )

    # ------------------------------------------------------------------
    # PREGUNTA 2: impacto en maximum_slope
    # ------------------------------------------------------------------

    banner("PREGUNTA 2: FRACCION DE PARES EXCLUIDOS DE MAXIMUM_SLOPE")

    for nombre, resumen in [("TRAIN", resumen_train), ("TEST", resumen_test)]:

        con_pares = resumen[resumen["n_pares_totales"] > 0].copy()
        con_pares["frac_excluida"] = (
            con_pares["n_pares_excluidos_max_slope"]
            / con_pares["n_pares_totales"]
        )

        print()
        print(f"--- {nombre} ---")
        print("Fraccion de pares consecutivos excluidos (dt=0) por objeto:")
        print(con_pares["frac_excluida"].describe().to_string())

        objetos_mitad_excluida = int((con_pares["frac_excluida"] >= 0.5).sum())
        print()
        print(
            f"Objetos donde se excluye >=50% de los pares: "
            f"{objetos_mitad_excluida:,} / {len(con_pares):,} "
            f"({100.0 * objetos_mitad_excluida / len(con_pares):.2f}%)"
        )

    # ------------------------------------------------------------------
    # PRECISION EFECTIVA DE MJD
    # ------------------------------------------------------------------

    banner("PRECISION EFECTIVA DE MJD (MENOR SALTO NO NULO POR OBJETO)")

    gaps_test = resumen_test["gap_mjd_minimo_no_nulo"].dropna()

    print()
    print("Estadisticas del hueco minimo no nulo entre mjd distintos (TEST):")
    print(gaps_test.describe().to_string())
    print()
    print(
        "Referencia: 1 dia = 1.0 en mjd. 0.000694 ~ 1 minuto. "
        "0.0104 ~ 15 minutos. 0.042 ~ 1 hora."
    )

    # ------------------------------------------------------------------
    # GUARDAR RESULTADOS
    # ------------------------------------------------------------------

    resumen_train["conjunto"] = "TRAIN"
    resumen_test["conjunto"] = "TEST"

    resumen_total = pd.concat([resumen_train, resumen_test], ignore_index=True)
    resumen_total.to_csv(SALIDA_RESUMEN_OBJETO, index=False)

    global_stats = []

    for nombre, resumen in [("TRAIN", resumen_train), ("TEST", resumen_test)]:

        total_grupos = resumen["n_grupos_intermedios"].sum()
        grupos_constantes = resumen["n_grupos_target_constante"].sum()
        grupos_distintos = resumen["n_grupos_target_distinto"].sum()

        con_pares = resumen[resumen["n_pares_totales"] > 0].copy()
        frac_excluida_media = (
            con_pares["n_pares_excluidos_max_slope"]
            / con_pares["n_pares_totales"]
        ).mean()

        global_stats.append({
            "conjunto": nombre,
            "total_grupos_intermedios": int(total_grupos),
            "grupos_target_constante": int(grupos_constantes),
            "grupos_target_distinto": int(grupos_distintos),
            "pct_grupos_target_constante": round(
                100.0 * grupos_constantes / max(total_grupos, 1), 2
            ),
            "frac_pares_excluidos_media": frac_excluida_media,
        })

    pd.DataFrame(global_stats).to_csv(SALIDA_RESUMEN_GLOBAL, index=False)

    todos_ejemplos = ejemplos_train + ejemplos_test

    if todos_ejemplos:
        muestra = pd.DataFrame(todos_ejemplos).head(N_EJEMPLOS_A_GUARDAR)
        muestra.to_csv(SALIDA_EJEMPLOS, index=False)

    print()
    print("Archivos guardados:")
    print(" ", SALIDA_RESUMEN_OBJETO)
    print(" ", SALIDA_RESUMEN_GLOBAL)
    if todos_ejemplos:
        print(" ", SALIDA_EJEMPLOS)

    print()
    print("INTERPRETACION RAPIDA:")
    print()
    print(
        "- Si la mayoria de los grupos intermedios tienen target "
        "CONSTANTE: hay registros duplicados de verdad en el dataset "
        "y conviene deduplicar (quedarse con un punto por mjd) antes "
        "de calcular ninguna feature en 21/22/23."
    )
    print(
        "- Si la mayoria tienen target DISTINTO: no son duplicados, "
        "son mediciones reales que comparten mjd por redondeo. Mirad "
        "el hueco minimo no nulo (arriba) para saber a que escala de "
        "tiempo esta pasando esto."
    )
    print(
        "- Si 'objetos donde se excluye >=50% de los pares' es alto, "
        "maximum_slope se esta calculando sobre menos de la mitad de "
        "la informacion disponible en muchos objetos. Eso explicaria "
        "por que salio como una de las variables menos utiles en la "
        "ablacion del 18: no es poco informativa, es que se le esta "
        "tirando la mayor parte del dato."
    )


if __name__ == "__main__":
    main()