# -*- coding: utf-8 -*-

"""
======================================================================
verificar_paquete.py - EL NUEVO PAQUETE REPRODUCE LOS NUMEROS ORIGINALES
======================================================================

Antes de construir deteccion.py (el siguiente paso del roadmap) hay
que confirmar que anomaly_detector/features.py, que es una
REESCRITURA de la logica de 20-24b sobre el esquema canonico, produce
exactamente los mismos numeros que el codigo original. Un refactor
que "deberia dar lo mismo" no vale como verificacion -- ya hemos visto
varias veces en esta sesion que un detalle pequeño (un shuffle, un
filtro, un orden) cambia resultados de forma significativa.

Este script:

  1. Calcula las 12 variables con el metodo ORIGINAL (la misma copia
     literal de codigo usada en 20-24b, sin pasar por el paquete).
  2. Calcula las mismas 12 variables con el paquete NUEVO
     (anomaly_detector.adapters.cargar_train_test_starembed +
     extraer_features_dataset).
  3. Compara ambos resultados objeto a objeto y variable a variable.

Si las diferencias son cero (o del orden de 1e-9, ruido de coma
flotante), el paquete queda verificado y es seguro seguir construyendo
sobre el. Si hay diferencias mayores, hay un bug en el refactor que
hay que corregir antes de continuar -- NO se debe seguir con
deteccion.py hasta que este script pase limpio.

Requiere que `anomaly_detector/` este en el mismo directorio que este
script (o en el PYTHONPATH).
======================================================================
"""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ======================================================================
# AÑADIR EL DIRECTORIO RAIZ AL PATH PARA IMPORTAR anomaly_detector
# ======================================================================

# Ruta relativa desde la ubicación del script (auditorias/05_verificacion_paquete)
BASE_DIR = Path(__file__).resolve().parents[2]  # Sube 2 niveles hasta Astronomia/

# Añadir BASE_DIR al path para poder importar anomaly_detector
sys.path.insert(0, str(BASE_DIR))

from anomaly_detector.adapters import cargar_train_test_starembed
from anomaly_detector.features import extraer_features_dataset, VARIABLES_BASE


# ======================================================================
# CONFIGURACION
# ======================================================================

DATA_DIR = BASE_DIR / "dataset" / "ZTF_40k_StarEmbed" / "data"

# Directorio de resultados dentro de la misma carpeta que el script
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "verificacion_"

MAX_TRAIN = 25000
MAX_TEST = 8000

BANDS_PRIORIDAD = ["r", "g", "i"]

TOLERANCIA = 1e-9

# Diferencias YA investigadas, explicadas y aceptadas como comportamiento
# correcto del paquete nuevo (no como bug). Formato: {etiqueta: {id_objeto: motivo}}.
# Si en una ejecucion futura aparece una diferencia en un id_objeto que
# esta aqui, se reporta igual (para que quede constancia) pero NO hace
# fallar el veredicto final. Cualquier id_objeto con diferencia que NO
# este en esta lista SI hace fallar el veredicto -- la lista es un
# allowlist explicito, no un silenciador general.
EXCEPCIONES_CONOCIDAS = {
    "TRAIN": {
        "CSS_J053809.1-070525": (
            "banda 'r' con un solo punto finito (n=1); el metodo original "
            "la aceptaba igualmente (solo comprobaba que la clave existiera), "
            "produciendo features degeneradas (std=0, skew/kurtosis sin "
            "sentido). El paquete nuevo la descarta via LightCurve.validar() "
            "(minimo 2 puntos) y cae correctamente a la banda 'g' (37 puntos "
            "reales). Comportamiento nuevo mas correcto, no un bug. "
            "Verificado en la sesion de auditoria, ver whitepaper.md."
        ),
    },
}

SALIDA_DIFERENCIAS = RESULTADOS_DIR / f"{PREFIJO}diferencias.csv"


def banner(texto):
    print()
    print("=" * 70)
    print(texto)
    print("=" * 70)


# ======================================================================
# METODO ORIGINAL (copia literal de la logica de 20-24b, SIN pasar
# por el paquete, a proposito -- es el patron de referencia contra el
# que se compara)
# ======================================================================

POSIBLES_COLUMNAS_ID = ["object_id", "source_id", "sourceid", "id"]


def _columna_id_original(df):
    for col in POSIBLES_COLUMNAS_ID:
        if col in df.columns:
            return col
    return None


def _fila_cruda_por_id(id_objeto, df_crudo):
    """
    Localiza la fila cruda correspondiente a un id_objeto para
    diagnostico. Nunca lanza excepcion -- si no puede localizarla
    (por ejemplo, porque ninguna columna de id reconocida existe y
    el id no es convertible a indice numerico), devuelve None y el
    llamador simplemente se salta el diagnostico crudo, no revienta
    la verificacion por esto.
    """

    col_id = _columna_id_original(df_crudo)

    if col_id:
        fila_cruda = df_crudo[df_crudo[col_id].astype(str) == str(id_objeto)]
        return fila_cruda if len(fila_cruda) > 0 else None

    try:
        idx = int(id_objeto)
        return df_crudo.iloc[[idx]]
    except (ValueError, IndexError):
        return None


def obtener_banda_original(bands):

    if bands is None:
        return None, None

    try:
        if isinstance(bands, dict):
            for b in BANDS_PRIORIDAD:
                if b in bands and bands[b] is not None:
                    return bands[b], b
        return None, None
    except Exception:
        return None, None


def obtener_curva_cruda_original(fila):

    bands = fila.get("bands_data", None)
    banda, banda_nombre = obtener_banda_original(bands)

    if banda is None:
        return np.array([]), np.array([]), None

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

    return x, t, banda_nombre


def calcular_features_original(fila, id_objeto):

    x, t, banda_nombre = obtener_curva_cruda_original(fila)

    if len(x) == 0:
        vals = {v: np.nan for v in VARIABLES_BASE}
        vals["period"] = np.nan
        vals["id_objeto"] = id_objeto
        vals["curva_vacia"] = True
        vals["banda_usada"] = banda_nombre
        vals["n_puntos_usados"] = 0
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
        "id_objeto": id_objeto,
        "curva_vacia": False,
        "banda_usada": banda_nombre,
        "n_puntos_usados": len(x),
    }


def extraer_features_original(df, etiqueta):

    filas = []
    n = len(df)

    col_id = _columna_id_original(df)

    for i, (_, fila) in enumerate(df.iterrows()):
        if i % 5000 == 0:
            print(f"  [ORIGINAL {etiqueta}] {i:,}/{n:,}")
        id_objeto = str(fila[col_id]) if col_id else str(i)
        filas.append(calcular_features_original(fila, id_objeto))

    return pd.DataFrame(filas)


def cargar_dataframes_originales():

    rutas_train = [
        DATA_DIR / "train-00000-of-00002.parquet",
        DATA_DIR / "train-00001-of-00002.parquet",
    ]
    ruta_test = DATA_DIR / "test-00000-of-00001.parquet"

    frames = [pd.read_parquet(r) for r in rutas_train]
    train = pd.concat(frames, ignore_index=True).iloc[:MAX_TRAIN].copy()

    test = pd.read_parquet(ruta_test).iloc[:MAX_TEST].copy()

    return train, test


# ======================================================================
# COMPARACION
# ======================================================================

def comparar(df_original, df_nuevo, variables, etiqueta, df_crudo=None):

    banner(f"COMPARANDO ORIGINAL vs PAQUETE NUEVO - {etiqueta}")

    # ------------------------------------------------------------------
    # Diagnostico de id_objeto duplicados (causa mas probable de que
    # el numero de FILAS no coincida con el numero de ids UNICOS)
    # ------------------------------------------------------------------

    dup_original = df_original["id_objeto"][df_original["id_objeto"].duplicated(keep=False)]
    dup_nuevo = df_nuevo["id_objeto"][df_nuevo["id_objeto"].duplicated(keep=False)]

    if len(dup_original) > 0 or len(dup_nuevo) > 0:

        print(
            f"AVISO: hay id_objeto DUPLICADOS. "
            f"original: {df_original['id_objeto'].duplicated().sum()} filas duplicadas "
            f"({dup_original.nunique()} ids distintos afectados). "
            f"nuevo: {df_nuevo['id_objeto'].duplicated().sum()} filas duplicadas "
            f"({dup_nuevo.nunique()} ids distintos afectados)."
        )

        if len(dup_original) > 0:

            id_ejemplo = dup_original.iloc[0]
            filas_dup = df_original[df_original["id_objeto"] == id_ejemplo]

            print()
            print(f"  Ejemplo: id_objeto='{id_ejemplo}' aparece {len(filas_dup)} veces en el metodo ORIGINAL.")

            # ¿son realmente el mismo objeto (features identicas) o dos
            # objetos distintos que comparten id por error/coincidencia?
            columnas_comparables = [c for c in variables if c in filas_dup.columns]
            iguales_entre_si = filas_dup[columnas_comparables].nunique().eq(1).all()

            if iguales_entre_si:
                print(
                    "  Las filas duplicadas tienen EXACTAMENTE las mismas features "
                    "-> son el mismo objeto repetido (registro duplicado), no dos "
                    "objetos distintos con id compartido por error."
                )
            else:
                print(
                    "  Las filas duplicadas tienen features DISTINTAS entre si "
                    "-> son dos objetos DIFERENTES que comparten id_objeto. "
                    "Esto es mas serio: cualquier logica que use id_objeto como "
                    "clave unica (incluido este mismo script, y el futuro cruce "
                    "con catalogos externos) esta perdiendo informacion real."
                )
                print()
                print(filas_dup[columnas_comparables].to_string())

        print()
        print(
            "  Politica aplicada para poder seguir comparando: se conserva SOLO "
            "la primera aparicion de cada id_objeto duplicado en ambos lados "
            "(mismo criterio que usa Dataset.ids_unicos() en el paquete nuevo, "
            "documentado en schema.py). El resto de la comparacion refleja esa "
            "decision, no los datos crudos completos."
        )

        df_original = df_original.drop_duplicates(subset="id_objeto", keep="first")
        df_nuevo = df_nuevo.drop_duplicates(subset="id_objeto", keep="first")

    ids_original = set(df_original["id_objeto"])
    ids_nuevo = set(df_nuevo["id_objeto"])

    solo_original = ids_original - ids_nuevo
    solo_nuevo = ids_nuevo - ids_original

    if solo_original or solo_nuevo:

        print(
            f"AVISO: los conjuntos de id_objeto NO coinciden "
            f"(original={len(ids_original)}, nuevo={len(ids_nuevo)})."
        )
        print(f"  Objetos solo en ORIGINAL (el paquete nuevo los perdio): {len(solo_original)}")
        print(f"  Objetos solo en NUEVO (no deberia pasar, investigar): {len(solo_nuevo)}")

        if solo_original:
            ejemplo_id = sorted(solo_original)[0]
            fila_original = df_original[df_original["id_objeto"] == ejemplo_id].iloc[0]
            print()
            print(f"  Diagnostico del primer objeto perdido (id_objeto={ejemplo_id}):")
            print(f"    curva_vacia segun metodo original: {fila_original['curva_vacia']}")

            if df_crudo is not None:
                fila_cruda = _fila_cruda_por_id(ejemplo_id, df_crudo)

                if fila_cruda is not None and len(fila_cruda) > 0:
                    bands = fila_cruda.iloc[0].get("bands_data", "<<columna ausente>>")
                    print(f"    tipo de bands_data: {type(bands)}")
                    if isinstance(bands, dict):
                        print(f"    claves de bands_data: {list(bands.keys())}")
                        for k, v in bands.items():
                            if v is None:
                                print(f"      banda '{k}': None")
                            elif isinstance(v, dict):
                                target_k = v.get("target", [])
                                mjd_k = v.get("mjd", [])
                                if target_k is None:
                                    target_k = []
                                if mjd_k is None:
                                    mjd_k = []
                                print(f"      banda '{k}': claves={list(v.keys())}, "
                                      f"len(target)={len(target_k)}, "
                                      f"len(mjd)={len(mjd_k)}")

        if len(solo_original) > 5:
            print(f"  ... y {len(solo_original) - 1} objetos mas solo en ORIGINAL (ver CSV completo).")

    # Comparacion SOLO sobre los id_objeto presentes en ambos lados.
    # Esto es lo que realmente responde "¿el refactor cambio algun
    # numero para los objetos que SI proceso en los dos metodos?" --
    # independiente de si hay objetos que uno descarta y el otro no,
    # que es una pregunta distinta (ver aviso de arriba).
    comunes = ids_original & ids_nuevo

    df_o = df_original[df_original["id_objeto"].isin(comunes)].set_index("id_objeto").sort_index()
    df_n = df_nuevo[df_nuevo["id_objeto"].isin(comunes)].set_index("id_objeto").sort_index()

    print()
    print(f"Comparando los {len(comunes)} objetos presentes en AMBOS metodos...")

    filas_diferencias = []
    todo_ok = True
    ids_con_diferencia = set()

    for var in variables + ["period"]:

        if var not in df_o.columns or var not in df_n.columns:
            print(f"  {var}: AUSENTE en alguno de los dos conjuntos, se omite.")
            continue

        a = df_o[var].values.astype(float)
        b = df_n[var].values.astype(float)

        ambos_nan = np.isnan(a) & np.isnan(b)
        diff = np.full_like(a, np.nan)
        validos = ~ambos_nan
        diff[validos] = np.abs(a[validos] - b[validos])

        diff_max = np.nanmax(diff) if np.any(validos) else 0.0
        diff_media = np.nanmean(diff) if np.any(validos) else 0.0
        n_distintos = int(np.sum(diff[validos] > TOLERANCIA))

        estado = "OK" if n_distintos == 0 else "DIFERENCIA"
        if n_distintos > 0:
            todo_ok = False
            mask_dif = np.zeros(len(diff), dtype=bool)
            mask_dif[validos] = diff[validos] > TOLERANCIA
            ids_con_diferencia.update(df_o.index[mask_dif].tolist())

        print(
            f"  {var:28s} max={diff_max:.3e}  media={diff_media:.3e}  "
            f"objetos_distintos={n_distintos:5d}  [{estado}]"
        )

        filas_diferencias.append({
            "conjunto": etiqueta,
            "variable": var,
            "diff_max": diff_max,
            "diff_media": diff_media,
            "objetos_distintos": n_distintos,
            "estado": estado,
            "n_solo_original": len(solo_original),
            "n_solo_nuevo": len(solo_nuevo),
        })

    if ids_con_diferencia:

        print()
        print(f"Objetos con al menos una diferencia numerica: {len(ids_con_diferencia)}")

        excepciones_etiqueta = EXCEPCIONES_CONOCIDAS.get(etiqueta, {})
        ids_sin_explicar = ids_con_diferencia - set(excepciones_etiqueta.keys())
        ids_explicadas = ids_con_diferencia & set(excepciones_etiqueta.keys())

        if ids_explicadas:
            print()
            print(f"  {len(ids_explicadas)} de ellas ya estan DOCUMENTADAS Y ACEPTADAS (no cuentan como fallo):")
            for id_objeto in ids_explicadas:
                print(f"    - {id_objeto}: {excepciones_etiqueta[id_objeto]}")

        if ids_sin_explicar:
            print()
            print(f"  {len(ids_sin_explicar)} diferencias SIN EXPLICAR todavia -- estas si son un problema:")

        for id_objeto in list(ids_sin_explicar)[:5]:

            print()
            print(f"--- Diagnostico de banda para id_objeto='{id_objeto}' ---")

            banda_original = df_o.loc[id_objeto, "banda_usada"] if "banda_usada" in df_o.columns else "?"
            n_original = df_o.loc[id_objeto, "n_puntos_usados"] if "n_puntos_usados" in df_o.columns else "?"
            banda_nueva = df_n.loc[id_objeto, "banda_usada"] if "banda_usada" in df_n.columns else "?"
            n_nueva = df_n.loc[id_objeto, "n_puntos_usados"] if "n_puntos_usados" in df_n.columns else "?"

            print(f"  Metodo ORIGINAL: banda='{banda_original}', n_puntos={n_original}")
            print(f"  Paquete NUEVO  : banda='{banda_nueva}', n_puntos={n_nueva}")

            if banda_original != banda_nueva:
                print(
                    "  -> CONFIRMADO: los dos metodos eligieron una banda distinta "
                    "para este objeto. Revisa mas abajo el contenido crudo de "
                    "bands_data para entender por que."
                )

            if df_crudo is not None:
                fila_cruda = _fila_cruda_por_id(id_objeto, df_crudo)

                if fila_cruda is not None and len(fila_cruda) > 0:
                    bands = fila_cruda.iloc[0].get("bands_data", "<<columna ausente>>")
                    if isinstance(bands, dict):
                        for k, v in bands.items():
                            if v is None:
                                print(f"    banda '{k}': None")
                            elif isinstance(v, dict):
                                target = v.get("target", [])
                                mjd = v.get("mjd", [])
                                if target is None:
                                    target = []
                                if mjd is None:
                                    mjd = []
                                n_finitos_t = int(np.sum(np.isfinite(np.asarray(target, dtype=float)))) if len(target) else 0
                                n_finitos_m = int(np.sum(np.isfinite(np.asarray(mjd, dtype=float)))) if len(mjd) else 0
                                print(
                                    f"    banda '{k}': len(target)={len(target)} "
                                    f"(finitos={n_finitos_t}), len(mjd)={len(mjd)} "
                                    f"(finitos={n_finitos_m})"
                                )

        if len(ids_sin_explicar) > 5:
            print(f"  ... y {len(ids_sin_explicar) - 5} objetos mas sin explicar (ver CSV).")

        # Las diferencias en la allowlist no cuentan como fallo; cualquier
        # diferencia fuera de la allowlist si lo hace.
        todo_ok = len(ids_sin_explicar) == 0

    print()
    if todo_ok and not solo_original and not solo_nuevo:
        print(f"RESULTADO {etiqueta}: TODO OK. Mismos objetos, mismos numeros "
              f"(o diferencias unicamente en casos ya documentados y aceptados).")
    elif todo_ok:
        print(
            f"RESULTADO {etiqueta}: los objetos EN COMUN dan numeros identicos "
            f"(o solo difieren en casos documentados), pero los conjuntos de "
            f"objetos NO coinciden ({len(solo_original)} solo en original, "
            f"{len(solo_nuevo)} solo en nuevo). Esto sigue siendo un problema "
            f"a resolver antes de continuar -- ver diagnostico arriba."
        )
    else:
        print(
            f"RESULTADO {etiqueta}: HAY DIFERENCIAS NUMERICAS SIN EXPLICAR "
            f"ademas de posibles objetos no coincidentes. No continuar con "
            f"deteccion.py."
        )

    return pd.DataFrame(filas_diferencias), (todo_ok and not solo_original and not solo_nuevo)


# ======================================================================
# MAIN
# ======================================================================

def main():

    inicio = time.time()

    banner("VERIFICACION DEL PAQUETE anomaly_detector CONTRA EL METODO ORIGINAL")

    print()
    print("--- Cargando datos con el metodo ORIGINAL (copia literal de 20-24b) ---")
    train_df, test_df = cargar_dataframes_originales()

    print()
    print("Extrayendo features con el metodo ORIGINAL...")
    X_train_original = extraer_features_original(train_df, "TRAIN")
    X_test_original = extraer_features_original(test_df, "TEST")

    print()
    print("--- Cargando datos con el PAQUETE NUEVO (anomaly_detector) ---")
    dataset_train, dataset_test = cargar_train_test_starembed(
        DATA_DIR, max_train=MAX_TRAIN, max_test=MAX_TEST
    )

    print()
    print("Extrayendo features con el PAQUETE NUEVO...")
    X_train_nuevo = extraer_features_dataset(dataset_train)
    X_test_nuevo = extraer_features_dataset(dataset_test)

    resultados = []
    todos_ok = True

    r_train, ok_train = comparar(
        X_train_original, X_train_nuevo, VARIABLES_BASE, "TRAIN", df_crudo=train_df
    )
    resultados.append(r_train)
    todos_ok = todos_ok and ok_train

    r_test, ok_test = comparar(
        X_test_original, X_test_nuevo, VARIABLES_BASE, "TEST", df_crudo=test_df
    )
    resultados.append(r_test)
    todos_ok = todos_ok and ok_test

    tabla_final = pd.concat(resultados, ignore_index=True)
    tabla_final.to_csv(SALIDA_DIFERENCIAS, index=False)
    print()
    print("Guardado:", SALIDA_DIFERENCIAS)

    if todos_ok:
        banner("VEREDICTO FINAL: PAQUETE VERIFICADO. Seguro continuar con deteccion.py.")
    else:
        banner(
            "VEREDICTO FINAL: NO VERIFICADO. Hay diferencias numericas y/o "
            "objetos que no coinciden entre metodos. NO continuar con "
            "deteccion.py hasta resolver esto -- ver diagnostico arriba "
            "y el CSV de diferencias."
        )

    print()
    print(f"Tiempo total: {time.time() - inicio:.2f} segundos")


if __name__ == "__main__":
    main()