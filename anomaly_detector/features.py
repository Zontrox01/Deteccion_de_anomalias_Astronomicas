# -*- coding: utf-8 -*-

"""
======================================================================
features.py - EXTRACCION DE FEATURES SOBRE EL ESQUEMA CANONICO
======================================================================

Las 12 variables y sus formulas son exactamente las validadas en la
sesion de auditoria (20-24b, con los controles 16b/16c/16d confirmando
que eta y maximum_slope son de fiar sobre curvas correctamente
ordenadas). Lo unico que cambia respecto a los scripts originales es
que ahora operan sobre `schema.LightCurve` en vez de sobre una fila
de un parquet concreto -- por eso ya no hay ningun `obtener_banda`
ni `obtener_curva_cruda` aqui: eso es responsabilidad exclusiva del
adaptador.

Nota sobre 'period'
--------------------
No se calcula en este modulo. Si el adaptador de origen lo ha puesto
en `curva.metadata['period']` (StarEmbed lo hace), `extraer_features`
lo incluye en el resultado; si no esta disponible, se omite del
diccionario de salida (no se rellena con NaN ni con un valor
inventado -- ausencia real, no un placeholder).

Nota sobre la limpieza previa
-------------------------------
`extraer_features` asume que la curva que recibe ya esta limpia y
ordenada (ver `LightCurve.limpia()`). No vuelve a ordenar ni a filtrar
NaN internamente -- eso ya causo una confusion real durante la
auditoria (16b) sobre si el orden se garantizaba en el propio calculo
o en la carga de datos. Aqui queda como contrato explicito: quien
llama decide cuando limpiar, esta funcion no lo hace por su cuenta.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .schema import LightCurve


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


def _stats_basicas(x: np.ndarray) -> dict:

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


def _eta(x: np.ndarray) -> float:
    """
    Von Neumann ratio. Requiere que x este en orden temporal real
    (ver LightCurve.limpia() y el control 16b, que confirmo que esto
    es seguro sobre StarEmbed una vez la curva esta ordenada).
    """

    if len(x) > 1 and np.std(x) > 1e-12:
        diferencias = np.diff(x)
        std = np.std(x)
        return float(np.sum(diferencias ** 2) / ((len(x) - 1) * std ** 2))

    return 0.0


def _maximum_slope(x: np.ndarray, t: np.ndarray) -> float:
    """
    Requiere t ordenado. El filtro dt > 0 descarta pares con mismo
    timestamp -- ver whitepaper.md, limitaciones conocidas: en
    StarEmbed esto descarta de media ~16% de los pares por la
    cuantizacion de mjd a 1/256 dia (hallazgo del control 16d). Otro
    dataset con mjd de precision completa no tendria este problema.
    """

    if len(t) > 1:
        dt = np.diff(t)
        dy = np.diff(x)
        valid = dt > 0
        if np.any(valid):
            return float(np.max(np.abs(dy[valid] / dt[valid])))
        return 0.0

    return 0.0


def extraer_features(curva: LightCurve) -> dict:
    """
    Calcula las 12 variables validadas sobre una LightCurve ya
    limpia y ordenada. Si `curva.metadata` contiene 'period', se
    incluye en el resultado bajo la misma clave.

    Devuelve un diccionario; no un array ni un DataFrame, para que
    quien llama decida el formato de agregacion (una fila de
    DataFrame, un tensor, lo que haga falta) sin que este modulo
    tenga una opinion sobre ello.
    """

    x = curva.magnitud
    t = curva.tiempo

    if len(x) == 0:
        valores = {v: np.nan for v in VARIABLES_BASE}
    else:
        valores = _stats_basicas(x)
        valores["eta"] = _eta(x)
        valores["maximum_slope"] = _maximum_slope(x, t)

    if "period" in curva.metadata:
        valores["period"] = curva.metadata["period"]

    return valores


def extraer_features_dataset(dataset, usar_banda_principal: bool = True) -> "pd.DataFrame":
    """
    Extrae features para todos los objetos de un Dataset.

    usar_banda_principal: si True (por defecto, y el comportamiento
    equivalente al de los scripts 20-24b), usa solo
    `dataset.banda_principal(id_objeto)` por objeto. Si False, extrae
    features de TODAS las bandas disponibles (una fila por banda) --
    util para analisis multi-banda futuros, pero no es lo que
    consumen todavia el clasificador de control ni el detector de
    anomalias, que esperan una fila por objeto.

    Import de pandas local a la funcion a proposito: el resto del
    modulo (LightCurve, extraer_features) no depende de pandas, solo
    esta funcion de conveniencia lo necesita.
    """

    import pandas as pd

    filas = []

    if usar_banda_principal:
        for id_objeto in dataset.ids_unicos():
            curva = dataset.banda_principal(id_objeto)
            if curva is None:
                continue
            curva_limpia = curva.limpia()
            fila = extraer_features(curva_limpia)
            fila["id_objeto"] = id_objeto
            fila["clase"] = curva.clase_conocida
            fila["banda_usada"] = curva.banda
            fila["n_puntos_usados"] = curva_limpia.n_observaciones()
            filas.append(fila)
    else:
        for curva in dataset:
            curva_limpia = curva.limpia()
            fila = extraer_features(curva_limpia)
            fila["id_objeto"] = curva.id_objeto
            fila["banda"] = curva.banda
            fila["clase"] = curva.clase_conocida
            filas.append(fila)

    return pd.DataFrame(filas)
