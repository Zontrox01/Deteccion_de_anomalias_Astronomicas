# -*- coding: utf-8 -*-

"""
======================================================================
features_forma.py - FEATURES DE FORMA DE CURVA (EXPERIMENTAL)
======================================================================

Extension experimental de features.py, separada a proposito en su
propio modulo: a diferencia de las 12 variables de VARIABLES_BASE
(validadas exhaustivamente sobre StarEmbed en la sesion de auditoria,
ver whitepaper.md seccion 4), estas features de forma de curva son una
HIPOTESIS EN PRUEBA -- se añaden aqui para poder evaluarlas sin tocar
el espacio de features ya validado, y solo se promoverian a
VARIABLES_BASE si demuestran mejorar resultados reales.

Motivacion (whitepaper.md, seccion 4.9): el AUC de deteccion OOD
(0.6608) es bajo especificamente en clases OOD que son variantes
sutiles de una clase conocida (Blazhko de RRab, HADS/ELL cercanas a
RS CVn/EW), y se descarto con evidencia que el problema fuera `period`
(ver comparar_periodo_anom.py). La hipotesis que se prueba aqui: esas
clases se diferencian por la FORMA de la curva (asimetria del pliegue
en fase, armonicos secundarios), no por su amplitud/dispersion global
(que ya capturan las 12 variables base) ni por su periodo.

Metodo: descomposicion de Fourier del plegado en fase (metodologia
estandar en clasificacion de RR Lyrae desde Simon & Lee 1981, tambien
usada para caracterizar morfologia de binarias eclipsantes). Requiere
`period` (ya disponible en LightCurve.metadata para StarEmbed).

Parametros extraidos (K armonicos, por defecto K=3):
  - fourier_amplitud_1  : amplitud del primer armonico (A1)
  - fourier_R21, R31    : razones de amplitud A2/A1, A3/A1 -- forma
                          relativa de la curva, independiente de su
                          amplitud absoluta
  - fourier_phi21, phi31: diferencias de fase entre armonicos
                          (envueltas a [0, 2π))
  - fourier_residual_std: dispersion de los puntos respecto al modelo
                          periodico ajustado -- deberia ser mayor en
                          estrellas con amplitud variable ciclo a
                          ciclo (Blazhko), aunque el Fourier de un
                          solo periodo no vea la modulacion en si
"""

from __future__ import annotations

import numpy as np

from .schema import LightCurve


N_ARMONICOS_DEFECTO = 3
MIN_PUNTOS_POR_ARMONICO = 4  # minimo de puntos para fiarse del ajuste

CLAVES_FOURIER = [
    "fourier_amplitud_1",
    "fourier_R21",
    "fourier_R31",
    "fourier_phi21",
    "fourier_phi31",
    "fourier_residual_std",
]


def _envolver_angulo(angulo: float) -> float:
    """Envuelve un angulo en radianes al rango [0, 2*pi)."""
    return float(np.mod(angulo, 2 * np.pi))


def extraer_features_fourier(
    curva: LightCurve,
    n_armonicos: int = N_ARMONICOS_DEFECTO,
) -> dict:
    """
    Calcula los parametros de Fourier del plegado en fase de una
    LightCurve YA LIMPIA (ver LightCurve.limpia() -- esta funcion no
    filtra NaN ni ordena, igual que extraer_features() en
    features.py, por el mismo motivo: responsabilidad explicita de
    quien llama, no implicita aqui).

    Devuelve NaN en todas las claves (no lanza excepcion) si:
      - no hay periodo valido en curva.metadata['period'], o
      - hay menos puntos que n_armonicos * MIN_PUNTOS_POR_ARMONICO
        (ajuste no fiable con pocos puntos).
    """

    vacio = {k: np.nan for k in CLAVES_FOURIER}

    periodo = curva.metadata.get("period") if curva.metadata else None

    if periodo is None or not np.isfinite(periodo) or periodo <= 0:
        return vacio

    t = curva.tiempo
    x = curva.magnitud

    if len(t) < n_armonicos * MIN_PUNTOS_POR_ARMONICO:
        return vacio

    fase = np.mod((t - t[0]) / periodo, 1.0)

    columnas = [np.ones_like(fase)]
    for k in range(1, n_armonicos + 1):
        columnas.append(np.cos(2 * np.pi * k * fase))
        columnas.append(np.sin(2 * np.pi * k * fase))
    diseno = np.column_stack(columnas)

    try:
        coef, _, rango, _ = np.linalg.lstsq(diseno, x, rcond=None)
    except np.linalg.LinAlgError:
        return vacio

    if rango < diseno.shape[1]:
        # sistema mal condicionado (p. ej. fase muy poco muestreada)
        return vacio

    amplitudes = []
    fases_k = []

    for k in range(1, n_armonicos + 1):
        ak = coef[1 + 2 * (k - 1)]
        bk = coef[2 + 2 * (k - 1)]
        amplitudes.append(float(np.hypot(ak, bk)))
        fases_k.append(float(np.arctan2(bk, ak)))

    modelo = diseno @ coef
    residual_std = float(np.std(x - modelo))

    A1 = amplitudes[0]

    if A1 > 1e-9:
        R21 = amplitudes[1] / A1 if n_armonicos >= 2 else np.nan
        R31 = amplitudes[2] / A1 if n_armonicos >= 3 else np.nan
    else:
        R21 = np.nan
        R31 = np.nan

    phi21 = _envolver_angulo(fases_k[1] - 2 * fases_k[0]) if n_armonicos >= 2 else np.nan
    phi31 = _envolver_angulo(fases_k[2] - 3 * fases_k[0]) if n_armonicos >= 3 else np.nan

    return {
        "fourier_amplitud_1": A1,
        "fourier_R21": R21,
        "fourier_R31": R31,
        "fourier_phi21": phi21,
        "fourier_phi31": phi31,
        "fourier_residual_std": residual_std,
    }


def extraer_features_fourier_dataset(dataset, n_armonicos: int = N_ARMONICOS_DEFECTO):
    """
    Extrae features de Fourier para todos los objetos de un Dataset,
    usando la banda principal de cada uno (mismo criterio que
    features.extraer_features_dataset). Devuelve un DataFrame con
    'id_objeto' + las columnas de CLAVES_FOURIER, pensado para
    cruzarse (merge por id_objeto) con el resultado de
    extraer_features_dataset().
    """

    import pandas as pd

    filas = []

    for id_objeto in dataset.ids_unicos():

        curva = dataset.banda_principal(id_objeto)
        if curva is None:
            continue

        curva_limpia = curva.limpia()
        fila = extraer_features_fourier(curva_limpia, n_armonicos=n_armonicos)
        fila["id_objeto"] = id_objeto
        filas.append(fila)

    return pd.DataFrame(filas)
