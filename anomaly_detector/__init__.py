# -*- coding: utf-8 -*-

"""
Deteccion de anomalias astronomicas en curvas de luz.

Estructura del paquete:
  - schema.py    : esquema interno canonico (LightCurve, Dataset).
  - adapters/    : un modulo por dataset externo soportado.
  - features.py  : extraccion de las 12 variables validadas.
  - (pendiente)  : deteccion.py -- Isolation Forest + LOF, global y
                    por clase (empaquetado de 24/24b, siguiente paso).

Ver whitepaper.md en la raiz del proyecto para el contexto completo
de por que esta estructurado asi.
"""

from .schema import LightCurve, Dataset, CurvaInvalidaError
from .features import extraer_features, extraer_features_dataset, VARIABLES_BASE
from .features_forma import (
    extraer_features_fourier,
    extraer_features_fourier_dataset,
    CLAVES_FOURIER,
)
from .deteccion import (
    detectar_anomalias,
    detectar_global,
    detectar_por_clase,
    detectar_ood_multiclase,
    entrenar_clasificador_control,
)

__all__ = [
    "LightCurve",
    "Dataset",
    "CurvaInvalidaError",
    "extraer_features",
    "extraer_features_dataset",
    "VARIABLES_BASE",
    "extraer_features_fourier",
    "extraer_features_fourier_dataset",
    "CLAVES_FOURIER",
    "detectar_anomalias",
    "detectar_global",
    "detectar_por_clase",
    "detectar_ood_multiclase",
    "entrenar_clasificador_control",
]
