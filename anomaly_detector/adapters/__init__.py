# -*- coding: utf-8 -*-

"""
Adaptadores de dataset. Cada uno traduce un formato externo concreto
al esquema interno canonico (ver ../schema.py).

Adaptadores disponibles:
  - starembed.AdaptadorStarEmbed : ZTF / StarEmbed (parquet).

Para anadir soporte a un dataset nuevo, ver base.AdaptadorDataset.
"""

from .base import AdaptadorDataset
from .starembed import AdaptadorStarEmbed, cargar_train_test_starembed
from .tabular_generico import AdaptadorTabularGenerico
from .gaia import AdaptadorGaiaDR3, cargar_gaia_dr3
from .registro import (
    FuenteDatos,
    ParametroConfig,
    registrar_fuente,
    fuentes_disponibles,
    fuentes_declarativas,
    fuentes_python_registradas,
)

__all__ = [
    "AdaptadorDataset",
    "AdaptadorStarEmbed",
    "cargar_train_test_starembed",
    "AdaptadorTabularGenerico",
    "AdaptadorGaiaDR3",
    "cargar_gaia_dr3",
    "FuenteDatos",
    "ParametroConfig",
    "registrar_fuente",
    "fuentes_disponibles",
    "fuentes_declarativas",
    "fuentes_python_registradas",
]
