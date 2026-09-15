# -*- coding: utf-8 -*-

"""
======================================================================
adapters/starembed.py - ADAPTADOR PARA ZTF / StarEmbed
======================================================================

Traduce los parquet de StarEmbed (train-*.parquet, test-*.parquet) al
esquema interno canonico. La logica de acceso a la curva cruda es la
misma que se valido exhaustivamente en 20-23 y en los controles de
integridad 16b/16c/16d -- no se ha cambiado ningun criterio numerico,
solo se ha movido a esta clase para que el resto del sistema deje de
depender de la estructura concreta de StarEmbed.

Diferencia deliberada respecto a los scripts de experimentacion:
------------------------------------------------------------------
Los scripts 20-24b solo usaban la banda prioritaria ('r' > 'g' > 'i')
y descartaban el resto. Este adaptador, en cambio, emite UNA
LightCurve por cada banda disponible del objeto (todas comparten el
mismo id_objeto). La decision de que banda usar como "principal" se
delega a `Dataset.banda_principal()`, para no perder informacion en
la capa de adaptacion que quiza haga falta mas adelante (por ejemplo,
si el detector de anomalias evoluciona a usar varias bandas a la vez).
Si se quiere reproducir EXACTAMENTE el comportamiento de 20-24b hoy,
usar `dataset.banda_principal(id_objeto)` en vez de iterar
`dataset.curvas` directamente.
"""

from __future__ import annotations

import os
from typing import Iterator, Optional

import pandas as pd

from ..schema import LightCurve
from .base import AdaptadorDataset
from .registro import registrar_fuente, ParametroConfig


BANDAS_CONOCIDAS = ("r", "g", "i")

POSIBLES_COLUMNAS_ID = ["object_id", "source_id", "sourceid", "id"]


@registrar_fuente(
    id="starembed",
    nombre="StarEmbed (ZTF, 40k curvas multibanda)",
    descripcion=(
        "Dataset curado de ~40.000 curvas de luz de ZTF, con 7 clases "
        "conocidas (EW, EA, RRab, RRc, RRd, RS CVn, LPV) mas un split "
        "'anom' de 10 clases OOD reales. Ficheros parquet locales."
    ),
    parametros=[
        ParametroConfig(
            nombre="rutas_parquet",
            etiqueta="Ficheros parquet (train + test, o el que corresponda)",
            tipo="ruta_archivo",
            requerido=True,
            ayuda=(
                "Acepta varios ficheros (p. ej. las dos partes de train). "
                "La interfaz debe permitir seleccionar mas de uno."
            ),
        ),
        ParametroConfig(
            nombre="max_objetos",
            etiqueta="Maximo de objetos a cargar (vacio = todos)",
            tipo="numero",
            requerido=False,
        ),
    ],
)
class AdaptadorStarEmbed(AdaptadorDataset):
    """
    Parametros
    ----------
    rutas_parquet : uno o varios ficheros parquet de StarEmbed
        (p. ej. las dos partes de train, o el fichero de test).
    max_objetos : si se indica, trunca a los primeros N objetos
        (equivalente al `iloc[:MAX_TRAIN]` usado en 20-24b).
        Ojo: la representatividad de ese truncado esta verificada
        SOLO para StarEmbed (ver whitepaper.md, control 20b) -- si
        se usa max_objetos con otro dataset, no se puede asumir lo
        mismo sin repetir ese control.
    columna_clase : nombre de la columna de etiqueta de clase.
        Por defecto 'class_str', como en StarEmbed.
    """

    nombre = "starembed"

    def __init__(
        self,
        rutas_parquet: list[str] | str,
        max_objetos: Optional[int] = None,
        columna_clase: str = "class_str",
        saltar_invalidas: bool = True,
    ):
        super().__init__(saltar_invalidas=saltar_invalidas)

        if isinstance(rutas_parquet, str):
            rutas_parquet = [rutas_parquet]

        self.rutas_parquet = rutas_parquet
        self.max_objetos = max_objetos
        self.columna_clase = columna_clase

    def _columna_id(self, df: pd.DataFrame) -> Optional[str]:
        for col in POSIBLES_COLUMNAS_ID:
            if col in df.columns:
                return col
        return None

    def _cargar_dataframe(self) -> pd.DataFrame:

        frames = [pd.read_parquet(ruta) for ruta in self.rutas_parquet]
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

        if self.max_objetos is not None:
            df = df.iloc[: self.max_objetos].copy()

        return df

    def iter_curvas(self) -> Iterator[LightCurve]:

        df = self._cargar_dataframe()

        col_id = self._columna_id(df)
        tiene_clase = self.columna_clase in df.columns
        tiene_periodo = "period" in df.columns

        for i, (_, fila) in enumerate(df.iterrows()):

            id_objeto = str(fila[col_id]) if col_id else str(i)

            bands = fila.get("bands_data", None)

            if bands is None or not isinstance(bands, dict):
                self.n_descartados += 1
                continue

            clase = str(fila[self.columna_clase]) if tiene_clase else None

            metadata = {}
            if tiene_periodo:
                metadata["period"] = fila.get("period")

            for banda_nombre, banda_datos in bands.items():

                if banda_datos is None:
                    continue

                target = banda_datos.get("target", [])
                mjd = banda_datos.get("mjd", [])

                if target is None or mjd is None:
                    continue

                curva = LightCurve(
                    id_objeto=id_objeto,
                    tiempo=mjd,
                    magnitud=target,
                    banda=banda_nombre,
                    clase_conocida=clase,
                    es_flujo=False,
                    metadata=dict(metadata),
                )

                yield from self._emitir(curva)


def cargar_train_test_starembed(
    base_dir: str,
    max_train: Optional[int] = 25000,
    max_test: Optional[int] = 8000,
):
    """
    Helper de conveniencia que reproduce la carga usada en 20-24b:
    dos ficheros de train concatenados + un fichero de test, con el
    mismo truncado por defecto. Devuelve (dataset_train, dataset_test).

    Esto NO es parte de la interfaz generica del adaptador -- es un
    atajo especifico de StarEmbed para no tener que repetir las rutas
    de los tres ficheros en cada script/notebook que trabaje con este
    dataset concreto.
    """

    rutas_train = [
        os.path.join(base_dir, "train-00000-of-00002.parquet"),
        os.path.join(base_dir, "train-00001-of-00002.parquet"),
    ]
    ruta_test = os.path.join(base_dir, "test-00000-of-00001.parquet")

    adaptador_train = AdaptadorStarEmbed(rutas_train, max_objetos=max_train)
    adaptador_test = AdaptadorStarEmbed(ruta_test, max_objetos=max_test)

    return adaptador_train.cargar(), adaptador_test.cargar()
