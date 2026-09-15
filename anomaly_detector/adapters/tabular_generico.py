# -*- coding: utf-8 -*-

"""
======================================================================
adapters/tabular_generico.py - ADAPTADOR DIRIGIDO POR MANIFIESTO YAML
======================================================================

Interpreta un manifiesto YAML (ver fuentes/EJEMPLO_generico.yaml para
la plantilla comentada) y traduce un fichero tabular (CSV o parquet)
en formato "largo" -- una fila por combinacion (objeto, epoca, banda)
-- al esquema canonico, sin que el contribuidor de la fuente tenga
que escribir ninguna linea de Python.

Por que "formato largo" y no el formato de StarEmbed (arrays
anidados por objeto)
------------------------------------------------------------------
El formato de StarEmbed (una fila por objeto, con un diccionario de
bandas conteniendo arrays) es COMODO mais NO es el mas comun en la
practica -- la mayoria de exports CSV de surveys y pipelines de
fotometria dan una fila por punto observado (object_id, mjd, mag,
mag_err, filtro...). Ese es el caso que este adaptador cubre, porque
es el que mas fuentes nuevas van a poder aportar sin escribir codigo.
Formatos con arrays anidados (como StarEmbed) siguen necesitando un
adaptador Python dedicado (Nivel 2, ver adapters/base.py) -- un YAML
no puede expresar esa estructura de forma razonable.

Contrato del manifiesto (campos reconocidos)
----------------------------------------------
nombre, descripcion, id (opcional, por defecto el nombre del fichero)
acceso.formato: "csv" | "parquet"
formato_datos: "largo"  (unico soportado por ahora, ver arriba)
mapeo_columnas:
  id_objeto: <nombre de columna>       (obligatorio)
  tiempo: <nombre de columna>          (obligatorio)
  magnitud: <nombre de columna>        (obligatorio)
  error: <nombre de columna>           (opcional)
  banda: <nombre de columna>           (opcional; si no esta, usar banda_fija)
  clase_conocida: <nombre de columna>  (opcional)
banda_fija: <string>                   (usada solo si no hay columna de banda)
unidades.offset_tiempo: <float>        (se SUMA al valor crudo de tiempo;
                                         por defecto 0.0. Sirve para
                                         convertir JD a MJD, aplicar un
                                         offset de referencia, etc.)
valores_nulos: [<string>, ...]         (valores a tratar como NaN antes
                                         de convertir a float)
"""

from __future__ import annotations

import os
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from ..schema import LightCurve
from .base import AdaptadorDataset


FORMATOS_DATOS_SOPORTADOS = ("largo",)


class ManifiestoInvalidoError(ValueError):
    pass


class AdaptadorTabularGenerico(AdaptadorDataset):

    def __init__(
        self,
        manifiesto_path: str,
        ruta_datos: Optional[str] = None,
        saltar_invalidas: bool = True,
    ):
        super().__init__(saltar_invalidas=saltar_invalidas)

        self.manifiesto_path = manifiesto_path
        self.manifiesto = self._cargar_manifiesto(manifiesto_path)

        self.ruta_datos = ruta_datos or self.manifiesto.get("acceso", {}).get(
            "ruta_defecto"
        )

        if not self.ruta_datos:
            raise ManifiestoInvalidoError(
                f"No se indico 'ruta_datos' y el manifiesto "
                f"({manifiesto_path}) tampoco define "
                f"acceso.ruta_defecto. Hace falta saber que fichero leer."
            )

        self.nombre = self.manifiesto.get("nombre", "tabular_generico")

        formato_datos = self.manifiesto.get("formato_datos", "largo")
        if formato_datos not in FORMATOS_DATOS_SOPORTADOS:
            raise ManifiestoInvalidoError(
                f"formato_datos='{formato_datos}' no soportado. "
                f"Soportados: {FORMATOS_DATOS_SOPORTADOS}. Para formatos "
                f"con arrays anidados por objeto, se necesita un "
                f"adaptador Python (Nivel 2), no un manifiesto YAML."
            )

        self._validar_mapeo()

    # ------------------------------------------------------------------

    def _cargar_manifiesto(self, ruta: str) -> dict:

        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "Hace falta 'pyyaml' para usar AdaptadorTabularGenerico "
                "(pip install pyyaml)."
            ) from e

        with open(ruta, "r", encoding="utf-8") as f:
            manifiesto = yaml.safe_load(f)

        if not manifiesto:
            raise ManifiestoInvalidoError(f"{ruta} esta vacio o no es YAML valido.")

        return manifiesto

    def _validar_mapeo(self) -> None:

        mapeo = self.manifiesto.get("mapeo_columnas", {})

        obligatorios = ["id_objeto", "tiempo", "magnitud"]
        faltantes = [c for c in obligatorios if c not in mapeo]

        if faltantes:
            raise ManifiestoInvalidoError(
                f"El manifiesto {self.manifiesto_path} no define "
                f"mapeo_columnas para: {faltantes} (obligatorios)."
            )

        if "banda" not in mapeo and not self.manifiesto.get("banda_fija"):
            raise ManifiestoInvalidoError(
                f"El manifiesto {self.manifiesto_path} no define ni "
                f"mapeo_columnas.banda ni banda_fija -- hace falta uno "
                f"de los dos para saber a que banda pertenece cada curva."
            )

    # ------------------------------------------------------------------

    def _cargar_tabla(self) -> pd.DataFrame:

        formato = self.manifiesto.get("acceso", {}).get("formato", "csv")

        if formato == "csv":
            df = pd.read_csv(self.ruta_datos)
        elif formato == "parquet":
            df = pd.read_parquet(self.ruta_datos)
        else:
            raise ManifiestoInvalidoError(
                f"acceso.formato='{formato}' no soportado (usa 'csv' o 'parquet')."
            )

        valores_nulos = self.manifiesto.get("valores_nulos", [])

        if valores_nulos:
            df = df.replace(valores_nulos, np.nan)

        return df

    # ------------------------------------------------------------------

    def iter_curvas(self) -> Iterator[LightCurve]:

        df = self._cargar_tabla()
        mapeo = self.manifiesto["mapeo_columnas"]

        col_id = mapeo["id_objeto"]
        col_tiempo = mapeo["tiempo"]
        col_magnitud = mapeo["magnitud"]
        col_error = mapeo.get("error")
        col_banda = mapeo.get("banda")
        col_clase = mapeo.get("clase_conocida")

        banda_fija = self.manifiesto.get("banda_fija")
        offset_tiempo = float(
            self.manifiesto.get("unidades", {}).get("offset_tiempo", 0.0)
        )

        for col, etiqueta in [
            (col_id, "id_objeto"), (col_tiempo, "tiempo"), (col_magnitud, "magnitud")
        ]:
            if col not in df.columns:
                raise ManifiestoInvalidoError(
                    f"La columna '{col}' (mapeo_columnas.{etiqueta}) no "
                    f"existe en el fichero de datos. Columnas disponibles: "
                    f"{list(df.columns)}"
                )

        for id_objeto, grupo in df.groupby(col_id, sort=False):

            if col_banda and col_banda in grupo.columns:
                sub_grupos = grupo.groupby(col_banda, sort=False)
            else:
                sub_grupos = [(banda_fija, grupo)]

            for banda_nombre, sub in sub_grupos:

                tiempo = pd.to_numeric(sub[col_tiempo], errors="coerce").to_numpy(dtype=float)
                tiempo = tiempo + offset_tiempo

                magnitud = pd.to_numeric(sub[col_magnitud], errors="coerce").to_numpy(dtype=float)

                error = None
                if col_error and col_error in sub.columns:
                    error = pd.to_numeric(sub[col_error], errors="coerce").to_numpy(dtype=float)

                clase = None
                if col_clase and col_clase in sub.columns and len(sub) > 0:
                    valor_clase = sub[col_clase].iloc[0]
                    clase = str(valor_clase) if pd.notna(valor_clase) else None

                curva = LightCurve(
                    id_objeto=str(id_objeto),
                    tiempo=tiempo,
                    magnitud=magnitud,
                    error=error,
                    banda=str(banda_nombre) if banda_nombre is not None else None,
                    clase_conocida=clase,
                    metadata={"fuente": self.nombre},
                )

                yield from self._emitir(curva)
