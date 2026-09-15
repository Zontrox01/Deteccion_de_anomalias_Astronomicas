# -*- coding: utf-8 -*-

"""
======================================================================
adapters/base.py - INTERFAZ QUE DEBE CUMPLIR CUALQUIER ADAPTADOR
======================================================================

Un adaptador traduce un dataset externo (con el formato/esquema que
sea) al esquema interno canonico (`schema.LightCurve` /
`schema.Dataset`). Es el UNICO punto del sistema que puede conocer
detalles especificos de un dataset concreto.

Para soportar un dataset nuevo:
  1. Crear una clase que herede de `AdaptadorDataset`.
  2. Implementar `cargar()`.
  3. No tocar nada fuera de `adapters/` -- si hace falta tocar
     `features.py` o el detector de anomalias para que un dataset
     nuevo funcione, es una señal de que el esquema canonico se ha
     quedado corto y hay que ampliarlo en `schema.py` (afectando a
     TODOS los adaptadores), no parchear un caso concreto.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from ..schema import LightCurve, Dataset, CurvaInvalidaError


class AdaptadorDataset(ABC):
    """
    Interfaz base. Los adaptadores concretos heredan de esta clase.

    Se ofrecen dos formas de consumir un adaptador:

      - `cargar()` -> Dataset completo en memoria. Comodo para
        datasets que ya caben en memoria (como StarEmbed con
        25k/8k objetos) y para trabajar interactivamente.
      - `iter_curvas()` -> generador perezoso, objeto a objeto.
        Pensado para datasets grandes donde cargar todo de golpe no
        es viable. `cargar()` esta implementado por defecto en
        terminos de `iter_curvas()`, asi que un adaptador nuevo solo
        tiene que implementar esta ultima.
    """

    #: Nombre corto identificativo del adaptador (usado en logs y en
    #: la interfaz para que el astronomo sepa que formato ha cargado).
    nombre: str = "adaptador_base"

    def __init__(self, saltar_invalidas: bool = True):
        """
        saltar_invalidas: si True (por defecto), los objetos que no
        pasan `LightCurve.validar()` se descartan silenciosamente
        (se cuentan y se pueden consultar despues via
        `self.n_descartados`). Si False, la primera curva invalida
        detiene la carga con CurvaInvalidaError -- util durante el
        desarrollo de un adaptador nuevo, para no acostumbrarse a
        ignorar datos malformados sin darse cuenta.
        """
        self.saltar_invalidas = saltar_invalidas
        self.n_descartados = 0

    @abstractmethod
    def iter_curvas(self) -> Iterator[LightCurve]:
        """
        Debe devolver (yield) instancias de LightCurve YA validadas
        (llamar a `curva.validar()` es responsabilidad del adaptador,
        antes de hacer yield -- ver `_emitir` como helper).
        """
        raise NotImplementedError

    def _emitir(self, curva: LightCurve) -> Iterator[LightCurve]:
        """
        Helper para que los adaptadores concretos validen de forma
        consistente. Uso tipico dentro de iter_curvas():

            for fila in self._filas_crudas():
                curva = LightCurve(...)
                yield from self._emitir(curva)
        """
        try:
            curva.validar()
            yield curva
        except CurvaInvalidaError as e:
            self.n_descartados += 1
            if not self.saltar_invalidas:
                raise
            # Si se decide saltar, no se hace yield de nada para esta
            # curva. El descarte queda contabilizado en
            # self.n_descartados para que se pueda reportar despues.

    def cargar(self) -> Dataset:
        """
        Materializa todo el dataset en memoria. Implementado en
        terminos de iter_curvas(), no hace falta sobreescribirlo en
        adaptadores concretos salvo que haya una optimizacion
        especifica del formato de origen.
        """

        curvas = list(self.iter_curvas())

        return Dataset(nombre=self.nombre, curvas=curvas)
