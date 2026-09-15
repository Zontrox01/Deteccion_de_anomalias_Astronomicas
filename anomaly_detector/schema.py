# -*- coding: utf-8 -*-

"""
======================================================================
schema.py - ESQUEMA INTERNO CANONICO
======================================================================

Este modulo define la unica forma en la que el resto del sistema
(extraccion de features, deteccion de anomalias, interfaz PySide6)
entiende una curva de luz. Ningun modulo aguas abajo de un adaptador
debe conocer nada especifico de StarEmbed, ZTF, ni de ningun otro
dataset concreto -- todo pasa antes por aqui.

Decision de diseño (ver whitepaper.md, seccion 3): la capa de
adaptacion es lo unico que cambia por dataset. Todo lo demas opera
exclusivamente sobre `LightCurve` y `Dataset`.

Por que un dataclass y no un DataFrame de pandas directamente
------------------------------------------------------------------
Se eligio un objeto explicito en vez de pasar un DataFrame con
columnas "bien conocidas" porque:

  1. Hace el contrato explicito: un adaptador que no rellena `tiempo`
     y `magnitud` correctamente falla al construir el objeto, no en
     mitad del calculo de una feature 40 lineas mas abajo.
  2. Permite validar la curva en un unico sitio (`LightCurve.validar`)
     en vez de repetir comprobaciones de NaN/longitud en cada script,
     que es exactamente lo que pasaba en el pipeline de StarEmbed
     (cada script de 20 a 24b repetia su propia version de
     `obtener_curva_cruda`).
  3. No ata el sistema a que el dato de origen sea siempre tabular
     (un FITS o una API devuelven la curva de otra forma).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Iterator, Iterable

import numpy as np


# ======================================================================
# EXCEPCIONES
# ======================================================================

class CurvaInvalidaError(ValueError):
    """
    Se lanza cuando una curva no cumple el contrato minimo del
    esquema canonico (longitudes desiguales, sin puntos finitos, etc).
    Los adaptadores deben capturar esto si quieren saltarse objetos
    invalidos en vez de detener la carga completa; por defecto, un
    objeto invalido detiene la construccion de la curva.
    """
    pass


# ======================================================================
# LIGHTCURVE: LA UNIDAD MINIMA DEL SISTEMA
# ======================================================================

@dataclass
class LightCurve:
    """
    Representacion canonica de una curva de luz, independiente del
    dataset de origen.

    Campos obligatorios
    --------------------
    id_objeto : identificador del objeto. Debe ser estable y, siempre
        que el dataset de origen lo permita, trazable a un catalogo
        externo real (por ejemplo, una designacion CSS_J... como en
        StarEmbed) -- esto es lo que permite luego el cruce con
        VSX/SIMBAD/Gaia sin tener que volver a los datos crudos.
    tiempo : array de tiempos (mjd o equivalente), en DIAS. Si el
        dataset de origen usa otra unidad, el adaptador debe convertir
        antes de construir el objeto -- este esquema no hace
        conversion de unidades por si mismo.
    magnitud : array de magnitudes (o flujo, ver `es_flujo`). Debe
        tener la misma longitud que `tiempo`.

    Campos opcionales
    -------------------
    error : incertidumbre por punto, si el dataset la proporciona.
        Necesaria para un stetson_K "de verdad" (ver whitepaper.md,
        seccion 6, sobre la aproximacion usada mientras no este
        disponible).
    banda : filtro/banda fotometrica de esta curva concreta (una
        LightCurve es una banda; un objeto con varias bandas se
        representa como varias LightCurve con el mismo id_objeto,
        ver `Dataset.curvas_de`).
    clase_conocida : etiqueta de clase, si el dataset viene etiquetado
        (entrenamiento/validacion). None si el dataset no tiene
        etiquetas (caso real de uso: un survey nuevo sin clasificar).
    es_flujo : si True, `magnitud` contiene flujo en vez de magnitud.
        Lo consume la capa de features para decidir si tiene sentido
        `percent_amplitude` tal como esta definida (pensada para
        magnitudes). Por defecto False.
    metadata : cualquier informacion adicional que el adaptador quiera
        conservar sin que el esquema canonico tenga que anticiparla
        (coordenadas, periodo precalculado, campo de observacion...).
        Las features y el detector de anomalias NO leen `metadata`
        directamente salvo que se les indique explicitamente (por
        ejemplo, `period` se lee de aqui si esta presente, ver
        features.py).
    """

    id_objeto: str
    tiempo: np.ndarray
    magnitud: np.ndarray
    error: Optional[np.ndarray] = None
    banda: Optional[str] = None
    clase_conocida: Optional[str] = None
    es_flujo: bool = False
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.tiempo = np.asarray(self.tiempo, dtype=float)
        self.magnitud = np.asarray(self.magnitud, dtype=float)

        if self.error is not None:
            self.error = np.asarray(self.error, dtype=float)

    def validar(self, minimo_puntos: int = 2) -> None:
        """
        Comprueba el contrato minimo del esquema. Lanza
        CurvaInvalidaError con un mensaje especifico si algo falla,
        en vez de dejar que el error aparezca mas tarde y mas lejos
        de la causa real (la leccion del pipeline de StarEmbed: los
        problemas de datos hay que atraparlos en la frontera, no en
        mitad de un calculo de eta).
        """

        if len(self.tiempo) != len(self.magnitud):
            raise CurvaInvalidaError(
                f"{self.id_objeto}: tiempo y magnitud tienen longitudes "
                f"distintas ({len(self.tiempo)} vs {len(self.magnitud)})."
            )

        if self.error is not None and len(self.error) != len(self.tiempo):
            raise CurvaInvalidaError(
                f"{self.id_objeto}: error tiene longitud distinta a "
                f"tiempo/magnitud ({len(self.error)} vs {len(self.tiempo)})."
            )

        n_finitos = int(np.sum(np.isfinite(self.tiempo) & np.isfinite(self.magnitud)))

        if n_finitos < minimo_puntos:
            raise CurvaInvalidaError(
                f"{self.id_objeto}: solo {n_finitos} puntos finitos, "
                f"se requieren al menos {minimo_puntos}."
            )

    def limpia(self) -> "LightCurve":
        """
        Devuelve una NUEVA LightCurve con solo los puntos finitos en
        tiempo y magnitud, y ordenada cronologicamente.

        No muta el objeto original -- el pipeline de StarEmbed tuvo
        varios sustos por confundir "ya esta ordenado" con "lo
        ordeno yo aqui sin decirselo a nadie" (ver 16b). Aqui queda
        explicito: si quieres la version limpia, la pides.
        """

        mask = np.isfinite(self.tiempo) & np.isfinite(self.magnitud)

        if self.error is not None:
            mask = mask & np.isfinite(self.error)

        t = self.tiempo[mask]
        x = self.magnitud[mask]
        e = self.error[mask] if self.error is not None else None

        orden = np.argsort(t, kind="mergesort")

        return LightCurve(
            id_objeto=self.id_objeto,
            tiempo=t[orden],
            magnitud=x[orden],
            error=e[orden] if e is not None else None,
            banda=self.banda,
            clase_conocida=self.clase_conocida,
            es_flujo=self.es_flujo,
            metadata=dict(self.metadata),
        )

    def n_observaciones(self) -> int:
        return int(np.sum(np.isfinite(self.tiempo) & np.isfinite(self.magnitud)))

    def __len__(self) -> int:
        return len(self.tiempo)


# ======================================================================
# DATASET: COLECCION DE OBJETOS, POSIBLEMENTE MULTI-BANDA
# ======================================================================

@dataclass
class Dataset:
    """
    Coleccion de LightCurve, con utilidades para trabajar por objeto
    cuando un mismo objeto tiene varias bandas.

    Se mantiene deliberadamente simple (una lista, no un indice
    complejo) porque el patron de acceso real hasta ahora siempre ha
    sido "recorrer todo una vez" (extraccion de features) -- no hace
    falta una estructura mas sofisticada todavia. Si en el futuro
    hace falta acceso aleatorio frecuente por id_objeto, se anade un
    indice interno sin cambiar la interfaz publica.
    """

    nombre: str
    curvas: list[LightCurve] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.curvas)

    def __iter__(self) -> Iterator[LightCurve]:
        return iter(self.curvas)

    def ids_unicos(self) -> list[str]:
        vistos = []
        vistos_set = set()
        for curva in self.curvas:
            if curva.id_objeto not in vistos_set:
                vistos.append(curva.id_objeto)
                vistos_set.add(curva.id_objeto)
        return vistos

    def curvas_de(self, id_objeto: str) -> list[LightCurve]:
        """Todas las bandas disponibles para un objeto concreto."""
        return [c for c in self.curvas if c.id_objeto == id_objeto]

    def banda_principal(
        self, id_objeto: str, prioridad: Iterable[str] = ("r", "g", "i")
    ) -> Optional[LightCurve]:
        """
        Devuelve la curva de la banda preferida disponible para un
        objeto, siguiendo el mismo criterio de prioridad que se uso
        en todo el pipeline de StarEmbed (20-24b: intentar 'r', si no
        'g', si no 'i'). Si el objeto no tiene ninguna banda de la
        lista de prioridad, devuelve la primera que tenga.
        """

        disponibles = {c.banda: c for c in self.curvas_de(id_objeto) if c.banda}

        for banda in prioridad:
            if banda in disponibles:
                return disponibles[banda]

        curvas = self.curvas_de(id_objeto)
        return curvas[0] if curvas else None

    def clases_conocidas(self) -> set[str]:
        return {
            c.clase_conocida for c in self.curvas if c.clase_conocida is not None
        }

    def es_supervisado(self) -> bool:
        """
        True si TODOS los objetos tienen clase_conocida. Un dataset
        "real" de descubrimiento (sin etiquetas) debe dar False aqui,
        y el pipeline de deteccion de anomalias debe poder operar
        igualmente -- ahora mismo (24/24b) el detector se entrena
        contra clases conocidas, asi que un dataset totalmente sin
        etiquetar no tiene todavia un camino definido. Ver
        whitepaper.md, seccion 7 (pendiente).
        """

        return all(c.clase_conocida is not None for c in self.curvas)
