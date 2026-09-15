# -*- coding: utf-8 -*-

"""
======================================================================
adapters/registro.py - REGISTRO DE FUENTES DE DATOS (para el selector)
======================================================================

Combina dos tipos de fuente de datos en una unica lista, pensada para
alimentar directamente un combo/selector en la interfaz PySide6:

  - Fuentes DECLARATIVAS: manifiestos YAML en un directorio (por
    defecto `fuentes/` en la raiz del proyecto). No requieren escribir
    Python -- ver tabular_generico.py.
  - Fuentes PYTHON: clases AdaptadorDataset registradas con el
    decorador @registrar_fuente. Para logica que un YAML no puede
    expresar (paginacion remota, autenticacion, estructuras anidadas).

Ninguna de las dos formas de registrar sabe nada de la otra -- ambas
producen el mismo tipo de objeto (`FuenteDatos`), que es lo unico que
consume la interfaz. Anadir una fuente nueva de cualquiera de los dos
tipos NUNCA requiere tocar este archivo ni ningun otro fuera de:
  - un YAML nuevo en fuentes/, o
  - un modulo Python nuevo con la clase decorada.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .base import AdaptadorDataset


# ======================================================================
# TIPOS
# ======================================================================

@dataclass
class ParametroConfig:
    """
    Describe un parametro que el usuario debe (o puede) rellenar para
    usar una fuente concreta -- p. ej. la ruta a un fichero de datos,
    o una clave de API. La interfaz genera el formulario a partir de
    esto; ninguna fuente concreta tiene que preocuparse de como se
    renderiza.
    """

    nombre: str  # nombre del argumento que recibira la fabrica
    etiqueta: str  # texto mostrado al usuario
    tipo: str  # "ruta_archivo" | "ruta_carpeta" | "texto" | "numero" | "credencial"
    requerido: bool = True
    valor_por_defecto: Optional[str] = None
    ayuda: Optional[str] = None


@dataclass
class FuenteDatos:
    """
    Una entrada del selector de "Fuente de datos". `fabrica` se llama
    con los parametros que el usuario haya rellenado (segun
    `parametros`) y debe devolver un AdaptadorDataset ya listo para
    `.cargar()` o `.iter_curvas()`.
    """

    id: str
    nombre: str
    descripcion: str
    tipo: str  # "declarativa" | "python"
    fabrica: Callable[..., AdaptadorDataset]
    parametros: list[ParametroConfig] = field(default_factory=list)
    origen: Optional[str] = None  # ruta del YAML o modulo Python, para diagnostico


# ======================================================================
# REGISTRO DE FUENTES PYTHON (NIVEL 2)
# ======================================================================

_REGISTRO_PYTHON: dict[str, FuenteDatos] = {}


def registrar_fuente(
    id: str,
    nombre: str,
    descripcion: str,
    parametros: Optional[list[ParametroConfig]] = None,
):
    """
    Decorador para registrar un adaptador Python como fuente de datos
    seleccionable. Uso:

        @registrar_fuente(
            id="gaia_dr3",
            nombre="Gaia DR3 (variables clasificadas)",
            descripcion="Fotometria por epoca via DataLink, TAP+.",
            parametros=[
                ParametroConfig("credenciales_usuario", "Usuario Gaia (opcional)",
                                 tipo="texto", requerido=False),
            ],
        )
        class AdaptadorGaiaDR3(AdaptadorDataset):
            ...

    La clase decorada debe seguir siendo instanciable con los nombres
    de `parametros` como argumentos de __init__ (ademas de los que ya
    tenga AdaptadorDataset, como saltar_invalidas).
    """

    def decorador(clase_adaptador):

        if not issubclass(clase_adaptador, AdaptadorDataset):
            raise TypeError(
                f"{clase_adaptador} debe heredar de AdaptadorDataset "
                f"para poder registrarse como fuente de datos."
            )

        _REGISTRO_PYTHON[id] = FuenteDatos(
            id=id,
            nombre=nombre,
            descripcion=descripcion,
            tipo="python",
            fabrica=clase_adaptador,
            parametros=parametros or [],
            origen=clase_adaptador.__module__,
        )

        return clase_adaptador

    return decorador


def fuentes_python_registradas() -> list[FuenteDatos]:
    return list(_REGISTRO_PYTHON.values())


# ======================================================================
# FUENTES DECLARATIVAS (NIVEL 1, YAML)
# ======================================================================

DIRECTORIO_FUENTES_POR_DEFECTO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "fuentes",
)


def _construir_fuente_desde_yaml(ruta_yaml: str) -> Optional[FuenteDatos]:
    """
    Devuelve None (en vez de lanzar) si el YAML no se puede leer o le
    faltan campos obligatorios -- un manifiesto mal formado de un
    contribuidor no debe tumbar el selector entero para todo el mundo,
    solo dejar de aparecer esa fuente concreta. El motivo se imprime
    como aviso para que sea facil de depurar.
    """

    # import diferido: quien no use fuentes declarativas no necesita
    # pyyaml instalado
    try:
        import yaml
    except ImportError:
        print(
            "AVISO: pyyaml no esta instalado, no se pueden cargar "
            "fuentes declarativas (YAML). Instala con: pip install pyyaml"
        )
        return None

    try:
        with open(ruta_yaml, "r", encoding="utf-8") as f:
            manifiesto = yaml.safe_load(f)
    except Exception as e:
        print(f"AVISO: no se pudo leer {ruta_yaml}: {e}")
        return None

    campos_obligatorios = ["nombre", "mapeo_columnas"]
    faltantes = [c for c in campos_obligatorios if c not in (manifiesto or {})]

    if faltantes:
        print(
            f"AVISO: {ruta_yaml} no es un manifiesto valido, "
            f"faltan campos: {faltantes}. Se omite esta fuente."
        )
        return None

    # import diferido para evitar import circular (tabular_generico
    # tambien importa de este paquete)
    from .tabular_generico import AdaptadorTabularGenerico

    id_fuente = manifiesto.get("id") or Path(ruta_yaml).stem

    parametros = [
        ParametroConfig(
            nombre="ruta_datos",
            etiqueta=f"Fichero de datos ({manifiesto.get('acceso', {}).get('formato', 'csv')})",
            tipo="ruta_archivo",
            requerido=True,
            ayuda=f"Fichero que sigue el mapeo de columnas definido en {Path(ruta_yaml).name}",
        )
    ]

    def fabrica(ruta_datos, saltar_invalidas=True, _ruta_yaml=ruta_yaml):
        return AdaptadorTabularGenerico(
            manifiesto_path=_ruta_yaml,
            ruta_datos=ruta_datos,
            saltar_invalidas=saltar_invalidas,
        )

    return FuenteDatos(
        id=id_fuente,
        nombre=manifiesto.get("nombre", id_fuente),
        descripcion=manifiesto.get("descripcion", ""),
        tipo="declarativa",
        fabrica=fabrica,
        parametros=parametros,
        origen=ruta_yaml,
    )


def fuentes_declarativas(directorio: Optional[str] = None) -> list[FuenteDatos]:

    directorio = directorio or DIRECTORIO_FUENTES_POR_DEFECTO

    if not os.path.isdir(directorio):
        return []

    fuentes = []

    for nombre_archivo in sorted(os.listdir(directorio)):

        if not nombre_archivo.endswith((".yaml", ".yml")):
            continue

        ruta = os.path.join(directorio, nombre_archivo)
        fuente = _construir_fuente_desde_yaml(ruta)

        if fuente is not None:
            fuentes.append(fuente)

    return fuentes


# ======================================================================
# REGISTRO COMBINADO (LO QUE CONSUME LA INTERFAZ)
# ======================================================================

def fuentes_disponibles(directorio_manifiestos: Optional[str] = None) -> list[FuenteDatos]:
    """
    Lista combinada de fuentes declarativas + Python, ordenada por
    nombre. Esto es lo unico que la interfaz PySide6 necesita llamar
    para poblar el selector de "Fuente de datos":

        for f in fuentes_disponibles():
            combo.addItem(f.nombre, f.id)
    """

    todas = fuentes_declarativas(directorio_manifiestos) + fuentes_python_registradas()

    ids_vistos = set()
    sin_duplicados = []
    for f in todas:
        if f.id in ids_vistos:
            print(
                f"AVISO: id de fuente duplicado '{f.id}' "
                f"(origen: {f.origen}). Se ignora la segunda aparicion."
            )
            continue
        ids_vistos.add(f.id)
        sin_duplicados.append(f)

    return sorted(sin_duplicados, key=lambda f: f.nombre)
