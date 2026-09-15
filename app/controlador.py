# -*- coding: utf-8 -*-

r"""
======================================================================
controlador.py - LOGICA DE NEGOCIO DE LA APP (SIN Qt)
======================================================================

Deliberadamente sin ningun import de PySide6. Esta separacion no es
un capricho de estilo: es la unica forma de poder probar la logica
real de la app en un entorno sin pantalla (como el sandbox donde se
escribio este codigo) -- todo lo que hay aqui se ha ejecutado de
verdad contra el paquete anomaly_detector antes de entregarse. Los
ficheros que si usan Qt (ventana_principal.py, widgets/) solo hacen
de "pegamento" visual sobre estas funciones, y no se han podido
probar en este entorno por falta de PySide6/pantalla -- ver
whitepaper.md seccion 5 para el detalle de que queda pendiente de
verificar en tu maquina.

Alcance de esta primera version (MVP), deliberadamente simplificado:
  - Deteccion GLOBAL (Isolation Forest + LOF sobre todo el conjunto
    cargado, no por clase) -- no hay todavia un concepto de "TRAIN de
    referencia" separado en la interfaz, asi que se usa el propio
    conjunto cargado como referencia y evaluacion a la vez
    (autodeteccion). `detectar_por_clase()`/`detectar_ood_multiclase()`
    (mas potentes, ya construidos y verificados en anomaly_detector)
    quedan para una siguiente iteracion, cuando la interfaz permita
    elegir un conjunto de referencia separado.
  - Una unica tabla ordenable con un score de anomalia y una columna
    booleana `candidato_fuerte`, no las dos secciones "seguros"/
    "ambiguos" del diseño completo (whitepaper.md seccion 5.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from anomaly_detector import extraer_features_dataset, VARIABLES_BASE
from anomaly_detector.adapters import fuentes_disponibles
from anomaly_detector.adapters.registro import FuenteDatos
from anomaly_detector.deteccion import detectar_global
from anomaly_detector.schema import Dataset


# ======================================================================
# RESULTADO DE UN ANALISIS (lo que la ventana principal muestra)
# ======================================================================

@dataclass
class ResultadoAnalisis:
    """
    Todo lo que la interfaz necesita tras cargar y analizar una
    fuente: la tabla de resultados (para la vista de tabla) y el
    Dataset original (para poder dibujar la curva de luz de
    cualquier objeto cuando el usuario lo seleccione).
    """

    tabla: pd.DataFrame
    dataset: Dataset
    variables_usadas: list[str]
    fuente_id: str
    fuente_nombre: str


# ======================================================================
# LISTADO DE FUENTES (alimenta el selector)
# ======================================================================

def listar_fuentes() -> list[FuenteDatos]:
    """Wrapper directo -- existe para que la ventana principal no
    tenga que importar anomaly_detector.adapters.registro ella misma,
    manteniendo toda la superficie de import de Qt reducida a este
    modulo y a los widgets."""
    return fuentes_disponibles()


# ======================================================================
# CARGA + ANALISIS (lo que dispara el boton "Cargar y analizar")
# ======================================================================

def _convertir_parametros(fuente: FuenteDatos, valores_texto: dict) -> dict:
    """
    Los widgets de la interfaz siempre devuelven texto (QLineEdit).
    Aqui se convierte cada valor al tipo que espera la fabrica del
    adaptador, segun el tipo declarado en cada ParametroConfig.
    """

    convertidos = {}

    for parametro in fuente.parametros:

        valor_texto = valores_texto.get(parametro.nombre, "")

        if valor_texto == "" or valor_texto is None:
            if parametro.requerido:
                raise ValueError(
                    f"Falta el parametro obligatorio '{parametro.etiqueta}'."
                )
            continue

        if parametro.tipo == "numero":
            try:
                # int si es un numero entero limpio, float si no
                convertidos[parametro.nombre] = (
                    int(valor_texto) if str(valor_texto).strip().lstrip("-").isdigit()
                    else float(valor_texto)
                )
            except ValueError:
                raise ValueError(
                    f"'{parametro.etiqueta}' debe ser un numero, se recibio: {valor_texto!r}"
                )
        elif parametro.tipo == "ruta_archivo":
            # el widget de ruta puede devolver varias rutas separadas
            # por ';' si el dialogo permite seleccion multiple (ver
            # widgets/formulario_fuente.py) -- StarEmbed necesita una
            # lista, la mayoria de fuentes solo un fichero.
            rutas = [r.strip() for r in str(valor_texto).split(";") if r.strip()]
            convertidos[parametro.nombre] = rutas if len(rutas) > 1 else rutas[0]
        else:
            convertidos[parametro.nombre] = valor_texto

    return convertidos


def cargar_y_analizar(
    fuente: FuenteDatos,
    valores_parametros: dict,
    top_percentil: float = 0.02,
) -> ResultadoAnalisis:
    """
    Punto de entrada principal: construye el adaptador, carga el
    Dataset, extrae features, ejecuta deteccion global de anomalias, y
    devuelve un ResultadoAnalisis listo para mostrar.

    Pensado para ejecutarse en un hilo aparte (ver
    widgets o ventana_principal.HiloAnalisis) -- puede tardar desde
    segundos (StarEmbed, fichero local) hasta minutos (Gaia DR3, ver
    whitepaper.md seccion 3, rendimiento medido ~0.90 s/objeto).
    """

    parametros = _convertir_parametros(fuente, valores_parametros)

    adaptador = fuente.fabrica(**parametros)
    dataset = adaptador.cargar()

    if len(dataset) == 0:
        raise ValueError(
            "La fuente no devolvio ningun objeto valido. Revisa los "
            "parametros (ruta del fichero, filtros de clase/confianza)."
        )

    X = extraer_features_dataset(dataset)

    if len(X) == 0:
        raise ValueError(
            "Se cargaron objetos pero no se pudo extraer features de "
            "ninguno (curvas demasiado cortas o sin datos validos)."
        )

    variables = list(VARIABLES_BASE)
    if "period" in X.columns and X["period"].notna().any():
        variables = variables + ["period"]

    # Limpieza identica al resto del pipeline (mediana de la propia
    # muestra, ya que aqui no hay un TRAIN separado)
    X_limpio = X.copy()
    X_limpio[variables] = X_limpio[variables].replace([np.inf, -np.inf], np.nan)
    medianas = X_limpio[variables].median()
    X_limpio[variables] = X_limpio[variables].fillna(medianas).fillna(0.0)

    resultado_deteccion = detectar_global(
        X_limpio, X_limpio, variables, top_percentil=top_percentil
    )

    tabla = pd.DataFrame({
        "id_objeto": X["id_objeto"].values,
        "clase": X["clase"].values if "clase" in X.columns else None,
    })
    tabla["score_anomalia"] = resultado_deteccion["score_if"].values
    tabla["score_lof"] = resultado_deteccion["score_lof"].values
    tabla["candidato_fuerte"] = resultado_deteccion["candidato_fuerte_global"].values

    tabla = tabla.sort_values("score_anomalia", ascending=False).reset_index(drop=True)

    return ResultadoAnalisis(
        tabla=tabla,
        dataset=dataset,
        variables_usadas=variables,
        fuente_id=fuente.id,
        fuente_nombre=fuente.nombre,
    )


# ======================================================================
# CURVA DE LUZ DE UN OBJETO CONCRETO (para el grafico al seleccionar fila)
# ======================================================================

def obtener_curva_para_grafico(
    resultado: ResultadoAnalisis, id_objeto: str
) -> tuple[np.ndarray, np.ndarray, Optional[str]]:
    """
    Devuelve (tiempo, magnitud, banda) de la banda principal de un
    objeto, ya limpia y ordenada, lista para dibujar.
    """

    curva = resultado.dataset.banda_principal(id_objeto)

    if curva is None:
        raise ValueError(f"No se encontro ninguna curva para '{id_objeto}'.")

    curva_limpia = curva.limpia()

    return curva_limpia.tiempo, curva_limpia.magnitud, curva_limpia.banda


# ======================================================================
# EXPORTACION
# ======================================================================

def exportar_csv(resultado: ResultadoAnalisis, ruta: str) -> None:
    resultado.tabla.to_csv(ruta, index=False)
