# -*- coding: utf-8 -*-

r"""
======================================================================
adapters/gaia.py - ADAPTADOR PARA Gaia DR3 (Nivel 2, respaldado por red)
======================================================================

Primer adaptador "de red" del proyecto (whitepaper.md, seccion 3 y
5.0): a diferencia de StarEmbed (parquet estatico local), este consulta
dos servicios remotos de la ESA/Gaia archive en cada carga:

  1. TAP/ADQL: tabla `gaiadr3.vari_classifier_result` -- que fuentes
     estan clasificadas y con que clase (best_class_name).
  2. DataLink: fotometria por epoca de esas fuentes concretas
     (retrieval_type='EPOCH_PHOTOMETRY'), en lotes.

Formato real de la tabla de fotometria por epoca (CONFIRMADO por
diagnostico_gaia.py contra datos reales el 31/08/2026, no por
suposicion -- ver whitepaper.md seccion 4 para el detalle completo):

    transit_id, g_transit_time, g_transit_mag, g_transit_flux,
    g_transit_flux_error, bp_obs_time, bp_mag, bp_flux,
    bp_flux_error, rp_obs_time, rp_mag, rp_flux, rp_flux_error,
    variability_flag_g_reject, variability_flag_bp_reject,
    variability_flag_rp_reject, rejected_by_photometry, ...

Es un formato "ancho": cada fila (transit_id) trae las tres bandas
como columnas separadas, con un tiempo ligeramente distinto por banda
dentro del mismo transito -- NO hay una columna 'band' que mezcle
filas de distintas bandas (a diferencia de como se diseño
originalmente, antes del diagnostico). No hay columna de error de
magnitud directa, solo flux_error por banda.

Tiempo: en dias (unidad 'd'), consistente con dias desde el 1 de enero
de 2010 T00:00:00 TCB (JD de referencia 2455197.5). Confirmado por
diagnostico: valores ~1709-1871 dias corresponden a mediados de 2014,
cuando Gaia empezo a operar cientificamente -- coherente con la
hipotesis. Se convierte a MJD sumando OFFSET_TIEMPO_GAIA_A_MJD.

Lo que SIGUE sin verificar contra un volumen grande de datos reales
(solo probado con 5 fuentes en el diagnostico): el comportamiento del
adaptador a gran escala (miles de fuentes, paginacion por lotes,
limites de tasa del servicio DataLink), y si `rejected_by_photometry`
es el filtro de calidad correcto o hace falta combinarlo con los flags
por banda de forma distinta. Si algo falla en una carga grande, es
aqui donde mirar primero.
"""

from __future__ import annotations

import re
import time
from typing import Iterator, Optional

from ..schema import LightCurve
from .base import AdaptadorDataset
from .registro import registrar_fuente, ParametroConfig


TABLA_CLASIFICACION = "gaiadr3.vari_classifier_result"

# MJD = tiempo_crudo_dias + OFFSET (dias desde 2010-01-01T00:00:00 TCB,
# JD=2455197.5, hasta la epoca de referencia MJD). Ver docstring arriba.
OFFSET_TIEMPO_GAIA_A_MJD = 55197.0

TAMANIO_LOTE_DATALINK_DEFECTO = 200

PAUSA_ENTRE_LOTES_SEGUNDOS = 1.0

PATRON_CLAVE_DATALINK = re.compile(r"Gaia DR3\s+(\d+)")

# (nombre_banda, columna_tiempo, columna_magnitud, columna_flag_reject_banda)
ESPECIFICACION_BANDAS = [
    ("g", "g_transit_time", "g_transit_mag", "variability_flag_g_reject"),
    ("bp", "bp_obs_time", "bp_mag", "variability_flag_bp_reject"),
    ("rp", "rp_obs_time", "rp_mag", "variability_flag_rp_reject"),
]


@registrar_fuente(
    id="gaia_dr3",
    nombre="Gaia DR3 (variables clasificadas, via TAP + DataLink)",
    descripcion=(
        "Fuente en linea (no fichero local): consulta la tabla de "
        "clasificacion de estrellas variables de Gaia DR3 y descarga "
        "fotometria por epoca (bandas G/BP/RP) por lotes. Verificado a "
        "pequena escala (5 objetos) el 31/08/2026, ver whitepaper.md "
        "seccion 3."
    ),
    parametros=[
        ParametroConfig(
            nombre="max_objetos",
            etiqueta="Numero maximo de objetos a consultar",
            tipo="numero",
            requerido=False,
            valor_por_defecto="500",
            ayuda="Cuidado con dejarlo sin limite -- puede ser muy lento.",
        ),
        ParametroConfig(
            nombre="umbral_confianza_clase",
            etiqueta="Confianza minima de clasificacion (0-1)",
            tipo="numero",
            requerido=False,
            valor_por_defecto="0.5",
        ),
    ],
)
class AdaptadorGaiaDR3(AdaptadorDataset):
    """
    Parametros
    ----------
    max_objetos : cuantas fuentes clasificadas pedir (TOP N en la
        consulta TAP). None = sin limite (¡miles de horas de descarga
        si se deja sin limite en la practica -- usar con cuidado!).
        Por defecto 500, pensado para pruebas razonables.
    umbral_confianza_clase : filtro sobre best_class_score (0-1).
        Por defecto 0.5. Subirlo da fuentes mas fiables pero menos
        volumen.
    clases : lista opcional de nombres de clase (best_class_name) para
        restringir la consulta a clases concretas (p. ej. ['RR', 'CEP']).
        None = todas las clases disponibles.
    tamanio_lote : cuantas fuentes pedir por llamada a DataLink. El
        servicio tiene limites de tamaño de peticion; 200 es
        conservador. Si falla por timeout, bajar este numero.
    """

    nombre = "gaia_dr3"

    def __init__(
        self,
        max_objetos: Optional[int] = 500,
        umbral_confianza_clase: float = 0.5,
        clases: Optional[list[str]] = None,
        tamanio_lote: int = TAMANIO_LOTE_DATALINK_DEFECTO,
        saltar_invalidas: bool = True,
    ):
        super().__init__(saltar_invalidas=saltar_invalidas)
        self.max_objetos = max_objetos
        self.umbral_confianza_clase = umbral_confianza_clase
        self.clases = clases
        self.tamanio_lote = tamanio_lote

        # Instrumentacion para diagnosticar perdidas a escala (un lote
        # que falla por red no lanza CurvaInvalidaError -- nunca llega
        # a _emitir(), asi que no lo cuenta n_descartados. Sin este
        # registro aparte, un fallo de red se confundiria con "menos
        # objetos de los pedidos" sin saber por que.
        self.ids_solicitados: list[int] = []
        self.lotes_fallidos: list[dict] = []

    # ------------------------------------------------------------------

    def _consultar_clasificacion(self):

        from astroquery.gaia import Gaia

        condiciones = [f"best_class_score > {self.umbral_confianza_clase}"]

        if self.clases:
            lista = ", ".join(f"'{c}'" for c in self.clases)
            condiciones.append(f"best_class_name IN ({lista})")

        limite = f"TOP {self.max_objetos}" if self.max_objetos else ""

        query = f"""
        SELECT {limite} source_id, best_class_name, best_class_score
        FROM {TABLA_CLASIFICACION}
        WHERE {" AND ".join(condiciones)}
        """

        print("Consultando clasificacion de Gaia DR3 (TAP)...")

        job = Gaia.launch_job_async(query)
        tabla = job.get_results()

        print(f"  {len(tabla)} fuentes clasificadas encontradas.")

        return tabla

    def _descargar_fotometria_lote(self, ids_lote):

        from astroquery.gaia import Gaia

        try:
            return Gaia.load_data(
                ids=list(ids_lote),
                data_release="Gaia DR3",
                retrieval_type="EPOCH_PHOTOMETRY",
                data_structure="INDIVIDUAL",
                verbose=False,
            )
        except Exception as e:
            print(f"  AVISO: fallo al descargar lote de {len(ids_lote)} fuentes: {e}")
            self.lotes_fallidos.append({
                "ids": list(ids_lote),
                "error": str(e),
            })
            return {}

    def _extraer_source_id_de_clave(self, clave):

        m = PATRON_CLAVE_DATALINK.search(str(clave))
        return int(m.group(1)) if m else None

    def _asegurar_tabla_astropy(self, tabla):
        """
        Gaia.load_data() puede devolver, segun la version de
        astroquery/astropy, astropy.table.Table directamente O
        astropy.io.votable.tree.TableElement (el objeto VOTable
        crudo, sin convertir). El segundo tipo no tiene .colnames ni
        .to_pandas() -- hay que convertirlo primero con .to_table().
        Confirmado necesario con error real: 'TableElement' object
        has no attribute 'colnames'.
        """

        if hasattr(tabla, "colnames"):
            return tabla

        if hasattr(tabla, "to_table"):
            return tabla.to_table()

        raise TypeError(f"Tipo de tabla no reconocido: {type(tabla)}")

    def _parsear_tabla_fuente(self, source_id, tabla, clase):
        """
        Traduce la tabla de fotometria por epoca de UNA fuente (formato
        ancho, tres bandas como columnas separadas) a hasta 3 LightCurve
        (una por banda disponible), filtrando por los flags de calidad.
        """

        try:
            tabla = self._asegurar_tabla_astropy(tabla)
        except Exception as e:
            print(f"  AVISO: no se pudo convertir la tabla de {source_id} a formato tabla: {e}")
            return []

        columnas = tabla.colnames

        try:
            df = tabla.to_pandas()
        except Exception as e:
            print(f"  AVISO: no se pudo convertir a pandas la tabla de {source_id}: {e}")
            return []

        hay_flag_global = "rejected_by_photometry" in df.columns

        curvas = []

        for nombre_banda, col_tiempo, col_mag, col_flag_banda in ESPECIFICACION_BANDAS:

            if col_tiempo not in columnas or col_mag not in columnas:
                continue

            mask_valido = df[col_mag].notna() & df[col_tiempo].notna()

            if col_flag_banda in df.columns:
                mask_valido &= ~df[col_flag_banda].astype(bool)

            if hay_flag_global:
                mask_valido &= ~df["rejected_by_photometry"].astype(bool)

            sub = df[mask_valido]

            if len(sub) == 0:
                continue

            tiempo = sub[col_tiempo].to_numpy(dtype=float) + OFFSET_TIEMPO_GAIA_A_MJD
            magnitud = sub[col_mag].to_numpy(dtype=float)

            curva = LightCurve(
                id_objeto=str(source_id),
                tiempo=tiempo,
                magnitud=magnitud,
                banda=nombre_banda,
                clase_conocida=clase,
                metadata={},
            )
            curvas.append(curva)

        return curvas

    # ------------------------------------------------------------------

    def iter_curvas(self) -> Iterator[LightCurve]:

        tabla_clases = self._consultar_clasificacion()

        if len(tabla_clases) == 0:
            print("AVISO: la consulta de clasificacion no devolvio ninguna fuente.")
            return

        source_ids = [int(sid) for sid in tabla_clases["source_id"]]
        clases_por_id = {
            int(sid): str(clase)
            for sid, clase in zip(tabla_clases["source_id"], tabla_clases["best_class_name"])
        }

        self.ids_solicitados = list(source_ids)

        n = len(source_ids)

        for inicio in range(0, n, self.tamanio_lote):

            lote = source_ids[inicio: inicio + self.tamanio_lote]
            print(f"Descargando fotometria por epoca: {inicio}/{n}...")

            datalink = self._descargar_fotometria_lote(lote)

            for clave, tablas in datalink.items():

                source_id = self._extraer_source_id_de_clave(clave)

                if source_id is None or source_id not in clases_por_id:
                    print(f"  AVISO: no se pudo asociar la clave '{clave}' a un source_id conocido, se omite.")
                    continue

                clase = clases_por_id[source_id]

                for tabla in tablas:
                    for curva in self._parsear_tabla_fuente(source_id, tabla, clase):
                        yield from self._emitir(curva)

            if inicio + self.tamanio_lote < n:
                time.sleep(PAUSA_ENTRE_LOTES_SEGUNDOS)


def cargar_gaia_dr3(
    max_objetos: Optional[int] = 500,
    umbral_confianza_clase: float = 0.5,
    clases: Optional[list[str]] = None,
):
    """Atajo de conveniencia, mismo patron que cargar_train_test_starembed()."""

    adaptador = AdaptadorGaiaDR3(
        max_objetos=max_objetos,
        umbral_confianza_clase=umbral_confianza_clase,
        clases=clases,
    )
    return adaptador.cargar()
