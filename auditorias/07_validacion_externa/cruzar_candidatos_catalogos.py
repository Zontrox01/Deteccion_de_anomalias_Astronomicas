# -*- coding: utf-8 -*-

r"""
======================================================================
cruzar_candidatos_catalogos.py - VALIDACION EXTERNA (SIMBAD + VSX)
======================================================================

Cierra la ultima pieza pendiente de la fase de deteccion de anomalias
sobre StarEmbed (whitepaper.md, seccion 4.5 y "proximos pasos" §7,
punto 2): cruzar los candidatos ya identificados contra catalogos
externos reales, para saber si son objetos ya conocidos (con una
clasificacion distinta a las 7 de StarEmbed -- validacion positiva
del metodo) o genuinamente no catalogados (candidatos a objeto nuevo,
mucho mas interesantes, pero tambien mas dudosos sin mas contexto).

Metodo
------
1. Extrae RA/Dec de cada id_objeto usando la convencion de nombres de
   Catalina Surveys (CSS_Jhhmmss.s+ddmmss) -- no hace falta tocar el
   pipeline ni el parquet, las coordenadas ya estan en el nombre.
2. Busqueda por cono (5 arcsec de radio) en:
   - SIMBAD (identificacion general + tipo de objeto)
   - VSX/AAVSO via VizieR (catalogo B/vsx/vsx -- clasificacion
     especifica de estrellas variables, mas relevante aqui que SIMBAD)
3. Guarda un CSV con, para cada candidato: si se encontro
   contrapartida en cada catalogo, su identificador, y su
   clasificacion/tipo si esta disponible.

IMPORTANTE -- este script depende de servicios externos (SIMBAD y
VizieR) y NO se ha podido ejecutar ni verificar en el entorno donde se
escribio (sin acceso a red). A diferencia de todos los scripts
anteriores de esta sesion, este no viene con la garantia de "probado
antes de entregar". Si da algun error de conexion o de la API de
astroquery, pega el traceback completo y se corrige con el mismo
criterio que los scripts anteriores -- por diagnostico real, no por
suposicion.

Requiere: pip install astroquery --break-system-packages (o sin ese
flag si no da conflicto en tu entorno)
======================================================================
"""

import os
import re
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

try:
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from astroquery.simbad import Simbad
    from astroquery.vizier import Vizier
except ImportError as e:
    print(
        "Falta instalar astropy/astroquery. Ejecuta:\n"
        "  pip install astropy astroquery --break-system-packages\n"
        f"Error original: {e}"
    )
    sys.exit(1)


# ======================================================================
# CONFIGURACION
# ======================================================================

# Ruta del script
RUTA_SCRIPT = Path(__file__).resolve()

# Ruta raiz del proyecto (Astronomia)
BASE_PROYECTO = RUTA_SCRIPT.parents[2]  # Sube 2 niveles hasta Astronomia/

# Directorio de resultados dentro de la misma carpeta que el script
CARPETA_SALIDA = RUTA_SCRIPT.parent / "resultados"
CARPETA_SALIDA.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "cruce_catalogos_"

# Ruta al CSV con los candidatos finales (en 04_deteccion_anomalias/resultados)
RUTA_CANDIDATOS = BASE_PROYECTO / "auditorias" / "04_deteccion_anomalias" / "resultados" / "24c_candidatos_interseccion.csv"

COLUMNA_ID = "id_objeto"

# Si el CSV tiene una columna que marca los candidatos "interesantes"
# (no ambiguos), se puede filtrar solo a esos. Pon None para cruzar
# TODOS los candidatos del CSV sin filtrar.
COLUMNA_FILTRO = "candidato_robusto_interesante"  # o None

RADIO_BUSQUEDA_ARCSEC = 5.0

PAUSA_ENTRE_CONSULTAS_SEGUNDOS = 1.0  # cortesia con los servicios externos

RUTA_SALIDA = CARPETA_SALIDA / f"{PREFIJO}resultado.csv"


# ======================================================================
# PARSEO DE COORDENADAS DESDE EL ID CSS_J...
# ======================================================================

PATRON_CSS = re.compile(
    r"CSS_J(\d{2})(\d{2})(\d{2}\.?\d*)([+-])(\d{2})(\d{2})(\d{2}\.?\d*)"
)


def parsear_coordenadas_css(id_objeto: str):
    """
    Devuelve (ra_deg, dec_deg) o None si el id no sigue la convencion
    CSS_Jhhmmss.s+ddmmss de Catalina Surveys.
    """

    m = PATRON_CSS.match(id_objeto)
    if not m:
        return None

    h, mi, s, signo, d, dm, ds = m.groups()

    ra_deg = (int(h) + int(mi) / 60 + float(s) / 3600) * 15.0
    dec_deg = int(d) + int(dm) / 60 + float(ds) / 3600

    if signo == "-":
        dec_deg = -dec_deg

    return ra_deg, dec_deg


# ======================================================================
# CONSULTAS A CATALOGOS
# ======================================================================

def consultar_simbad(coord: "SkyCoord"):
    """
    Devuelve (encontrado, id_simbad, tipo_objeto) o (False, None, None).
    """

    try:
        Simbad.reset_votable_fields()
        Simbad.add_votable_fields("otype")
        resultado = Simbad.query_region(coord, radius=RADIO_BUSQUEDA_ARCSEC * u.arcsec)
    except Exception as e:
        print(f"    [SIMBAD] error de consulta: {e}")
        return False, None, None

    if resultado is None or len(resultado) == 0:
        return False, None, None

    fila = resultado[0]
    main_id = str(fila["main_id"]) if "main_id" in resultado.colnames else None
    tipo = str(fila["otype"]) if "otype" in resultado.colnames else None

    return True, main_id, tipo


def consultar_vsx(coord: "SkyCoord"):
    """
    Devuelve (encontrado, nombre_vsx, tipo_variable) o (False, None, None).
    Catalogo VSX/AAVSO via VizieR: B/vsx/vsx.
    """

    try:
        vizier = Vizier(columns=["Name", "Type", "Period"])
        resultado = vizier.query_region(
            coord, radius=RADIO_BUSQUEDA_ARCSEC * u.arcsec, catalog="B/vsx/vsx"
        )
    except Exception as e:
        print(f"    [VSX/VizieR] error de consulta: {e}")
        return False, None, None

    if not resultado or len(resultado) == 0:
        return False, None, None

    tabla = resultado[0]
    if len(tabla) == 0:
        return False, None, None

    fila = tabla[0]
    nombre = str(fila["Name"]) if "Name" in tabla.colnames else None
    tipo = str(fila["Type"]) if "Type" in tabla.colnames else None

    return True, nombre, tipo


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print("CRUCE DE CANDIDATOS CON CATALOGOS EXTERNOS (SIMBAD + VSX)")
    print("=" * 70)

    if not RUTA_CANDIDATOS.exists():
        print(f"No se encuentra {RUTA_CANDIDATOS}. Ajusta RUTA_CANDIDATOS arriba.")
        print(f"Ruta esperada: {RUTA_CANDIDATOS}")
        return

    df = pd.read_csv(RUTA_CANDIDATOS)

    if COLUMNA_ID not in df.columns:
        print(f"El CSV no tiene una columna '{COLUMNA_ID}'. Columnas disponibles: {list(df.columns)}")
        return

    if COLUMNA_FILTRO and COLUMNA_FILTRO in df.columns:
        antes = len(df)
        df = df[df[COLUMNA_FILTRO] == True].copy()
        print(f"Filtrado por '{COLUMNA_FILTRO}': {len(df)} de {antes} candidatos.")

    print(f"\nCandidatos a cruzar: {len(df)}\n")

    filas_resultado = []

    for i, id_objeto in enumerate(df[COLUMNA_ID]):

        print(f"[{i+1}/{len(df)}] {id_objeto}")

        coords = parsear_coordenadas_css(id_objeto)

        if coords is None:
            print("    no sigue la convencion CSS_J..., se omite")
            filas_resultado.append({
                "id_objeto": id_objeto, "ra_deg": None, "dec_deg": None,
                "simbad_encontrado": None, "simbad_id": None, "simbad_tipo": None,
                "vsx_encontrado": None, "vsx_nombre": None, "vsx_tipo": None,
            })
            continue

        ra_deg, dec_deg = coords
        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)

        simbad_ok, simbad_id, simbad_tipo = consultar_simbad(coord)
        print(f"    SIMBAD: {'encontrado -> ' + str(simbad_id) + ' (' + str(simbad_tipo) + ')' if simbad_ok else 'sin contrapartida'}")

        time.sleep(PAUSA_ENTRE_CONSULTAS_SEGUNDOS)

        vsx_ok, vsx_nombre, vsx_tipo = consultar_vsx(coord)
        print(f"    VSX   : {'encontrado -> ' + str(vsx_nombre) + ' (' + str(vsx_tipo) + ')' if vsx_ok else 'sin contrapartida'}")

        time.sleep(PAUSA_ENTRE_CONSULTAS_SEGUNDOS)

        filas_resultado.append({
            "id_objeto": id_objeto,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "simbad_encontrado": simbad_ok,
            "simbad_id": simbad_id,
            "simbad_tipo": simbad_tipo,
            "vsx_encontrado": vsx_ok,
            "vsx_nombre": vsx_nombre,
            "vsx_tipo": vsx_tipo,
        })

    resultado = pd.DataFrame(filas_resultado)
    resultado.to_csv(RUTA_SALIDA, index=False)

    print()
    print("=" * 70)
    print("RESUMEN")
    print("=" * 70)

    n_total = len(resultado)
    n_sin_coords = resultado["ra_deg"].isna().sum()
    n_con_vsx = (resultado["vsx_encontrado"] == True).sum()
    n_con_simbad = (resultado["simbad_encontrado"] == True).sum()
    n_sin_ninguno = (
        (resultado["vsx_encontrado"] == False)
        & (resultado["simbad_encontrado"] == False)
    ).sum()

    print(f"Total candidatos             : {n_total}")
    print(f"Sin coordenadas parseables   : {n_sin_coords}")
    print(f"Con contrapartida en VSX     : {n_con_vsx}")
    print(f"Con contrapartida en SIMBAD  : {n_con_simbad}")
    print(f"SIN contrapartida en ninguno : {n_sin_ninguno}  <- los mas interesantes")

    if n_sin_ninguno > 0:
        print()
        print("Candidatos sin contrapartida en ningun catalogo:")
        sin_match = resultado[
            (resultado["vsx_encontrado"] == False)
            & (resultado["simbad_encontrado"] == False)
        ]
        print(sin_match[["id_objeto", "ra_deg", "dec_deg"]].to_string(index=False))

    print()
    print("Guardado:", RUTA_SALIDA)


if __name__ == "__main__":
    main()