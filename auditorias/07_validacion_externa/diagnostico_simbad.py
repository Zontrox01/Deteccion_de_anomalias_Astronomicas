# -*- coding: utf-8 -*-

"""
======================================================================
diagnostico_simbad.py - PRUEBA DE CONEXION A SIMBAD
======================================================================

Script de diagnóstico para verificar que la conexión a SIMBAD funciona
correctamente y para inspeccionar las columnas que devuelve la consulta.

Útil para depurar problemas en cruzar_candidatos_catalogos.py.
======================================================================
"""

import sys
import time
from pathlib import Path

# ======================================================================
# CONFIGURACION
# ======================================================================

# Ruta del script
RUTA_SCRIPT = Path(__file__).resolve()

# Directorio de resultados dentro de la misma carpeta que el script
CARPETA_SALIDA = RUTA_SCRIPT.parent / "resultados"
CARPETA_SALIDA.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "diagnostico_simbad_"

RUTA_SALIDA = CARPETA_SALIDA / f"{PREFIJO}resultado.txt"

# Coordenadas de ejemplo (CSS_J103704.9-103738)
RA_EJEMPLO = 159.27041666666668
DEC_EJEMPLO = -10.627777777777778
RADIO_BUSQUEDA_ARCSEC = 5.0


def main():

    print("=" * 70)
    print("DIAGNOSTICO DE CONEXION A SIMBAD")
    print("=" * 70)

    try:
        from astropy.coordinates import SkyCoord
        from astropy import units as u
        from astroquery.simbad import Simbad
    except ImportError as e:
        print(f"ERROR: No se puede importar astropy/astroquery: {e}")
        print("Ejecuta: pip install astropy astroquery --break-system-packages")
        return

    print("\nConfigurando SIMBAD...")
    Simbad.reset_votable_fields()
    Simbad.add_votable_fields("otype")
    print("  -> otype añadido a los campos solicitados.")

    coord = SkyCoord(ra=RA_EJEMPLO * u.deg, dec=DEC_EJEMPLO * u.deg)
    print(f"\nCoordenadas de prueba: RA={RA_EJEMPLO:.6f} deg, DEC={DEC_EJEMPLO:.6f} deg")
    print(f"Radio de busqueda: {RADIO_BUSQUEDA_ARCSEC} arcsec")

    print("\nConsultando SIMBAD...")

    try:
        resultado = Simbad.query_region(coord, radius=RADIO_BUSQUEDA_ARCSEC * u.arcsec)
    except Exception as e:
        print(f"ERROR en la consulta: {e}")
        # Guardar el error en el archivo de salida
        with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
            f.write("ERROR EN CONSULTA SIMBAD\n")
            f.write("=" * 70 + "\n")
            f.write(f"Coordenadas: RA={RA_EJEMPLO:.6f}, DEC={DEC_EJEMPLO:.6f}\n")
            f.write(f"Radio: {RADIO_BUSQUEDA_ARCSEC} arcsec\n")
            f.write("\n")
            f.write(f"Error: {e}\n")
        print(f"\nError guardado en: {RUTA_SALIDA}")
        return

    print("\n" + "-" * 70)
    print("RESULTADO DE LA CONSULTA")
    print("-" * 70)

    if resultado is None:
        print("  -> No se encontraron resultados.")
        with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
            f.write("CONSULTA SIMBAD - SIN RESULTADOS\n")
            f.write("=" * 70 + "\n")
            f.write(f"Coordenadas: RA={RA_EJEMPLO:.6f}, DEC={DEC_EJEMPLO:.6f}\n")
            f.write(f"Radio: {RADIO_BUSQUEDA_ARCSEC} arcsec\n")
            f.write("\n")
            f.write("No se encontraron objetos en esta region.\n")
        print(f"\nResultado guardado en: {RUTA_SALIDA}")
        return

    print("\nColumnas disponibles:")
    for col in resultado.colnames:
        print(f"  - {col}")

    print("\n" + "-" * 70)
    print("DATOS (primeras filas)")
    print("-" * 70)

    # Mostrar las primeras 10 filas
    n_filas = min(10, len(resultado))
    print(f"Total de objetos encontrados: {len(resultado)}")
    print(f"Mostrando primeras {n_filas} filas:\n")

    # Crear una representación legible
    lines = []
    lines.append("CONSULTA SIMBAD - RESULTADOS")
    lines.append("=" * 70)
    lines.append(f"Coordenadas: RA={RA_EJEMPLO:.6f}, DEC={DEC_EJEMPLO:.6f}")
    lines.append(f"Radio: {RADIO_BUSQUEDA_ARCSEC} arcsec")
    lines.append(f"Objetos encontrados: {len(resultado)}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("COLUMNAS DISPONIBLES:")
    for col in resultado.colnames:
        lines.append(f"  - {col}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("DATOS (primeras filas):")
    lines.append("")

    # Mostrar los datos en formato tabular
    for i in range(n_filas):
        fila = resultado[i]
        line = f"  [{i+1}] "
        for col in resultado.colnames:
            val = fila[col]
            # Truncar valores largos
            if isinstance(val, str) and len(val) > 40:
                val = val[:37] + "..."
            line += f"{col}={val}  "
        print(line)
        lines.append(line)

    print("\n" + "-" * 70)
    print("DIAGNOSTICO COMPLETADO")
    print("-" * 70)

    # Verificar que la columna 'otype' está presente
    if "otype" in resultado.colnames:
        print("\n  ✓ La columna 'otype' está presente.")
        tipos = set(resultado["otype"])
        print(f"  Tipos encontrados: {sorted(tipos)}")
    else:
        print("\n  ✗ La columna 'otype' NO está presente.")
        print("  Esto puede afectar a cruzar_candidatos_catalogos.py.")

    # Guardar resultados en archivo
    with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"\nResultado completo guardado en: {RUTA_SALIDA}")


if __name__ == "__main__":
    main()