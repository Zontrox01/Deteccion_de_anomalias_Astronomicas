# -*- coding: utf-8 -*-

"""
======================================================================
diagnostico_gaia_dr3.py - DIAGNOSTICO DE CONEXION A GAIA DR3
======================================================================

Script de diagnóstico para verificar que la conexión a Gaia DR3
funciona correctamente y para inspeccionar las columnas que devuelve
la consulta de fotometría por época.

Útil para preparar la validación con Gaia DR3.
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
PREFIJO = "diagnostico_gaia_dr3_"

RUTA_SALIDA = CARPETA_SALIDA / f"{PREFIJO}resultado.txt"


def main():

    print("=" * 70)
    print("DIAGNOSTICO DE CONEXION A GAIA DR3")
    print("=" * 70)

    try:
        from astroquery.gaia import Gaia
        from astropy.io.votable import parse_single_table
    except ImportError as e:
        print(f"ERROR: No se puede importar astroquery/astropy: {e}")
        print("Ejecuta: pip install astropy astroquery --break-system-packages")
        with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
            f.write("ERROR DE IMPORTACION\n")
            f.write("=" * 70 + "\n")
            f.write(f"Error: {e}\n")
            f.write("\nEjecuta: pip install astropy astroquery --break-system-packages\n")
        print(f"\nError guardado en: {RUTA_SALIDA}")
        return

    # Configurar timeout
    Gaia.TIMEOUT = 300

    lines = []
    lines.append("DIAGNOSTICO DE CONEXION A GAIA DR3")
    lines.append("=" * 70)
    lines.append("")

    print("\nConsultando 5 fuentes variables de Gaia...")
    lines.append("Consultando 5 fuentes variables de Gaia...")

    query = """
    SELECT TOP 5 source_id, best_class_name, best_class_score
    FROM gaiadr3.vari_classifier_result
    WHERE best_class_score > 0.9
    ORDER BY source_id
    """

    try:
        job = Gaia.launch_job_async(query)
        tabla = job.get_results()
    except Exception as e:
        print(f"ERROR en la consulta: {e}")
        lines.append(f"ERROR en la consulta: {e}")
        with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nError guardado en: {RUTA_SALIDA}")
        return

    if len(tabla) == 0:
        print("  -> No se encontraron fuentes variables.")
        lines.append("  -> No se encontraron fuentes variables.")
        with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nResultado guardado en: {RUTA_SALIDA}")
        return

    source_ids = list(tabla["source_id"])
    print(f"\nSource IDs obtenidos: {source_ids}")
    lines.append(f"\nSource IDs obtenidos: {source_ids}")

    print("\nDescargando fotometría por época...")
    lines.append("\nDescargando fotometría por época...")

    try:
        datalink = Gaia.load_data(
            ids=source_ids,
            data_release="Gaia DR3",
            retrieval_type="EPOCH_PHOTOMETRY",
            data_structure="INDIVIDUAL",
            verbose=False,
        )
    except Exception as e:
        print(f"ERROR al descargar fotometría: {e}")
        lines.append(f"ERROR al descargar fotometría: {e}")
        with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nError guardado en: {RUTA_SALIDA}")
        return

    print(f"\nClaves disponibles en datalink: {list(datalink.keys())}")
    lines.append(f"\nClaves disponibles en datalink: {list(datalink.keys())}")

    # Tomar la primera fuente
    primera_clave = list(datalink.keys())[0]
    primer_valor = datalink[primera_clave]

    print(f"\nProcesando clave: {primera_clave}")
    lines.append(f"\nProcesando clave: {primera_clave}")

    # Convertir VOTable a Table de Astropy
    try:
        votable = primer_valor[0]
        try:
            tabla_astropy = votable.to_table()
        except Exception:
            # Fallback: usar parse_single_table
            tabla_astropy = parse_single_table(votable).to_table()
    except Exception as e:
        print(f"ERROR al procesar VOTable: {e}")
        lines.append(f"ERROR al procesar VOTable: {e}")
        with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nError guardado en: {RUTA_SALIDA}")
        return

    print(f"\nColumnas disponibles en los datos de fotometría por época:")
    lines.append(f"\nColumnas disponibles en los datos de fotometría por época:")

    for col in tabla_astropy.colnames:
        print(f"  - {col} ({tabla_astropy[col].dtype})")
        lines.append(f"  - {col} ({tabla_astropy[col].dtype})")

    print(f"\nNúmero de mediciones: {len(tabla_astropy)}")
    lines.append(f"\nNúmero de mediciones: {len(tabla_astropy)}")

    # Mostrar muestra de datos
    print("\nMuestra de los datos (primeras 5 filas):")
    lines.append("\nMuestra de los datos (primeras 5 filas):")
    lines.append("")

    # Convertir a string legible
    for i in range(min(5, len(tabla_astropy))):
        fila = tabla_astropy[i]
        line = f"  [{i+1}] "
        for col in tabla_astropy.colnames:
            val = fila[col]
            if isinstance(val, (bytes, bytearray)):
                try:
                    val = str(val, 'utf-8')
                except Exception:
                    val = str(val)
            elif hasattr(val, 'value'):
                val = val.value
            # Truncar valores largos
            if isinstance(val, str) and len(str(val)) > 40:
                val = str(val)[:37] + "..."
            elif isinstance(val, float):
                val = f"{val:.6f}"
            line += f"{col}={val}  "
        print(line)
        lines.append(line)

    # Verificar columnas importantes
    print("\n" + "-" * 70)
    lines.append("\n" + "-" * 70)

    columnas_esperadas = ["mag", "time", "band", "flux", "flux_error"]
    columnas_presentes = [col for col in columnas_esperadas if col in tabla_astropy.colnames]
    columnas_faltantes = [col for col in columnas_esperadas if col not in tabla_astropy.colnames]

    print("VERIFICACION DE COLUMNAS IMPORTANTES:")
    lines.append("VERIFICACION DE COLUMNAS IMPORTANTES:")

    if columnas_presentes:
        print(f"  ✓ Columnas encontradas: {columnas_presentes}")
        lines.append(f"  ✓ Columnas encontradas: {columnas_presentes}")

    if columnas_faltantes:
        print(f"  ✗ Columnas faltantes: {columnas_faltantes}")
        lines.append(f"  ✗ Columnas faltantes: {columnas_faltantes}")

    print("\n" + "-" * 70)
    print("DIAGNOSTICO COMPLETADO")
    print("-" * 70)

    lines.append("\n" + "-" * 70)
    lines.append("DIAGNOSTICO COMPLETADO")
    lines.append("-" * 70)

    # Guardar resultados en archivo
    with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nResultado completo guardado en: {RUTA_SALIDA}")


if __name__ == "__main__":
    main()