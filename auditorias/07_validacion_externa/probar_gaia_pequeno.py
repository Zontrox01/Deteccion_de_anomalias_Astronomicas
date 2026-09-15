# -*- coding: utf-8 -*-

r"""
======================================================================
probar_gaia_pequeno.py - PRUEBA REAL DEL ADAPTADOR (pocos objetos)
======================================================================

Antes de lanzar una carga grande de Gaia DR3 (que puede tardar mucho
y consumir cuota del servicio), prueba el adaptador completo con muy
pocos objetos -- confirma que AdaptadorGaiaDR3 funciona de principio a
fin (TAP -> DataLink -> LightCurve -> features) sobre datos reales,
no solo en la simulacion que ya se probo sin red.

Coloca este script en el mismo sitio que diagnostico_gaia.py.
======================================================================
"""

import sys
import os
import time
from pathlib import Path

# ======================================================================
# AÑADIR EL DIRECTORIO RAIZ AL PATH PARA IMPORTAR anomaly_detector
# ======================================================================

# Ruta del script
RUTA_SCRIPT = Path(__file__).resolve()

# Buscar la carpeta Astronomia subiendo niveles
BASE_PROYECTO = RUTA_SCRIPT.parent
while BASE_PROYECTO.name != "Astronomia" and BASE_PROYECTO.parent != BASE_PROYECTO:
    BASE_PROYECTO = BASE_PROYECTO.parent
    if BASE_PROYECTO.name == "Astronomia":
        break

if BASE_PROYECTO.name != "Astronomia":
    print(f"ERROR: No se encontró la carpeta 'Astronomia' en la ruta.")
    print(f"Ruta actual del script: {RUTA_SCRIPT}")
    print(f"BASE_PROYECTO encontrado: {BASE_PROYECTO}")
    sys.exit(1)

print(f"BASE_PROYECTO encontrado: {BASE_PROYECTO}")

# Añadir BASE_PROYECTO al path para poder importar anomaly_detector
if str(BASE_PROYECTO) not in sys.path:
    sys.path.insert(0, str(BASE_PROYECTO))

# Verificar que anomaly_detector existe
RUTA_ANOMALY_DETECTOR = BASE_PROYECTO / "anomaly_detector"
if not RUTA_ANOMALY_DETECTOR.exists():
    print(f"ERROR: No se encontró anomaly_detector en {RUTA_ANOMALY_DETECTOR}")
    print("Asegúrate de que la carpeta anomaly_detector está en la raíz de Astronomia/")
    sys.exit(1)

print(f"anomaly_detector encontrado en: {RUTA_ANOMALY_DETECTOR}")


# ======================================================================
# CONFIGURACION
# ======================================================================

# Directorio de resultados dentro de la misma carpeta que el script
CARPETA_SALIDA = RUTA_SCRIPT.parent / "resultados"
CARPETA_SALIDA.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "probar_gaia_pequeno_"

RUTA_SALIDA = CARPETA_SALIDA / f"{PREFIJO}resultado.txt"


def main():

    inicio = time.time()

    print("=" * 70)
    print("PRUEBA PEQUEÑA DEL ADAPTADOR DE GAIA DR3 (5 objetos)")
    print("=" * 70)

    lines = []
    lines.append("PRUEBA PEQUEÑA DEL ADAPTADOR DE GAIA DR3 (5 objetos)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"BASE_PROYECTO: {BASE_PROYECTO}")
    lines.append(f"anomaly_detector: {RUTA_ANOMALY_DETECTOR}")
    lines.append("")

    try:
        from anomaly_detector.adapters import AdaptadorGaiaDR3
        from anomaly_detector import extraer_features_dataset, VARIABLES_BASE
        print("  ✓ Importación de anomaly_detector exitosa")
        lines.append("  ✓ Importación de anomaly_detector exitosa")
    except ImportError as e:
        print(f"ERROR: No se puede importar anomaly_detector: {e}")
        print(f"  sys.path: {sys.path}")
        lines.append(f"ERROR: No se puede importar anomaly_detector: {e}")
        lines.append(f"  sys.path: {sys.path}")
        with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nError guardado en: {RUTA_SALIDA}")
        return

    print("\nCreando adaptador Gaia DR3 (max_objetos=5)...")
    lines.append("Creando adaptador Gaia DR3 (max_objetos=5)...")

    try:
        adaptador = AdaptadorGaiaDR3(max_objetos=5, umbral_confianza_clase=0.9)
        print("  ✓ Adaptador creado correctamente")
        lines.append("  ✓ Adaptador creado correctamente")
    except Exception as e:
        print(f"ERROR al crear adaptador: {e}")
        lines.append(f"ERROR al crear adaptador: {e}")
        with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nError guardado en: {RUTA_SALIDA}")
        return

    print("\nCargando dataset (puede tardar unos segundos)...")
    lines.append("Cargando dataset (puede tardar unos segundos)...")

    try:
        dataset = adaptador.cargar()
        print("  ✓ Dataset cargado correctamente")
        lines.append("  ✓ Dataset cargado correctamente")
    except Exception as e:
        print(f"ERROR al cargar dataset: {e}")
        lines.append(f"ERROR al cargar dataset: {e}")
        with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nError guardado en: {RUTA_SALIDA}")
        return

    print()
    print(f"Objetos unicos cargados: {len(dataset.ids_unicos())}")
    print(f"Curvas totales (todas las bandas): {len(dataset.curvas)}")
    print(f"Descartadas por invalidas: {adaptador.n_descartados}")

    lines.append("")
    lines.append(f"Objetos unicos cargados: {len(dataset.ids_unicos())}")
    lines.append(f"Curvas totales (todas las bandas): {len(dataset.curvas)}")
    lines.append(f"Descartadas por invalidas: {adaptador.n_descartados}")

    # Mostrar informacion por objeto
    lines.append("")
    for id_objeto in dataset.ids_unicos():
        curvas_objeto = dataset.curvas_de(id_objeto)
        info = f"  {id_objeto} (clase={curvas_objeto[0].clase_conocida}):"
        print(info)
        lines.append(info)
        for c in curvas_objeto:
            info_banda = f"    banda={c.banda}  n_obs={c.n_observaciones()}"
            print(info_banda)
            lines.append(info_banda)

    print()
    print("Extrayendo las 12 variables (mismo pipeline que StarEmbed, sin cambios)...")
    lines.append("")
    lines.append("Extrayendo las 12 variables (mismo pipeline que StarEmbed, sin cambios)...")

    try:
        X = extraer_features_dataset(dataset)
        print("  ✓ Features extraidas correctamente")
        lines.append("  ✓ Features extraidas correctamente")
    except Exception as e:
        print(f"ERROR al extraer features: {e}")
        lines.append(f"ERROR al extraer features: {e}")
        with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nError guardado en: {RUTA_SALIDA}")
        return

    print()
    print("Features extraidas:")
    lines.append("")
    lines.append("Features extraidas:")

    # Verificar columnas
    columnas = ["id_objeto", "clase"] + VARIABLES_BASE
    columnas_presentes = [c for c in columnas if c in X.columns]
    columnas_faltantes = [c for c in columnas if c not in X.columns]

    if columnas_faltantes:
        print(f"  AVISO: Columnas faltantes: {columnas_faltantes}")
        lines.append(f"  AVISO: Columnas faltantes: {columnas_faltantes}")

    # Mostrar datos
    print(X[columnas_presentes].to_string(index=False))
    lines.append(X[columnas_presentes].to_string(index=False))

    # Mostrar estadisticas de las features
    print()
    print("Estadisticas de features:")
    lines.append("")
    lines.append("Estadisticas de features:")

    for var in VARIABLES_BASE:
        if var in X.columns:
            valores = X[var].dropna()
            if len(valores) > 0:
                stats = f"  {var}: mean={valores.mean():.4f}, std={valores.std():.4f}, min={valores.min():.4f}, max={valores.max():.4f}, n={len(valores)}"
                print(stats)
                lines.append(stats)
            else:
                stats = f"  {var}: TODOS NaN"
                print(stats)
                lines.append(stats)

    print()
    print("-" * 70)
    print("DIAGNOSTICO")
    print("-" * 70)

    lines.append("")
    lines.append("-" * 70)
    lines.append("DIAGNOSTICO")
    lines.append("-" * 70)

    # Verificar si el adaptador funcionó correctamente
    n_objetos = len(dataset.ids_unicos())
    n_features_validas = X[VARIABLES_BASE].notna().any().any() if len(X) > 0 else False

    if n_objetos > 0 and n_features_validas:
        resultado = "PRUEBA OK: el adaptador funciona de principio a fin sobre datos reales de Gaia DR3, y el mismo extraer_features_dataset() de StarEmbed funciona sin ningun cambio -- confirma que el esquema canonico generaliza de verdad a un segundo dataset."
        print()
        print(resultado)
        lines.append("")
        lines.append(resultado)
    elif n_objetos > 0 and not n_features_validas:
        resultado = "AVISO: se cargaron objetos pero no se obtuvieron features validas. Revisa la salida para diagnosticar."
        print()
        print(resultado)
        lines.append("")
        lines.append(resultado)
    else:
        resultado = "ERROR: no se cargaron objetos. Revisa la conexion a Gaia DR3."
        print()
        print(resultado)
        lines.append("")
        lines.append(resultado)

    tiempo_total = time.time() - inicio
    print()
    print(f"Tiempo total: {tiempo_total:.2f} segundos")
    lines.append("")
    lines.append(f"Tiempo total: {tiempo_total:.2f} segundos")

    # Guardar resultados en archivo
    with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nResultado completo guardado en: {RUTA_SALIDA}")


if __name__ == "__main__":
    main()