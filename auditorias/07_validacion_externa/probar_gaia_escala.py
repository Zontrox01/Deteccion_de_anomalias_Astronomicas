# -*- coding: utf-8 -*-

r"""
======================================================================
probar_gaia_escala.py - PRUEBA A MAYOR ESCALA (paginacion, fallos de red)
======================================================================

probar_gaia_pequeno.py ya confirmo que el adaptador funciona de
principio a fin con 5 objetos. Este script sube la escala (por
defecto 300 objetos, mas del tamaño de un lote de DataLink -- ver
TAMANIO_LOTE_DATALINK_DEFECTO=200 en gaia.py) para confirmar lo que
una prueba de 5 objetos no puede probar:

  1. Que la paginacion por lotes funciona correctamente (mas de una
     llamada a Gaia.load_data()).
  2. Cuantos objetos, si alguno, se pierden por fallos de red en un
     lote concreto (usando la instrumentacion nueva de gaia.py:
     ids_solicitados y lotes_fallidos).
  3. Cuanto tarda en la practica cargar un volumen mayor -- para
     saber si es viable pedir mas en el futuro, o si hace falta
     paralelizar/optimizar antes de construir la interfaz encima.
  4. Si hay variedad real de clases en el volumen pedido (a 5 objetos
     solo salieron LPV y ECL; con mas volumen deberia verse mas
     diversidad de las clases de Gaia).

No hace falta que termine sin ningun fallo para ser util -- si algun
lote falla, este script lo va a decir con precision (cuantos objetos,
que error), que es exactamente la informacion que hace falta para
decidir si el adaptador esta listo para la interfaz o necesita
reintentos automaticos antes.
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
    sys.exit(1)

# Añadir BASE_PROYECTO al path para poder importar anomaly_detector
if str(BASE_PROYECTO) not in sys.path:
    sys.path.insert(0, str(BASE_PROYECTO))

# ======================================================================
# CONFIGURACION
# ======================================================================

MAX_OBJETOS = 300
UMBRAL_CONFIANZA = 0.9

# Directorio de resultados dentro de la misma carpeta que el script
CARPETA_SALIDA = RUTA_SCRIPT.parent / "resultados"
CARPETA_SALIDA.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "probar_gaia_escala_"

RUTA_SALIDA = CARPETA_SALIDA / f"{PREFIJO}resultado.txt"


def main():

    inicio = time.time()

    print("=" * 70)
    print(f"PRUEBA A MAYOR ESCALA DEL ADAPTADOR DE GAIA DR3 ({MAX_OBJETOS} objetos)")
    print("=" * 70)

    lines = []
    lines.append("=" * 70)
    lines.append(f"PRUEBA A MAYOR ESCALA DEL ADAPTADOR DE GAIA DR3 ({MAX_OBJETOS} objetos)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"BASE_PROYECTO: {BASE_PROYECTO}")
    lines.append("")

    try:
        from anomaly_detector.adapters import AdaptadorGaiaDR3
        from anomaly_detector import extraer_features_dataset, VARIABLES_BASE
        print("  ✓ Importación de anomaly_detector exitosa")
        lines.append("  ✓ Importación de anomaly_detector exitosa")
    except ImportError as e:
        print(f"ERROR: No se puede importar anomaly_detector: {e}")
        lines.append(f"ERROR: No se puede importar anomaly_detector: {e}")
        with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nError guardado en: {RUTA_SALIDA}")
        return

    print("\nCreando adaptador Gaia DR3...")
    lines.append("Creando adaptador Gaia DR3...")

    try:
        adaptador = AdaptadorGaiaDR3(
            max_objetos=MAX_OBJETOS,
            umbral_confianza_clase=UMBRAL_CONFIANZA,
        )
        print("  ✓ Adaptador creado correctamente")
        lines.append("  ✓ Adaptador creado correctamente")
    except Exception as e:
        print(f"ERROR al crear adaptador: {e}")
        lines.append(f"ERROR al crear adaptador: {e}")
        with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nError guardado en: {RUTA_SALIDA}")
        return

    print("\nCargando dataset (puede tardar varios minutos)...")
    lines.append("Cargando dataset (puede tardar varios minutos)...")

    try:
        dataset = adaptador.cargar()
        duracion = time.time() - inicio
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
    print("=" * 70)
    print("RESULTADO DE LA CARGA")
    print("=" * 70)

    lines.append("")
    lines.append("=" * 70)
    lines.append("RESULTADO DE LA CARGA")
    lines.append("=" * 70)

    n_solicitados = len(adaptador.ids_solicitados)
    n_cargados = len(dataset.ids_unicos())
    n_curvas = len(dataset.curvas)
    n_descartados_validacion = adaptador.n_descartados

    print(f"Objetos solicitados (clasificados en Gaia)  : {n_solicitados}")
    print(f"Objetos con al menos una curva valida        : {n_cargados}")
    print(f"Curvas totales (todas las bandas)            : {n_curvas}")
    print(f"Descartados por LightCurve.validar()         : {n_descartados_validacion}")
    print(f"Tiempo total                                  : {duracion:.1f}s "
          f"({duracion/max(n_solicitados,1):.2f}s por objeto solicitado)")

    lines.append(f"Objetos solicitados (clasificados en Gaia)  : {n_solicitados}")
    lines.append(f"Objetos con al menos una curva valida        : {n_cargados}")
    lines.append(f"Curvas totales (todas las bandas)            : {n_curvas}")
    lines.append(f"Descartados por LightCurve.validar()         : {n_descartados_validacion}")
    lines.append(f"Tiempo total                                  : {duracion:.1f}s "
                 f"({duracion/max(n_solicitados,1):.2f}s por objeto solicitado)")

    print()
    if adaptador.lotes_fallidos:
        n_ids_en_lotes_fallidos = sum(len(l["ids"]) for l in adaptador.lotes_fallidos)
        print(
            f"AVISO: {len(adaptador.lotes_fallidos)} lote(s) fallaron por red, "
            f"afectando a {n_ids_en_lotes_fallidos} objetos solicitados:"
        )
        lines.append("")
        lines.append(
            f"AVISO: {len(adaptador.lotes_fallidos)} lote(s) fallaron por red, "
            f"afectando a {n_ids_en_lotes_fallidos} objetos solicitados:"
        )
        for i, lote in enumerate(adaptador.lotes_fallidos):
            print(f"  Lote {i+1}: {len(lote['ids'])} ids, error: {lote['error']}")
            lines.append(f"  Lote {i+1}: {len(lote['ids'])} ids, error: {lote['error']}")
    else:
        print("Ningun lote fallo por red -- paginacion completa sin incidentes.")
        lines.append("Ningun lote fallo por red -- paginacion completa sin incidentes.")

    # Objetos solicitados que ni siquiera aparecen en el dataset final,
    # y que TAMPOCO estan explicados por un lote fallido conocido --
    # estos son la perdida "silenciosa" real que hay que investigar.
    ids_cargados = set(dataset.ids_unicos())
    ids_en_lotes_fallidos = set()
    for lote in adaptador.lotes_fallidos:
        ids_en_lotes_fallidos.update(str(i) for i in lote["ids"])

    ids_solicitados_str = set(str(i) for i in adaptador.ids_solicitados)
    ids_perdidos_sin_explicar = ids_solicitados_str - ids_cargados - ids_en_lotes_fallidos

    print()
    print(f"Objetos solicitados pero SIN curva final y SIN lote fallido conocido: "
          f"{len(ids_perdidos_sin_explicar)}")
    lines.append("")
    lines.append(
        f"Objetos solicitados pero SIN curva final y SIN lote fallido conocido: "
        f"{len(ids_perdidos_sin_explicar)}"
    )
    if ids_perdidos_sin_explicar and len(ids_perdidos_sin_explicar) <= 20:
        print("  IDs:", sorted(ids_perdidos_sin_explicar))
        lines.append(f"  IDs: {sorted(ids_perdidos_sin_explicar)}")
    elif ids_perdidos_sin_explicar:
        print(f"  (demasiados para listar, {len(ids_perdidos_sin_explicar)} en total)")
        lines.append(f"  (demasiados para listar, {len(ids_perdidos_sin_explicar)} en total)")

    # ------------------------------------------------------------------
    # DIVERSIDAD DE CLASES
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("DIVERSIDAD DE CLASES OBTENIDAS")
    print("=" * 70)

    lines.append("")
    lines.append("=" * 70)
    lines.append("DIVERSIDAD DE CLASES OBTENIDAS")
    lines.append("=" * 70)

    clases = {}
    for id_objeto in dataset.ids_unicos():
        curva = dataset.banda_principal(id_objeto)
        clase = curva.clase_conocida if curva else "?"
        clases[clase] = clases.get(clase, 0) + 1

    for clase, n in sorted(clases.items(), key=lambda x: -x[1]):
        print(f"  {clase:15s}: {n}")
        lines.append(f"  {clase:15s}: {n}")

    # ------------------------------------------------------------------
    # EXTRACCION DE FEATURES A ESCALA
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("EXTRAYENDO FEATURES (mismo pipeline que StarEmbed)")
    print("=" * 70)

    lines.append("")
    lines.append("=" * 70)
    lines.append("EXTRAYENDO FEATURES (mismo pipeline que StarEmbed)")
    lines.append("=" * 70)

    inicio_features = time.time()

    try:
        X = extraer_features_dataset(dataset)
        duracion_features = time.time() - inicio_features
        print(f"Features extraidas para {len(X)} objetos en {duracion_features:.1f}s")
        lines.append(f"Features extraidas para {len(X)} objetos en {duracion_features:.1f}s")
    except Exception as e:
        print(f"ERROR al extraer features: {e}")
        lines.append(f"ERROR al extraer features: {e}")
        with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nError guardado en: {RUTA_SALIDA}")
        return

    n_con_nan = X[VARIABLES_BASE].isna().any(axis=1).sum()
    print(f"Objetos con algun NaN en las 12 variables: {n_con_nan} / {len(X)}")
    lines.append(f"Objetos con algun NaN en las 12 variables: {n_con_nan} / {len(X)}")

    print()
    print("Resumen estadistico de las 12 variables (para detectar valores")
    print("degenerados/absurdos a simple vista, no solo NaN):")
    print(X[VARIABLES_BASE].describe().to_string())

    lines.append("")
    lines.append("Resumen estadistico de las 12 variables:")
    lines.append(X[VARIABLES_BASE].describe().to_string())

    print()
    print("=" * 70)
    print("VEREDICTO")
    print("=" * 70)

    lines.append("")
    lines.append("=" * 70)
    lines.append("VEREDICTO")
    lines.append("=" * 70)

    tasa_exito = n_cargados / max(n_solicitados, 1)
    print(f"Tasa de exito (cargados / solicitados): {tasa_exito*100:.1f}%")
    lines.append(f"Tasa de exito (cargados / solicitados): {tasa_exito*100:.1f}%")

    if tasa_exito > 0.95 and not adaptador.lotes_fallidos:
        mensaje = (
            "-> El adaptador se comporta de forma fiable a esta escala. "
            "Listo para considerar en la interfaz, con la reserva de que "
            "300 objetos sigue siendo modesto comparado con un uso real "
            "(miles de objetos)."
        )
        print(mensaje)
        lines.append(mensaje)
    elif tasa_exito > 0.8:
        mensaje = (
            "-> Funciona razonablemente pero con perdidas no triviales. "
            "Revisar los lotes fallidos y los ids sin explicar antes de "
            "construir la interfaz encima -- probablemente convenga "
            "añadir reintentos automaticos por lote."
        )
        print(mensaje)
        lines.append(mensaje)
    else:
        mensaje = (
            "-> Tasa de exito baja. No construir la interfaz sobre esto "
            "todavia -- hace falta diagnosticar por que se pierden tantos "
            "objetos antes de seguir."
        )
        print(mensaje)
        lines.append(mensaje)

    # Guardar resultados en archivo
    with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nResultado completo guardado en: {RUTA_SALIDA}")


if __name__ == "__main__":
    main()