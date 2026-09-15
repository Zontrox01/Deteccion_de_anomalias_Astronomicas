# -*- coding: utf-8 -*-

"""
======================================================================
probar_deteccion.py - PIPELINE COMPLETO SOBRE DATOS REALES + COMPARACION
======================================================================

deteccion.py (dentro de anomaly_detector/) es un MODULO -- solo define
funciones, no hace nada por si solo. Este script es el "conductor"
que lo usa de verdad:

  1. Carga TRAIN/TEST reales de StarEmbed via el adaptador.
  2. Extrae las 12 variables (ya verificadas, ver verificar_paquete.py).
  3. Ejecuta detectar_anomalias() -- el pipeline completo (24+24b+control)
     empaquetado en una sola llamada.
  4. Compara los id_objeto marcados como candidato_interesante contra
     los 8 "interesantes" + 7 "ambiguos" ya conocidos de
     24c_candidatos_interseccion.csv, para confirmar que el paquete
     reproduce el mismo resultado que el pipeline de scripts sueltos
     (24 -> 24b -> cruce manual), no solo que las features coinciden.

Coloca este script en el mismo directorio que verificar_paquete.py
(junto a la carpeta anomaly_detector/ y los parquet de StarEmbed).
======================================================================
"""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ======================================================================
# AÑADIR EL DIRECTORIO RAIZ AL PATH PARA IMPORTAR anomaly_detector
# ======================================================================

# Ruta relativa desde la ubicación del script (auditorias/05_verificacion_paquete)
BASE_DIR = Path(__file__).resolve().parents[2]  # Sube 2 niveles hasta Astronomia/

# Añadir BASE_DIR al path para poder importar anomaly_detector
sys.path.insert(0, str(BASE_DIR))

# Ahora podemos importar desde el paquete anomaly_detector
from anomaly_detector.adapters import cargar_train_test_starembed
from anomaly_detector import extraer_features_dataset, detectar_anomalias, VARIABLES_BASE


# ======================================================================
# CONFIGURACION
# ======================================================================

# Directorio de resultados dentro de la misma carpeta que el script
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "probar_deteccion_"

DATA_DIR = BASE_DIR / "dataset" / "ZTF_40k_StarEmbed" / "data"

# Directorio donde están los resultados del 24c (en 04_deteccion_anomalias)
RESULTADOS_24_DIR = BASE_DIR / "auditorias" / "04_deteccion_anomalias" / "resultados"

MAX_TRAIN = 25000
MAX_TEST = 8000

CLASES_DIFICILES = ["RRd", "RS CVn"]

RUTA_24C = RESULTADOS_24_DIR / "24c_candidatos_interseccion.csv"
SALIDA = RESULTADOS_DIR / f"{PREFIJO}resultado.csv"


def banner(texto):
    print()
    print("=" * 70)
    print(texto)
    print("=" * 70)


# ======================================================================
# MAIN
# ======================================================================

def main():

    inicio = time.time()

    banner("CARGANDO DATOS REALES VIA EL ADAPTADOR")

    train, test = cargar_train_test_starembed(
        DATA_DIR, max_train=MAX_TRAIN, max_test=MAX_TEST
    )
    print(f"TRAIN: {len(train)} curvas cargadas (multi-banda)")
    print(f"TEST : {len(test)} curvas cargadas (multi-banda)")

    banner("EXTRAYENDO FEATURES (banda principal por objeto)")

    X_train = extraer_features_dataset(train)
    X_test = extraer_features_dataset(test)
    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")

    banner("EJECUTANDO detectar_anomalias() -- pipeline completo (24+24b+control)")

    variables = list(VARIABLES_BASE)
    if "period" in X_train.columns and "period" in X_test.columns:
        variables = variables + ["period"]
        print("'period' disponible -- incluida en el espacio de features "
              "(equivalente a la version CON periodo de 24/24b, que es la "
              "que genero 24c_candidatos_interseccion.csv).")
    else:
        print("'period' NO disponible -- ejecutando solo con las 12 variables "
              "base (equivalente a la version SIN periodo de 24).")

    resultado = detectar_anomalias(
        X_train, X_test, variables,
        columna_clase="clase",
        columna_id="id_objeto",
        clases_dificiles=CLASES_DIFICILES,
    )

    print(f"Balanced accuracy del RF de control: {resultado.attrs.get('balanced_accuracy_control'):.4f}")
    print()
    print("Candidatos por categoria:")
    print(f"  candidato_fuerte_global    : {resultado['candidato_fuerte_global'].sum()}")
    print(f"  candidato_fuerte_por_clase : {resultado['candidato_fuerte_por_clase'].sum()}")
    print(f"  candidato_robusto (ambos)  : {resultado['candidato_robusto'].sum()}")
    print(f"  candidato_interesante      : {resultado['candidato_interesante'].sum()}")

    resultado.to_csv(SALIDA, index=False)
    print()
    print("Guardado:", SALIDA)

    # ------------------------------------------------------------------
    # COMPARAR CONTRA 24c (si existe)
    # ------------------------------------------------------------------

    if RUTA_24C.exists():

        banner("COMPARANDO CONTRA 24c_candidatos_interseccion.csv")

        ref = pd.read_csv(RUTA_24C)

        ids_ref_robustos = set(ref.loc[ref["candidato_robusto"], "id_objeto"])
        ids_ref_interesantes = set(
            ref.loc[ref["candidato_robusto_interesante"], "id_objeto"]
        ) if "candidato_robusto_interesante" in ref.columns else set()

        ids_nuevo_robustos = set(
            resultado.loc[resultado["candidato_robusto"], "id_objeto"]
        )
        ids_nuevo_interesantes = set(
            resultado.loc[resultado["candidato_interesante"], "id_objeto"]
        )

        print()
        print(f"Candidatos robustos -- 24c (referencia): {len(ids_ref_robustos)}")
        print(f"Candidatos robustos -- paquete nuevo    : {len(ids_nuevo_robustos)}")
        print(f"En comun                                : {len(ids_ref_robustos & ids_nuevo_robustos)}")

        solo_ref = ids_ref_robustos - ids_nuevo_robustos
        solo_nuevo = ids_nuevo_robustos - ids_ref_robustos

        if solo_ref:
            print(f"  Solo en 24c (el paquete nuevo no los marca): {sorted(solo_ref)}")
        if solo_nuevo:
            print(f"  Solo en el paquete nuevo (24c no los marcaba): {sorted(solo_nuevo)}")

        print()
        print(f"Candidatos interesantes -- 24c (referencia): {len(ids_ref_interesantes)}")
        print(f"Candidatos interesantes -- paquete nuevo   : {len(ids_nuevo_interesantes)}")
        print(f"En comun                                   : {len(ids_ref_interesantes & ids_nuevo_interesantes)}")

        if ids_ref_interesantes == ids_nuevo_interesantes:
            print()
            print(
                "COINCIDENCIA EXACTA en candidatos interesantes. El pipeline "
                "empaquetado reproduce el resultado del 24+24b+24c."
            )
        else:
            print()
            print(
                "NO hay coincidencia exacta. Esto no es necesariamente un bug: "
                "Isolation Forest y LOF tienen componentes aleatorios "
                "(bootstrap, muestreo interno) y aunque random_state esta fijo "
                "en ambos, la version de scikit-learn, el numero de hilos "
                "(n_jobs=-1) u otras diferencias de entorno pueden alterar "
                "ligeramente el resultado en los limites del top 2%/percentil "
                "98. Revisa si los conjuntos son PARECIDOS (solapamiento alto) "
                "o completamente distintos -- lo primero es normal, lo segundo "
                "si merece investigarse."
            )

    else:
        print()
        print(
            f"No se encontro {RUTA_24C} en el directorio -- si quieres la "
            f"comparacion contra el resultado ya conocido, colocalo ahi."
        )
        print(f"Ruta esperada: {RUTA_24C}")

    print()
    print(f"Tiempo total: {time.time() - inicio:.2f} segundos")


if __name__ == "__main__":
    main()