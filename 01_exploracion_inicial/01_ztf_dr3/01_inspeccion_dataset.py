from pathlib import Path
import numpy as np
import os
import sys
import time
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Ruta relativa desde la ubicación del script
BASE_DIR = Path(__file__).resolve().parents[3]  # Sube 3 niveles hasta Astronomia/

DATASETS = {
    "m31": {
        "feature": BASE_DIR / "dataset" / "ZTF_DR3" / "feature_m31.dat",
        "names": BASE_DIR / "dataset" / "ZTF_DR3" / "feature_m31.name",
        "oid": BASE_DIR / "dataset" / "ZTF_DR3" / "oid_m31.dat",
    },
    "deep": {
        "feature": BASE_DIR / "dataset" / "ZTF_DR3" / "feature_deep.dat",
        "names": BASE_DIR / "dataset" / "ZTF_DR3" / "feature_deep.name",
        "oid": BASE_DIR / "dataset" / "ZTF_DR3" / "oid_deep.dat",
    },
    "disk": {
        "feature": BASE_DIR / "dataset" / "ZTF_DR3" / "feature_disk.dat",
        "names": BASE_DIR / "dataset" / "ZTF_DR3" / "feature_disk.name",
        "oid": BASE_DIR / "dataset" / "ZTF_DR3" / "oid_disk.dat",
    },
}

# Tamaño de bloque para estadísticas.
# No necesitamos procesar todo de golpe.
CHUNK_SIZE = 100_000


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def tamaño_mb(path):
    return path.stat().st_size / (1024 ** 2)


def comprobar_archivos(nombre, config):
    print("\n" + "=" * 70)
    print(f"COMPROBANDO DATASET: {nombre.upper()}")
    print("=" * 70)

    todos_ok = True

    for clave, archivo in config.items():
        path = BASE_DIR / archivo if not isinstance(archivo, Path) else archivo

        if path.exists():
            print(
                f"  OK  {str(archivo):<25} "
                f"{tamaño_mb(path):>10.2f} MB"
            )
        else:
            print(f"  ERROR: NO ENCONTRADO -> {path}")
            todos_ok = False

    return todos_ok

def cargar_nombres(path):
    with open(path, "r", encoding="utf-8") as f:
        nombres = f.read().split()

    return nombres


def crear_dtype(nombres):
    return [(nombre, np.float32) for nombre in nombres]


def obtener_num_objetos(oid_path):
    """
    Calcula el número de objetos a partir del tamaño del fichero.
    oid es uint64 = 8 bytes.
    """

    tamaño = oid_path.stat().st_size

    if tamaño % np.dtype(np.uint64).itemsize != 0:
        raise ValueError(
            f"El tamaño de {oid_path.name} no es múltiplo de 8 bytes."
        )

    return tamaño // np.dtype(np.uint64).itemsize


def abrir_dataset(config):
    feature_path = BASE_DIR / config["feature"]
    names_path = BASE_DIR / config["names"]
    oid_path = BASE_DIR / config["oid"]

    nombres = cargar_nombres(names_path)

    dtype = crear_dtype(nombres)

    n_objetos = obtener_num_objetos(oid_path)

    oid = np.memmap(
        oid_path,
        mode="r",
        dtype=np.uint64
    )

    feature = np.memmap(
        feature_path,
        mode="r",
        dtype=dtype,
        shape=(n_objetos,)
    )

    return oid, feature, nombres


def analizar_dataset(nombre, config):

    print("\n")
    print("#" * 70)
    print(f"# DATASET: {nombre.upper()}")
    print("#" * 70)

    inicio = time.time()

    # --------------------------------------------------------
    # Abrir datos
    # --------------------------------------------------------

    oid, feature, nombres = abrir_dataset(config)

    n = len(oid)

    print(f"\nNúmero de objetos : {n:,}")
    print(f"Número de features: {len(nombres)}")

    print("\nFeatures:")
    for i, nombre_feature in enumerate(nombres, start=1):
        print(f"  {i:2d}. {nombre_feature}")

    # --------------------------------------------------------
    # Comprobación de tamaño
    # --------------------------------------------------------

    feature_path = BASE_DIR / config["feature"]
    oid_path = BASE_DIR / config["oid"]

    print("\nTamaños:")
    print(f"  Features: {tamaño_mb(feature_path):,.2f} MB")
    print(f"  OID     : {tamaño_mb(oid_path):,.2f} MB")

    print("\nTipos:")
    print(f"  OID     : {oid.dtype}")
    print(f"  Features: {feature.dtype}")

    # --------------------------------------------------------
    # Estadísticas
    # --------------------------------------------------------

    print("\nCalculando estadísticas...")
    print(f"Procesando en bloques de {CHUNK_SIZE:,} objetos.")

    estadisticas = {}

    for nombre_feature in nombres:
        estadisticas[nombre_feature] = {
            "min": np.inf,
            "max": -np.inf,
            "sum": 0.0,
            "sum_sq": 0.0,
            "validos": 0,
            "nan": 0,
            "inf": 0,
        }

    # --------------------------------------------------------
    # Procesamiento por bloques
    # --------------------------------------------------------

    for inicio_bloque in range(0, n, CHUNK_SIZE):

        fin_bloque = min(
            inicio_bloque + CHUNK_SIZE,
            n
        )

        datos = feature[inicio_bloque:fin_bloque]

        for nombre_feature in nombres:

            valores = datos[nombre_feature]

            es_nan = np.isnan(valores)
            es_inf = np.isinf(valores)
            es_validos = np.isfinite(valores)

            stats = estadisticas[nombre_feature]

            num_nan = int(np.sum(es_nan))
            num_inf = int(np.sum(es_inf))

            stats["nan"] += num_nan
            stats["inf"] += num_inf

            validos = valores[es_validos]

            if len(validos) == 0:
                continue

            stats["validos"] += len(validos)

            minimo = float(np.min(validos))
            maximo = float(np.max(validos))

            if minimo < stats["min"]:
                stats["min"] = minimo

            if maximo > stats["max"]:
                stats["max"] = maximo

            v64 = validos.astype(np.float64)

            stats["sum"] += float(np.sum(v64))
            stats["sum_sq"] += float(np.sum(v64 * v64))

        if inicio_bloque % (CHUNK_SIZE * 5) == 0:
            porcentaje = 100 * fin_bloque / n

            print(
                f"  Progreso: "
                f"{fin_bloque:,}/{n:,} "
                f"({porcentaje:5.1f} %)"
            )

    # --------------------------------------------------------
    # Mostrar resultados
    # --------------------------------------------------------

    print("\n")
    print("-" * 120)
    print("ESTADÍSTICAS DE FEATURES")
    print("-" * 120)

    encabezado = (
        f"{'FEATURE':<25}"
        f"{'MIN':>14}"
        f"{'MAX':>14}"
        f"{'MEDIA':>14}"
        f"{'STD':>14}"
        f"{'NaN':>12}"
        f"{'INF':>12}"
        f"{'VÁLIDOS':>14}"
    )

    print(encabezado)
    print("-" * 120)

    constantes = []
    problemas = []

    for nombre_feature in nombres:

        stats = estadisticas[nombre_feature]

        validos = stats["validos"]

        if validos > 0:

            media = stats["sum"] / validos

            varianza = (
                stats["sum_sq"] / validos
                - media * media
            )

            varianza = max(varianza, 0.0)

            std = np.sqrt(varianza)

            minimo = stats["min"]
            maximo = stats["max"]

        else:
            media = np.nan
            std = np.nan
            minimo = np.nan
            maximo = np.nan

        print(
            f"{nombre_feature:<25}"
            f"{minimo:>14.5g}"
            f"{maximo:>14.5g}"
            f"{media:>14.5g}"
            f"{std:>14.5g}"
            f"{stats['nan']:>12,}"
            f"{stats['inf']:>12,}"
            f"{validos:>14,}"
        )

        if (
            validos > 0
            and np.isfinite(minimo)
            and np.isfinite(maximo)
            and minimo == maximo
        ):
            constantes.append(nombre_feature)

        if stats["nan"] > 0 or stats["inf"] > 0:
            problemas.append(
                (
                    nombre_feature,
                    stats["nan"],
                    stats["inf"]
                )
            )

    print("-" * 120)

    # --------------------------------------------------------
    # Features constantes
    # --------------------------------------------------------

    print("\nFEATURES CONSTANTES:")

    if constantes:
        for nombre_feature in constantes:
            print(f"  - {nombre_feature}")
    else:
        print("  Ninguna.")

    # --------------------------------------------------------
    # NaN / Inf
    # --------------------------------------------------------

    print("\nFEATURES CON NaN/INF:")

    if problemas:
        for nombre_feature, nan, inf in problemas:
            print(
                f"  - {nombre_feature}: "
                f"NaN={nan:,}, INF={inf:,}"
            )
    else:
        print("  Ninguna.")

    # --------------------------------------------------------
    # Comprobación de OID
    # --------------------------------------------------------

    print("\nOID:")

    print(f"  Primer OID : {oid[0]}")
    print(f"  Último OID : {oid[-1]}")

    muestra = min(100_000, n)

    oid_muestra = np.asarray(oid[:muestra])

    duplicados_muestra = (
        len(oid_muestra)
        - len(np.unique(oid_muestra))
    )

    print(
        f"  Duplicados en primeros "
        f"{muestra:,}: {duplicados_muestra:,}"
    )

    # --------------------------------------------------------
    # Guardar resultados en CSV
    # --------------------------------------------------------
    
    resultados_dir = Path(__file__).resolve().parent / "resultados"
    resultados_dir.mkdir(exist_ok=True)
    
    csv_path = resultados_dir / f"01_inspeccion_{nombre}.csv"
    
    datos_csv = []
    for nombre_feature in nombres:
        stats = estadisticas[nombre_feature]
        validos = stats["validos"]
        if validos > 0:
            media = stats["sum"] / validos
            varianza = max(stats["sum_sq"] / validos - media * media, 0.0)
            std = np.sqrt(varianza)
        else:
            media = np.nan
            std = np.nan
        
        datos_csv.append({
            "feature": nombre_feature,
            "min": stats["min"],
            "max": stats["max"],
            "media": media,
            "std": std,
            "nan": stats["nan"],
            "inf": stats["inf"],
            "validos": validos,
            "es_constante": nombre_feature in constantes
        })
    
    df = pd.DataFrame(datos_csv)
    df.to_csv(csv_path, index=False)
    print(f"\nResultados guardados en: {csv_path}")

    # --------------------------------------------------------
    # Tiempo
    # --------------------------------------------------------

    duracion = time.time() - inicio

    print(
        f"\nTiempo de análisis: "
        f"{duracion:.2f} segundos"
    )

    return {
        "nombre": nombre,
        "n_objetos": n,
        "n_features": len(nombres),
        "features": nombres,
        "constantes": constantes,
        "problemas": problemas,
    }


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    print("=" * 70)
    print("INSPECCIÓN DEL DATASET ZTF DR3")
    print("=" * 70)

    print(f"\nDirectorio de trabajo:")
    print(f"  {BASE_DIR}")

    # --------------------------------------------------------
    # Comprobar directorio
    # --------------------------------------------------------

    if not BASE_DIR.exists():

        print(
            "\nERROR: el directorio no existe:"
        )

        print(BASE_DIR)

        sys.exit(1)

    # --------------------------------------------------------
    # Comprobar archivos
    # --------------------------------------------------------

    print("\nCOMPROBACIÓN DE ARCHIVOS")

    datasets_validos = []

    for nombre, config in DATASETS.items():

        if comprobar_archivos(nombre, config):
            datasets_validos.append(nombre)

    if not datasets_validos:

        print(
            "\nERROR: no se ha encontrado ningún dataset completo."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Analizar datasets
    # --------------------------------------------------------

    resultados = []

    for nombre in datasets_validos:

        resultado = analizar_dataset(
            nombre,
            DATASETS[nombre]
        )

        resultados.append(resultado)

    # --------------------------------------------------------
    # RESUMEN FINAL
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)

    total_objetos = 0

    for resultado in resultados:

        n = resultado["n_objetos"]

        total_objetos += n

        print(
            f"\n{resultado['nombre'].upper()}:"
        )

        print(
            f"  Objetos : {n:,}"
        )

        print(
            f"  Features: "
            f"{resultado['n_features']}"
        )

        print(
            f"  Constantes: "
            f"{len(resultado['constantes'])}"
        )

        print(
            f"  Features con NaN/INF: "
            f"{len(resultado['problemas'])}"
        )

    print("\n" + "-" * 70)

    print(
        f"TOTAL DE OBJETOS: "
        f"{total_objetos:,}"
    )

    print("=" * 70)

    print(
        "\nInspección terminada correctamente."
    )


if __name__ == "__main__":
    main()