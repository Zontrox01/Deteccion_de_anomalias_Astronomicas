from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.preprocessing import RobustScaler
    from sklearn.decomposition import PCA
except ImportError:
    print("\nERROR: falta scikit-learn.")
    print("Instálalo con:")
    print("    pip install scikit-learn")
    raise


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Ruta relativa desde la ubicación del script
BASE_DIR = Path(__file__).resolve().parents[3]  # Sube 3 niveles hasta Astronomia/

# Directorio de resultados dentro de la misma carpeta que el script
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "02_"

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

# Objetos utilizados de cada dataset para el análisis PCA.
# 50.000 por dataset = 150.000 objetos.
SAMPLE_SIZE = 50_000

RANDOM_SEED = 42

# Número de componentes que queremos estudiar.
N_COMPONENTS = 42

# Umbrales orientativos para señalar correlaciones fuertes.
CORRELACION_FUERTE = 0.90
CORRELACION_MODERADA = 0.75


# ============================================================
# FUNCIONES
# ============================================================

def cargar_dataset(config):
    """
    Abre el dataset utilizando memmap.
    No carga el fichero completo en RAM.
    """

    feature_path = config["feature"] if isinstance(config["feature"], Path) else BASE_DIR / config["feature"]
    names_path = config["names"] if isinstance(config["names"], Path) else BASE_DIR / config["names"]
    oid_path = config["oid"] if isinstance(config["oid"], Path) else BASE_DIR / config["oid"]

    with open(names_path, "r", encoding="utf-8") as f:
        nombres = f.read().split()

    dtype = [(nombre, np.float32) for nombre in nombres]

    n_objetos = oid_path.stat().st_size // 8

    oid = np.memmap(
        oid_path,
        mode="r",
        dtype=np.uint64
    )

    features = np.memmap(
        feature_path,
        mode="r",
        dtype=dtype,
        shape=(n_objetos,)
    )

    return oid, features, nombres


def structured_to_matrix(features, nombres, indices):
    """
    Convierte únicamente la muestra seleccionada a una matriz
    NumPy convencional.
    """

    matriz = np.empty(
        (len(indices), len(nombres)),
        dtype=np.float32
    )

    for j, nombre in enumerate(nombres):
        matriz[:, j] = features[nombre][indices]

    return matriz


def seleccionar_muestra(n, cantidad, rng):
    """
    Selecciona una muestra aleatoria sin cargar los datos.
    """

    cantidad = min(cantidad, n)

    return rng.choice(
        n,
        size=cantidad,
        replace=False
    )


def guardar_correlacion(matriz, nombres, dataset):
    """
    Calcula y guarda la matriz de correlación.
    """

    correlacion = np.corrcoef(
        matriz.astype(np.float64),
        rowvar=False
    )

    ruta = RESULTADOS_DIR / f"{PREFIJO}correlacion_{dataset}.png"

    fig, ax = plt.subplots(
        figsize=(15, 13)
    )

    im = ax.imshow(
        correlacion,
        interpolation="nearest",
        aspect="auto",
        vmin=-1,
        vmax=1
    )

    ax.set_title(
        f"Correlación de features - {dataset.upper()}"
    )

    ax.set_xticks(range(len(nombres)))
    ax.set_yticks(range(len(nombres)))

    ax.set_xticklabels(
        nombres,
        rotation=90,
        fontsize=7
    )

    ax.set_yticklabels(
        nombres,
        fontsize=7
    )

    fig.colorbar(im, ax=ax, label="Correlación")

    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()

    return correlacion


def analizar_correlaciones(
    correlacion,
    nombres,
    dataset,
    informe
):

    informe.write("\n")
    informe.write("=" * 70 + "\n")
    informe.write(
        f"CORRELACIONES - {dataset.upper()}\n"
    )
    informe.write("=" * 70 + "\n")

    pares_fuertes = []
    pares_moderados = []

    for i in range(len(nombres)):

        for j in range(i + 1, len(nombres)):

            r = correlacion[i, j]
            ar = abs(r)

            if ar >= CORRELACION_FUERTE:

                pares_fuertes.append(
                    (
                        ar,
                        r,
                        nombres[i],
                        nombres[j]
                    )
                )

            elif ar >= CORRELACION_MODERADA:

                pares_moderados.append(
                    (
                        ar,
                        r,
                        nombres[i],
                        nombres[j]
                    )
                )

    pares_fuertes.sort(reverse=True)
    pares_moderados.sort(reverse=True)

    informe.write(
        "\nPARES CON |r| >= "
        f"{CORRELACION_FUERTE}:\n\n"
    )

    if pares_fuertes:

        for ar, r, a, b in pares_fuertes:

            informe.write(
                f"{a:<40} "
                f"{b:<40} "
                f"r={r: .4f}\n"
            )

    else:

        informe.write("Ninguno.\n")

    informe.write(
        "\nPARES CON "
        f"{CORRELACION_MODERADA} <= |r| < "
        f"{CORRELACION_FUERTE}:\n\n"
    )

    for ar, r, a, b in pares_moderados[:100]:

        informe.write(
            f"{a:<40} "
            f"{b:<40} "
            f"r={r: .4f}\n"
        )


def calcular_estadisticas(matriz, nombres):

    resultados = []

    for i, nombre in enumerate(nombres):

        x = matriz[:, i].astype(np.float64)

        resultados.append(
            {
                "nombre": nombre,
                "min": np.min(x),
                "p01": np.percentile(x, 1),
                "p25": np.percentile(x, 25),
                "mediana": np.percentile(x, 50),
                "p75": np.percentile(x, 75),
                "p99": np.percentile(x, 99),
                "max": np.max(x),
                "media": np.mean(x),
                "std": np.std(x),
            }
        )

    return resultados


def analizar_asimetria(
    estadisticas,
    dataset,
    informe
):

    informe.write("\n")
    informe.write("=" * 70 + "\n")
    informe.write(
        f"DISTRIBUCIONES - {dataset.upper()}\n"
    )
    informe.write("=" * 70 + "\n")

    informe.write(
        "\nLas siguientes features presentan una diferencia "
        "especialmente grande entre percentiles y extremos:\n\n"
    )

    for s in estadisticas:

        rango = s["max"] - s["min"]
        iqr = s["p75"] - s["p25"]

        if iqr <= 0:
            continue

        ratio = rango / iqr

        if ratio > 50:

            informe.write(
                f"{s['nombre']:<40} "
                f"rango/IQR={ratio:>10.2f}\n"
            )


def guardar_distribuciones(
    matriz,
    nombres,
    dataset
):

    """
    Guarda histogramas de las 42 features.

    Para evitar que las variables con escalas enormes
    oculten las demás, cada histograma utiliza su propia escala.
    """

    n = len(nombres)

    filas = 7
    columnas = 6

    fig, axes = plt.subplots(
        filas,
        columnas,
        figsize=(18, 20)
    )

    axes = axes.flatten()

    for i, nombre in enumerate(nombres):

        x = matriz[:, i].astype(np.float64)

        ax = axes[i]

        # Percentiles para evitar que unos pocos extremos
        # destruyan la visualización.
        p01 = np.percentile(x, 1)
        p99 = np.percentile(x, 99)

        x_plot = x[
            (x >= p01) &
            (x <= p99)
        ]

        ax.hist(
            x_plot,
            bins=50
        )

        ax.set_title(
            nombre,
            fontsize=8
        )

        ax.tick_params(
            axis="both",
            labelsize=7
        )

    fig.suptitle(
        f"Distribuciones de features - {dataset.upper()}",
        fontsize=16
    )

    plt.tight_layout()

    ruta = (
        RESULTADOS_DIR /
        f"{PREFIJO}distribuciones_{dataset}.png"
    )

    plt.savefig(
        ruta,
        dpi=150
    )

    plt.close()


def realizar_pca(
    muestras,
    nombres,
    informe
):

    """
    Une las muestras de los tres datasets y realiza:

    1. RobustScaler
    2. PCA
    """

    print("\n")
    print("=" * 70)
    print("PCA")
    print("=" * 70)

    informe.write("\n")
    informe.write("=" * 70 + "\n")
    informe.write("PCA GLOBAL\n")
    informe.write("=" * 70 + "\n")

    matriz = np.vstack(muestras)

    print(
        f"\nMatriz PCA: "
        f"{matriz.shape[0]:,} objetos x "
        f"{matriz.shape[1]} features"
    )

    # --------------------------------------------------------
    # RobustScaler
    # --------------------------------------------------------

    print("\nAplicando RobustScaler...")

    scaler = RobustScaler()

    matriz_escalada = scaler.fit_transform(
        matriz
    )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    print("Calculando PCA...")

    pca = PCA(
        n_components=N_COMPONENTS,
        random_state=RANDOM_SEED
    )

    componentes = pca.fit_transform(
        matriz_escalada
    )

    varianza = pca.explained_variance_ratio_

    acumulada = np.cumsum(varianza)

    # --------------------------------------------------------
    # Informe
    # --------------------------------------------------------

    informe.write(
        "\nVARIANZA EXPLICADA:\n\n"
    )

    informe.write(
        f"{'COMPONENTE':<15}"
        f"{'VARIANZA':>15}"
        f"{'ACUMULADA':>15}\n"
    )

    informe.write("-" * 50 + "\n")

    for i in range(len(varianza)):

        informe.write(
            f"{i + 1:<15}"
            f"{varianza[i]:>14.6f}"
            f"{acumulada[i]:>14.6f}\n"
        )

    # --------------------------------------------------------
    # Número de componentes para ciertos niveles
    # --------------------------------------------------------

    informe.write("\n")

    for objetivo in [0.80, 0.90, 0.95, 0.99]:

        cantidad = (
            np.argmax(acumulada >= objetivo)
            + 1
        )

        informe.write(
            f"Componentes para {objetivo:.0%}: "
            f"{cantidad}\n"
        )

    # --------------------------------------------------------
    # PCA - gráfico de varianza
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(10, 6)
    )

    componentes_x = np.arange(
        1,
        len(varianza) + 1
    )

    plt.plot(
        componentes_x,
        acumulada,
        marker="o"
    )

    plt.axhline(
        0.80,
        linestyle="--"
    )

    plt.axhline(
        0.90,
        linestyle="--"
    )

    plt.axhline(
        0.95,
        linestyle="--"
    )

    plt.axhline(
        0.99,
        linestyle="--"
    )

    plt.xlabel(
        "Número de componentes"
    )

    plt.ylabel(
        "Varianza acumulada"
    )

    plt.title(
        "PCA - Varianza explicada acumulada"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        RESULTADOS_DIR / f"{PREFIJO}pca_varianza.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # PCA 2D
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(10, 8)
    )

    plt.scatter(
        componentes[:len(muestras[0]), 0],
        componentes[:len(muestras[0]), 1],
        s=3,
        alpha=0.35,
        label="M31"
    )

    inicio = len(muestras[0])
    fin = inicio + len(muestras[1])

    plt.scatter(
        componentes[inicio:fin, 0],
        componentes[inicio:fin, 1],
        s=3,
        alpha=0.35,
        label="DEEP"
    )

    inicio = fin

    plt.scatter(
        componentes[inicio:, 0],
        componentes[inicio:, 1],
        s=3,
        alpha=0.35,
        label="DISK"
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.title(
        "PCA - Primeros dos componentes"
    )

    plt.legend()

    plt.grid(
        alpha=0.2
    )

    plt.tight_layout()

    plt.savefig(
        RESULTADOS_DIR / f"{PREFIJO}pca_2d.png",
        dpi=150
    )

    plt.close()

    return scaler, pca


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    print("=" * 70)
    print("ANÁLISIS DE DISTRIBUCIONES Y PCA - ZTF DR3")
    print("=" * 70)

    print(
        f"\nResultados: {RESULTADOS_DIR}"
    )

    informe_path = (
        RESULTADOS_DIR /
        f"{PREFIJO}informe_analisis.txt"
    )

    muestras = []

    nombres_globales = None

    with open(
        informe_path,
        "w",
        encoding="utf-8"
    ) as informe:

        informe.write(
            "ANÁLISIS DEL DATASET ZTF DR3\n"
        )

        informe.write(
            f"Muestra por dataset: {SAMPLE_SIZE:,}\n"
        )

        informe.write(
            f"Semilla aleatoria: {RANDOM_SEED}\n"
        )

        # ====================================================
        # DATASETS
        # ====================================================

        for nombre_dataset, config in DATASETS.items():

            print("\n")
            print("#" * 70)
            print(
                f"DATASET: {nombre_dataset.upper()}"
            )
            print("#" * 70)

            oid, features, nombres = cargar_dataset(
                config
            )

            if nombres_globales is None:
                nombres_globales = nombres

            elif nombres != nombres_globales:

                raise RuntimeError(
                    "Las features no coinciden entre datasets."
                )

            n = len(oid)

            print(
                f"Objetos disponibles: {n:,}"
            )

            indices = seleccionar_muestra(
                n,
                SAMPLE_SIZE,
                rng
            )

            print(
                f"Seleccionando {len(indices):,} objetos..."
            )

            matriz = structured_to_matrix(
                features,
                nombres,
                indices
            )

            print("Calculando correlaciones...")

            correlacion = guardar_correlacion(
                matriz,
                nombres,
                nombre_dataset
            )

            analizar_correlaciones(
                correlacion,
                nombres,
                nombre_dataset,
                informe
            )

            print(
                "Analizando distribuciones..."
            )

            estadisticas = calcular_estadisticas(
                matriz,
                nombres
            )

            analizar_asimetria(
                estadisticas,
                nombre_dataset,
                informe
            )

            guardar_distribuciones(
                matriz,
                nombres,
                nombre_dataset
            )

            muestras.append(matriz)

            informe.write("\n")
            informe.write(
                f"Dataset {nombre_dataset.upper()}: "
                f"{n:,} objetos\n"
            )

        # ====================================================
        # PCA GLOBAL
        # ====================================================

        realizar_pca(
            muestras,
            nombres_globales,
            informe
        )

        informe.write("\n")
        informe.write("=" * 70 + "\n")
        informe.write("FIN DEL ANÁLISIS\n")
        informe.write("=" * 70 + "\n")

    print("\n")
    print("=" * 70)
    print("ANÁLISIS TERMINADO")
    print("=" * 70)

    print(
        f"\nResultados guardados en:"
    )

    print(
        f"  {RESULTADOS_DIR}"
    )

    print(
        "\nArchivos generados:"
    )

    for archivo in sorted(
        RESULTADOS_DIR.iterdir()
    ):

        print(
            f"  - {archivo.name}"
        )

    print(
        "\nIMPORTANTE:"
    )

    print(
        "No se ha modificado ninguno de los datasets originales."
    )


if __name__ == "__main__":
    main()