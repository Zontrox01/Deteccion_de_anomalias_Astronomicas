from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Ruta relativa desde la ubicación del script
BASE_DIR = Path(__file__).resolve().parents[3]  # Sube 3 niveles hasta Astronomia/

# Directorio de resultados dentro de la misma carpeta que el script
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "03_"

DATASETS = {
    "M31": {
        "feature": BASE_DIR / "dataset" / "ZTF_DR3" / "feature_m31.dat",
        "names": BASE_DIR / "dataset" / "ZTF_DR3" / "feature_m31.name",
        "oid": BASE_DIR / "dataset" / "ZTF_DR3" / "oid_m31.dat",
    },
    "DEEP": {
        "feature": BASE_DIR / "dataset" / "ZTF_DR3" / "feature_deep.dat",
        "names": BASE_DIR / "dataset" / "ZTF_DR3" / "feature_deep.name",
        "oid": BASE_DIR / "dataset" / "ZTF_DR3" / "oid_deep.dat",
    },
    "DISK": {
        "feature": BASE_DIR / "dataset" / "ZTF_DR3" / "feature_disk.dat",
        "names": BASE_DIR / "dataset" / "ZTF_DR3" / "feature_disk.name",
        "oid": BASE_DIR / "dataset" / "ZTF_DR3" / "oid_disk.dat",
    },
}

SAMPLE_SIZE = 50_000
RANDOM_SEED = 42

# Vecinos utilizados para estudiar densidad.
K_NEIGHBORS = 20


# ============================================================
# CARGA
# ============================================================

def cargar_dataset(config):

    feature_path = config["feature"] if isinstance(config["feature"], Path) else BASE_DIR / config["feature"]
    names_path = config["names"] if isinstance(config["names"], Path) else BASE_DIR / config["names"]
    oid_path = config["oid"] if isinstance(config["oid"], Path) else BASE_DIR / config["oid"]

    with open(names_path, "r", encoding="utf-8") as f:
        nombres = f.read().split()

    dtype = [
        (nombre, np.float32)
        for nombre in nombres
    ]

    n = oid_path.stat().st_size // 8

    oid = np.memmap(
        oid_path,
        mode="r",
        dtype=np.uint64
    )

    features = np.memmap(
        feature_path,
        mode="r",
        dtype=dtype,
        shape=(n,)
    )

    return oid, features, nombres


def seleccionar_indices(n, cantidad, rng):

    cantidad = min(n, cantidad)

    return rng.choice(
        n,
        size=cantidad,
        replace=False
    )


def convertir_matriz(features, nombres, indices):

    matriz = np.empty(
        (len(indices), len(nombres)),
        dtype=np.float32
    )

    for i, nombre in enumerate(nombres):
        matriz[:, i] = features[nombre][indices]

    return matriz


# ============================================================
# PCA
# ============================================================

def ejecutar_pca(matriz):

    scaler = RobustScaler()

    escalada = scaler.fit_transform(matriz)

    pca = PCA(
        n_components=10,
        random_state=RANDOM_SEED
    )

    componentes = pca.fit_transform(escalada)

    return scaler, pca, componentes


# ============================================================
# DISTANCIAS ENTRE CENTROIDES
# ============================================================

def centroides(componentes, etiquetas):

    centros = {}

    for etiqueta in np.unique(etiquetas):

        centros[etiqueta] = np.mean(
            componentes[etiquetas == etiqueta],
            axis=0
        )

    return centros


def distancias_centros(centros):

    nombres = list(centros.keys())

    resultado = {}

    for i in range(len(nombres)):

        for j in range(i + 1, len(nombres)):

            a = nombres[i]
            b = nombres[j]

            distancia = np.linalg.norm(
                centros[a] - centros[b]
            )

            resultado[
                f"{a} <-> {b}"
            ] = distancia

    return resultado


# ============================================================
# SILHOUETTE
# ============================================================

def calcular_silhouette(componentes, etiquetas):

    # Usamos los primeros componentes porque son los que
    # contienen la estructura dominante.
    n_componentes = min(10, componentes.shape[1])

    datos = componentes[:, :n_componentes]

    return silhouette_score(
        datos,
        etiquetas
    )


# ============================================================
# DENSIDAD
# ============================================================

def analizar_densidad(componentes, etiquetas, informe):

    informe.write("\n")
    informe.write("=" * 70 + "\n")
    informe.write("ANÁLISIS DE DENSIDAD POR POBLACIÓN\n")
    informe.write("=" * 70 + "\n")

    for etiqueta in np.unique(etiquetas):

        datos = componentes[
            etiquetas == etiqueta
        ]

        n_vecinos = min(
            K_NEIGHBORS,
            len(datos) - 1
        )

        nn = NearestNeighbors(
            n_neighbors=n_vecinos + 1
        )

        nn.fit(datos[:, :10])

        distancias, _ = nn.kneighbors(
            datos[:, :10]
        )

        # Primera columna = el propio objeto.
        dist_k = distancias[:, -1]

        percentiles = np.percentile(
            dist_k,
            [1, 5, 25, 50, 75, 95, 99]
        )

        informe.write(
            f"\n{etiqueta}\n"
        )

        informe.write(
            f"Objetos: {len(datos):,}\n"
        )

        informe.write(
            "Distancia al vecino K:\n"
        )

        informe.write(
            f"  P01 = {percentiles[0]:.5f}\n"
            f"  P05 = {percentiles[1]:.5f}\n"
            f"  P25 = {percentiles[2]:.5f}\n"
            f"  P50 = {percentiles[3]:.5f}\n"
            f"  P75 = {percentiles[4]:.5f}\n"
            f"  P95 = {percentiles[5]:.5f}\n"
            f"  P99 = {percentiles[6]:.5f}\n"
        )


# ============================================================
# GRÁFICO PCA 2D
# ============================================================

def grafico_pca(componentes, etiquetas):

    fig = plt.figure(
        figsize=(11, 9)
    )

    for etiqueta in np.unique(etiquetas):

        datos = componentes[
            etiquetas == etiqueta
        ]

        plt.scatter(
            datos[:, 0],
            datos[:, 1],
            s=3,
            alpha=0.25,
            label=etiqueta
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.title(
        "ZTF DR3 - Separación de poblaciones"
    )

    plt.legend()

    plt.grid(
        alpha=0.2
    )

    plt.tight_layout()

    plt.savefig(
        RESULTADOS_DIR / f"{PREFIJO}poblaciones_pca_2d.png",
        dpi=150
    )

    plt.close()


# ============================================================
# GRÁFICO PCA 3D
# ============================================================

def grafico_pca_3d(componentes, etiquetas):

    fig = plt.figure(
        figsize=(11, 9)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    for etiqueta in np.unique(etiquetas):

        datos = componentes[
            etiquetas == etiqueta
        ]

        ax.scatter(
            datos[:, 0],
            datos[:, 1],
            datos[:, 2],
            s=3,
            alpha=0.20,
            label=etiqueta
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    ax.set_title(
        "ZTF DR3 - PCA 3D"
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTADOS_DIR / f"{PREFIJO}poblaciones_pca_3d.png",
        dpi=150
    )

    plt.close()


# ============================================================
# GRÁFICO DE CENTROIDES
# ============================================================

def grafico_centroides(centros):

    nombres = list(centros.keys())

    x = [
        centros[n][0]
        for n in nombres
    ]

    y = [
        centros[n][1]
        for n in nombres
    ]

    plt.figure(
        figsize=(9, 7)
    )

    plt.scatter(
        x,
        y,
        s=100
    )

    for nombre, px, py in zip(
        nombres,
        x,
        y
    ):

        plt.annotate(
            nombre,
            (px, py),
            xytext=(8, 8),
            textcoords="offset points"
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.title(
        "Centroides de las poblaciones"
    )

    plt.grid(
        alpha=0.2
    )

    plt.tight_layout()

    plt.savefig(
        RESULTADOS_DIR / f"{PREFIJO}centroides_poblaciones.png",
        dpi=150
    )

    plt.close()


# ============================================================
# PRINCIPAL
# ============================================================

def main():

    print("=" * 70)
    print("TERCER BISTURÍ")
    print("ESTRUCTURA DE POBLACIONES - ZTF DR3")
    print("=" * 70)

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    matrices = []
    etiquetas = []

    nombres_globales = None

    informe_path = (
        RESULTADOS_DIR /
        f"{PREFIJO}informe_poblaciones.txt"
    )

    with open(
        informe_path,
        "w",
        encoding="utf-8"
    ) as informe:

        informe.write(
            "ANÁLISIS DE ESTRUCTURA DE POBLACIONES\n"
        )

        informe.write(
            f"Muestra por dataset: {SAMPLE_SIZE:,}\n"
        )

        # ----------------------------------------------------
        # CARGAR LOS TRES DATASETS
        # ----------------------------------------------------

        for nombre, config in DATASETS.items():

            print("\n" + "-" * 70)
            print(
                f"Cargando {nombre}..."
            )

            oid, features, nombres = cargar_dataset(
                config
            )

            if nombres_globales is None:

                nombres_globales = nombres

            elif nombres != nombres_globales:

                raise RuntimeError(
                    "Los datasets tienen diferentes features."
                )

            indices = seleccionar_indices(
                len(oid),
                SAMPLE_SIZE,
                rng
            )

            matriz = convertir_matriz(
                features,
                nombres,
                indices
            )

            matrices.append(
                matriz
            )

            etiquetas.extend(
                [nombre] * len(matriz)
            )

            print(
                f"  {len(matriz):,} objetos"
            )

        etiquetas = np.array(
            etiquetas
        )

        matriz_global = np.vstack(
            matrices
        )

        print("\n")
        print("=" * 70)
        print("MATRIZ GLOBAL")
        print("=" * 70)

        print(
            f"Objetos: {len(matriz_global):,}"
        )

        print(
            f"Features: {matriz_global.shape[1]}"
        )

        # ----------------------------------------------------
        # PCA
        # ----------------------------------------------------

        print("\nCalculando PCA...")

        scaler, pca, componentes = ejecutar_pca(
            matriz_global
        )

        informe.write("\n")
        informe.write("=" * 70 + "\n")
        informe.write("PCA\n")
        informe.write("=" * 70 + "\n")

        varianza = pca.explained_variance_ratio_

        acumulada = np.cumsum(
            varianza
        )

        for i, (v, a) in enumerate(
            zip(varianza, acumulada),
            start=1
        ):

            informe.write(
                f"PC{i:02d}: "
                f"{v:.6f} "
                f"(acumulada={a:.6f})\n"
            )

        # ----------------------------------------------------
        # SILHOUETTE
        # ----------------------------------------------------

        print(
            "\nCalculando separación de poblaciones..."
        )

        silhouette = calcular_silhouette(
            componentes,
            etiquetas
        )

        informe.write("\n")
        informe.write(
            f"Silhouette score: "
            f"{silhouette:.6f}\n"
        )

        print(
            f"Silhouette score: {silhouette:.6f}"
        )

        # ----------------------------------------------------
        # CENTROIDES
        # ----------------------------------------------------

        centros = centroides(
            componentes,
            etiquetas
        )

        informe.write("\n")
        informe.write("=" * 70 + "\n")
        informe.write("CENTROIDES\n")
        informe.write("=" * 70 + "\n")

        for nombre, centro in centros.items():

            informe.write(
                f"\n{nombre}\n"
            )

            for i in range(
                min(10, len(centro))
            ):

                informe.write(
                    f"  PC{i + 1}: "
                    f"{centro[i]:.6f}\n"
                )

        # ----------------------------------------------------
        # DISTANCIAS
        # ----------------------------------------------------

        distancias = distancias_centros(
            centros
        )

        informe.write("\n")
        informe.write("=" * 70 + "\n")
        informe.write("DISTANCIAS ENTRE CENTROIDES\n")
        informe.write("=" * 70 + "\n")

        for pareja, distancia in distancias.items():

            informe.write(
                f"{pareja}: "
                f"{distancia:.6f}\n"
            )

        # ----------------------------------------------------
        # DENSIDAD
        # ----------------------------------------------------

        print(
            "\nAnalizando densidades..."
        )

        analizar_densidad(
            componentes,
            etiquetas,
            informe
        )

        # ----------------------------------------------------
        # GRÁFICOS
        # ----------------------------------------------------

        print(
            "\nGenerando gráficos..."
        )

        grafico_pca(
            componentes,
            etiquetas
        )

        grafico_pca_3d(
            componentes,
            etiquetas
        )

        grafico_centroides(
            centros
        )

        # ----------------------------------------------------
        # CONCLUSIÓN AUTOMÁTICA
        # ----------------------------------------------------

        informe.write("\n")
        informe.write("=" * 70 + "\n")
        informe.write("INTERPRETACIÓN PRELIMINAR\n")
        informe.write("=" * 70 + "\n")

        if silhouette < 0.10:

            informe.write(
                "\nLa separación entre poblaciones es MUY débil.\n"
            )

        elif silhouette < 0.25:

            informe.write(
                "\nLa separación entre poblaciones es débil/moderada.\n"
            )

        elif silhouette < 0.50:

            informe.write(
                "\nExiste una separación apreciable entre poblaciones.\n"
            )

        else:

            informe.write(
                "\nExiste una separación fuerte entre poblaciones.\n"
            )

        informe.write(
            "\nIMPORTANTE:\n"
        )

        informe.write(
            "El resultado NO demuestra que las poblaciones "
            "representen clases astronómicas distintas.\n"
        )

        informe.write(
            "La separación puede deberse a selección, "
            "instrumentación, distribución espacial u otros efectos.\n"
        )

    # --------------------------------------------------------
    # FIN
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TERCER BISTURÍ TERMINADO")
    print("=" * 70)

    print(
        f"\nResultados en:\n"
        f"{RESULTADOS_DIR}"
    )

    print(
        "\nArchivos:"
    )

    for archivo in sorted(
        RESULTADOS_DIR.iterdir()
    ):

        print(
            f"  {archivo.name}"
        )


if __name__ == "__main__":
    main()