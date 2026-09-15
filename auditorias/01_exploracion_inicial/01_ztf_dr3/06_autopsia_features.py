import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp, kruskal
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Ruta relativa desde la ubicación del script
BASE_DIR = Path(__file__).resolve().parents[3]  # Sube 3 niveles hasta Astronomia/

# Directorio de resultados dentro de la misma carpeta que el script
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "06_"

MUESTRA_POR_DATASET = 50000
RANDOM_STATE = 42

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
    }
}


# ============================================================
# CARGA DE DATASET
# ============================================================

def cargar_dataset(nombre, info):

    feature_path = info["feature"]
    names_path = info["names"]
    oid_path = info["oid"]

    with open(names_path, "r", encoding="utf-8") as f:
        names = f.read().split()

    dtype = [(name, np.float32) for name in names]

    oid = np.memmap(
        oid_path,
        mode="r",
        dtype=np.uint64
    )

    features = np.memmap(
        feature_path,
        mode="r",
        dtype=dtype,
        shape=oid.shape
    )

    n = len(oid)

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    if n > MUESTRA_POR_DATASET:

        indices = rng.choice(
            n,
            size=MUESTRA_POR_DATASET,
            replace=False
        )

    else:

        indices = np.arange(n)

    X = np.column_stack([
        features[name][indices]
        for name in names
    ]).astype(np.float64)

    print(
        f"{nombre:5s}: "
        f"{len(X):,} objetos"
    )

    return X, names


# ============================================================
# FUNCIÓN AUXILIAR
# ============================================================

def auc_simetrico(y_true, valores):
    """
    AUC siempre expresado entre 0.5 y 1.
    0.5 = ninguna capacidad discriminante
    1.0 = separación perfecta
    """

    auc = roc_auc_score(
        y_true,
        valores
    )

    return max(
        auc,
        1.0 - auc
    )


# ============================================================
# INICIO
# ============================================================

print("=" * 78)
print("AUTOPSIA INDIVIDUAL DE FEATURES")
print("=" * 78)

inicio = time.time()


# ============================================================
# CARGAR DATOS
# ============================================================

datos = {}

feature_names = None

for nombre, info in DATASETS.items():

    X, names = cargar_dataset(
        nombre,
        info
    )

    datos[nombre] = X

    if feature_names is None:

        feature_names = names

    elif names != feature_names:

        raise RuntimeError(
            "Los datasets no tienen las mismas features."
        )


n_features = len(feature_names)

print()
print(f"Features analizadas: {n_features}")


# ============================================================
# ETIQUETAS PARA AUC
# ============================================================

pares = [
    ("M31", "DEEP"),
    ("M31", "DISK"),
    ("DEEP", "DISK")
]


# ============================================================
# ESTRUCTURA DE RESULTADOS
# ============================================================

resultados = []


# ============================================================
# ANALISIS FEATURE POR FEATURE
# ============================================================

print()
print("=" * 78)
print("ANÁLISIS FEATURE POR FEATURE")
print("=" * 78)

for i, feature in enumerate(feature_names):

    print(
        f"[{i+1:02d}/{n_features}] "
        f"{feature}"
    )

    valores_m31 = datos["M31"][:, i]
    valores_deep = datos["DEEP"][:, i]
    valores_disk = datos["DISK"][:, i]

    # --------------------------------------------------------
    # KRUSKAL-WALLIS
    # --------------------------------------------------------

    kw_stat, kw_p = kruskal(
        valores_m31,
        valores_deep,
        valores_disk
    )

    # --------------------------------------------------------
    # MEDIANAS
    # --------------------------------------------------------

    med_m31 = np.median(valores_m31)
    med_deep = np.median(valores_deep)
    med_disk = np.median(valores_disk)

    # --------------------------------------------------------
    # RANGO INTERCUARTÍLICO GLOBAL
    # --------------------------------------------------------

    todos = np.concatenate([
        valores_m31,
        valores_deep,
        valores_disk
    ])

    q25 = np.percentile(
        todos,
        25
    )

    q75 = np.percentile(
        todos,
        75
    )

    iqr = q75 - q25

    if iqr == 0:
        iqr = 1e-12

    # --------------------------------------------------------
    # DIFERENCIAS DE MEDIANAS NORMALIZADAS
    # --------------------------------------------------------

    diff_m31_deep = (
        abs(med_m31 - med_deep)
        / iqr
    )

    diff_m31_disk = (
        abs(med_m31 - med_disk)
        / iqr
    )

    diff_deep_disk = (
        abs(med_deep - med_disk)
        / iqr
    )

    # --------------------------------------------------------
    # RESULTADOS POR PAREJA
    # --------------------------------------------------------

    fila = {
        "feature": feature,

        "median_M31": med_m31,
        "median_DEEP": med_deep,
        "median_DISK": med_disk,

        "KW_stat": kw_stat,
        "KW_pvalue": kw_p,

        "M31_DEEP_KS": np.nan,
        "M31_DEEP_KS_pvalue": np.nan,
        "M31_DEEP_AUC": np.nan,
        "M31_DEEP_median_diff_IQR": diff_m31_deep,

        "M31_DISK_KS": np.nan,
        "M31_DISK_KS_pvalue": np.nan,
        "M31_DISK_AUC": np.nan,
        "M31_DISK_median_diff_IQR": diff_m31_disk,

        "DEEP_DISK_KS": np.nan,
        "DEEP_DISK_KS_pvalue": np.nan,
        "DEEP_DISK_AUC": np.nan,
        "DEEP_DISK_median_diff_IQR": diff_deep_disk
    }

    # ========================================================
    # M31 vs DEEP
    # ========================================================

    ks, p = ks_2samp(
        valores_m31,
        valores_deep
    )

    y = np.concatenate([
        np.zeros(len(valores_m31)),
        np.ones(len(valores_deep))
    ])

    x = np.concatenate([
        valores_m31,
        valores_deep
    ])

    auc = auc_simetrico(
        y,
        x
    )

    fila["M31_DEEP_KS"] = ks
    fila["M31_DEEP_KS_pvalue"] = p
    fila["M31_DEEP_AUC"] = auc

    # ========================================================
    # M31 vs DISK
    # ========================================================

    ks, p = ks_2samp(
        valores_m31,
        valores_disk
    )

    y = np.concatenate([
        np.zeros(len(valores_m31)),
        np.ones(len(valores_disk))
    ])

    x = np.concatenate([
        valores_m31,
        valores_disk
    ])

    auc = auc_simetrico(
        y,
        x
    )

    fila["M31_DISK_KS"] = ks
    fila["M31_DISK_KS_pvalue"] = p
    fila["M31_DISK_AUC"] = auc

    # ========================================================
    # DEEP vs DISK
    # ========================================================

    ks, p = ks_2samp(
        valores_deep,
        valores_disk
    )

    y = np.concatenate([
        np.zeros(len(valores_deep)),
        np.ones(len(valores_disk))
    ])

    x = np.concatenate([
        valores_deep,
        valores_disk
    ])

    auc = auc_simetrico(
        y,
        x
    )

    fila["DEEP_DISK_KS"] = ks
    fila["DEEP_DISK_KS_pvalue"] = p
    fila["DEEP_DISK_AUC"] = auc

    resultados.append(
        fila
    )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    resultados
)


# ============================================================
# SCORE GLOBAL
# ============================================================

df["AUC_media"] = df[
    [
        "M31_DEEP_AUC",
        "M31_DISK_AUC",
        "DEEP_DISK_AUC"
    ]
].mean(axis=1)

df["KS_media"] = df[
    [
        "M31_DEEP_KS",
        "M31_DISK_KS",
        "DEEP_DISK_KS"
    ]
].mean(axis=1)

df["diferencia_mediana_media"] = df[
    [
        "M31_DEEP_median_diff_IQR",
        "M31_DISK_median_diff_IQR",
        "DEEP_DISK_median_diff_IQR"
    ]
].mean(axis=1)


# ============================================================
# RANKING POR AUC
# ============================================================

df_auc = df.sort_values(
    "AUC_media",
    ascending=False
).reset_index(drop=True)

df_auc["ranking_AUC"] = (
    np.arange(len(df_auc)) + 1
)


# ============================================================
# RANKING POR KS
# ============================================================

df_ks = df.sort_values(
    "KS_media",
    ascending=False
).reset_index(drop=True)

df_ks["ranking_KS"] = (
    np.arange(len(df_ks)) + 1
)


# ============================================================
# MOSTRAR RANKING
# ============================================================

print()
print("=" * 78)
print("RANKING GLOBAL POR CAPACIDAD DISCRIMINANTE")
print("=" * 78)

print()
print(
    f"{'RANGO':<6}"
    f"{'FEATURE':<40}"
    f"{'AUC':>10}"
    f"{'KS':>10}"
)

print("-" * 70)

for i, row in df_auc.iterrows():

    print(
        f"{i+1:<6}"
        f"{row['feature']:<40}"
        f"{row['AUC_media']:>10.5f}"
        f"{row['KS_media']:>10.5f}"
    )


# ============================================================
# RANKING POR PAREJAS
# ============================================================

for dataset_a, dataset_b in pares:

    columna_auc = (
        f"{dataset_a}_{dataset_b}_AUC"
    )

    columna_ks = (
        f"{dataset_a}_{dataset_b}_KS"
    )

    temp = df.sort_values(
        columna_auc,
        ascending=False
    )

    print()
    print("=" * 78)
    print(
        f"RANKING: {dataset_a} vs {dataset_b}"
    )
    print("=" * 78)

    print()
    print(
        f"{'RANGO':<6}"
        f"{'FEATURE':<40}"
        f"{'AUC':>10}"
        f"{'KS':>10}"
    )

    print("-" * 70)

    for j, (_, row) in enumerate(
        temp.head(15).iterrows(),
        start=1
    ):

        print(
            f"{j:<6}"
            f"{row['feature']:<40}"
            f"{row[columna_auc]:>10.5f}"
            f"{row[columna_ks]:>10.5f}"
        )


# ============================================================
# GUARDAR CSV COMPLETO
# ============================================================

csv_path = RESULTADOS_DIR / f"{PREFIJO}autopsia_features.csv"

df_auc.to_csv(
    csv_path,
    index=False
)

print()
print("=" * 78)
print("ARCHIVO GENERADO")
print("=" * 78)

print()
print(
    f"Resultados completos:"
)
print(
    csv_path
)


# ============================================================
# GRÁFICO
# ============================================================

TOP = 15

top = df_auc.head(TOP)

plt.figure(
    figsize=(12, 8)
)

plt.barh(
    top["feature"][::-1],
    top["AUC_media"][::-1]
)

plt.axvline(
    0.5,
    linestyle="--"
)

plt.xlabel(
    "AUC medio entre datasets"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Capacidad individual de discriminación entre datasets"
)

plt.tight_layout()

grafico_path = RESULTADOS_DIR / f"{PREFIJO}ranking_features_AUC.png"

plt.savefig(
    grafico_path,
    dpi=150
)

plt.close()


# ============================================================
# SEGUNDO GRÁFICO: KS
# ============================================================

top_ks = df_ks.head(TOP)

plt.figure(
    figsize=(12, 8)
)

plt.barh(
    top_ks["feature"][::-1],
    top_ks["KS_media"][::-1]
)

plt.xlabel(
    "KS medio"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Diferencia estadística entre datasets"
)

plt.tight_layout()

grafico_ks_path = RESULTADOS_DIR / f"{PREFIJO}ranking_features_KS.png"

plt.savefig(
    grafico_ks_path,
    dpi=150
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 78)
print("FINAL")
print("=" * 78)

print()
print(
    f"CSV : {csv_path}"
)

print(
    f"PNG : {grafico_path}"
)

print(
    f"PNG : {grafico_ks_path}"
)

print()
print(
    f"Tiempo total: "
    f"{time.time() - inicio:.2f} s"
)

print("=" * 78)