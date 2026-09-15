import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Ruta relativa desde la ubicación del script
BASE_DIR = Path(__file__).resolve().parents[3]  # Sube 3 niveles hasta Astronomia/

# Directorio de resultados dentro de la misma carpeta que el script
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "09_"

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

N_MUESTRA = 50000
RANDOM_STATE = 42

# Features que han resultado más discriminantes.
FEATURES_CONTROL = [
    "eta_e",
    "cusum",
    "linear_trend_sigma",
    "linear_fit_slope_sigma",
    "period_0",
    "period_1",
    "period_2",
    "period_s_to_n_0",
    "period_s_to_n_1",
    "period_s_to_n_2",
    "mean",
    "weighted_mean",
    "stetson_K",
    "maximum_slope",
]

# Número de vecinos candidatos.
K_VECINOS = 20

# Distancia máxima permitida en espacio normalizado.
# Se calcula de forma relativa posteriormente.
FACTOR_DISTANCIA = 1.5

SALIDA_CSV = RESULTADOS_DIR / f"{PREFIJO}matching_vecinos_resultados.csv"


# ============================================================
# CARGA
# ============================================================

def cargar_dataset(nombre):

    cfg = DATASETS[nombre]

    feature_path = cfg["feature"]
    names_path = cfg["names"]
    oid_path = cfg["oid"]

    with open(names_path, "r", encoding="utf-8") as f:
        nombres = f.read().split()

    oids = np.memmap(
        oid_path,
        dtype=np.uint64,
        mode="r"
    )

    dtype = [
        (nombre, np.float32)
        for nombre in nombres
    ]

    features = np.memmap(
        feature_path,
        dtype=np.dtype(dtype),
        mode="r",
        shape=(len(oids),)
    )

    return features, nombres


def obtener_muestra(nombre):

    features, nombres = cargar_dataset(nombre)

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    n = min(
        N_MUESTRA,
        len(features)
    )

    indices = rng.choice(
        len(features),
        size=n,
        replace=False
    )

    X = np.column_stack([
        np.asarray(
            features[f][indices],
            dtype=np.float64
        )
        for f in FEATURES_CONTROL
    ])

    return X


# ============================================================
# LIMPIEZA
# ============================================================

def limpiar(X):

    X = np.asarray(
        X,
        dtype=np.float64
    )

    X[~np.isfinite(X)] = np.nan

    for j in range(X.shape[1]):

        mediana = np.nanmedian(
            X[:, j]
        )

        if not np.isfinite(mediana):
            mediana = 0.0

        X[
            np.isnan(X[:, j]),
            j
        ] = mediana

    return X


# ============================================================
# MATCHING
# ============================================================

def matching_por_vecinos(Xs):

    """
    Construye un conjunto balanceado buscando objetos
    comparables entre los tres datasets.

    Para cada objeto de M31 se busca el vecino más cercano
    en DEEP y DISK.

    Se trabaja en espacio RobustScaler para reducir
    la influencia de escalas y valores extremos.
    """

    print()
    print("=" * 78)
    print("MATCHING POR VECINOS MÁS CERCANOS")
    print("=" * 78)

    # --------------------------------------------------------
    # Escalado conjunto
    # --------------------------------------------------------

    X_total = np.vstack(Xs)

    scaler = RobustScaler()

    X_total_s = scaler.fit_transform(
        X_total
    )

    n0 = len(Xs[0])
    n1 = len(Xs[1])
    n2 = len(Xs[2])

    X0 = X_total_s[
        :n0
    ]

    X1 = X_total_s[
        n0:n0 + n1
    ]

    X2 = X_total_s[
        n0 + n1:
    ]

    # --------------------------------------------------------
    # Vecinos
    # --------------------------------------------------------

    print()
    print("Buscando vecinos M31 -> DEEP...")

    nn_deep = NearestNeighbors(
        n_neighbors=K_VECINOS,
        algorithm="auto",
        n_jobs=-1
    )

    nn_deep.fit(X1)

    dist_deep, idx_deep = nn_deep.kneighbors(
        X0
    )

    print("Buscando vecinos M31 -> DISK...")

    nn_disk = NearestNeighbors(
        n_neighbors=K_VECINOS,
        algorithm="auto",
        n_jobs=-1
    )

    nn_disk.fit(X2)

    dist_disk, idx_disk = nn_disk.kneighbors(
        X0
    )

    # --------------------------------------------------------
    # Distancia combinada
    # --------------------------------------------------------

    mejor_deep = dist_deep[:, 0]
    mejor_disk = dist_disk[:, 0]

    distancia_combinada = (
        mejor_deep +
        mejor_disk
    )

    limite = np.percentile(
        distancia_combinada,
        25
    ) * FACTOR_DISTANCIA

    candidatos = np.flatnonzero(
        distancia_combinada <= limite
    )

    print()
    print(
        f"Distancia M31-DEEP P50 : "
        f"{np.median(mejor_deep):.4f}"
    )

    print(
        f"Distancia M31-DISK P50 : "
        f"{np.median(mejor_disk):.4f}"
    )

    print(
        f"Distancia combinada P25 : "
        f"{np.percentile(distancia_combinada, 25):.4f}"
    )

    print(
        f"Distancia combinada P50 : "
        f"{np.percentile(distancia_combinada, 50):.4f}"
    )

    print(
        f"Distancia combinada P75 : "
        f"{np.percentile(distancia_combinada, 75):.4f}"
    )

    print(
        f"Límite utilizado          : "
        f"{limite:.4f}"
    )

    print()
    print(
        f"Candidatos M31 válidos: "
        f"{len(candidatos):,}"
    )

    # --------------------------------------------------------
    # Construcción de triples
    # --------------------------------------------------------

    triples = []

    usados_deep = set()
    usados_disk = set()

    # Los mejores candidatos primero.
    orden = candidatos[
        np.argsort(
            distancia_combinada[candidatos]
        )
    ]

    for i in orden:

        # Buscamos el primer DEEP no utilizado.
        deep_idx = None

        for k in idx_deep[i]:

            k = int(k)

            if k not in usados_deep:
                deep_idx = k
                break

        if deep_idx is None:
            continue

        # Buscamos el primer DISK no utilizado.
        disk_idx = None

        for k in idx_disk[i]:

            k = int(k)

            if k not in usados_disk:
                disk_idx = k
                break

        if disk_idx is None:
            continue

        triples.append(
            (
                int(i),
                deep_idx,
                disk_idx
            )
        )

        usados_deep.add(
            deep_idx
        )

        usados_disk.add(
            disk_idx
        )

    print()
    print(
        f"TRIPLETAS CONSEGUIDAS: "
        f"{len(triples):,}"
    )

    return triples


# ============================================================
# CONSTRUCCIÓN DE DATASET MATCHED
# ============================================================

def construir_dataset_matched(
    Xs,
    triples
):

    X_m31 = []
    X_deep = []
    X_disk = []

    for i, j, k in triples:

        X_m31.append(
            Xs[0][i]
        )

        X_deep.append(
            Xs[1][j]
        )

        X_disk.append(
            Xs[2][k]
        )

    return [
        np.asarray(X_m31),
        np.asarray(X_deep),
        np.asarray(X_disk),
    ]


# ============================================================
# COMPARACIÓN DE DISTRIBUCIONES
# ============================================================

def comparar_distribuciones(
    Xs,
    nombre
):

    print()
    print("=" * 78)
    print(
        f"DISTRIBUCIONES - {nombre}"
    )
    print("=" * 78)

    filas = []

    for j, feature in enumerate(
        FEATURES_CONTROL
    ):

        valores = [
            np.median(X[:, j])
            for X in Xs
        ]

        p10 = [
            np.percentile(
                X[:, j],
                10
            )
            for X in Xs
        ]

        p90 = [
            np.percentile(
                X[:, j],
                90
            )
            for X in Xs
        ]

        filas.append({
            "feature": feature,
            "M31_median": valores[0],
            "DEEP_median": valores[1],
            "DISK_median": valores[2],
            "M31_P10": p10[0],
            "DEEP_P10": p10[1],
            "DISK_P10": p10[2],
            "M31_P90": p90[0],
            "DEEP_P90": p90[1],
            "DISK_P90": p90[2],
        })

        print()
        print(feature)

        print(
            f"  M31  median={valores[0]: .5g} "
            f"P10={p10[0]: .5g} "
            f"P90={p90[0]: .5g}"
        )

        print(
            f"  DEEP median={valores[1]: .5g} "
            f"P10={p10[1]: .5g} "
            f"P90={p90[1]: .5g}"
        )

        print(
            f"  DISK median={valores[2]: .5g} "
            f"P10={p10[2]: .5g} "
            f"P90={p90[2]: .5g}"
        )

    return filas


# ============================================================
# CLASIFICACIÓN
# ============================================================

def clasificar(
    Xs,
    nombre
):

    print()
    print("=" * 78)
    print(
        f"CLASIFICACIÓN - {nombre}"
    )
    print("=" * 78)

    X = np.vstack(Xs)

    y = np.concatenate([
        np.full(
            len(Xs[0]),
            "M31"
        ),
        np.full(
            len(Xs[1]),
            "DEEP"
        ),
        np.full(
            len(Xs[2]),
            "DISK"
        ),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    inicio = time.time()

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )

    rf.fit(
        X_train,
        y_train
    )

    pred = rf.predict(
        X_test
    )

    acc_rf = accuracy_score(
        y_test,
        pred
    )

    tiempo_rf = (
        time.time() -
        inicio
    )

    print(
        f"Random Forest : "
        f"{acc_rf:.6f} "
        f"({tiempo_rf:.2f} s)"
    )

    # --------------------------------------------------------
    # LOGÍSTICA
    # --------------------------------------------------------

    inicio = time.time()

    scaler = RobustScaler()

    X_train_s = scaler.fit_transform(
        X_train
    )

    X_test_s = scaler.transform(
        X_test
    )

    log = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        random_state=RANDOM_STATE
    )

    log.fit(
        X_train_s,
        y_train
    )

    pred_log = log.predict(
        X_test_s
    )

    acc_log = accuracy_score(
        y_test,
        pred_log
    )

    tiempo_log = (
        time.time() -
        inicio
    )

    print(
        f"Logística     : "
        f"{acc_log:.6f} "
        f"({tiempo_log:.2f} s)"
    )

    return acc_rf, acc_log


# ============================================================
# MAIN
# ============================================================

def main():

    inicio_total = time.time()

    print("=" * 78)
    print("MATCHING DE OBJETOS ENTRE DATASETS")
    print("=" * 78)

    print()
    print(
        "Objetivo:"
    )

    print(
        "Construir triples M31-DEEP-DISK de objetos"
    )

    print(
        "estadísticamente comparables y comprobar"
    )

    print(
        "si el dataset de origen continúa siendo"
    )

    print(
        "identificable."
    )

    print()
    print(
        f"Muestra inicial: {N_MUESTRA:,} por dataset"
    )

    print(
        f"Features de control: "
        f"{len(FEATURES_CONTROL)}"
    )

    # --------------------------------------------------------
    # CARGA
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("CARGA DE DATOS")
    print("=" * 78)

    Xs = []

    for nombre in [
        "M31",
        "DEEP",
        "DISK"
    ]:

        print(
            f"\nDataset: {nombre}"
        )

        X = obtener_muestra(
            nombre
        )

        X = limpiar(
            X
        )

        print(
            f"Objetos: {len(X):,}"
        )

        print(
            f"Features: {X.shape[1]}"
        )

        Xs.append(X)

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    acc_rf_base, acc_log_base = clasificar(
        Xs,
        "BASELINE"
    )

    # --------------------------------------------------------
    # MATCHING
    # --------------------------------------------------------

    triples = matching_por_vecinos(
        Xs
    )

    if len(triples) < 100:

        raise RuntimeError(
            "Se han encontrado menos de 100 triples. "
            "El matching no es suficientemente fiable."
        )

    X_matched = construir_dataset_matched(
        Xs,
        triples
    )

    # --------------------------------------------------------
    # DISTRIBUCIONES
    # --------------------------------------------------------

    comparar_distribuciones(
        Xs,
        "ANTES DEL MATCHING"
    )

    comparar_distribuciones(
        X_matched,
        "DESPUÉS DEL MATCHING"
    )

    # --------------------------------------------------------
    # CLASIFICACIÓN MATCHED
    # --------------------------------------------------------

    acc_rf_match, acc_log_match = clasificar(
        X_matched,
        "MATCHING"
    )

    # --------------------------------------------------------
    # RESULTADOS
    # --------------------------------------------------------

    df = pd.DataFrame([
        {
            "experimento": "BASELINE",
            "objetos_por_dataset": len(Xs[0]),
            "RF": acc_rf_base,
            "LOG": acc_log_base,
        },
        {
            "experimento": "MATCHING_VECINOS",
            "objetos_por_dataset": len(X_matched[0]),
            "RF": acc_rf_match,
            "LOG": acc_log_match,
        },
    ])

    df.to_csv(
        SALIDA_CSV,
        index=False
    )

    # --------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("RESUMEN FINAL")
    print("=" * 78)

    print()
    print(
        f"{'EXPERIMENTO':25s}"
        f"{'OBJETOS':>12s}"
        f"{'RF':>12s}"
        f"{'LOG':>12s}"
    )

    print("-" * 78)

    for _, r in df.iterrows():

        print(
            f"{r['experimento']:25s}"
            f"{int(r['objetos_por_dataset']):12,}"
            f"{r['RF']:12.6f}"
            f"{r['LOG']:12.6f}"
        )

    print()
    print(
        "Accuracy aleatoria: 0.333333"
    )

    print()
    print(
        "Diferencia RF:"
    )

    print(
        f"  Baseline -> Matching: "
        f"{acc_rf_base - acc_rf_match:+.6f}"
    )

    print()
    print(
        "Diferencia Logística:"
    )

    print(
        f"  Baseline -> Matching: "
        f"{acc_log_base - acc_log_match:+.6f}"
    )

    print()
    print("=" * 78)
    print("INTERPRETACIÓN")
    print("=" * 78)

    print()

    if acc_rf_match > 0.90:

        print(
            "La clasificación continúa siendo MUY alta "
            "después del matching."
        )

        print(
            "La separación no parece explicarse "
            "únicamente por diferencias marginales "
            "de las features controladas."
        )

    elif acc_rf_match > 0.60:

        print(
            "La clasificación disminuye claramente, "
            "pero permanece por encima del azar."
        )

        print(
            "Existe evidencia de estructura residual, "
            "aunque parte de la separación original "
            "sí estaba asociada a las distribuciones "
            "controladas."
        )

    else:

        print(
            "La capacidad de clasificación cae "
            "considerablemente después del matching."
        )

        print(
            "Esto indica que una parte importante "
            "de la separación original estaba "
            "relacionada con las diferencias "
            "estadísticas controladas."
        )

    print()
    print(
        "IMPORTANTE:"
    )

    print(
        "El matching no demuestra que M31, DEEP y DISK "
        "sean poblaciones astronómicas diferentes."
    )

    print(
        "Solo permite evaluar si la procedencia del "
        "dataset sigue siendo detectable después de "
        "comparar objetos estadísticamente similares."
    )

    print()
    print(
        "Triples generados:"
        f" {len(triples):,}"
    )

    print()
    print(
        "Resultados:"
    )

    print(
        SALIDA_CSV
    )

    print()
    print(
        f"Tiempo total: "
        f"{time.time() - inicio_total:.2f} segundos"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()