import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Ruta relativa desde la ubicación del script
BASE_DIR = Path(__file__).resolve().parents[3]  # Sube 3 niveles hasta Astronomia/

# Directorio de resultados dentro de la misma carpeta que el script
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "08_"

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

# Features que han aparecido repetidamente como
# responsables de la separación.
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

# Número de bins utilizados para igualar distribuciones.
N_BINS = 20

SALIDA_CSV = RESULTADOS_DIR / f"{PREFIJO}control_estructura_resultados.csv"


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

    dtype = [(n, np.float32) for n in nombres]

    features = np.memmap(
        feature_path,
        dtype=np.dtype(dtype),
        mode="r",
        shape=(len(oids),)
    )

    return features, nombres


def obtener_muestra(nombre, n=N_MUESTRA):
    features, nombres = cargar_dataset(nombre)

    indices = np.random.default_rng(RANDOM_STATE).choice(
        len(features),
        size=min(n, len(features)),
        replace=False
    )

    X = np.column_stack([
        np.asarray(features[f][indices], dtype=np.float64)
        for f in FEATURES_CONTROL
    ])

    return X


# ============================================================
# LIMPIEZA
# ============================================================

def limpiar(X):
    X = np.asarray(X, dtype=np.float64)

    X[~np.isfinite(X)] = np.nan

    # Sustituimos NaN por la mediana de cada feature.
    for j in range(X.shape[1]):
        col = X[:, j]
        mediana = np.nanmedian(col)

        if not np.isfinite(mediana):
            mediana = 0.0

        col[np.isnan(col)] = mediana

    return X


# ============================================================
# CREAR BINS GLOBALES
# ============================================================

def crear_bins_globales(Xs):
    """
    Crea límites de bins a partir de la distribución conjunta
    de los tres datasets.

    Se utilizan percentiles para evitar que los valores extremos
    dominen la discretización.
    """

    bins = []

    X_total = np.vstack(Xs)

    for j in range(X_total.shape[1]):

        valores = X_total[:, j]

        p01 = np.percentile(valores, 1)
        p99 = np.percentile(valores, 99)

        if p01 == p99:
            limites = np.array(
                [p01 - 0.5, p99 + 0.5]
            )
        else:
            limites = np.linspace(
                p01,
                p99,
                N_BINS + 1
            )

        bins.append(limites)

    return bins


# ============================================================
# HISTOGRAMA
# ============================================================

def obtener_bins(X, limites):

    indices = np.zeros(len(X), dtype=np.int32)

    for j, edges in enumerate(limites):

        valores = X[:, j]

        b = np.digitize(
            valores,
            edges[1:-1],
            right=False
        )

        indices = indices * N_BINS + b

    return indices


# ============================================================
# CONTROL DE DISTRIBUCIÓN
# ============================================================

def igualar_distribuciones(Xs):
    """
    Realiza un submuestreo para que los datasets tengan
    una representación comparable en el espacio conjunto
    de las features de control.

    No genera datos artificiales.
    Solamente descarta objetos.
    """

    limites = crear_bins_globales(Xs)

    bins_por_dataset = [
        obtener_bins(X, limites)
        for X in Xs
    ]

    # Conteos por bin
    conteos = []

    for b in bins_por_dataset:
        valores, counts = np.unique(
            b,
            return_counts=True
        )

        conteos.append(
            dict(zip(valores, counts))
        )

    # Bins presentes en todos
    comunes = set(conteos[0])

    for c in conteos[1:]:
        comunes &= set(c)

    if not comunes:
        raise RuntimeError(
            "No existen regiones comunes entre los tres datasets."
        )

    rng = np.random.default_rng(RANDOM_STATE)

    indices_finales = [
        [] for _ in Xs
    ]

    for bin_id in comunes:

        cantidades = [
            conteo.get(bin_id, 0)
            for conteo in conteos
        ]

        n = min(cantidades)

        if n <= 0:
            continue

        for i, b in enumerate(bins_por_dataset):

            candidatos = np.flatnonzero(b == bin_id)

            seleccion = rng.choice(
                candidatos,
                size=n,
                replace=False
            )

            indices_finales[i].extend(
                seleccion.tolist()
            )

    X_equilibradas = []

    for X, indices in zip(Xs, indices_finales):

        indices = np.asarray(indices)

        X_equilibradas.append(
            X[indices]
        )

    return X_equilibradas


# ============================================================
# CLASIFICACIÓN
# ============================================================

def clasificar(Xs, nombre_experimento):

    X = np.vstack(Xs)

    y = np.concatenate([
        np.full(len(Xs[0]), "M31"),
        np.full(len(Xs[1]), "DEEP"),
        np.full(len(Xs[2]), "DISK"),
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

    rf.fit(X_train, y_train)

    pred = rf.predict(X_test)

    acc_rf = accuracy_score(
        y_test,
        pred
    )

    tiempo_rf = time.time() - inicio

    print(
        f"Random Forest : {acc_rf:.6f} "
        f"({tiempo_rf:.2f} s)"
    )

    # --------------------------------------------------------
    # LOGÍSTICA
    # --------------------------------------------------------

    inicio = time.time()

    scaler = RobustScaler()

    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    log = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        random_state=RANDOM_STATE
    )

    log.fit(X_train_s, y_train)

    pred_log = log.predict(X_test_s)

    acc_log = accuracy_score(
        y_test,
        pred_log
    )

    tiempo_log = time.time() - inicio

    print(
        f"Logística     : {acc_log:.6f} "
        f"({tiempo_log:.2f} s)"
    )

    return {
        "experimento": nombre_experimento,
        "objetos_M31": len(Xs[0]),
        "objetos_DEEP": len(Xs[1]),
        "objetos_DISK": len(Xs[2]),
        "features": X.shape[1],
        "RF": acc_rf,
        "LOG": acc_log,
    }


# ============================================================
# DISTANCIAS ENTRE DISTRIBUCIONES
# ============================================================

def resumen_distribuciones(Xs):

    print()
    print("=" * 78)
    print("DISTRIBUCIONES DESPUÉS DEL CONTROL")
    print("=" * 78)

    for j, feature in enumerate(FEATURES_CONTROL):

        print()
        print(feature)

        for nombre, X in zip(
            ["M31", "DEEP", "DISK"],
            Xs
        ):

            print(
                f"  {nombre:5s} "
                f"mediana={np.median(X[:, j]): .5g} "
                f"P10={np.percentile(X[:, j], 10): .5g} "
                f"P90={np.percentile(X[:, j], 90): .5g}"
            )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    inicio_total = time.time()

    print("=" * 78)
    print("CONTROL DE ESTRUCTURA ENTRE DATASETS")
    print("=" * 78)

    print()
    print("Objetivo:")
    print("Comprobar si la separación entre M31, DEEP y DISK")
    print("permanece cuando se igualan las distribuciones")
    print("de las features más discriminantes.")

    print()
    print(f"Muestra inicial por dataset : {N_MUESTRA:,}")
    print(f"Random State                : {RANDOM_STATE}")

    # --------------------------------------------------------
    # CARGA
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("CARGA")
    print("=" * 78)

    Xs = []

    for nombre in ["M31", "DEEP", "DISK"]:

        print(f"\nDataset: {nombre}")

        X = obtener_muestra(nombre)
        X = limpiar(X)

        print(f"Objetos: {len(X):,}")
        print(f"Features: {X.shape[1]}")

        Xs.append(X)

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("EXPERIMENTO 1 - BASELINE")
    print("=" * 78)

    resultados = []

    resultado = clasificar(
        Xs,
        "BASELINE"
    )

    resultados.append(resultado)

    # --------------------------------------------------------
    # IGUALACIÓN
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("EXPERIMENTO 2 - DISTRIBUCIONES IGUALADAS")
    print("=" * 78)

    print()
    print(
        "Se realiza submuestreo conjunto sobre las "
        f"{len(FEATURES_CONTROL)} features de control."
    )

    X_control = igualar_distribuciones(Xs)

    print()
    print("Objetos después del control:")

    for nombre, X in zip(
        ["M31", "DEEP", "DISK"],
        X_control
    ):
        print(
            f"{nombre:5s}: {len(X):,}"
        )

    if min(len(X) for X in X_control) < 1000:
        print()
        print(
            "ADVERTENCIA: el control ha eliminado "
            "una cantidad muy grande de objetos."
        )

    resumen_distribuciones(X_control)

    print()
    print("=" * 78)
    print("CLASIFICACIÓN DESPUÉS DEL CONTROL")
    print("=" * 78)

    resultado = clasificar(
        X_control,
        "DISTRIBUCIONES_IGUALADAS"
    )

    resultados.append(resultado)

    # --------------------------------------------------------
    # CONTROL MÁS ESTRICTO
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("EXPERIMENTO 3 - CONTROL ESTRICTO")
    print("=" * 78)

    # Repetimos el procedimiento utilizando únicamente
    # las features que mostraron mayor capacidad discriminante.

    FEATURES_ORIGINALES = FEATURES_CONTROL.copy()

    features_backup = FEATURES_CONTROL.copy()

    FEATURES_CONTROL[:] = [
        "eta_e",
        "cusum",
        "linear_trend_sigma",
        "linear_fit_slope_sigma",
        "period_0",
        "period_1",
        "period_2",
    ]

    Xs_estricto = []

    # Las columnas están en el orden original.
    columnas = [
        FEATURES_ORIGINALES.index(f)
        for f in FEATURES_CONTROL
    ]

    for X in Xs:
        Xs_estricto.append(
            X[:, columnas]
        )

    X_estricto = igualar_distribuciones(
        Xs_estricto
    )

    print()
    print("Features utilizadas:")

    for f in FEATURES_CONTROL:
        print(f"  {f}")

    print()
    print("Objetos después del control:")

    for nombre, X in zip(
        ["M31", "DEEP", "DISK"],
        X_estricto
    ):
        print(
            f"{nombre:5s}: {len(X):,}"
        )

    resultado = clasificar(
        X_estricto,
        "CONTROL_ESTRICTO"
    )

    resultados.append(resultado)

    FEATURES_CONTROL[:] = features_backup

    # --------------------------------------------------------
    # RESULTADOS
    # --------------------------------------------------------

    df = pd.DataFrame(resultados)

    df.to_csv(
        SALIDA_CSV,
        index=False
    )

    print()
    print("=" * 78)
    print("RESUMEN FINAL")
    print("=" * 78)

    print()
    print(
        f"{'EXPERIMENTO':35s}"
        f"{'OBJETOS':>12s}"
        f"{'RF':>12s}"
        f"{'LOG':>12s}"
    )

    print("-" * 78)

    for r in resultados:

        total = (
            r["objetos_M31"]
            + r["objetos_DEEP"]
            + r["objetos_DISK"]
        )

        print(
            f"{r['experimento']:35s}"
            f"{total:12,}"
            f"{r['RF']:12.6f}"
            f"{r['LOG']:12.6f}"
        )

    print()
    print("=" * 78)
    print("INTERPRETACIÓN")
    print("=" * 78)

    baseline = resultados[0]["RF"]
    control = resultados[1]["RF"]
    estricto = resultados[2]["RF"]

    print()
    print(
        f"RF baseline             : {baseline:.6f}"
    )

    print(
        f"RF distribuciones igualadas: {control:.6f}"
    )

    print(
        f"RF control estricto     : {estricto:.6f}"
    )

    print()
    print(
        f"Reducción baseline -> igualado: "
        f"{baseline - control:.6f}"
    )

    print(
        f"Reducción baseline -> estricto: "
        f"{baseline - estricto:.6f}"
    )

    print()
    print(
        "Referencia aleatoria: 0.333333"
    )

    print()
    print(
        "IMPORTANTE:"
    )
    print(
        "Este experimento NO demuestra por sí solo que las"
    )
    print(
        "poblaciones sean astronómicamente diferentes."
    )
    print(
        "Su objetivo es comprobar cuánto de la separación"
    )
    print(
        "desaparece al controlar las distribuciones."
    )

    print()
    print(
        f"Resultados guardados en:"
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