import os
import time
from pathlib import Path
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
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
PREFIJO = "05_"

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
# CARGA
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

    rng = np.random.default_rng(RANDOM_STATE)

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
    ]).astype(np.float32)

    print(
        f"{nombre:5s}: {len(X):,} objetos | "
        f"{len(names)} features"
    )

    return X, names


# ============================================================
# CARGAR LOS TRES DATASETS
# ============================================================

print("=" * 78)
print("ABLACIÓN DE FEATURES")
print("=" * 78)

inicio_total = time.time()

X_list = []
y_list = []

feature_names = None

for label, info in DATASETS.items():

    X, names = cargar_dataset(label, info)

    if feature_names is None:
        feature_names = names
    else:
        if names != feature_names:
            raise RuntimeError(
                "Los datasets no tienen las mismas features."
            )

    X_list.append(X)
    y_list.extend([label] * len(X))


X = np.vstack(X_list)

y = np.array(y_list)

print()
print(f"Objetos totales : {len(X):,}")
print(f"Features         : {X.shape[1]}")

# ============================================================
# TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


# ============================================================
# FUNCIONES
# ============================================================

def evaluar(Xtr, Xte, descripcion):

    print()
    print("-" * 78)
    print(descripcion)
    print("-" * 78)

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    t0 = time.time()

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt"
    )

    rf.fit(Xtr, y_train)

    pred = rf.predict(Xte)

    acc_rf = accuracy_score(y_test, pred)

    tiempo_rf = time.time() - t0

    print(
        f"Random Forest : {acc_rf:.6f} "
        f"({tiempo_rf:.2f} s)"
    )

    # --------------------------------------------------------
    # LOGÍSTICA
    # --------------------------------------------------------

    t0 = time.time()

    modelo_log = Pipeline([
        (
            "scaler",
            RobustScaler()
        ),
        (
            "logistic",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ])

    modelo_log.fit(Xtr, y_train)

    pred = modelo_log.predict(Xte)

    acc_log = accuracy_score(y_test, pred)

    tiempo_log = time.time() - t0

    print(
        f"Logística     : {acc_log:.6f} "
        f"({tiempo_log:.2f} s)"
    )

    return acc_rf, acc_log


# ============================================================
# EXPERIMENTOS
# ============================================================

resultados = []


# ------------------------------------------------------------
# EXPERIMENTO 1
# TODAS LAS FEATURES
# ------------------------------------------------------------

rf, log = evaluar(
    X_train,
    X_test,
    "EXPERIMENTO 1 - LAS 42 FEATURES"
)

resultados.append(
    ("42 features", rf, log)
)


# ============================================================
# GRUPOS DE FEATURES
# ============================================================

grupos = {

    "sin_eta_e": [
        "eta_e"
    ],

    "sin_eta_e_medias": [
        "eta_e",
        "mean",
        "weighted_mean"
    ],

    "sin_eta_e_tendencias": [
        "eta_e",
        "linear_fit_slope",
        "linear_fit_slope_sigma",
        "linear_trend",
        "linear_trend_sigma"
    ],

    "sin_eta_e_medias_tendencias": [
        "eta_e",
        "mean",
        "weighted_mean",
        "linear_fit_slope",
        "linear_fit_slope_sigma",
        "linear_trend",
        "linear_trend_sigma"
    ],

    "sin_magnitud_escala": [
        "eta_e",
        "mean",
        "weighted_mean",
        "standard_deviation",
        "amplitude",
        "percent_amplitude",
        "median_absolute_deviation",
        "inter_percentile_range_25",
        "inter_percentile_range_10"
    ],

    "solo_temporales": [
        "eta_e",
        "mean",
        "weighted_mean",
        "standard_deviation",
        "amplitude",
        "percent_amplitude",
        "median_absolute_deviation",
        "inter_percentile_range_25",
        "inter_percentile_range_10",
        "linear_fit_slope",
        "linear_fit_slope_sigma",
        "linear_fit_reduced_chi2",
        "linear_trend",
        "linear_trend_sigma",
        "maximum_slope",
        "period_0",
        "period_s_to_n_0",
        "period_1",
        "period_s_to_n_1",
        "period_2",
        "period_s_to_n_2",
        "periodogram_amplitude",
        "periodogram_standard_deviation",
        "periodogram_eta",
        "cusum",
        "skew",
        "kurtosis",
        "stetson_K",
        "chi2"
    ]
}


# ============================================================
# EJECUTAR ABLACIONES
# ============================================================

for nombre_grupo, eliminar in grupos.items():

    indices = [
        i
        for i, name in enumerate(feature_names)
        if name not in eliminar
    ]

    print()
    print(
        f"Eliminadas: {len(eliminar)} features"
    )

    print(
        "Features utilizadas:",
        len(indices)
    )

    rf, log = evaluar(
        X_train[:, indices],
        X_test[:, indices],
        f"ABLACIÓN - {nombre_grupo}"
    )

    resultados.append(
        (
            nombre_grupo,
            rf,
            log
        )
    )


# ============================================================
# TEST DE ETIQUETAS ALEATORIAS
# ============================================================

print()
print("=" * 78)
print("PRUEBA DE CONTROL - ETIQUETAS ALEATORIAS")
print("=" * 78)

rng = np.random.default_rng(RANDOM_STATE)

y_random = y.copy()

rng.shuffle(y_random)

Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
    X,
    y_random,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_random
)

rf_random = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    max_features="sqrt"
)

t0 = time.time()

rf_random.fit(Xtr_r, ytr_r)

pred_random = rf_random.predict(Xte_r)

acc_random = accuracy_score(
    yte_r,
    pred_random
)

print(
    f"Accuracy con etiquetas aleatorias: "
    f"{acc_random:.6f}"
)

print(
    "Esperada aproximadamente: 0.333333"
)

# ============================================================
# GUARDAR RESULTADOS EN CSV
# ============================================================

csv_path = RESULTADOS_DIR / f"{PREFIJO}resultados_ablacion.csv"

with open(csv_path, "w", encoding="utf-8") as f:
    f.write("experimento,accuracy_rf,accuracy_logistica\n")
    for nombre, rf, log in resultados:
        f.write(f"{nombre},{rf:.10f},{log:.10f}\n")
    
    # Añadir el control de etiquetas aleatorias
    f.write(f"etiquetas_aleatorias,{acc_random:.10f},-\n")

print()
print(f"Resultados guardados en: {csv_path}")

# ============================================================
# RESUMEN
# ============================================================

print()
print("=" * 78)
print("RESUMEN DE ABLACIÓN")
print("=" * 78)

print()
print(
    f"{'EXPERIMENTO':40s}"
    f"{'RF':>12s}"
    f"{'LOG':>12s}"
)

print("-" * 66)

for nombre, rf, log in resultados:

    print(
        f"{nombre:40s}"
        f"{rf:12.6f}"
        f"{log:12.6f}"
    )

# Añadir el control al resumen
print(
    f"{'etiquetas_aleatorias':40s}"
    f"{acc_random:12.6f}"
    f"{'':>12s}"
)

print()
print("=" * 78)

print(
    f"Tiempo total: "
    f"{time.time() - inicio_total:.2f} s"
)

print("=" * 78)