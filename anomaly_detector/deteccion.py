# -*- coding: utf-8 -*-

"""
======================================================================
deteccion.py - DETECCION DE ANOMALIAS (empaquetado de 24 + 24b)
======================================================================

Empaqueta la logica validada en los scripts 24 (deteccion global) y
24b (deteccion normalizada por clase) para que opere sobre el
DataFrame que produce `features.extraer_features_dataset`, en vez de
sobre columnas especificas de StarEmbed.

Ninguna formula ha cambiado respecto a 24/24b. Lo que cambia es que
ahora la funcion recibe DataFrames de features ya extraidas (de
cualquier dataset que pase por un adaptador) en vez de leer parquet
directamente.

Dos modos, igual que en la sesion de auditoria (ver whitepaper.md,
seccion 4.4):

  - `detectar_global`: cada objeto se compara contra TODO el conjunto
    de referencia (equivalente al script 24).
  - `detectar_por_clase`: cada objeto se compara solo contra los de
    su propia clase (equivalente al 24b). Neutraliza el sesgo de que
    clases pequeñas parezcan anomalas por pura escasez de referencia
    (hallazgo real: LPV en StarEmbed, ver whitepaper.md seccion 4.4).

`clasificador_control` es opcional pero recomendado: sin el no se
puede poblar `mal_clasificado`, que es la señal que separa "anomalia
real" de "sesgo de clasificador conocido disfrazado de anomalia"
(ver whitepaper.md, criterio de candidato robusto).
"""

from __future__ import annotations

from typing import Optional, Iterable

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score


RANDOM_STATE_DEFAULT = 42

# Minimo de objetos de referencia necesarios para entrenar un detector
# fiable solo con esa clase (ver 24b). Por debajo de esto, la clase se
# marca como sin base suficiente en vez de forzar un resultado ruidoso.
MIN_OBJETOS_POR_CLASE_DEFAULT = 50


# ======================================================================
# CLASIFICADOR DE CONTROL (OPCIONAL)
# ======================================================================

def entrenar_clasificador_control(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    variables: list[str],
    random_state: int = RANDOM_STATE_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Random Forest de referencia (mismo criterio que 21/24/24b):
    n_estimators=200, class_weight='balanced_subsample'.

    Devuelve (predicciones, mascara_mal_clasificado, balanced_accuracy).
    No es el detector de anomalias -- es el control que permite marcar
    que candidatos son, en realidad, solo objetos que el clasificador
    supervisado confunde de clase.
    """

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_features="sqrt",
    )

    rf.fit(X_train[variables], y_train)
    pred = rf.predict(X_test[variables])

    balanced = balanced_accuracy_score(y_test, pred)
    mal_clasificado = (pred != y_test.values)

    return pred, mal_clasificado, balanced


# ======================================================================
# DETECCION GLOBAL (equivalente al script 24)
# ======================================================================

def detectar_global(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    variables: list[str],
    random_state: int = RANDOM_STATE_DEFAULT,
    top_percentil: float = 0.02,
) -> pd.DataFrame:
    """
    Isolation Forest + LOF entrenados sobre TODO X_train, puntuando
    TODO X_test. Score invertido (mayor = mas anomalo, ver 24).

    Devuelve un DataFrame indexado igual que X_test, con:
      score_if, score_lof, rank_if, rank_lof, top_if, top_lof,
      candidato_fuerte_global (top_if Y top_lof a la vez).

    top_percentil: fraccion superior considerada "top" por cada
    metodo (0.02 = top 2%, igual que 24).
    """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[variables])
    X_test_scaled = scaler.transform(X_test[variables])

    iso = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X_train_scaled)
    score_if = -iso.score_samples(X_test_scaled)

    lof = LocalOutlierFactor(
        n_neighbors=min(20, max(len(X_train) - 1, 1)),
        novelty=True,
        contamination="auto",
        n_jobs=-1,
    )
    lof.fit(X_train_scaled)
    score_lof = -lof.score_samples(X_test_scaled)

    resultado = pd.DataFrame(
        {
            "score_if": score_if,
            "score_lof": score_lof,
        },
        index=X_test.index,
    )

    resultado["rank_if"] = resultado["score_if"].rank(ascending=False, method="min").astype(int)
    resultado["rank_lof"] = resultado["score_lof"].rank(ascending=False, method="min").astype(int)

    umbral_n = max(int(len(resultado) * top_percentil), 1)

    resultado["top_if"] = resultado["rank_if"] <= umbral_n
    resultado["top_lof"] = resultado["rank_lof"] <= umbral_n
    resultado["candidato_fuerte_global"] = resultado["top_if"] & resultado["top_lof"]

    return resultado


# ======================================================================
# DETECCION POR CLASE (equivalente al script 24b)
# ======================================================================

def detectar_por_clase(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    variables: list[str],
    random_state: int = RANDOM_STATE_DEFAULT,
    percentil_candidato: float = 98.0,
    min_objetos_por_clase: int = MIN_OBJETOS_POR_CLASE_DEFAULT,
) -> pd.DataFrame:
    """
    Un Isolation Forest + LOF INDEPENDIENTE por clase, entrenado solo
    con los objetos de esa clase en X_train, puntuando los objetos de
    esa misma clase en X_test. Score normalizado a percentil DENTRO
    de la clase (0-100), para que sea comparable entre clases de
    tamaños muy distintos (ver whitepaper.md, hallazgo LPV).

    Clases con menos de `min_objetos_por_clase` en TRAIN se marcan
    con `base_suficiente=False` y NO se les asigna score (NaN) --
    no se fuerza un resultado ruidoso con muy poca referencia.

    Devuelve un DataFrame indexado igual que X_test, con:
      score_if_percentil_clase, score_lof_percentil_clase,
      n_train_clase, base_suficiente, candidato_fuerte_por_clase.
    """

    clases = sorted(y_train.unique())

    filas = []

    for clase in clases:

        mask_train = (y_train.values == clase)
        n_train_clase = int(mask_train.sum())

        mask_test = (y_test.values == clase)

        if n_train_clase < min_objetos_por_clase:
            for idx in X_test.index[mask_test]:
                filas.append({
                    "index_test": idx,
                    "score_if_percentil_clase": np.nan,
                    "score_lof_percentil_clase": np.nan,
                    "n_train_clase": n_train_clase,
                    "base_suficiente": False,
                })
            continue

        scaler = StandardScaler()
        X_train_clase = scaler.fit_transform(X_train.loc[mask_train, variables])
        X_test_clase = scaler.transform(X_test.loc[mask_test, variables])

        iso = IsolationForest(
            n_estimators=300,
            contamination="auto",
            random_state=random_state,
            n_jobs=-1,
        )
        iso.fit(X_train_clase)
        score_if = -iso.score_samples(X_test_clase)

        n_vecinos = min(20, max(n_train_clase - 1, 1))
        lof = LocalOutlierFactor(
            n_neighbors=n_vecinos,
            novelty=True,
            contamination="auto",
            n_jobs=-1,
        )
        lof.fit(X_train_clase)
        score_lof = -lof.score_samples(X_test_clase)

        percentil_if = pd.Series(score_if).rank(pct=True).values * 100
        percentil_lof = pd.Series(score_lof).rank(pct=True).values * 100

        indices_test_clase = X_test.index[mask_test]

        for j, idx in enumerate(indices_test_clase):
            filas.append({
                "index_test": idx,
                "score_if_percentil_clase": percentil_if[j],
                "score_lof_percentil_clase": percentil_lof[j],
                "n_train_clase": n_train_clase,
                "base_suficiente": True,
            })

    resultado = pd.DataFrame(filas).set_index("index_test")
    resultado.index.name = X_test.index.name

    resultado["candidato_fuerte_por_clase"] = (
        (resultado["score_if_percentil_clase"] >= percentil_candidato)
        & (resultado["score_lof_percentil_clase"] >= percentil_candidato)
        & (resultado["base_suficiente"])
    )

    return resultado


# ======================================================================
# PIPELINE COMPLETO (equivalente a 24 + 24b + interseccion, ver 24c)
# ======================================================================

def detectar_anomalias(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    variables: list[str],
    columna_clase: str = "clase",
    columna_id: str = "id_objeto",
    clases_dificiles: Optional[Iterable[str]] = None,
    random_state: int = RANDOM_STATE_DEFAULT,
    top_percentil_global: float = 0.02,
    percentil_candidato_por_clase: float = 98.0,
    min_objetos_por_clase: int = MIN_OBJETOS_POR_CLASE_DEFAULT,
    usar_clasificador_control: bool = True,
) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo: deteccion global + deteccion por
    clase + (opcional) clasificador de control, y devuelve una unica
    tabla con todo lo necesario para decidir candidatos, equivalente
    a lo que 24c_candidatos_interseccion.csv construia a mano cruzando
    los CSV de 24 y 24b.

    Columnas de salida (una fila por objeto de X_test):
      id_objeto, clase, score_if, score_lof, candidato_fuerte_global,
      score_if_percentil_clase, score_lof_percentil_clase,
      candidato_fuerte_por_clase, candidato_robusto (global Y por clase),
      [si usar_clasificador_control] prediccion, mal_clasificado,
      es_clase_dificil, candidato_interesante (robusto, no dificil,
      no mal clasificado).

    Requiere que X_train y X_test tengan las columnas `columna_clase`
    y `columna_id` (las produce `features.extraer_features_dataset`).
    """

    if columna_clase not in X_train.columns or columna_clase not in X_test.columns:
        raise ValueError(
            f"Falta la columna de clase '{columna_clase}' en X_train/X_test. "
            f"La deteccion por clase y el clasificador de control la necesitan."
        )

    y_train = X_train[columna_clase].astype(str)
    y_test = X_test[columna_clase].astype(str)

    clases_dificiles = set(clases_dificiles) if clases_dificiles else set()

    resultado_global = detectar_global(
        X_train, X_test, variables,
        random_state=random_state, top_percentil=top_percentil_global,
    )

    resultado_por_clase = detectar_por_clase(
        X_train, y_train, X_test, y_test, variables,
        random_state=random_state,
        percentil_candidato=percentil_candidato_por_clase,
        min_objetos_por_clase=min_objetos_por_clase,
    )

    tabla = pd.DataFrame({
        "id_objeto": X_test[columna_id].values,
        "clase": y_test.values,
    }, index=X_test.index)

    tabla = tabla.join(resultado_global[["score_if", "score_lof", "candidato_fuerte_global"]])
    tabla = tabla.join(resultado_por_clase[[
        "score_if_percentil_clase", "score_lof_percentil_clase",
        "n_train_clase", "base_suficiente", "candidato_fuerte_por_clase",
    ]])

    tabla["candidato_robusto"] = (
        tabla["candidato_fuerte_global"] & tabla["candidato_fuerte_por_clase"]
    )

    if usar_clasificador_control:

        pred, mal_clasificado, balanced = entrenar_clasificador_control(
            X_train, y_train, X_test, y_test, variables, random_state=random_state,
        )

        tabla["prediccion"] = pred
        tabla["mal_clasificado"] = mal_clasificado
        tabla["es_clase_dificil"] = y_test.isin(clases_dificiles).values

        tabla["candidato_interesante"] = (
            tabla["candidato_robusto"]
            & (~tabla["mal_clasificado"])
            & (~tabla["es_clase_dificil"])
        )

        tabla.attrs["balanced_accuracy_control"] = balanced

    return tabla


# ======================================================================
# DETECCION OOD "MULTI-CLASE" (metodo de referencia del paper StarEmbed)
# ======================================================================

def detectar_ood_multiclase(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    variables: list[str],
    random_state: int = RANDOM_STATE_DEFAULT,
    min_objetos_por_clase: int = MIN_OBJETOS_POR_CLASE_DEFAULT,
) -> pd.DataFrame:
    """
    Deteccion de anomalias fuera de distribucion (OOD), replicando el
    metodo de referencia de los propios autores de StarEmbed
    ("multi-class isolation forest", Gupta et al. 2025, ver
    whitepaper.md seccion 4.8): un Isolation Forest independiente por
    cada clase conocida de TRAIN, y el score final de cada objeto de
    evaluacion es el MINIMO de sus scores contra los N detectores.

    Diferencia clave con `detectar_por_clase()`: esa funcion puntua
    cada objeto SOLO contra el detector de su propia clase verdadera
    (pensada para detectar outliers dentro de las mismas 7 clases
    conocidas). Esta funcion puntua cada objeto contra TODOS los
    detectores, sin asumir que su clase verdadera esta entre las de
    entrenamiento -- es la que corresponde cuando se evaluan objetos
    de clases genuinamente nuevas (como el split `anom` de StarEmbed).

    Devuelve un DataFrame indexado igual que X_eval, con:
      score_min (mayor = mas anomalo -- ni su mejor clase le sienta bien),
      clase_mas_cercana (la clase de entrenamiento con score mas bajo,
      es decir, la que mejor explica al objeto),
      score_<clase> para cada clase de entrenamiento usada (para poder
      inspeccionar el detalle, no solo el resumen).
    """

    clases = sorted(y_train.unique())

    modelos = {}
    scalers = {}

    for clase in clases:

        mask = (y_train.values == clase)
        n_clase = int(mask.sum())

        if n_clase < min_objetos_por_clase:
            print(
                f"  AVISO: clase '{clase}' tiene solo {n_clase} objetos "
                f"en TRAIN (< {min_objetos_por_clase}), se omite su "
                f"detector -- no participara en el minimo."
            )
            continue

        scaler = StandardScaler()
        X_clase = scaler.fit_transform(X_train.loc[mask, variables])

        iso = IsolationForest(
            n_estimators=300,
            contamination="auto",
            random_state=random_state,
            n_jobs=-1,
        )
        iso.fit(X_clase)

        scalers[clase] = scaler
        modelos[clase] = iso

    if not modelos:
        raise ValueError(
            "Ninguna clase de TRAIN tiene objetos suficientes para "
            "entrenar un detector. Revisa min_objetos_por_clase."
        )

    scores_por_clase = {}

    for clase, iso in modelos.items():
        X_eval_escalado = scalers[clase].transform(X_eval[variables])
        scores_por_clase[clase] = -iso.score_samples(X_eval_escalado)

    clases_usadas = list(modelos.keys())
    matriz_scores = np.column_stack([scores_por_clase[c] for c in clases_usadas])

    idx_min = np.argmin(matriz_scores, axis=1)
    score_min = matriz_scores[np.arange(len(matriz_scores)), idx_min]
    clase_mas_cercana = np.array(clases_usadas)[idx_min]

    resultado = pd.DataFrame(
        {
            "score_min": score_min,
            "clase_mas_cercana": clase_mas_cercana,
        },
        index=X_eval.index,
    )

    for clase in clases_usadas:
        resultado[f"score_{clase}"] = scores_por_clase[clase]

    resultado["rank_percentil"] = resultado["score_min"].rank(pct=True) * 100

    return resultado
