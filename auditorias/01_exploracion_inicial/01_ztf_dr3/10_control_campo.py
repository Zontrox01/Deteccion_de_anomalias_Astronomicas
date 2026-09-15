import numpy as np
from pathlib import Path
from collections import Counter


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Ruta relativa desde la ubicación del script
BASE_DIR = Path(__file__).resolve().parents[3]  # Sube 3 niveles hasta Astronomia/

# Directorio de resultados dentro de la misma carpeta que el script
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

# Prefijo para todos los archivos generados
PREFIJO = "10_"

DATASETS = {
    "M31": BASE_DIR / "dataset" / "ZTF_DR3" / "oid_m31.dat",
    "DEEP": BASE_DIR / "dataset" / "ZTF_DR3" / "oid_deep.dat",
    "DISK": BASE_DIR / "dataset" / "ZTF_DR3" / "oid_disk.dat",
}


# ============================================================
# CARGA
# ============================================================

def cargar_oids(ruta):
    if not ruta.exists():
        raise FileNotFoundError(f"No existe:\n{ruta}")

    return np.memmap(
        ruta,
        mode="r",
        dtype=np.dtype("<u8")
    )


# ============================================================
# EXTRACCIÓN DEL CAMPO
# ============================================================

def extraer_field_id(oids):
    """
    Los OID ZTF de estos datasets tienen el field_id
    en la parte superior del identificador.

    Ejemplo:

        577108400005378
        ^^^
        campo 577
    """

    return np.asarray(oids // 10**12, dtype=np.int64)


# ============================================================
# RESUMEN
# ============================================================

def resumen_dataset(nombre, oids):

    fields = extraer_field_id(oids)

    contador = Counter(fields.tolist())

    print()
    print("=" * 70)
    print(f"DATASET: {nombre}")
    print("=" * 70)

    print(f"Objetos: {len(oids):,}")
    print(f"Campos diferentes: {len(contador):,}")

    print()
    print("Campos:")

    for field, cantidad in sorted(
        contador.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        porcentaje = cantidad / len(oids) * 100

        print(
            f"  FIELD {field:4d} : "
            f"{cantidad:10,} objetos "
            f"({porcentaje:7.3f} %)"
        )

    return set(contador.keys())


# ============================================================
# GUARDAR RESULTADOS EN CSV
# ============================================================

def guardar_resultados(campos, pares, todos, diagnostico):
    """
    Guarda un resumen de los resultados en un archivo CSV.
    """

    csv_path = RESULTADOS_DIR / f"{PREFIJO}control_campo_resultados.csv"

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("dataset,campos,objetos\n")
        
        for nombre, conjunto in campos.items():
            # Contamos cuántos objetos tiene cada dataset
            # (necesitamos los oids originales)
            pass  # Esto se manejará en main

    # Mejor guardamos un resumen en texto
    txt_path = RESULTADOS_DIR / f"{PREFIJO}control_campo_resumen.txt"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("CONTROL DE IDENTIDAD DE CAMPO ZTF\n")
        f.write("=" * 70 + "\n\n")

        for nombre, conjunto in campos.items():
            f.write(f"{nombre}:\n")
            f.write(f"  Campos: {sorted(conjunto)}\n")
            f.write(f"  Número de campos: {len(conjunto)}\n\n")

        f.write("=" * 70 + "\n")
        f.write("SOLAPAMIENTO ENTRE DATASETS\n")
        f.write("=" * 70 + "\n\n")

        for a, b in pares:
            comunes = campos[a] & campos[b]
            f.write(f"{a} <-> {b}:\n")
            f.write(f"  Campos comunes: {len(comunes)}\n")
            if comunes:
                f.write(f"  {sorted(comunes)}\n")
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("ESTRUCTURA GLOBAL\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Campos totales utilizados: {len(todos)}\n\n")

        f.write("=" * 70 + "\n")
        f.write("DIAGNÓSTICO\n")
        f.write("=" * 70 + "\n\n")
        f.write(diagnostico)

    return csv_path, txt_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CONTROL DE IDENTIDAD DE CAMPO ZTF")
    print("=" * 70)

    print()
    print("Objetivo:")
    print("Determinar si M31, DEEP y DISK corresponden")
    print("a campos ZTF diferentes y si la etiqueta del")
    print("dataset está directamente relacionada con FIELDID.")

    print()
    print("=" * 70)
    print("CARGA DE OID")
    print("=" * 70)

    oids = {}

    for nombre, ruta in DATASETS.items():

        print()
        print(f"{nombre}")
        print(f"Ruta: {ruta}")

        oids[nombre] = cargar_oids(ruta)

        print(f"Objetos: {len(oids[nombre]):,}")

        if len(oids[nombre]) == 0:
            raise RuntimeError(
                f"{nombre}: archivo OID vacío."
            )

    # --------------------------------------------------------
    # CAMPOS
    # --------------------------------------------------------

    campos = {}

    for nombre in DATASETS:

        campos[nombre] = resumen_dataset(
            nombre,
            oids[nombre]
        )

    # --------------------------------------------------------
    # INTERSECCIONES
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SOLAPAMIENTO ENTRE DATASETS")
    print("=" * 70)

    pares = [
        ("M31", "DEEP"),
        ("M31", "DISK"),
        ("DEEP", "DISK"),
    ]

    for a, b in pares:

        comunes = campos[a] & campos[b]

        print()
        print(f"{a} <-> {b}")

        print(
            f"Campos comunes: {len(comunes)}"
        )

        if comunes:
            print(
                "  ",
                sorted(comunes)
            )

    # --------------------------------------------------------
    # UNION
    # --------------------------------------------------------

    todos = (
        campos["M31"]
        | campos["DEEP"]
        | campos["DISK"]
    )

    print()
    print("=" * 70)
    print("ESTRUCTURA GLOBAL")
    print("=" * 70)

    print(
        f"Campos totales utilizados: {len(todos)}"
    )

    # --------------------------------------------------------
    # DIAGNÓSTICO
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DIAGNÓSTICO")
    print("=" * 70)

    un_campo = all(
        len(campos[nombre]) == 1
        for nombre in DATASETS
    )

    campos_disjuntos = (
        len(
            campos["M31"]
            & campos["DEEP"]
        ) == 0
        and
        len(
            campos["M31"]
            & campos["DISK"]
        ) == 0
        and
        len(
            campos["DEEP"]
            & campos["DISK"]
        ) == 0
    )

    diagnostico = ""

    if un_campo and campos_disjuntos:

        print()
        print("RESULTADO: MUY IMPORTANTE")
        print()
        print(
            "Cada dataset corresponde a un campo ZTF diferente."
        )

        print()
        print(
            "Por tanto, el clasificador está distinguiendo"
        )
        print(
            "tres campos observacionales diferentes."
        )

        print()
        print(
            "La elevada accuracy NO puede interpretarse"
        )
        print(
            "como evidencia de tres poblaciones astronómicas."
        )

        print()
        print(
            "La investigación debe detenerse aquí respecto"
        )
        print(
            "a interpretar M31/DEEP/DISK como clases físicas."
        )

        diagnostico = (
            "RESULTADO: MUY IMPORTANTE\n"
            "Cada dataset corresponde a un campo ZTF diferente.\n"
            "Por tanto, el clasificador está distinguiendo\n"
            "tres campos observacionales diferentes.\n"
            "La elevada accuracy NO puede interpretarse\n"
            "como evidencia de tres poblaciones astronómicas.\n"
            "La investigación debe detenerse aquí respecto\n"
            "a interpretar M31/DEEP/DISK como clases físicas."
        )

    elif campos_disjuntos:

        print()
        print(
            "RESULTADO: CAMPOS SEPARADOS"
        )

        print(
            "Los datasets utilizan conjuntos de campos"
        )
        print(
            "sin solapamiento."
        )

        print()
        print(
            "Todavía merece la pena realizar un control"
        )
        print(
            "por FIELDID antes de extraer conclusiones."
        )

        diagnostico = (
            "RESULTADO: CAMPOS SEPARADOS\n"
            "Los datasets utilizan conjuntos de campos\n"
            "sin solapamiento.\n"
            "Todavía merece la pena realizar un control\n"
            "por FIELDID antes de extraer conclusiones."
        )

    else:

        print()
        print(
            "RESULTADO: HAY SOLAPAMIENTO DE CAMPOS"
        )

        print(
            "Esto abre una puerta para un verdadero"
        )
        print(
            "GroupShuffleSplit por campo."
        )

        print()
        print(
            "La investigación puede continuar."
        )

        diagnostico = (
            "RESULTADO: HAY SOLAPAMIENTO DE CAMPOS\n"
            "Esto abre una puerta para un verdadero\n"
            "GroupShuffleSplit por campo.\n"
            "La investigación puede continuar."
        )

    # --------------------------------------------------------
    # GUARDAR RESULTADOS
    # --------------------------------------------------------

    csv_path, txt_path = guardar_resultados(
        campos,
        pares,
        todos,
        diagnostico
    )

    print()
    print("=" * 70)
    print("RESULTADOS GUARDADOS")
    print("=" * 70)
    print()
    print(f"CSV: {csv_path}")
    print(f"TXT: {txt_path}")

    print()
    print("=" * 70)
    print("FIN")
    print("=" * 70)


if __name__ == "__main__":
    main()