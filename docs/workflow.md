# Detector de Anomalías Astronómicas

Herramienta para ayudar a un astrónomo a encontrar, dentro de un dataset de curvas
de luz, los objetos cuyo comportamiento no encaja con las clases de variabilidad
conocidas — candidatos a estudiar con prioridad.

## Estructura de carpetas

Ver `docs/FILES.md`, sección 0, para el árbol completo. Resumen: `anomaly_detector/`
y `fuentes/` van en la raíz (son código de producción, no auditoría); `docs/` tiene
los tres documentos de control; `dataset/` los datos crudos; `Auditorías/` todo el
histórico de scripts y CSV de la fase de validación. `reorganizar_proyecto.py`
(raíz) automatiza el movimiento desde la carpeta plana original.

## Hallazgo importante: el dataset tiene un cuarto split, `anom`

Además de `train`/`validation`/`test`, StarEmbed publica un benchmark de detección
de anomalías real: 1.087 objetos de 10 clases de variables que los propios autores
excluyeron del entrenamiento por tener pocos ejemplos (documentado en el paper
original, arXiv 2510.06200). Esto convierte todo el trabajo de detección de
anomalías (24, 24b, 24c) de "candidatos sin verdad fundamental" a "validable contra
ground truth real". El esquema (`sourceid`, `bands_data` con claves `g`/`i`/`r`,
`period`, `class_str`, `ra`/`dec`) está confirmado compatible con `AdaptadorStarEmbed`
sin cambios. Ver `docs/whitepaper.md`, sección 4.8, para el detalle completo y las
citas obligatorias del dataset. **Siguiente paso recomendado del proyecto.**

## Estado actual

La parte científica del proyecto (extracción de features de curvas de luz +
detección de anomalías no supervisada) está **validada sobre StarEmbed y
sobre Gaia DR3** (dos fuentes reales, arquitecturas deliberadamente
distintas — fichero local vs. red), con una auditoría exhaustiva de
integridad de datos y una primera validación cuantitativa con ground truth
real (ver `whitepaper.md`, secciones 3 y 4).

**Ya existe:**

- Generalización a un segundo dataset (Gaia DR3), validada a escala (300
  objetos, 100% de éxito) con arquitectura de adaptador completamente
  distinta a StarEmbed.
- Validación externa de candidatos con catálogos reales (SIMBAD, VSX/AAVSO).
- Primera versión de la interfaz PySide6 (`app/`) — controlador probado de
  extremo a extremo, resto de ficheros con sintaxis verificada pero
  **pendiente de probar en un entorno real con pantalla** (ver
  `whitepaper.md` sección 5.2).

**Todavía pendiente:**

- Probar `app/` de verdad, con PySide6 instalado y pantalla — es lo primero
  que hacer al retomar el proyecto.
- Ampliar la interfaz al diseño completo (dos secciones "seguros"/"ambiguos",
  detección por clase/OOD en vez de la detección global auto-referencial
  del MVP actual).

Ver la sección "Próximos pasos" de `whitepaper.md` para la hoja de ruta completa.

## Qué leer primero

- **`whitepaper.md`** — por qué el proyecto está diseñado así, qué se ha
  validado y con qué evidencia, qué limitaciones se conocen. Léelo antes de
  tocar código si no has seguido todo el proceso de auditoría.
- **`FILES.md`** — inventario de todos los scripts, qué hace cada uno y si
  está auditado o no. Consúltalo antes de usar un script que no reconozcas.
- Este documento — cómo poner en marcha lo que ya existe.

## Requisitos

```
python >= 3.9
pandas
numpy
scikit-learn
scipy
```

(Sin fichero `requirements.txt` todavía — pendiente de crear cuando se empaquete
el pipeline como módulo, ver `whitepaper.md` sección 6.)

## Cómo ejecutar el pipeline actual (StarEmbed)

Los datos esperados son los parquet de StarEmbed (`train-00000-of-00002.parquet`,
`train-00001-of-00002.parquet`, `test-00000-of-00001.parquet`) en la ruta
configurada como `BASE_DIR` al principio de cada script — ajústala antes de
ejecutar.

Orden recomendado si se parte de cero:

1. `20_validacion_robustez.py` — pipeline base de clasificación de referencia.
2. `21_validacion_por_clase.py`, `22_analisis_dependencia_observaciones.py`,
   `23_estabilidad_features_por_clase.py` — validación de las 12 variables.
3. (Opcional, solo si se cambia de dataset o se sospecha de la calidad de los
   datos) `16b`, `16c`, `16d`, `20b` — controles de integridad. Sobre StarEmbed
   ya están superados, no hace falta repetirlos salvo que cambie el dataset de
   origen.
4. `24_deteccion_anomalias.py` — detección global de anomalías.
5. `24b_anomalias_por_clase.py` — detección normalizada por clase (requiere
   haber ejecutado el 24 primero, compara contra su salida).

Cada script imprime en consola una interpretación de sus propios resultados y
guarda uno o varios CSV en `BASE_DIR`.

## Candidatos actuales (StarEmbed)

`24c_candidatos_interseccion.csv` — cruce directo de las salidas del 24
(global) y el 24b (por clase). 15 objetos son candidatos robustos (fuertes en
ambos métodos), identificados con su designación real de Catalina Sky Survey
(`CSS_J...`):

- **8 "interesantes"** (`candidato_robusto_interesante=True`): ni clase
  difícil ni mal clasificados por el RF de control. Lista principal para
  inspección visual o cruce con catálogos externos (VSX/SIMBAD/Gaia DR3).
- **7 "ambiguos"** (`candidato_robusto_interesante=False`): el RF se
  equivoca de clase en estos, y aun así ambos detectores de anomalías
  coinciden en marcarlos. No se descartan — ver `whitepaper.md` sección 4.5
  para la interpretación y sección 5 para cómo se presentarán en la futura
  interfaz (sección "ambiguos", nunca oculta del todo).

## Sistema de fuentes de datos (para el futuro selector de la interfaz)

El proyecto va a ser de código abierto, así que el mecanismo para añadir un
dataset nuevo está pensado para que la comunidad lo use sin tocar el núcleo
del proyecto. Dos niveles (detalle completo en `whitepaper.md`, sección 5.0):

- **Declarativas (YAML, sin Python)** — para CSV/parquet en formato "largo"
  (una fila por objeto-época). Copia `fuentes/EJEMPLO_generico.yaml`,
  edítalo con el mapeo de tus columnas, y ya aparece disponible:

  ```python
  from anomaly_detector.adapters import fuentes_disponibles

  for f in fuentes_disponibles():
      print(f.id, f.nombre, f.tipo)  # "declarativa" o "python"
  ```

- **Python (`@registrar_fuente`)** — para fuentes con lógica real
  (paginación remota, estructuras anidadas). StarEmbed es el ejemplo ya
  construido (`adapters/starembed.py`), aunque todavía se usa directamente
  vía `cargar_train_test_starembed()` en vez de pasar por el registro —
  pendiente de formalizar.

## Paquete reutilizable (`anomaly_detector/`)

**Completo y verificado de punta a punta** contra los datos reales de
StarEmbed — tanto las features (`verificar_paquete.py`) como el pipeline de
detección completo (`probar_deteccion.py`, 100% de recall sobre los
candidatos ya conocidos de `24c_candidatos_interseccion.csv`). Ver
`whitepaper.md`, secciones 4.6 y 4.7.

```python
from anomaly_detector.adapters import cargar_train_test_starembed
from anomaly_detector import extraer_features_dataset, detectar_anomalias, VARIABLES_BASE

train, test = cargar_train_test_starembed(base_dir="C:/ruta/a/tus/parquet")
X_train = extraer_features_dataset(train)
X_test = extraer_features_dataset(test)

variables = VARIABLES_BASE + ["period"]  # si el dataset lo proporciona
resultado = detectar_anomalias(
    X_train, X_test, variables,
    clases_dificiles=["RRd", "RS CVn"],
)
# resultado ya trae candidato_robusto y candidato_interesante calculados
```

Esto reemplaza tanto la extracción manual de features como la lógica de
detección de los scripts 20-24b/24c: una llamada a `detectar_anomalias()`
sustituye al 24 + 24b + el cruce manual que hacía el 24c.

## Próximos pasos inmediatos

Ver `whitepaper.md`, sección 7, para la hoja de ruta completa. En resumen:

1. **Probar `app/` (la interfaz PySide6) en un entorno real con pantalla** —
   prioridad inmediata. `pip install -r docs/requirements.txt` y
   `python app\main.py` desde la raíz. Es razonable que aparezca algún
   ajuste en la primera ejecución, igual que pasó con el adaptador de Gaia.
2. Ampliar la interfaz al diseño completo (dos secciones "seguros"/
   "ambiguos", detección por clase/OOD en vez de la detección global
   auto-referencial del MVP actual).

## Mantenimiento de estos documentos

`whitepaper.md`, `FILES.md` y `readme.md` se actualizan juntos cada vez que el
proyecto cambia de fase o se añade un script relevante. No se reescriben desde
cero — se añade una entrada nueva al historial de cambios de cada uno y se
edita el contenido que haya quedado obsoleto.
