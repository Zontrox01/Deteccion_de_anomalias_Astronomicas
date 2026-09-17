# Detector de Anomalías Astronómicas

Herramienta de código abierto para ayudar a astrónomos a encontrar, dentro de un
catálogo de curvas de luz, los objetos cuyo comportamiento no encaja con las
clases de variabilidad conocidas — candidatos a estudiar con prioridad.

Funciona sobre **cualquier catálogo de curvas de luz**, no solo sobre los que
trae de serie: añadir una fuente de datos nueva no requiere modificar el núcleo
del proyecto (ver [Añadir tu propia fuente de datos](#añadir-tu-propia-fuente-de-datos)).

---

## Características

- **Detección de anomalías no supervisada** — Isolation Forest y Local Outlier
  Factor, en modo global y normalizado por clase, con controles para separar
  anomalía real de sesgo conocido del clasificador.
- **Multi-fuente desde el diseño** — incluye adaptadores para StarEmbed (ZTF,
  ~40.000 curvas multibanda) y Gaia DR3 (consulta en vivo vía TAP + DataLink).
  Añadir un CSV o parquet propio no requiere escribir código.
- **Validación externa integrada** — cruce automático de candidatos con SIMBAD y
  VSX/AAVSO para comprobar si un objeto ya está catalogado.
- **Interfaz gráfica** (PySide6) además del uso como biblioteca de Python.
- **Metodología documentada y auditada** — incluyendo los resultados negativos.
  Ver [`docs/whitepaper.md`](docs/whitepaper.md).

## Instalación

```bash
git clone <url-del-repositorio>
cd Astronomia
pip install -r docs/requirements.txt
```

Requiere Python 3.9 o superior.

## Uso

### Interfaz gráfica

```bash
python app/main.py
```

Selecciona una fuente de datos, rellena sus parámetros, pulsa **Cargar y
analizar**, y explora los resultados ordenados por score de anomalía. Al
seleccionar un objeto se muestra su curva de luz.

> **Nota:** la interfaz está en su primera versión y todavía no se ha probado de
> forma exhaustiva en todos los entornos. Ver [Estado del proyecto](#estado-del-proyecto).

### Como biblioteca

```python
from anomaly_detector.adapters import cargar_train_test_starembed
from anomaly_detector import (
    extraer_features_dataset, detectar_anomalias, VARIABLES_BASE,
)

train, test = cargar_train_test_starembed(base_dir="ruta/a/los/parquet")

X_train = extraer_features_dataset(train)
X_test = extraer_features_dataset(test)

resultado = detectar_anomalias(
    X_train, X_test,
    VARIABLES_BASE + ["period"],
    clases_dificiles=["RRd", "RS CVn"],
)

candidatos = resultado[resultado["candidato_interesante"]]
```

Consultar Gaia DR3 en vivo:

```python
from anomaly_detector.adapters import cargar_gaia_dr3

dataset = cargar_gaia_dr3(max_objetos=500, umbral_confianza_clase=0.9)
```

## Añadir tu propia fuente de datos

### Opción 1 — CSV o parquet (sin escribir código)

Si tus datos están en formato "largo" (una fila por observación), copia
[`fuentes/EJEMPLO_generico.yaml`](fuentes/EJEMPLO_generico.yaml), ajusta el mapeo
de columnas, y tu fuente aparecerá automáticamente en la aplicación:

```yaml
nombre: "Mi Survey"
formato_datos: largo
acceso:
  formato: csv
mapeo_columnas:
  id_objeto: "ID"
  tiempo: "mjd"
  magnitud: "mag_r"
  error: "mag_r_err"
  clase_conocida: "var_type"
banda_fija: "r"
unidades:
  offset_tiempo: 0.0      # -2400000.5 para convertir JD a MJD
```

### Opción 2 — Fuentes con lógica propia (Python)

Para formatos anidados, APIs remotas o autenticación, hereda de
`AdaptadorDataset` y registra la clase:

```python
from anomaly_detector.adapters import AdaptadorDataset, registrar_fuente
from anomaly_detector import LightCurve

@registrar_fuente(id="mi_fuente", nombre="Mi Fuente", descripcion="...")
class MiAdaptador(AdaptadorDataset):
    def iter_curvas(self):
        for objeto in mis_datos:
            curva = LightCurve(
                id_objeto=objeto.id,
                tiempo=objeto.mjd,           # en días
                magnitud=objeto.mag,
                banda="r",
                clase_conocida=objeto.tipo,  # opcional
            )
            yield from self._emitir(curva)
```

Ver `anomaly_detector/adapters/starembed.py` (fichero local) y
`anomaly_detector/adapters/gaia.py` (servicio remoto con paginación) como
referencias completas.

## Estado del proyecto

| Componente | Estado |
|---|---|
| Pipeline de features (12 variables) | Validado sobre dos datasets reales |
| Detección de anomalías | Validada con ground truth real (AUC 0.66, ver nota) |
| Adaptador StarEmbed | Validado (33.000 objetos) |
| Adaptador Gaia DR3 | Validado (300 objetos, 100% de éxito) |
| Sistema de fuentes plegable | Funcional |
| Interfaz PySide6 | Primera versión — pendiente de pruebas en más entornos |

**Sobre el AUC de 0.66:** la detección se validó contra el split `anom` de
StarEmbed (1.087 objetos de 10 clases fuera de distribución). El rendimiento es
modesto y la causa está documentada: el recall es inversamente proporcional a la
cercanía taxonómica entre la clase anómala y las clases de entrenamiento. Las
estrellas Blazhko (RR Lyrae con modulación) apenas se detectan, mientras que las
binarias post-envolvente-común se detectan bien. Se investigaron y descartaron
tres hipótesis de mejora, todas documentadas en el whitepaper. Esta limitación es
estructural del enfoque basado en estadísticos agregados por curva.

## Documentación

- [`docs/whitepaper.md`](docs/whitepaper.md) — fundamento científico, metodología,
  resultados de validación (incluidos los negativos) y limitaciones conocidas.
- [`docs/FILES.md`](docs/FILES.md) — inventario completo de scripts y módulos, con
  su estado de auditoría.
- [`docs/workflow.md`](docs/workflow.md) — guía práctica de uso y diario de
  desarrollo del proyecto.

## Contribuir

Las contribuciones son bienvenidas. Dos consideraciones:

- **Fuentes declarativas (YAML)** — son solo datos, no ejecutan código. Basta con
  un pull request añadiendo el manifiesto.
- **Adaptadores en Python** — ejecutan código en la máquina de quien los use, así
  que se pide que el pull request incluya un script de verificación con un
  dataset de ejemplo pequeño, siguiendo el patrón de los ya existentes en
  `auditorias/`.

## Créditos y citas

Este proyecto usa datos de terceros. Si lo utilizas en un trabajo publicado, cita
también las fuentes originales:

- **StarEmbed** — Li et al. 2025, [arXiv:2510.06200](https://arxiv.org/abs/2510.06200)
- **ZTF** — Bellm et al. 2018, *PASP* 131, 018002
- **Catalina Surveys** — Drake et al. 2014, *ApJS* 213, 9
- **Gaia** — [ESA Gaia Archive](https://gea.esac.esa.int/archive/)
- **SIMBAD / VizieR** — Centre de Données astronomiques de Strasbourg
- **VSX** — AAVSO International Variable Star Index

## Licencia

Pendiente de definir.
