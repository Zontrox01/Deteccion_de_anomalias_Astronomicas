# Whitepaper — Detector de Anomalías Astronómicas en Curvas de Luz

**Estado:** v2.3 — primera versión de la interfaz PySide6 construida (sección 5.2); pendiente de probar en un entorno real con pantalla
**Última actualización:** ver historial de cambios al final del documento

---

## 1. Objetivo del proyecto

Construir una herramienta (aplicación de escritorio en PySide6) que, dado un dataset de
curvas de luz astronómicas — en principio cualquiera, no solo ZTF/StarEmbed — sea capaz
de señalar a un astrónomo qué objetos presentan un comportamiento anómalo respecto a las
clases de variabilidad conocidas, para que puedan priorizarse para estudio.

El objetivo **no** es construir el mejor clasificador posible de clases conocidas.
Ese es un problema instrumental, necesario para tener un espacio de referencia fiable,
pero no es el fin. El fin es la detección de lo que no encaja.

## 2. Alcance actual vs. alcance objetivo

| | Alcance actual (validado) | Alcance objetivo |
|---|---|---|
| Datos | ZTF / StarEmbed (train-*.parquet, test-*.parquet) | Cualquier dataset de curvas de luz, formato variable |
| Interfaz | Scripts Python sueltos, ejecución manual, CSV de salida | Aplicación PySide6, carga de dataset, resultados visuales |
| Features | 12 variables + `period` (cuando existe), fijas | Mismas 12 + `period`; extensibles |
| Detección | Isolation Forest + LOF, global y por clase | Igual, empaquetado como pipeline reutilizable |
| Documentación | Este conjunto de documentos | Se mantiene igual, actualizado por fases |

## 3. Decisión de arquitectura validada: la capa de adaptación

Para soportar "cualquier datasheet" sin reescribir el pipeline cada vez, la propuesta es
una capa de adaptación entre el dato crudo y el resto del sistema:

```
dataset externo (parquet/CSV/FITS/...)
        │
        ▼
  ADAPTADOR (por formato)  →  normaliza a esquema interno comun
        │
        ▼
  ESQUEMA INTERNO CANONICO:
    - id_objeto
    - tiempo[]        (mjd o equivalente)
    - magnitud[]       (o flujo)
    - error[]           (opcional)
    - banda             (opcional)
    - clase_conocida     (opcional, si el dataset viene etiquetado)
        │
        ▼
  EXTRACCION DE FEATURES (12 variables, validadas en esta sesion)
        │
        ▼
  DETECCION DE ANOMALIAS (Isolation Forest + LOF, global y por clase)
        │
        ▼
  INTERFAZ PYSIDE6 (listado de candidatos, curvas, exportacion)
```

Cada dataset nuevo solo necesita un adaptador nuevo (una funcion que traduzca sus
columnas al esquema canonico). El resto del sistema no cambia. Esto era una decision
de diseño propuesta, no cerrada, en la version original de este documento.

**Confirmada con evidencia (31/08/2026):** `AdaptadorGaiaDR3`
(`anomaly_detector/adapters/gaia.py`) es el segundo adaptador real del
proyecto, y deliberadamente distinto en arquitectura al primero -- consulta
dos servicios remotos (TAP + DataLink) en vez de leer un parquet local, y
la tabla de fotometria por epoca de Gaia tiene un formato "ancho"
completamente distinto al de StarEmbed (bandas G/BP/RP como columnas
separadas en la misma fila, no un diccionario anidado por banda).
Probado a pequeña escala (5 fuentes reales) el mismo dia: **el esquema
canonico y `extraer_features_dataset()` funcionaron sin ningun cambio de
codigo**, y las features resultantes distinguen correctamente, con sentido
fisico real, entre una eclipsante (ECL, `eta=2.01`, `maximum_slope=8.03`,
picos y caidas bruscas por el eclipse) y variables de periodo largo (LPV,
`eta` entre 0.48 y 0.74, variacion suave). Esta es la prueba de
generalizacion que motivo la decision de arquitectura -- queda **validada**,
no solo propuesta. Ambos adaptadores (StarEmbed y Gaia DR3) estan
registrados en el sistema de fuentes (`@registrar_fuente`, seccion 5.0);
`fuentes_disponibles()` los expone junto al adaptador tabular generico de
ejemplo.

**Prueba a mayor escala completada (300 objetos, 31/08/2026): 100% de exito.**
`probar_gaia_escala.py` -- 300/300 objetos solicitados terminaron con curva
valida, 900 curvas totales (3 bandas × 300), 0 lotes fallidos por red, 0
perdidas silenciosas, 0 `NaN` en las 12 variables. Confirma que la
paginacion por lotes (`Gaia.load_data()` en tandas de 200) funciona
correctamente fuera de una prueba minima.

**Rendimiento medido:** 271s para 300 objetos (~0.90 s/objeto). Para un
volumen de miles de objetos, esto implica 15-75+ minutos de carga --
informacion de diseño real para la interfaz: cargar Gaia en vivo necesitara
progreso visible o carga en segundo plano, a diferencia de StarEmbed (fichero
local, carga casi instantanea).

**Diversidad de clases confirmada a escala** (con umbral de confianza 0.9):
LPV (103), ECL (87), AGN (44), S (30), SOLAR_LIKE (9), RS (8), RR (8),
DSCT|GDOR|SXPHE (7), BE|GCAS|SDOR|WR (3), YSO (1). Destaca `AGN` -- un
nucleo galactico activo no es una estrella, y aun asi el esquema canonico lo
proceso sin ningun error: la generalizacion no se limita a subtipos
estelares, funciona igual con un fenomeno astrofisico fundamentalmente
distinto. Es la prueba de generalizacion mas fuerte obtenida en toda la
sesion.

**Estadisticas de las 12 variables a escala:** rangos sanos, sin valores
degenerados ni infinitos (eta entre 0.08 y 2.91, kurtosis con cola pesada
hasta 36.5 -- plausible en astronomia, no un error de calculo). Sin ningun
`NaN`.

**Fase de Gaia DR3 cerrada.** No se considera necesaria una prueba a escala
todavia mayor (miles de objetos) antes de empezar a usar Gaia desde la
interfaz -- el resultado a 300 objetos, multi-lote, con 100% de exito, es
suficiente evidencia para la fase actual del proyecto.

## 4. Metodologia validada (resumen de la fase de auditoria)

Esta seccion resume, sin repetir el detalle completo, lo que se ha verificado con
evidencia (logs y CSV de ejecucion real) sobre el dataset ZTF/StarEmbed antes de
construir el detector de anomalias. El objetivo de incluirlo aqui es que cualquier
adaptador futuro para otro dataset pueda comprobar los mismos puntos.

### 4.1 Espacio de features (12 variables)

`median`, `standard_deviation`, `median_absolute_deviation`, `amplitude`,
`percent_amplitude`, `inter_percentile_range_25` (percentiles 12.5-87.5),
`skew`, `kurtosis`, `stetson_K` (version aproximada, sin errores fotometricos
por punto), `eta` (von Neumann ratio), `chi2`, `maximum_slope`.

Opcionalmente, `period`, cuando el dataset de origen lo proporciona precalculado
(no se recalcula internamente).

**Importancia relativa (ablacion, experimento 18):** `eta` es, con diferencia, la
variable mas influyente (retirarla cuesta 3-4 veces mas balanced accuracy que la
siguiente). `stetson_K` es la segunda mas influyente y la mas estable frente a
cambios en el numero de observaciones (experimento 23).

**Matiz importante (scripts 17 y 19):** esto es importancia *marginal dentro
del conjunto completo* (cuanto cuesta retirarla), no poder predictivo en
solitario. Usada sola, `eta` rinde de forma mediocre (balanced accuracy RF
~0.26-0.28) — `median_absolute_deviation` es la variable individualmente mas
fuerte (~0.32). La seleccion progresiva (`19`) confirma que `eta` aporta un
salto de señal notable en combinacion con otras (al añadirla a un subconjunto
ya seleccionado), pero no es la mejor variable univariante. Interpretacion:
`eta` es poco redundante con el resto del conjunto (aporta informacion que
ninguna otra variable cubre), lo que explica su alto coste de retirada en la
ablacion sin que sea, por si sola, la señal mas discriminante.

### 4.2 Controles de integridad de datos (superados)

- **Orden temporal** (16b): las curvas de ZTF/StarEmbed estan correctamente
  ordenadas cronologicamente (0 inversiones en 8.000/8.000 objetos de TEST).
  `eta` es de fiar.
- **Padding de cola** (16c): existe en el 1.1% de los objetos, impacto
  despreciable en features y clasificacion.
- **Empates de timestamp intermedios** (16d): explicados por cuantizacion
  deliberada de `mjd` a una rejilla de 1/256 dia (~5.6 min) en el propio
  dataset. El 95% son mediciones reales distintas, no duplicados. Efecto
  secundario: `maximum_slope` pierde de media ~16% de los pares por el filtro
  `dt > 0`, lo que probablemente explica su baja importancia relativa.
- **Leakage TRAIN/TEST** (12, 13): 55 curvas duplicadas detectadas por hash
  criptografico y eliminadas de TEST; el impacto en accuracy es marginal
  (0.9131 → 0.9128), confirmando que el leakage no estaba inflando resultados.
- **Representatividad del truncado `iloc[:25000]`/`iloc[:8000]`** (20b):
  confirmado mediante chi-cuadrado (p=0.99 TRAIN, p=0.9996 TEST) y analisis
  por deciles de posicion que el parquet ya viene efectivamente barajado.
  No hace falta re-muestrear.

### 4.3 Clasificador supervisado de referencia (control, no el objetivo final)

Random Forest (`n_estimators=200`, `class_weight="balanced_subsample"`,
`max_features="sqrt"`) sobre las 12 variables (+ `period` cuando se usa).
Balanced accuracy de referencia: ~0.64 sin `period`, ~0.73-0.74 con `period`.

**Limitacion conocida y persistente:** RRd y RS CVn se clasifican mal en
*todos* los regimenes de numero de observaciones (experimento 22), incluido
el de ≥1000 observaciones. No es un artefacto de muestra pequeña — es
solapamiento real en el espacio de features con otras clases. Cualquier
candidato a anomalia de estas dos clases debe tratarse con cautela adicional
(ver 4.4).

### 4.4 Deteccion de anomalias (scripts 24, 24b)

Dos detectores no supervisados, complementarios:

- **Isolation Forest**: anomalias globales (facilmente aislables).
- **Local Outlier Factor** (modo `novelty`): anomalias locales (densidad baja
  respecto a vecinos).

**Dos variantes, ambas necesarias:**

- **Global** (24): cada objeto de TEST se compara contra todo TRAIN.
- **Por clase** (24b): cada objeto se compara solo contra los de su propia
  clase, con el score normalizado a percentil dentro de la clase. Esto
  neutraliza el sesgo de que las clases pequeñas (p. ej. LPV, ~1% de TRAIN)
  parezcan anomalas por pura escasez de referencia, no por comportamiento real.

**Hallazgo de validacion clave:** en la primera ejecucion sobre StarEmbed,
LPV aparecia sobrerrepresentada 15-18x en el metodo global. El metodo por
clase (24b) elimino completamente esa señal (ratio 0.0x, 0 de 4 candidatos
LPV sobreviven al control) — confirmando que era un artefacto de tamaño de
muestra, no una anomalia real. Al mismo tiempo, el metodo por clase reveló
una señal nueva en RRc (0 candidatos en el metodo global, 10 en el metodo
por clase) — outliers dentro de una clase grande y aparentemente homogenea,
invisibles para el metodo global.

**Criterio de candidato robusto:** interseccion de ambos metodos (global Y
por clase), ademas de no ser clase dificil (RRd/RS CVn) ni estar mal
clasificado por el RF de control. Esto separa anomalia real de sesgo de
clasificador conocido.

**Pendiente de cierre:** validacion externa (cruce con catalogos VSX/SIMBAD/
Gaia DR3 variability) de los candidatos que sobreviven a ambos metodos, y/o
inspeccion visual de sus curvas de luz. Sin esto, "candidato" sigue
significando "estadisticamente atipico dentro de este espacio de 12
variables", no "anomalia astronomica confirmada".

### 4.5 Interseccion de metodos (script derivado, no un experimento nuevo)

Cruzando las salidas del 24 y el 24b (`24c_candidatos_interseccion.csv`) sobre
TEST de StarEmbed:

- **15 candidatos "robustos"**: fuertes en el metodo global Y en el metodo
  por clase simultaneamente.
- De esos, **8 son "robustos e interesantes"**: ademas no son clase dificil
  ni estan mal clasificados por el RF de control. Esta es la lista principal
  de salida de esta fase sobre StarEmbed.
- Los otros **7** son casos donde el RF se equivoca de clase Y ambos
  detectores de anomalias coinciden en marcarlos. Se guardan aparte
  (`candidato_robusto_interesante=False` en el mismo CSV) porque son
  ambiguos por diseño: podrian ser sesgo de clasificador disfrazado de
  anomalia (el patron que se ha descartado varias veces en esta auditoria),
  o podrian ser objetos genuinamente en la frontera entre clases conocidas
  — precisamente el perfil que cabria esperar de una clase no representada
  en el dataset de entrenamiento. No hay evidencia todavia para decidir
  cual de las dos explicaciones es la correcta en cada caso.

**Decision de producto (no de pipeline):** el pipeline calcula y expone
siempre ambos grupos completos; nunca decide ocultar los ambiguos. La
decision de que se muestra por defecto es de la interfaz (ver seccion 5).

**Cierre: validacion externa completada** (`cruzar_candidatos_catalogos.py`,
ver FILES.md seccion 7). Los 8 candidatos "interesantes" se cruzaron contra
SIMBAD y VSX/AAVSO usando coordenadas parseadas del propio identificador
`CSS_J...` (convencion Catalina Surveys). **Resultado: 8/8 con contrapartida
en ambos catalogos**, con clasificacion consistente en las tres fuentes
(SIMBAD, VSX, RF propio) para los 8 casos -- 7 son `EB*`/`EW` (binarias
eclipsantes de contacto), y uno, `CSS_J175050.6+500249`, resulto ser
`MX Her`, una estrella con nombre propio ya estudiada (`SB*` en SIMBAD,
`EA/SD` en VSX, coherente con la prediccion `EA` del RF).

**Ningun candidato de este lote es un objeto genuinamente no catalogado ni
de tipo taxonomico nuevo.** No es un resultado negativo: confirma que el
detector encuentra objetos reales y correctamente identificados por fuentes
externas independientes, no ruido estadistico -- pero en esta tanda concreta
son casos extremos dentro de clases conocidas (probablemente EW con
morfologia de curva inusual), no descubrimientos. Coherente con los limites
del espacio de 12 variables ya documentados en la seccion 4.9: el enfoque
detecta bien lo que se aparta del comportamiento tipico dentro del rango de
clases conocidas, no esta diseñado para encontrar una clase que no exista
en absoluto en ese espacio.

### 4.6 Verificacion del paquete reutilizable contra el metodo original

Antes de construir `deteccion.py` se verifico que `anomaly_detector/features.py`
(la reescritura de las 12 variables sobre el esquema canonico) reproduce
exactamente los mismos numeros que el metodo original de 20-24b, objeto a
objeto, sobre los datos reales de StarEmbed (`verificar_paquete.py`).

**TEST: verificado sin excepciones** — 8.000/8.000 objetos, 13 variables
(12 + `period`), diferencia maxima 0.0 en todas.

**TRAIN: verificado con una excepcion documentada** — 24.999/25.000 objetos
(el id_objeto restante es un duplicado exacto en el parquet de origen, mismo
objeto repetido con features identicas; se trata por separado, ver mas abajo)
dan numeros identicos. Un unico objeto, `CSS_J053809.1-070525`, difiere en
las 12 variables a la vez (no en una sola, lo que ya apuntaba a un problema
de datos de entrada, no de formula):

- El metodo original elige su banda `'r'`, que tiene **un solo punto finito**.
  El criterio original (`obtener_banda`) solo comprobaba que la clave de la
  banda existiera y no fuera `None` -- nunca comprobo si habia datos
  *suficientes*. Con 1 punto, `standard_deviation=0`, `skew`/`kurtosis` no
  tienen significado estadistico, y `eta` se calcula sobre un `diff` de
  longitud 0. Este objeto ha estado produciendo features degeneradas desde
  el experimento 20 sin que ninguna metrica agregada (balanced accuracy,
  chi-cuadrado del 20b, etc.) lo hiciera visible -- el efecto de 1 objeto
  entre 25.000 es invisible en cualquier metrica global.
- El paquete nuevo, via `LightCurve.validar()` (minimo 2 puntos finitos),
  descarta esa banda y cae correctamente a `'g'`, que tiene 37 puntos reales.

**Veredicto:** comportamiento nuevo mas correcto que el original, no un bug
del refactor. Se documenta aqui como excepcion aceptada (tambien registrada
en el `EXCEPCIONES_CONOCIDAS` de `verificar_paquete.py`, para que
ejecuciones futuras del script de verificacion no la reporten como fallo).
No se ha corregido retroactivamente en los resultados de 20-24b -- el
impacto de 1 objeto sobre 25.000 en metricas agregadas es nulo, y no
justifica reabrir esa fase ya cerrada.

**Duplicado detectado como efecto secundario:** el proceso de verificacion
tambien revelo que `CSS_J164349.2+401208` aparece dos veces en el parquet
de TRAIN, con features identicas en ambas apariciones (mismo objeto
repetido, no dos objetos distintos con id compartido por error). El paquete
nuevo, via `Dataset.ids_unicos()`, ya trata esto correctamente (conserva
una sola aparicion). No requiere accion adicional.

**Conclusion:** `anomaly_detector/features.py` queda verificado como
sustituto fiel (y, en el caso encontrado, mas correcto) del metodo usado en
20-24b. Seguro continuar construyendo `deteccion.py` sobre el.

### 4.7 Verificacion end-to-end del pipeline completo (features + deteccion)

Con `features.py` ya verificado (seccion 4.6), se verifico tambien
`deteccion.py` de punta a punta (`probar_deteccion.py`): pipeline completo
(adaptador -> features -> `detectar_anomalias()`, con `period` incluido)
sobre los datos reales de StarEmbed, comparado contra el resultado ya
conocido de 24+24b+24c (`24c_candidatos_interseccion.csv`).

**Resultado:** de los 15 candidatos "robustos" y 8 "interesantes" ya
conocidos, el pipeline empaquetado reproduce el 100% (ninguna perdida),
añadiendo 1 candidato robusto y 2 interesantes adicionales en el margen.
Diferencias de este tipo son esperables: Isolation Forest (muestreo
bootstrap interno) y LOF con `n_jobs=-1` no son perfectamente
deterministas incluso con `random_state` fijo, y con umbrales tan ajustados
(top 2% / percentil 98 sobre 8.000 objetos) algun objeto en el limite puede
cruzar la linea entre ejecuciones. Balanced accuracy del RF de control:
0.7395 (referencia: 0.7348) -- tambien coherente.

Uno de los dos candidatos "interesantes" nuevos, `CSS_J210351.9-005926`,
era antes uno de los 7 casos "ambiguos" (mal clasificado por el RF en la
ejecucion original) -- en esta ejecucion el RF si lo clasifica bien,
consistente con las pequeñas correcciones de calculo de features que ya
trae el paquete nuevo (ver seccion 4.6).

**Veredicto: pipeline completo (`anomaly_detector`) verificado de punta a
punta.** Seguro construir la interfaz PySide6 encima, o extender a un
segundo dataset.

### 4.8 Descubrimiento: StarEmbed publica un cuarto split, `anom`, con verdad fundamental real para OOD

Hasta ahora, todo el trabajo de deteccion de anomalias (4.4-4.7) operaba sin
verdad fundamental -- un candidato era "estadisticamente atipico dentro del
espacio de 12 variables", pendiente de validacion externa manual (cruce de
catalogos, inspeccion visual). Al inspeccionar el dataset descargado
completo se ha encontrado que StarEmbed publica, ademas de `train`/
`validation`/`test`, un cuarto split llamado `anom`.

**Verificado con `summary.anom.json`:** 1.087 objetos, 10 clases, ninguna
de las 7 clases de entrenamiento. La suma de `classCounts` (261+202+150+
141+108+77+61+57+23+7) es exactamente 1.087 -- consistencia interna
completa, no una muestra parcial:

| Clase | N | % |
|---|---|---|
| Beta_Lyrae | 261 | 24.0% |
| HADS | 202 | 18.6% |
| ELL | 150 | 13.8% |
| EA_UP | 141 | 13.0% |
| Cep-II | 108 | 9.9% |
| PCEB | 77 | 7.1% |
| Blazhko | 61 | 5.6% |
| ACEP | 57 | 5.2% |
| Hump | 23 | 2.1% |
| LADS | 7 | 0.6% |

**Confirmado contra la fuente primaria** (arXiv 2510.06200, "StarEmbed:
Benchmarking Time Series Foundation Models on Astronomical Observations
of Variable Stars"): los propios autores construyen este split a partir
de estrellas del catalogo CSPVS con demasiados pocos ejemplos como para
entrenar, y las definen explicitamente como fuentes fuera de distribucion
(OOD) para benchmarking. Las 10 clases citadas en el paper coinciden
exactamente, sin excepciones, con las de `summary.anom.json`.

**Metodo de referencia de los propios autores:** entrenan un Isolation
Forest independiente para cada una de las 7 clases de entrenamiento sobre
los embeddings, y usan el minimo de esos 7 scores como score de anomalia
final ("multi-class isolation forest", Gupta et al. 2025). Esto es
metodologicamente muy cercano al 24b (deteccion por clase) -- la
diferencia esta en la regla de combinacion: el 24b cruza global+por-clase
con percentiles, el paper toma el minimo de los scores por clase. Es una
referencia citable con la que comparar el diseño propio.

**Significado de las clases** (confirmado donde ha sido posible; no se
afirma con seguridad lo que no se ha podido verificar): HADS/LADS son
Delta Scuti de alta/baja amplitud respectivamente (division estandar por
amplitud de pulsacion, confirmado en literatura ZTF-CPVS); Cep-II son
Cefeidas Tipo II; ACEP, Cefeidas Anomalas; ELL, variables elipsoidales
(binarias deformadas por marea, sin eclipses); PCEB, binarias
post-envolvente-comun; Beta_Lyrae, binarias semi-separadas (clase propia,
distinta de EA/EW). **"EA_UP" y "Hump" no se han podido confirmar con
certeza** -- el paper las nombra pero no se ha localizado la definicion
exacta del criterio de clasificacion en las fuentes consultadas. No usar
estos dos nombres en comunicacion externa sin verificar antes.

**Impacto en el proyecto:** esto convierte la deteccion de anomalias de
"candidatos sin verdad fundamental" a "validable con ground truth real".
Pendiente (proximo paso recomendado, ver seccion 7): construir un script
que cargue `train` + `anom` via `anomaly_detector` y mida cuantos de los
1.087 objetos de `anom` quedarian correctamente señalados por
`detectar_por_clase()` -- la primera validacion con verdad fundamental de
todo el proyecto.

**Esquema confirmado con la documentacion oficial del dataset** (README de
StarEmbed en HuggingFace, transcrito integramente por el autor del
proyecto):

| Columna (nivel objeto) | Tipo | Descripcion |
|---|---|---|
| `sourceid` | string | id CSDR1 del objeto (p. ej. `CSS_J082956.4-044426`) |
| `bands_data` | dict (claves `g`, `i`, `r`) | curva de luz por banda, o `null` si falta |
| `period` | float64 | periodo catalogado, dias (rango ≈ 0.13-885) |
| `class_str` | string | EW, EA, RRab, RRc, RRd, RS CVn, LPV (train/val/test) |
| `ra`, `dec` | float64 | coordenadas J2000, grados decimales |

| Campo (dentro de cada banda) | Tipo | Descripcion |
|---|---|---|
| `target` | list[float] | magnitud (sistema AB) -- la curva de luz cruda |
| `past_feat_dynamic_real` | list[float] | `mag_err`, incertidumbre 1σ, alineada con `target` |
| `feat_dynamic_real` | list[float] | `delta_t` ya precalculado entre observaciones consecutivas |
| `mjd` | list[float] | tiempo de observacion (MJD), alineado con `target` |
| `length` | int64 | numero de observaciones de esa banda |

**Confirmado: las claves de `bands_data` son `g`, `i`, `r` -- exactamente
las que ya usa `AdaptadorStarEmbed` (`BANDAS_PRIORIDAD`).** El aviso
tecnico de una version anterior de este documento, que sugeria revisar
un posible mapeo `g_ZTF`/`i_ZTF`/`r_ZTF`, queda descartado: ese nombre
solo aparecia como etiqueta de visualizacion en `summary.anom.json`, no
como clave real del struct. **`AdaptadorStarEmbed` deberia funcionar
sobre `anom` sin ningun cambio.**

**Citas de la fuente** (obligatorias si el proyecto se publica, dataset
curado a partir de dos trabajos previos):

```bibtex
@article{StarEmbed,
  author = {{Li}, Weijian and {Chen}, Hong-Yu and {Rehemtulla}, Nabeel and
            {Shah}, Ved G. and {Wu}, Dennis and {Kim}, Dongho and
            {Lin}, Qinjie and {Miller}, Adam A. and {Liu}, Han},
  title = "{StarEmbed: Benchmarking Time Series Foundation Models on
            Astronomical Observations of Variable Stars}",
  journal = {arXiv e-prints}, year = 2025, eid = {arXiv:2510.06200},
  doi = {10.48550/arXiv.2510.06200}, archivePrefix = {arXiv},
  eprint = {2510.06200}, primaryClass = {astro-ph.SR}
}
@article{bellm2018zwicky,
  title={The Zwicky Transient Facility: system overview, performance,
         and first results},
  author={Bellm, Eric C and Kulkarni, Shrinivas R and Graham, Matthew J
          and others},
  journal={Publications of the Astronomical Society of the Pacific},
  volume={131}, number={995}, pages={018002}, year={2018}
}
@article{drake2014catalina,
  title={The catalina surveys periodic variable star catalog},
  author={Drake, AJ and Graham, MJ and Djorgovski, SG and others},
  journal={The Astrophysical Journal Supplement Series},
  volume={213}, number={1}, pages={9}, year={2014}
}
```

### 4.9 Resultado real de la validacion contra ground truth (`validar_contra_anom.py`)

Primera validacion de todo el proyecto contra verdad fundamental real,
ejecutada sobre datos reales de StarEmbed (`detectar_ood_multiclase()`,
7 detectores por clase de TRAIN, score minimo, con `period` incluido en
el espacio de features).

**Resultado global:**

| Metrica | Valor |
|---|---|
| AUC-ROC (TEST vs ANOM) | 0.6608 |
| Recall en ANOM a FP=1% (en TEST) | 3.96% |
| Recall en ANOM a FP=5% (en TEST) | 16.38% |
| Recall en ANOM a FP=10% (en TEST) | 27.78% |
| Recall en ANOM a FP=20% (en TEST) | 40.94% |

Un AUC de 0.66 es modesto -- muy por debajo del rendimiento en el caso
sintetico de control (0.99, seccion de pruebas de `deteccion.py`), y
lejos de una separacion fuerte. **No se maquilla este resultado: es la
cifra real, y es mas valiosa que un numero alto artificial porque
apunta a una limitacion concreta y explicable, no a ruido.**

**La causa tiene una explicacion astrofisica clara, verificada cruzando
`clase_real` contra `clase_mas_cercana`** (la clase de TRAIN cuyo
detector dio el score minimo, es decir, la que el sistema considera que
mejor "explica" a cada objeto OOD):

| Clase OOD | Recall (FP=5%) | Clase mas cercana dominante | Relacion taxonomica |
|---|---|---|---|
| PCEB | 40.3% | Repartida (sin dominancia clara) | Sin pariente cercano entre las 7 clases |
| EA_UP | 34.8% | EA (83%) | Subtipo peculiar de EA -- coherente con el propio nombre |
| Cep-II | 27.8% | RS CVn (44%), LPV (36%) | Cefeida, periodo largo -- se acerca a LPV en parte |
| ACEP | 26.3% | RRab (53%), RS CVn (23%) | Cefeida anomala, rango de periodo solapa con RRab |
| LADS | 14.3% | RS CVn (86%, N=7 muy bajo) | Delta Scuti baja amplitud |
| Beta_Lyrae | 10.7% | EA (43%), RS CVn (34%) | Binaria semi-separada -- pariente directo de EA |
| HADS | 5.9% | RS CVn (68%) | Delta Scuti alta amplitud |
| ELL | 5.3% | RS CVn (62%), EW (31%) | Variable elipsoidal, baja amplitud -- parecida a EW/RS CVn |
| Blazhko | 4.9% | RRab (90%) | **Es RR Lyrae ab con modulacion** -- pariente directo, casi identica en 12 estadisticos agregados |
| Hump | 4.3% | RS CVn (39%) | Clase sin definicion confirmada (ver seccion 4.8) |

**Patron claro y consistente: el recall es inversamente proporcional a
la cercania taxonomica con las 7 clases de entrenamiento.** Blazhko es,
por definicion, una RR Lyrae ab con modulacion de amplitud/fase en
escalas de semanas-meses -- una diferencia que 12 estadisticos
agregados por curva apenas capturan, de ahi el recall mas bajo de
todas (4.9%) y el 90% de asignacion a RRab. En el otro extremo, PCEB
(binarias post-envolvente-comun, sin pariente directo entre las 7
clases) es la mejor detectada (40.3%) precisamente porque no se
parece lo bastante a ninguna clase de referencia como para que ningun
detector la acepte como normal.

**Interpretacion honesta:** el espacio de 12 variables agregadas tiene
un techo estructural para detectar variantes sutiles *dentro* de una
familia de variabilidad conocida (Blazhko dentro de RR Lyrae, ELL
dentro del espacio EW/RS CVn). Detecta razonablemente bien
comportamientos *genuinamente* distintos (PCEB, EA_UP). Esto no es un
fallo del pipeline -- es el limite real y explicable de un enfoque
basado en estadisticos resumen de la curva, y es informacion valiosa
de cara a decidir si merece la pena ampliar el espacio de features
(por ejemplo, con coeficientes de Fourier del plegado en fase, que
capturarian mejor la forma fina de la curva) antes de confiar el
descubrimiento real en este detector tal como esta hoy.

**Hipotesis descartada con evidencia (no queda pendiente):** se probo
normalizar cada `score_<clase>` a percentil (0-100) antes de tomar el
minimo, por si la comparabilidad entre detectores de tamaños de clase muy
distintos (RRd ~85 objetos en TRAIN, EW miles) estuviera distorsionando
el `score_min` bruto. Resultado, calculado post-hoc sobre
`validacion_anom_scores.csv` sin necesidad de reentrenar: **el metodo
normalizado da PEOR resultado** (AUC 0.6082 frente a 0.6608 bruto;
recall a FP=5% 10.76% frente a 16.38% bruto). Explicacion: las clases con
pocos objetos de entrenamiento producen arboles de Isolation Forest poco
discriminativos, con puntuaciones brutas comprimidas en un rango
estrecho — casi ruido. Al convertir eso a percentil, ese ruido se
disfraza de señal significativa (cualquier conjunto tiene "un percentil
alto" por definicion). El metodo bruto, sin buscarlo, ya dejaba que los
detectores de clases grandes y bien entrenadas dominaran el minimo, en
vez de dar el mismo peso a un detector construido con ~85 ejemplos. El
patron de cercania taxonomica (seccion 4.9, tabla de `clase_mas_cercana`)
sigue siendo la explicacion dominante del AUC=0.66 -- no habia un
artefacto metodologico escondido detras.

**Contribucion de `period` al AUC, aislada (`comparar_periodo_anom.py`):**

| Metrica | Sin `period` | Con `period` | Diferencia |
|---|---|---|---|
| AUC-ROC global | 0.6442 | 0.6608 | +0.0166 |

A nivel agregado la diferencia parece pequeña, pero **esconde un efecto
real y contrapuesto que solo aparece al desglosar por clase OOD**:

| Clase OOD | Recall sin `period` | Recall con `period` | Diferencia |
|---|---|---|---|
| ACEP | 15.8% | 26.3% | **+10.5** |
| Cep-II | 17.6% | 27.8% | **+10.2** |
| ELL | 4.7% | 5.3% | +0.7 |
| PCEB | 40.3% | 40.3% | 0.0 |
| HADS | 5.9% | 5.9% | 0.0 |
| LADS | 14.3% | 14.3% | 0.0 |
| Beta_Lyrae | 11.1% | 10.7% | -0.4 |
| Blazhko | 6.6% | 4.9% | **-1.6** |
| Hump | 8.7% | 4.3% | -4.3 |
| EA_UP | 40.4% | 34.8% | **-5.7** |

**Interpretacion, clase por clase:**
- **ACEP y Cep-II mejoran mucho** (cefeidas, periodos tipicamente mucho
  mas largos que los de las 7 clases de entrenamiento -- el periodo por
  si solo las delata).
- **EA_UP empeora**, y es la que mejor se detectaba de las diez sin
  `period`: coherente con ser un subtipo de EA cuyo periodo cae dentro
  del rango normal de EA, asi que añadir `period` la acerca mas al
  detector de EA en vez de alejarla.
- **PCEB, HADS, LADS no cambian en absoluto** (recall identico hasta el
  ultimo decimal) -- `period` no mueve ni un objeto de lado del umbral
  para estas tres.
- **Blazhko empeora.** Esto responde directamente a la pregunta que
  motivo el experimento: Blazhko no se confunde con RRab por el periodo
  -- comparten practicamente el mismo rango por definicion (Blazhko es
  RRab con modulacion). `period` no solo no ayuda, activamente confunde
  un poco. El problema de Blazhko es de forma/modulacion de la curva, no
  de periodo.

**Conclusion, con evidencia y no solo intuicion:** para las tres clases
mas dificiles (Blazhko, HADS, ELL), `period` no resuelve nada -- en el
mejor caso (ELL) el efecto es insignificante, en el peor (Blazhko)
empeora. Esto justifica la tarea 2 (features de forma de curva): el
problema de esas tres clases es de forma, no de periodo.

**Resultado (`probar_features_forma.py`, 3 armonicos): hipotesis REFUTADA.**

| Configuracion | AUC |
|---|---|
| BASE (12 vars + period) | 0.6608 |
| BASE + Fourier (6 vars mas) | 0.6576 (peor) |
| SOLO Fourier | 0.5121 (practicamente azar) |

Las tres clases que motivaron el experimento **empeoran, no mejoran**:
Blazhko -3.3 puntos de recall (de 4.9% a 1.6%, la peor de las diez),
HADS -1.5, ELL -1.3. El daño es aun mayor en clases que antes iban
razonablemente bien gracias a `period` (seccion anterior): ACEP -22.8
puntos, Cep-II -17.6, EA_UP -10.6. Solo `Hump` mejora de forma notable
(+13 puntos), pero es la clase de definicion taxonomica no confirmada
(ver seccion 4.8) y con N=23 -- no se sobre-interpreta un resultado
aislado con esa base.

**Causa mas probable, sin necesidad de otra ronda para confirmarla:**
`SOLO_FOURIER` da AUC=0.51 (las 6 variables, por si solas, apenas
separan nada) y el daño se concentra justo en las clases donde `period`
si aportaba señal real -- consistente con dilucion por dimensionalidad:
añadir 6 dimensiones de señal debil a un espacio de 13 que ya
funcionaba diluye la señal existente, especialmente en las clases de
TRAIN mas pequeñas (RRd, RS CVn), donde Isolation Forest ya tenia poco
margen para discriminar con 13 dimensiones y tiene menos aun con 19.

**Decision: se cierra esta linea de investigacion sin una cuarta
iteracion (mas o menos armonicos, otra parametrizacion).** Continuar
ajustando esta misma tecnica indefinidamente seria precisamente el
patron de "bucle de validacion" que la auditoria original del proyecto
identifico como riesgo a evitar (ver seccion 8). Con tres experimentos
consecutivos bien documentados (normalizacion por percentil: refutada;
`period`: efecto real pero heterogeneo, no resuelve las tres clases
dificiles; forma de curva via Fourier: refutada, empeora las tres
clases dificiles), la conclusion que sostienen los datos es que
**AUC≈0.66 es, con el enfoque actual de estadisticos agregados por
curva, un techo estructural real** -- no un problema de ajuste fino
pendiente de resolver con una variante mas. Subirlo de forma sustancial
probablemente requeriria un enfoque distinto (representaciones
aprendidas de la serie temporal completa, no estadisticos hechos a
mano -- literalmente el tipo de modelos que el paper de StarEmbed
compara, ver seccion 4.8), que queda fuera del alcance actual del
proyecto y no se persigue en esta fase.

## 5. Diseno de la interfaz PySide6 — decisiones tomadas

### 5.0 Sistema de fuentes de datos plegable (decision de arquitectura, previa a la interfaz)

Motivada por que el software sera de codigo abierto en un repositorio publico,
abierto a contribuciones de la comunidad cientifica. Un selector de "Fuente
de datos" en la interfaz que solo soporte los adaptadores que yo mismo he
escrito no demuestra generalizacion real -- lo que hace falta es un
**mecanismo de extension** que cualquiera pueda usar sin tocar el nucleo del
proyecto ni requerir una nueva release para anadir un dataset.

**Dos niveles de fuente, segun la complejidad real del formato de origen:**

- **Nivel 1 — declarativas (YAML, sin escribir Python).** Para datasets
  tabulares donde cada fila es un punto de una curva (formato "largo": una
  fila por par objeto-epoca, el patron mas comun en exports CSV de surveys
  reales) y las columnas solo necesitan renombrarse/reescalarse. Un
  manifiesto YAML describe el mapeo de columnas al esquema canonico; lo
  interpreta un unico adaptador generico (`AdaptadorTabularGenerico`) que ya
  existe y sirve para cualquier manifiesto de este tipo. El usuario nunca
  toca codigo Python.
- **Nivel 2 — Python.** Para fuentes con logica real que un mapeo declarativo
  no puede expresar: paginacion contra un servicio remoto (Gaia DR3 via
  DataLink), estructuras anidadas (StarEmbed, bandas como diccionario),
  autenticacion, reintentos. Sigue siendo una clase que hereda de
  `AdaptadorDataset`, igual que el adaptador de StarEmbed ya construido.

**Registro comun:** `fuentes_disponibles()` combina ambos niveles (escanea
manifiestos YAML de un directorio + adaptadores Python registrados con un
decorador) y devuelve una lista de `FuenteDatos`, cada una con sus
`ParametroConfig` — la interfaz PySide6 genera el formulario de parametros
(rutas, credenciales) dinamicamente a partir de eso, sin tener logica
especifica de ninguna fuente hardcodeada en la UI. El selector de "Fuente de
datos" de la interfaz simplemente itera sobre esa lista.

**Gobernanza para un repositorio abierto:** las fuentes declarativas (YAML)
son intrinsecamente seguras (datos, no ejecutan nada) y se pueden aceptar
por PR con friccion minima. Las fuentes Python ejecutan codigo arbitrario en
la maquina de quien las use, asi que se exige el mismo estandar que ya se
aplico a StarEmbed en esta sesion: cualquier adaptador Python nuevo debe
venir acompañado de un script de verificacion (patron de
`verificar_paquete.py`) con un dataset de ejemplo pequeño, para que el CI del
repositorio pueda comprobar automaticamente que construye `LightCurve`
validas antes de aceptar el PR.

**Como encaja Gaia DR3 en esto:** sigue siendo el segundo dataset objetivo
(ver seccion 4, fuentes de datos), pero ahora se construye como la primera
fuente de Nivel 2 (requiere paginacion real contra DataLink), y se acompaña
de un manifiesto YAML de ejemplo (Nivel 1) para demostrar que el mecanismo
cubre ambos casos desde el principio.

### 5.1 Interfaz PySide6 — otras decisiones tomadas

- **Dos secciones siempre presentes**: "candidatos seguros" (robustos e
  interesantes) y "candidatos ambiguos" (robustos pero con clasificador
  dudoso). El pipeline nunca filtra los ambiguos fuera de la salida; la app
  decide como presentarlos, no si existen.
- **Comportamiento adaptativo de la seccion de ambiguos**: si el numero de
  candidatos seguros esta por debajo de un umbral configurable (no fijo en
  codigo, ajustable desde la propia interfaz), la seccion de ambiguos se
  muestra expandida por defecto. Si hay suficientes seguros, la seccion de
  ambiguos existe igual pero colapsada, con una etiqueta visible del tipo
  "+ N candidatos ambiguos" para que sigan siendo accesibles en un clic.
- Motivo: un umbral fijo que oculta candidatos por completo es fragil en los
  casos limite (una ejecucion con 9 candidatos "seguros" se comportaria de
  forma muy distinta a otra con 11, sin ninguna razon cientifica para ese
  corte) y, en datasets con mas solapamiento de clases que StarEmbed, podria
  descartar sistematicamente el tipo de candidato mas interesante — objetos
  en la frontera entre clases conocidas.

### 5.2 Primera implementacion real (`app/`, 31/08/2026)

Construida la primera version de la interfaz, con una separacion de
arquitectura deliberada: toda la logica de negocio vive en
`app/controlador.py`, **sin ningun import de PySide6** -- es la unica
forma de poder probar codigo real en un entorno sin pantalla (el
mismo criterio que ya se aplico en `deteccion.py`/`features.py`: lo
que no depende de Qt/red se prueba de verdad, lo que si depende se
verifica con el maximo cuidado posible y se marca explicitamente como
pendiente de confirmar en ejecucion real).

**Probado de extremo a extremo:** `controlador.py` (carga de fuente,
extraccion de features, deteccion, tabla de resultados, obtencion de
curva para grafico, exportacion), con datos sinteticos a traves del
adaptador tabular generico -- un objeto con amplitud 10x inyectado
queda correctamente primero en la tabla por `score_anomalia`.

**NO probado en ejecucion** (sin PySide6 ni pantalla en el entorno de
desarrollo): `main.py`, `ventana_principal.py`, `modelo_resultados.py`,
`widgets/formulario_fuente.py`, `widgets/grafico_curva.py`. Verificados
en sintaxis (los 8 ficheros) y en coherencia de nombres entre ficheros
(cada metodo que `ventana_principal.py` llama sobre un widget existe
con ese nombre exacto en el widget correspondiente) -- elimina una
categoria de errores tipicos, pero no sustituye la prueba real. Ver
FILES.md seccion 7 para el detalle completo por fichero.

**Simplificaciones deliberadas de esta primera version (MVP):**
- Deteccion GLOBAL (`detectar_global()`) sobre el propio conjunto
  cargado como referencia y evaluacion a la vez -- no hay todavia en
  la interfaz un concepto de "conjunto de referencia" separado de lo
  que se analiza. `detectar_por_clase()`/`detectar_ood_multiclase()`
  (ya construidos y verificados, mas potentes) quedan para una
  iteracion futura, cuando la interfaz permita elegir un TRAIN de
  referencia distinto del conjunto a evaluar.
- Una tabla unica ordenable con columna `candidato_fuerte`, no las dos
  secciones "seguros"/"ambiguos" con comportamiento adaptativo de la
  seccion 5.1 -- diseño completo, pendiente de implementar.

## 6. Limitaciones conocidas (a tener en cuenta al generalizar)

- `maximum_slope` pierde informacion sistematicamente por la cuantizacion de
  `mjd` en StarEmbed. Otro dataset con `mjd` de precision completa no tendria
  este problema — el adaptador deberia detectar la resolucion temporal
  efectiva del dataset de origen.
- `stetson_K`, tal como esta implementado, es una aproximacion (usa desviacion
  estandar global de la curva, no incertidumbre fotometrica por punto). No es
  directamente comparable con el indice de Stetson clasico de la literatura.
- `skew` es matematicamente inestable cuando la media esta cerca de cero
  (coeficiente de variacion puede dispararse sin que la variable sea inutil).
- El criterio de "candidato" es puramente estadistico (aislamiento en el
  espacio de features). No incorpora todavia contexto fisico ni validacion
  externa automatizada.

## 7. Proximos pasos

1. **Probar `app/` en un entorno real con PySide6 y pantalla** -- prioridad
   inmediata, es la unica forma de validar de verdad todo lo que no se pudo
   ejecutar durante el desarrollo (seccion 5.2). Esperable alguna ronda de
   ajuste, igual que con el adaptador de Gaia.
2. Ampliar la interfaz al diseño completo de la seccion 5.1 (dos secciones
   "seguros"/"ambiguos") y a deteccion por clase/OOD
   (`detectar_por_clase()`/`detectar_ood_multiclase()`, ya construidos y
   verificados) en vez de la deteccion global auto-referencial del MVP
   actual -- requiere primero decidir en la interfaz que es "el conjunto de
   referencia" frente a "lo que se analiza".
3. Registrar Gaia y StarEmbed permitiendo seleccion de multiples ficheros
   desde el propio dialogo de la interfaz de forma mas pulida (el MVP actual
   soporta seleccion multiple via `;`, funcional pero rudimentario).

**Completado** (referencia rapida, detalle en las secciones indicadas):
esquema canonico y adaptador StarEmbed (§3), pipeline empaquetado
`anomaly_detector` con extraccion de features y deteccion de anomalias
verificados end-to-end contra datos reales (§4.6-4.7), sistema de fuentes
de datos plegable con adaptador tabular generico (§5.0), estructura de
carpetas definitiva del proyecto (`FILES.md` §0), **primera validacion
cuantitativa con ground truth real, incluida la investigacion completa
del espacio de features (§4.9: normalizacion por percentil, contribucion
de `period`, y features de forma via Fourier — las tres cerradas con
evidencia, AUC≈0.66 documentado como techo estructural del enfoque
actual), y validacion externa de candidatos completada (§4.5: 8/8
confirmados en SIMBAD y VSX, sin descubrimientos de tipo nuevo en este
lote pero con validacion cruzada consistente en tres fuentes
independientes)**.

## 8. Antecedentes: dataset ZTF DR3 original, descartado (scripts 01-10)

Antes de trabajar sobre ZTF/StarEmbed, el proyecto empezo explorando un dataset
ZTF DR3 distinto, en formato binario propio (`feature_*.dat` + `*.name` +
`oid_*.dat`, leidos con `np.memmap`), organizado en tres subconjuntos: `M31`,
`DEEP` y `DISK`. Esta fase quedo completamente abandonada y no forma parte del
pipeline descrito en la seccion 4; se documenta aqui solo como antecedente,
porque su cierre explica en parte por que el proyecto migro a StarEmbed.

**Lo que se investigo:** una inspeccion inicial (01-03) mostro que un
clasificador (RF + regresion logistica) distinguia el subconjunto de origen
(M31/DEEP/DISK) de cada objeto con una accuracy muy alta (~99%) usando solo
las 42 features crudas del dataset (04). Esto disparo una investigacion de
posible sesgo: ¿son M31/DEEP/DISK poblaciones astronomicas realmente
distintas, o es un artefacto de como se construyo o instrumento el dataset?

**Controles aplicados, en orden:**
- **05-06**: ablacion por grupos de features y "autopsia" feature a feature
  (Kruskal-Wallis, KS, AUC simetrico). `06` identifico `eta_e` (variable de
  este dataset, no confundir con `eta`/von Neumann ratio de la seccion 4.1,
  que pertenece a un espacio de features distinto) como la feature
  individualmente mas discriminante entre subconjuntos.
- **07 (dos versiones)**: eliminacion progresiva de grupos de features
  sospechosas (`eta_e`, magnitud, tendencias, temporales, escala). La v1
  (unica con resultados validos; la v2 fallo en ejecucion) mostro que incluso
  quitando los 5 grupos a la vez la accuracy del RF se mantenia en 89.6%
  (baseline 99.02%) — la separacion no se explicaba por ninguno de estos
  grupos.
- **08-09**: intentos de igualar estadisticamente los tres subconjuntos antes
  de clasificar. `08` (igualacion de distribuciones por bins de percentiles)
  **fallo en ejecucion** porque no existia ninguna region de percentiles
  comun entre los tres subconjuntos. `09` (matching por vecino mas cercano)
  si se completo, pero incluso comparando los objetos mas parecidos posibles
  entre subconjuntos (4.596 tripletas de 50.000 objetos/dataset iniciales),
  la accuracy del RF solo bajaba de 98.81% a 97.24%.
- **10 (`10_control_campo.py`)**: extrajo el `field_id` de ZTF directamente
  del OID. **Resultado que cierra la investigacion:** M31, DEEP y DISK
  correspondian, cada uno, a un unico y distinto campo de observacion ZTF
  (695, 795 y 807 respectivamente), sin ningun solapamiento entre ellos. Esto
  explica retroactivamente tanto la altisima accuracy del `04` como los
  fallos/resultados nulos de `08` y `09`: no eran tres poblaciones
  astronomicas comparables sometidas a sesgos sutiles, sino tres campos de
  cielo distintos con sus propias condiciones de observacion — la
  clasificacion era trivial por construccion, no evidencia de estructura
  astrofisica.

**Por que es relevante para el proyecto actual:** esta fase confirmo por la
via dificil (varios controles, dos de ellos fallidos en el camino) algo que
condiciona el diseño de la seccion 3: cualquier adaptador futuro para un
dataset nuevo deberia comprobar pronto si sus subconjuntos/etiquetas
"naturales" son en realidad artefactos de instrumentacion (campo, CCD,
campaña de observacion) antes de tratarlos como clases o poblaciones de
interes cientifico. Detalle completo, script a script, en `FILES.md`
(seccion "Scripts de exploracion inicial").

## 9. Historial de cambios

- **v1** — Consolidacion inicial tras el cierre de la fase de validacion
  (scripts 16-24b). Documento creado a partir de la sesion de auditoria y
  desarrollo del detector de anomalias sobre ZTF/StarEmbed.
- **v1.1** — Añadida seccion 4.5 (resultado de la interseccion 24+24b:
  15 candidatos robustos, 8 interesantes, 7 ambiguos) y seccion 5 (decisiones
  de diseño de la interfaz PySide6 para presentar seguros/ambiguos sin que
  el pipeline oculte datos). Reordenado "proximos pasos" para priorizar el
  esquema canonico como bloqueo real antes del empaquetado y la interfaz.
- **v1.2** — Añadidas secciones 4.6 (verificacion de `features.py` contra el
  metodo original de 20-24b, sobre datos reales: TEST exacto, TRAIN con una
  excepcion documentada — banda degenerada de 1 punto) y 4.7 (verificacion
  end-to-end del pipeline completo `deteccion.py`, 100% de recall sobre los
  candidatos ya conocidos de `24c_candidatos_interseccion.csv`).
- **v1.3** — Añadida seccion 5.0 (sistema de fuentes de datos plegable, dos
  niveles: declarativo YAML y Python con registro, motivado por que el
  proyecto sera de codigo abierto). Renumerada la seccion 5 original a 5.1.
- **v1.4** — Fusion con la auditoria de scripts realizada por el autor:
  matiz en la seccion 4.1 (importancia *marginal* de `eta` dentro del
  conjunto completo, segun ablacion del 18, frente a su poder predictivo
  *individual* mediocre, segun 17/19 — no son contradictorios, ver texto).
  Añadida seccion 8, "Antecedentes: dataset ZTF DR3 original, descartado
  (scripts 01-10)" — la rama de investigacion previa a StarEmbed, cerrada
  tras confirmar en `10_control_campo.py` que los subconjuntos M31/DEEP/DISK
  eran campos de observacion ZTF disjuntos, no poblaciones astronomicas
  distintas. Renumerado "Historial de cambios" a seccion 9. `FILES.md`
  actualizado en paralelo con la auditoria detallada de los scripts 01-19.
- **v1.5** — Añadida seccion 4.8: descubrimiento del cuarto split de
  StarEmbed, `anom` (1.087 objetos, 10 clases OOD reales), confirmado
  contra la fuente primaria (arXiv 2510.06200). Esto habilita validar la
  deteccion de anomalias contra verdad fundamental real, en vez de solo
  contra inspeccion manual — proximo paso prioritario del proyecto (ver
  seccion 7). `FILES.md` actualizado con la estructura de carpetas
  definitiva del proyecto (nueva seccion 0).
- **v1.6** — Añadida seccion 4.9: resultado real de `validar_contra_anom.py`
  sobre datos reales (AUC-ROC 0.6608, recall 16.4% a FP=5%). Añadida
  `detectar_ood_multiclase()` a `anomaly_detector/deteccion.py` (replica el
  metodo de referencia del paper StarEmbed). Hallazgo principal: el recall
  por clase OOD es inversamente proporcional a la cercania taxonomica con
  las 7 clases de entrenamiento (Blazhko, subtipo de RRab con modulacion,
  recall 4.9%; PCEB, sin pariente cercano, recall 40.3%) — confirmado
  cruzando `clase_real` contra `clase_mas_cercana`. Limitacion estructural
  documentada, no oculta: el espacio de 12 variables agregadas no captura
  bien variantes sutiles dentro de una misma familia de variabilidad.
- **v1.7** — Descartada con evidencia (post-hoc, sin reentrenar) la
  hipotesis de que normalizar `score_min` por percentil dentro de cada
  clase mejoraria el resultado: empeora (AUC 0.6082 vs 0.6608 bruto).
  Explicado por que: clases con pocos objetos de entrenamiento producen
  detectores poco discriminativos cuyo ruido se disfraza de señal al
  pasar a percentil. Confirma que el patron de cercania taxonomica es la
  explicacion dominante, no un artefacto metodologico de comparabilidad
  entre clases de tamaño distinto.
- **v1.8** — Aislada la contribucion de `period` al AUC
  (`comparar_periodo_anom.py`): +0.0166 en AUC global, pero el numero
  agregado esconde un efecto real y contrapuesto por clase. Ayuda mucho
  en ACEP/Cep-II (periodos largos, distintivos), no cambia nada en
  PCEB/HADS/LADS, y empeora en Blazhko/EA_UP/Hump. Confirma con evidencia
  que el problema de Blazhko/HADS/ELL es de forma de curva, no de
  periodo -- justifica la siguiente tarea (features de forma, Fourier
  del plegado en fase) en vez de dejarla como intuicion sin respaldar.
- **v1.9** — Cerrada la investigacion del espacio de features (seccion
  4.9) con el tercer y ultimo experimento: features de forma de curva
  (`anomaly_detector/features_forma.py`, descomposicion de Fourier del
  plegado en fase, `probar_features_forma.py`). Resultado: hipotesis
  REFUTADA -- empeora el AUC global (0.6576 vs 0.6608) y empeora
  especificamente en las tres clases objetivo (Blazhko -3.3 puntos de
  recall, HADS -1.5, ELL -1.3). Decision documentada de no iterar una
  cuarta vez sobre la misma tecnica (evitar el patron de "bucle de
  validacion" ya identificado en la auditoria original, seccion 8).
  AUC≈0.66 queda establecido como techo estructural del enfoque actual
  de estadisticos agregados, con tres experimentos consecutivos como
  evidencia. Renumerados "proximos pasos" (seccion 7): la investigacion
  de features deja de ser prioridad, se retoma Gaia DR3 / interfaz
  PySide6.
- **v2.0** — Cerrada la validacion externa de candidatos (seccion 4.5,
  `cruzar_candidatos_catalogos.py`, FILES.md seccion 7): 8/8 candidatos
  con contrapartida consistente en SIMBAD y VSX/AAVSO, coincidente con
  la prediccion del RF propio en los 8 casos. Ninguno resulto ser un
  tipo taxonomico nuevo -- resultado que confirma que el detector
  encuentra objetos reales, no ruido, aunque este lote concreto no
  contenga un descubrimiento. Corregida de paso una referencia cruzada
  obsoleta (seccion 4.5 apuntaba a "seccion 8" para el diseño de la
  interfaz; es la seccion 5).
- **v2.1** — Construido y validado `AdaptadorGaiaDR3`
  (`anomaly_detector/adapters/gaia.py`), el segundo adaptador real del
  proyecto (seccion 3). Formato de fotometria por epoca confirmado con
  datos reales via `diagnostico_gaia.py` (tabla "ancha", bandas G/BP/RP
  como columnas separadas, distinto de lo asumido inicialmente).
  Corregidos dos bugs reales durante el desarrollo: nombres de columna
  incorrectos (fix directo tras diagnostico) y conversion
  `TableElement` -> `Table` necesaria antes de `.colnames` (fix con
  logica defensiva, probada con los tres casos posibles). Prueba a
  pequeña escala (5 fuentes) confirma que el esquema canonico y
  `extraer_features_dataset()` funcionan sin cambios, con features que
  distinguen correctamente (sentido fisico real) una eclipsante de
  variables de periodo largo. Registrados StarEmbed y Gaia DR3 en el
  sistema de fuentes (`@registrar_fuente`) -- pendiente historico
  cerrado. La decision de arquitectura de la seccion 3 pasa de
  "propuesta" a "validada con evidencia".
- **v2.2** — Prueba a mayor escala de Gaia DR3 completada (300 objetos,
  seccion 3): 100% de exito, 0 lotes fallidos, 0 perdidas silenciosas,
  0 `NaN`. Añadida instrumentacion a `gaia.py` (`ids_solicitados`,
  `lotes_fallidos`) para poder diagnosticar perdidas de red a escala,
  probada antes de entregar. Rendimiento medido: ~0.90 s/objeto --
  documentado como consideracion de diseño para la interfaz (carga en
  segundo plano necesaria para volumenes grandes). Confirmada
  diversidad real de clases, incluida `AGN` (nucleo galactico activo,
  no una estrella) procesada sin ningun error -- la prueba de
  generalizacion mas fuerte de toda la sesion. Fase de Gaia DR3 cerrada.
- **v2.3** — Construida la primera version de la interfaz PySide6
  (`app/`, seccion 5.2): `controlador.py` (logica de negocio, sin Qt)
  probado de extremo a extremo con datos sinteticos reales; el resto
  de ficheros (`main.py`, `ventana_principal.py`,
  `modelo_resultados.py`, `widgets/`) con sintaxis verificada y
  coherencia de nombres comprobada entre ficheros, pero sin ejecutar
  (sin PySide6/pantalla en el entorno de desarrollo -- mismo criterio
  de honestidad ya aplicado con el adaptador de Gaia). MVP con dos
  simplificaciones deliberadas: deteccion global auto-referencial en
  vez de por clase/OOD, y una tabla unica en vez de las dos secciones
  "seguros"/"ambiguos" del diseño completo de la seccion 5.1.
  `requirements.txt` actualizado con `PySide6` y `matplotlib`.
