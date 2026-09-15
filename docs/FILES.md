# FILES.md — Inventario del proyecto

**Estado:** v11 — primera versión de la interfaz PySide6 (app/, sección 7): controlador probado de extremo a extremo, resto de ficheros con sintaxis verificada pero sin ejecutar (sin PySide6/pantalla en el entorno de desarrollo)
**Leyenda de estado de auditoria:**
- ✅ **Auditado** — revisado en detalle (codigo y/o resultados) durante esta sesion.
- ⚪ **No auditado** — existe en el proyecto pero no se ha revisado su codigo ni sus
  resultados en esta sesion. La descripcion es una suposicion razonable a partir del
  nombre del archivo unicamente y DEBE confirmarse antes de darla por buena.
- 🗄️ **Historico/superado** — su resultado ha sido reemplazado por una version
  posterior; se conserva por trazabilidad, no como referencia activa.

---

## 0. Estructura de carpetas del proyecto

**Confirmada con `listar_estructura.py` el 27/08/2026** — árbol real, no una
propuesta. Cuatro incidencias detectadas al confirmarla, pendientes de que el
autor las corrija (no resueltas automáticamente, ver aviso tras el árbol):

```
C:\Users\Usuario\Documents\Astronomia\
│
├── readme.md
│
├── anomaly_detector\                    (paquete core)
│   ├── __init__.py
│   ├── schema.py
│   ├── features.py
│   ├── deteccion.py
│   └── adapters\
│       ├── __init__.py
│       ├── base.py
│       ├── starembed.py
│       ├── registro.py
│       └── tabular_generico.py
│
├── fuentes\
│   └── EJEMPLO_generico.yaml
│
├── app\                                 (interfaz PySide6 — primera versión, ver sección 7)
│   ├── __init__.py
│   ├── main.py                          (punto de entrada: python app\main.py)
│   ├── controlador.py                   (lógica de negocio, SIN Qt — probado de verdad)
│   ├── ventana_principal.py             (Qt — no probado en ejecución)
│   ├── modelo_resultados.py             (Qt — no probado en ejecución)
│   └── widgets\
│       ├── __init__.py
│       ├── formulario_fuente.py         (Qt — no probado en ejecución)
│       └── grafico_curva.py             (Qt — no probado en ejecución)
│
├── docs\
│   ├── FILES.md
│   └── whitepaper.md
│   └── requirements.txt
│
├── dataset\
│   ├── ZTF_40k_StarEmbed\
│   │   ├── README.md
│   │   ├── summary.{train,test,validation,anom}.json
│   │   └── data\
│   │       ├── train-0000{0,1}-of-00002.parquet
│   │       ├── test-00000-of-00001.parquet
│   │       ├── validation-00000-of-00001.parquet
│   │       └── anom-00000-of-00001.parquet    ← benchmark OOD real, ver whitepaper.md 4.8
│   │
│   └── ZTF_DR3\                         (descartado, ver whitepaper.md sección 8)
│       ├── feature_{deep,disk,m31}.dat / .name
│       └── oid_{deep,disk,m31}.dat
│
└── auditorias\
    ├── 01_exploracion_inicial\
    │   ├── 01_ztf_dr3\                   (scripts 01-10, salvo 07_controlar_sesgos_2.py — eliminado por el autor, ver nota)
    │   │   └── 01...10_control_grupos.py
    │   └── 02_starembed_ztf_40k\         (11, 14, 15, 17, 19)
    │
    ├── 02_validacion_robusta\            (12, 13, 16, 18, 20, 21, 22, 23)
    │   └── resultados\                   (CSV de esos scripts)
    │
    ├── 03_control_integridad_sesion\     (16b, 16c, 16d, 20b)
    │   └── resultados\
    │       └── (16c_diagnostico_padding.csv, junto con el resto de resultados)
    │
    ├── 04_deteccion_anomalias\           (24, 24b)
    │   └── resultados\                   (incluye 24c_candidatos_interseccion.csv)
    │
    └── 05_verificacion_paquete\          (verificar_paquete.py, probar_deteccion.py)
        └── resultados\
```

**Historial de incidencias detectadas al confirmar la estructura:**

1. ~~Falta `07_controlar_sesgos_2.py`~~ — **resuelto, no era una incidencia**:
   el autor lo eliminó intencionadamente tras confirmar que su ejecución daba
   errores y no producía resultados válidos (ya documentado en la sección 1,
   fila `07_controlar_sesgos_2.py`: "descartado, ejecución con errores").
2. ~~`16c_diagnostico_padding.py`/`.csv` intercambiados~~ — **resuelto**, cada
   uno está ya en su sitio correcto.
3. ~~`fuentes\` vacía~~ — **resuelto**, `EJEMPLO_generico.yaml` colocado.
4. ~~Copias sueltas de `FILES.md`/`readme.md`/`whitepaper.md` dentro de
   `anomaly_detector\`~~ — **resuelto**, eliminadas.
5. ~~`tabular_generico.py` y `registro.py` mal ubicados~~ — **resuelto y
   verificado el 28/08/2026**. Costó tres rondas porque se superponían dos
   problemas distintos:
   - `anomaly_detector\base.py` y `starembed.py` (duplicados directamente en
     la raíz del paquete, no solo en `adapters\`) reaparecían tras cada
     reorganización manual, y `registro.py`/`tabular_generico.py` quedaban
     mal ubicados al mismo nivel en vez de dentro de `adapters\`. Resuelto
     con `reparar_paquete.py` (script idempotente, con modo prueba).
   - `adapters\__init__.py` se guardó como `__init__ .py` (con un espacio
     antes de la extensión) — Windows lo permite sin avisar, pero para
     Python es un nombre de archivo distinto, así que el paquete no lo veía
     en absoluto (`ImportError: ... (unknown location)`, síntoma
     característico de un `__init__.py` ausente o vacío). Además había un
     `__init_1_.py` residual (513 bytes) de una descarga anterior con
     conflicto de nombre. Renombrado a `__init__.py` y borrado el residual.

   **Verificado en producción:**
   `python -c "from anomaly_detector.adapters import fuentes_disponibles; print(fuentes_disponibles())"`
   devuelve correctamente `FuenteDatos(id='ejemplo_generico', ...)` con su
   fábrica y parámetros — el sistema de fuentes plegable (whitepaper.md,
   sección 5.0) queda confirmado funcional, no solo documentado.

**Por qué `anomaly_detector\` y `fuentes\` van en la raíz y no en `auditorias\`
ni en `docs\`:** son código de producción, no auditoría — `main.py` y la
futura app los importan directamente, y `registro.py` calcula la ruta de
`fuentes\` en relación a su propia posición dentro del paquete. Moverlos
rompería esa ruta calculada.

**Por qué `validar_contra_anom.py` va en `auditorias\06_validacion_ground_truth\`
y no en la raíz:** es un script de auditoría (mide el rendimiento del
pipeline contra verdad fundamental real), no código de producción — el mismo
criterio que separa `anomaly_detector\` de `auditorias\` en el punto anterior.
Se corrige aquí un despiste de una ronda anterior, en la que se entregó
colocado en la raíz por error.

**Fase 06 — validación con ground truth real** (nueva, ver whitepaper.md
sección 4.8):

```
auditorias\06_validacion_ground_truth\
├── validar_contra_anom.py
├── comparar_periodo_anom.py
├── probar_features_forma.py
└── resultados\
    ├── validacion_anom_scores.csv
    ├── validacion_anom_umbrales.csv
    ├── validacion_anom_recall_por_clase.csv
    ├── comparacion_con_sin_period.csv
    └── forma_comparacion_final.csv

auditorias\07_validacion_externa\
├── cruzar_candidatos_catalogos.py
├── diagnostico_gaia_dr3.py          (renombrado desde diagnostico_gaia.py)
├── diagnostico_simbad.py            (nuevo — creado por el autor a partir
│                                      del snippet de diagnóstico dado en chat)
├── probar_gaia_escala.py            (reubicado aquí desde la raíz)
├── probar_gaia_pequeno.py           (reubicado aquí desde la raíz)
└── resultados\
    ├── cruce_catalogos_resultado.csv
    ├── diagnostico_gaia_dr3_resultado.txt
    ├── diagnostico_simbad_resultado.txt
    ├── probar_gaia_escala_resultado.txt
    └── probar_gaia_pequeno_resultado.txt
```

**Script de reorganización:** `reorganizar_proyecto.py` crea la estructura y
mueve los ficheros que estaban en una carpeta plana. **Script de verificación:**
`listar_estructura.py` genera un árbol fiable con indentación real, filtrando
`.zip`/`.pyc`/`__pycache__` y carpetas que parezcan backups sueltos.

**Sobre `BASE_DIR` en los scripts de `auditorias\`:** ~~pendiente de
actualizar~~ — **resuelto por el autor**: todos los scripts de auditoría
(01-24b, `verificar_paquete.py`, `probar_deteccion.py`) se han adaptado a la
nueva estructura de carpetas y re-ejecutado con éxito, generando sus CSV/PNG
de resultados en la `resultados\` correspondiente de cada fase (incluidas
`01_ztf_dr3\resultados\` y `02_starembed_ztf_40k\resultados\`, que no
existían en la confirmación anterior y ahora están pobladas). El árbol de
arriba refleja esto. `auditorias\` deja de ser solo un archivo histórico
inerte — es una fuente de verdad ejecutable, consistente con la estructura
actual del proyecto.

---

## 1. Scripts de exploracion inicial (01-19)

> Auditados en esta sesion: `01` a `10_control_campo.py` (ver tabla). El resto
> (`10_control_grupos.py` en adelante que no aparezcan marcados) siguen sin
> auditar; sus descripciones son una suposicion a partir del nombre de archivo
> y deben corregirse por el autor del proyecto.
>
> **Nota de cierre de esta rama:** los scripts `01`-`10` trabajan sobre un
> dataset ZTF DR3 previo (`m31`/`deep`/`disk`, formato binario memmap)
> **completamente distinto de StarEmbed** y ya abandonado. La investigacion de
> por que un clasificador distinguia estos tres subconjuntos con altisima
> accuracy (iniciada en `04`) queda **cerrada** por `10_control_campo.py`:
> M31, DEEP y DISK resultaron ser tres campos de observacion ZTF distintos y
> completamente disjuntos (campos 695, 795 y 807 respectivamente) — la
> separacion era identidad de campo, no evidencia de poblaciones astronomicas
> distintas. Esto es coherente con el fallo de `08` (sin solapamiento de
> distribuciones entre campos) y con que `09` no lograra reducir la
> clasificabilidad ni comparando los objetos mas similares posibles entre
> datasets. Se conserva por trazabilidad; no es referencia activa para el
> pipeline StarEmbed vigente.

| Archivo | Descripcion (sin confirmar) | Estado |
|---|---|---|
| `01_inspeccion_dataset.py` | Inspeccion de un dataset ZTF DR3 **distinto de StarEmbed**, en formato binario propio (memmap), con tres subconjuntos `m31`/`deep`/`disk` (`feature_*.dat` + `*.name` + `oid_*.dat`). Comprueba existencia/tamaño de archivos, calcula min/max/media/std/NaN/Inf por feature en bloques de 100k objetos, detecta features constantes, y comprueba duplicados de OID solo sobre una muestra de los primeros 100.000 (no sobre el total). Fase exploratoria anterior, abandonada en favor de StarEmbed — no forma parte del pipeline validado en `whitepaper.md` seccion 4. | ✅ Auditado — 🗄️ dataset descartado (proyecto migro a StarEmbed) |
| `02_analisis_distribuciones.py` | Continuacion de 01 sobre el mismo dataset ZTF DR3 descartado (`m31`/`deep`/`disk`). Sobre una muestra de 50k objetos por dataset (150k total): matriz de correlacion por dataset (heatmap + pares con \|r\|>=0.90 y 0.75-0.90), estadisticas por feature con deteccion de colas pesadas (ratio rango/IQR > 50), histogramas de las 42 features (recortados a p1-p99), y un PCA global (RobustScaler + 42 componentes) sobre la muestra combinada de los tres datasets, con varianza explicada acumulada y scatter PC1/PC2 coloreado por dataset de origen. | ✅ Auditado — 🗄️ dataset descartado (ver 01) |
| `03_estructura_poblaciones.py` | Continuacion de 02, mismo dataset descartado. PCA (RobustScaler, 10 componentes) sobre la muestra combinada M31/DEEP/DISK, y evalua si el PCA separa los tres datasets entre si (no clases astronomicas): silhouette score de las etiquetas de dataset en el espacio PCA, centroides y distancias euclidianas entre ellos, densidad local por dataset (percentiles de distancia al k=20-esimo vecino), graficos PCA 2D/3D y de centroides. Interpretacion automatica del silhouette (debil/moderada/fuerte) con aviso explicito de que la separacion no implica poblaciones astronomicas distintas (puede deberse a seleccion, instrumentacion, etc.). | ✅ Auditado — 🗄️ dataset descartado (ver 01) |
| `04_clasificador_datasets.py` | Continuacion de 03, mismo dataset descartado. Prueba de sesgo explicita (lo dice el propio docstring): entrena Random Forest (`n_estimators=150`) y regresion logistica (con RobustScaler) para predecir de que dataset (M31/DEEP/DISK) procede un objeto usando solo las 42 features, con split estratificado 80/20. Reporta accuracy, classification report, matriz de confusion (PNG) e importancia de features (CSV) para ambos modelos. Interpretacion automatica por umbrales de accuracy del RF (>=0.95 alerta de sesgo muy fuerte, >=0.75 moderado/fuerte, >=0.50 debil/moderado, si no, debil), con el mismo aviso de que una clasificacion alta no prueba diferencias astronomicas reales. | ✅ Auditado — 🗄️ dataset descartado (ver 01) |
| `05_ablacion_features.py` | Mismo dataset ZTF DR3 descartado. Continua el `04`: entrena RF (100 arboles) + regresion logistica (RobustScaler) para clasificar el origen (M31/DEEP/DISK) y prueba ablaciones sucesivas quitando grupos de features (`eta_e` sola; `eta_e`+medias; `eta_e`+tendencias; combinaciones; un grupo "sin magnitud/escala"; y un grupo "solo temporales" que conserva solo las features relacionadas con tiempo/periodo). Incluye control de etiquetas aleatorias (~0.333 esperado). Version temprana de esta ablacion — el analisis sistematico y mas cuidadoso esta en `07`/`07_2` (ver tambien 18, que es la ablacion final sobre el pipeline StarEmbed, no relacionada con este). | ✅ Auditado — 🗄️ dataset descartado (ver 01) |
| `06_autopsia_features.py` | Mismo dataset ZTF DR3 descartado. Analiza cada una de las 42 features **individualmente** (no un modelo conjunto): Kruskal-Wallis entre M31/DEEP/DISK, medianas y diferencia de medianas normalizada por IQR global, y para cada par de datasets (M31-DEEP, M31-DISK, DEEP-DISK) un test KS + un "AUC simetrico" (capacidad de esa sola feature para distinguir el par). Genera un ranking global y por pareja, CSV completo (`06_autopsia_features.csv`) y dos graficos de barras (top-15 por AUC y por KS). **Resultado clave que motiva el `07`/`07_2`: identifica `eta_e` como la feature individualmente mas sospechosa** (mayor poder discriminante entre datasets) — no confundir con `eta` (von Neumann ratio), que es una variable distinta del espacio de features de StarEmbed (ver `whitepaper.md` seccion 4.1). | ✅ Auditado — 🗄️ dataset descartado (ver 01) |
| `07_controlar_sesgos.py` | Mismo dataset ZTF DR3 descartado. Version 1 del control sistematico de sesgo: define 5 grupos de features (`eta_e`, magnitud, tendencias, temporales/periodos, escala) y ejecuta 9 experimentos acumulativos (desde baseline 42 features hasta un "CONTROL FUERTE" que quita los 5 grupos a la vez), con RF (`class_weight=None`) + regresion logistica en cada uno, accuracy y balanced accuracy, mas control de etiquetas aleatorias. Guarda `07_control_sesgos_resultados.csv`. **Resultado real (confirmado con el CSV de salida):** baseline RF 99.02%; incluso el "CONTROL FUERTE" (19 features, los 5 grupos eliminados a la vez) mantiene RF 89.6% — muy por encima del umbral de separacion fuerte (>0.80) que define el propio script. Quitar solo variables temporales apenas cambia nada (99.04%). Conclusion: la separacion entre M31/DEEP/DISK no se explica por magnitud, escala, tendencias ni periodicidad combinadas — persiste señal fuerte no explicada. **Nota:** usa el mismo nombre de fichero de salida que `07_controlar_sesgos_2.py` — si se ejecutaron en el mismo directorio, uno sobrescribe el CSV del otro; el CSV disponible corresponde a esta version (v1), no a la v2. | ✅ Auditado — 🗄️ dataset descartado (ver 01) |
| `07_controlar_sesgos_2.py` | Version 2 del control de sesgo (grupos `eta_e`, magnitud, tendencias, periodos, periodograma; RF con `class_weight="balanced"`). **Confirmado por el autor: la ejecucion dio errores** — por eso no existe un CSV de resultados propio (el `07_control_sesgos_resultados.csv` disponible es el de la v1). No se debe usar ni referenciar como fuente de resultados; el control de sesgo valido para este dataset es exclusivamente `07_controlar_sesgos.py` (v1). | 🗄️ Descartado — ejecucion con errores, sin resultados validos |
| `08_control_estructura.py` | Mismo dataset ZTF DR3 descartado. En vez de quitar grupos de features enteros (como el `07`), iguala las **distribuciones** de 14 features sospechosas (`eta_e`, `cusum`, tendencias, periodos, medias, `stetson_K`, `maximum_slope`) discretizando en bins de percentiles y submuestreando para que M31/DEEP/DISK queden representados por igual en cada bin conjunto. **Ejecucion real: el Experimento 1 (baseline) completo con RF 98.81% / LOG 85.60% (con aviso de no convergencia de la logistica), pero el Experimento 2 (distribuciones igualadas) fallo con `RuntimeError: No existen regiones comunes entre los tres datasets`** — es decir, ni siquiera existe solapamiento de distribuciones en el espacio de percentiles conjunto entre M31/DEEP/DISK. El script no llego a completarse. Este fallo es coherente con el hallazgo del `10_control_campo.py`: al ser tres campos ZTF completamente distintos, sus distribuciones marginales no se solapan en absoluto. | ✅ Auditado — ejecucion parcial (fallo en Experimento 2) — 🗄️ dataset descartado (ver 01) |
| `09_matching_vecinos.py` | Mismo dataset ZTF DR3 descartado. Alternativa al `08`: matching por vecino mas cercano (k-NN en RobustScaler, mismas 14 features) para emparejar cada objeto de M31 con su vecino mas similar en DEEP y en DISK. **Ejecucion real (`09_matching_vecinos_resultados.csv`): baseline RF 98.81% / LOG 85.60% (50.000 objetos/dataset); tras el matching, con solo 4.596 tripletas conseguidas, RF se mantiene en 97.24% / LOG 80.67%** — la separacion apenas cae. Ni siquiera comparando los objetos mas parecidos posibles entre datasets (via k-NN) desaparece la clasificabilidad. Coherente con `10_control_campo.py`: al ser campos disjuntos, no existen realmente objetos "comparables" entre datasets. | ✅ Auditado — resultados confirmados — 🗄️ dataset descartado (ver 01) |
| `10_control_campo.py` | Mismo dataset ZTF DR3 descartado. Extrae el `field_id` de ZTF directamente del OID (division entera por 10^12) para M31/DEEP/DISK y comprueba si cada dataset corresponde a un unico campo observacional y si esos campos son disjuntos entre datasets. **Ejecucion real — hallazgo definitivo para toda esta rama de investigacion (04 a 09): M31 = campo 695 (57.546 objetos), DEEP = campo 795 (406.611 objetos), DISK = campo 807 (1.790.565 objetos). Cada dataset es exactamente UN campo ZTF, y los tres campos son completamente disjuntos (0 solapamiento).** Conclusion textual del propio script: la separacion que detectaban los clasificadores del 04, 06, 07, 08 y 09 es simplemente identidad de campo observacional, no evidencia de poblaciones astronomicas distintas — "la investigacion debe detenerse aqui". Este resultado cierra la rama de investigacion de sesgo entre datasets y es coherente con el fallo del `08` y la persistencia de separacion en el `09`. | ✅ Auditado — resultados confirmados — CIERRA la investigacion de sesgo M31/DEEP/DISK — 🗄️ dataset descartado (ver 01) |
| `10_control_grupos.py` | Intento generico de control de fuga por grupos (buscar columnas candidatas de agrupacion como `field`, `observation`, `visit`, etc. y comparar split aleatorio vs `GroupShuffleSplit`/`GroupKFold`). **Confirmado por el autor: fallo, nunca se completo con exito.** Ademas, a diferencia de todos los demas scripts de esta rama, esperaba los datos en CSV (`M31.csv`/`DEEP.csv`/`DISK.csv`) en vez del formato `.dat`/memmap usado en el resto del proyecto — formato que no parece haber existido nunca en este proyecto. No usar ni referenciar; el control de campo valido es `10_control_campo.py`. | 🗄️ Descartado — ejecucion con errores, sin resultados validos |
| `11_clasificador_StarEmbed.py` | **Primer script del proyecto que usa StarEmbed** (hasta aqui todo trabajaba sobre el dataset ZTF DR3 ya descartado, ver seccion de antecedentes en `whitepaper.md`). Extrae estadisticas ad-hoc por banda (g/r/i por separado; no las 12 variables canonicas del pipeline final) y entrena RF+LOG para comprobar si las clases de StarEmbed son recuperables desde las curvas — presentado explicitamente como "referencia fisica independiente" tras la investigacion de sesgo M31/DEEP/DISK. Aqui aparece por primera vez la convencion de muestreo 25.000 TRAIN / 8.000 TEST reutilizada luego en `20_validacion_robustez.py`. **Resultado real (`11_StarEmbed_resultados.csv`): RF 91.28% / LOG 80.16% / control aleatorio 67.46%.** `period` es la feature mas importante del RF (16.2%), seguida de `skew` y `mad` en g/r. **Aviso metodologico:** el "control aleatorio" del script usa accuracy sin balancear; el 67.46% no es el azar teorico (1/7=14.3%) sino el resultado de un fuerte desequilibrio de clases (una clase domina ~67% de las muestras) — confirmado cruzando con el `14`/`15`, donde el mismo control con *balanced accuracy* da 0.143, el valor teorico correcto. La accuracy cruda del RF (91.3%) debe interpretarse con cautela frente a esa base real. | ✅ Auditado — resultados confirmados |
| `12_control_leakage_StarEmbed.py` | Control de leakage TRAIN/TEST | ✅ Auditado |
| `13_control_leakage_estricto.py` | Control de leakage, version estricta (limpieza de 55 curvas duplicadas) | ✅ Auditado |
| `14_control_morfologia.py` | Sobre StarEmbed. Separa la señal discriminante en 4 fuentes: periodo, estructura observacional (nº obs, duracion, cadencia, errores), estadisticas de magnitud, y "morfologia pura" (curva normalizada: magnitud a media 0/std 1, tiempo a [0,1], interpolada a 64 bins — elimina nivel absoluto y escala temporal). **Resultados reales (`14_control_morfologia_resultados.csv`, balanced accuracy RF/LOG):** SOLO PERIODO 0.42/0.49; SOLO OBSERVACION 0.19/0.34 (casi azar); SOLO ESTADISTICAS 0.52/0.72 (señal fuerte); MORFOLOGIA PURA 0.21/0.17 (cae casi al azar, que es 0.143 confirmado por ETIQUETAS_ALEATORIAS); MORFOLOGIA+PERIODO 0.39/0.41; TODO 0.58/0.76. **Conclusion (aplicando el propio criterio del script, RF_BAL de morfologia pura 0.213 no supera el umbral azar+0.10=0.243):** la clasificacion depende en gran medida de propiedades observacionales/estadisticas y no tanto de la forma pura de la curva una vez eliminados nivel y escala temporal. Coherente con el hallazgo del `15` de que normalizar solo el nivel fotometrico (sin tocar el tiempo) apenas afecta la señal — sugiere que lo que realmente se pierde en la "morfologia pura" es la informacion temporal real (cadencia/duracion), no el nivel de magnitud. | ✅ Auditado — resultados confirmados |
| `15_control_estadisticas.py` | Sobre StarEmbed. Descompone las estadisticas de magnitud (sin periodo, RA/DEC, sourceid ni morfologia) en 6 grupos: nivel fotometrico, dispersion, asimetria/forma, variabilidad, incertidumbre, observaciones/cadencia; mas una version "normalizada por objeto" (cada curva centrada/escalada individualmente, sin tocar el tiempo). **Resultados reales (`15_control_estadisticas_resultados.csv`, balanced accuracy RF/LOG):** VARIABILIDAD 0.55/0.69 (grupo individual mas fuerte); DISPERSION 0.47/0.61; NIVEL FOTOMETRICO 0.42/0.61; ASIMETRIA_FORMA 0.26/0.33; INCERTIDUMBRE 0.24/0.37; OBSERVACIONES_CADENCIA 0.17/0.17 (el mas debil, casi azar — coherente con el hallazgo equivalente del `14`); TODAS_ESTADISTICAS 0.60/0.74; **ESTADISTICAS_NORMALIZADAS 0.62/0.74 — practicamente igual o mejor que sin normalizar**, es decir, quitar el nivel fotometrico absoluto de cada curva no perjudica la clasificacion. Control aleatorio confirma balanced accuracy teorica 0.143 (1/7), lo que expone que el "control aleatorio" de `11` (accuracy cruda 67.46%) reflejaba desequilibrio de clases, no azar real. | ✅ Auditado — resultados confirmados |
| `16_control_orden_temporal.py` | Control de orden temporal — **insuficiente**: probado sobre 30 features distribucionales invariantes al orden, no detecta problemas reales de orden en `eta`/`maximum_slope`. Ver 16b. | ✅ Auditado — superado por 16b |
| `17_control_variables.py` | Sobre StarEmbed. Precursor de `19`: identifica que variables estadisticas individuales (18 candidatas ad-hoc; `eta_e` se menciona en el docstring como candidata pero nunca se llega a calcular ni testear, inconsistencia menor del propio script) llevan la señal discriminante, promediando estadisticas entre bandas g/r/i disponibles (a diferencia del criterio r>g>i de banda unica que se adopta despues en `19`/`20`). Usa truncado `iloc[:25000]`/`iloc[:8000]` (no muestreo aleatorio), la misma convencion que `19` y `20`. **Resultados reales (`17_control_variables_resultados.csv`, balanced accuracy RF/LOG):** individualmente, RANGO_PERCENTIL (0.43/0.57) y PERIODO (0.43/0.49) son los mas fuertes; MAD, ASIMETRIA (skew) y STETSON_K en rango medio (~0.28-0.36); `ETA` individual es mediocre (0.28/0.32, ni de lejos la mejor variable individual); N_OBSERVACIONES es la mas debil (0.15, practicamente azar). El grupo VARIABILIDAD (12 variables) da un salto grande a 0.665/0.775; añadir error fotometrico/n_obs/cadencia no aporta nada (0.665/0.776); TODAS_VARIABLES (22, incluye mean/median/period) alcanza el mejor resultado (0.749/0.805). Control aleatorio confirma el azar teorico (0.143). | ✅ Auditado — resultados confirmados |
| `19_seleccion_variables.py` | Sobre StarEmbed. **Este es el script que fija las 12 variables canonicas usadas desde el `20` en adelante** (lista y formulas identicas a las de `whitepaper.md` seccion 4.1, incluido el criterio de seleccion de banda r>g>i que se reutiliza despues) — confirma la sospecha de `FILES.md` anterior. Prueba variables individuales, forward selection, backward elimination y control aleatorio, todo con balanced accuracy. **Resultados reales (`19_seleccion_variables_resultados.csv`):** individualmente, `median_absolute_deviation` (MAD) es la variable mas fuerte (RF_BAL 0.320), no `eta` (0.257, en la mitad baja de las 12) — consistente con el hallazgo del `17`. La seleccion forward construye el conjunto añadiendo `mad` -> `skew` (salto grande, 0.320->0.543) -> `percent_amplitude` -> `amplitude` -> `eta` (aporta un buen salto aqui, 0.579->0.613) -> `kurtosis` -> `stetson_K` -> `median` -> `maximum_slope` (pico RF_BAL ~0.647 con 9 variables) -> `standard_deviation` -> `chi2` (pico LOG_BAL 0.742 con 11) -> `inter_percentile_range_25` (12ª y ultima, sin mejora, RF_BAL incluso baja ligeramente). La eliminacion backward confirma que `inter_percentile_range_25` es la primera candidata a eliminar del conjunto completo sin perdida, y que con solo 3 variables (`mad`+`skew`+`stetson_K`) ya se recupera la mayor parte de la señal (RF_BAL 0.562, LOG_BAL 0.657, frente a 0.64/0.74 del conjunto completo). Control aleatorio confirma el azar teorico (0.143). **Nota interpretativa (no contradice el whitepaper, lo matiza):** que `eta` sea "la variable mas importante con diferencia" en la ablacion del `18` (impacto de retirarla del modelo completo) es compatible con que aqui, en solitario, no sea la mejor predictora — su importancia parece venir de aportar informacion no redundante con el resto, no de ser la señal univariante mas fuerte. | ✅ Auditado — resultados confirmados — origen de las 12 variables canonicas |
| `18_ablacion_variabilidad.py` | Ablacion de las 12 variables sobre el modelo final. Resultado clave: `eta` es la variable mas importante con diferencia. | ✅ Auditado |


## 2. Scripts de validacion robusta (20-23)

| Archivo | Descripcion | Estado |
|---|---|---|
| `20_validacion_robustez.py` | Pipeline de referencia: carga `iloc[:25000]`/`iloc[:8000]`, extrae las 12 variables desde la curva cruda, entrena RF + regresion logistica. Version correcta, usada como base por 21-24b. | ✅ Auditado |
| `20_validacion_robustez malo.py` | Version descartada: asumia features precalculadas como columnas del parquet en vez de calcularlas desde la curva; ademas barajaba el TRAIN (shuffle) de forma inconsistente con el resto del pipeline. | ✅ Auditado — descartado, no usar |
| `21_validacion_por_clase.py` | Metricas por clase (precision/recall/F1, matrices de confusion). Hallazgo clave: RRd y RS CVn con recall muy bajo en RF (~0.22) pese a accuracy global del 87%. | ✅ Auditado |
| `22_analisis_dependencia_observaciones.py` | Rendimiento del clasificador por rango de numero de observaciones. Hallazgo: el mal rendimiento en RRd/RS CVn persiste en todos los rangos, no es un artefacto de poca muestra. | ✅ Auditado |
| `23_estabilidad_features_por_clase.py` | Estabilidad estadistica (CV) de las 6 variables mas importantes por clase y por regimen de observaciones. Hallazgo: `eta` muy dependiente del numero de observaciones; `stetson_K` muy estable. | ✅ Auditado |

## 3. Scripts de control de integridad de datos (16b-16d, 20b) — generados en esta sesion

| Archivo | Descripcion | Estado |
|---|---|---|
| `16b_control_orden_temporal_real.py` | Version correcta del control de orden temporal, aplicado a `eta`/`maximum_slope` reales. Resultado: 0 inversiones temporales en 8.000/8.000 objetos de TEST — `eta` confirmada fiable. | ✅ Auditado |
| `16c_diagnostico_padding.py` | Deteccion de padding de cola (mjd repetido + target constante al final de la curva). Resultado: 1.1% de objetos afectados, impacto despreciable. | ✅ Auditado |
| `16d_caracterizacion_empates.py` | Caracterizacion de empates de `mjd` intermedios (no en la cola). Resultado: 95% son mediciones reales distintas bajo cuantizacion de `mjd` a 1/256 dia, no duplicados. | ✅ Auditado |
| `20b_control_representatividad_muestra.py` | Comprueba si `iloc[:25000]`/`iloc[:8000]` introduce sesgo de muestreo (chi-cuadrado + analisis por deciles + comparacion practica). Resultado: sin sesgo detectable, el parquet ya viene barajado. | ✅ Auditado |

## 4. Scripts de deteccion de anomalias (24, 24b) — generados en esta sesion

| Archivo | Descripcion | Estado |
|---|---|---|
| `24_deteccion_anomalias.py` | Isolation Forest + LOF, comparacion global (cada objeto vs. todo TRAIN), con y sin `period`. Genera la primera lista de candidatos. | ✅ Auditado |
| `24b_anomalias_por_clase.py` | Isolation Forest + LOF entrenados por clase (cada objeto vs. su propia clase), normalizado a percentil. Descarta la sobrerrepresentacion de LPV del 24 (artefacto de muestra pequeña) y revela una señal nueva en RRc. | ✅ Auditado |
| `24c_candidatos_interseccion.csv` | No es un script, es un CSV derivado (cruce directo de las salidas del 24 y el 24b). 15 candidatos robustos (fuertes en ambos metodos), de los cuales 8 son "interesantes" (lista principal) y 7 son "ambiguos" (RF se equivoca de clase, ambos detectores coinciden en marcarlos). Ver `whitepaper.md` seccion 4.5. | ✅ Generado en esta sesion |

## 5. Validacion con ground truth real (auditorias/06_validacion_ground_truth) — generado en esta sesion

| Archivo | Descripcion | Estado |
|---|---|---|
| `validar_contra_anom.py` | Entrena `detectar_ood_multiclase()` (7 detectores, uno por clase de TRAIN, score minimo — replica el metodo de referencia del paper StarEmbed) y puntua TEST (in-distribution) + `anom` (1.087 objetos OOD reales) a la vez. Calcula AUC-ROC, recall a varios niveles de falsos positivos, y recall desglosado por cada una de las 10 clases OOD. Primera validacion de todo el proyecto contra verdad fundamental real, no solo inspeccion de candidatos. Probado con datos sinteticos (AUC=0.99 en un caso de separacion clara) antes de la entrega. | ✅ Probado — pendiente de ejecutar sobre datos reales |
| `resultados/validacion_anom_scores.csv` | Score `score_min` y clase mas cercana para cada uno de los 9.087 objetos evaluados (TEST + anom). | Pendiente de generar |
| `resultados/validacion_anom_umbrales.csv` | Recall en `anom` a distintos niveles de falsos positivos fijados sobre TEST (1%, 2%, 5%, 10%, 20%). | Pendiente de generar |
| `resultados/validacion_anom_recall_por_clase.csv` | Recall desglosado por cada una de las 10 clases OOD, al umbral de referencia (FP=5%). | Pendiente de generar |

## 6. Paquete reutilizable `anomaly_detector/` — generado en esta sesión

Primer paso del empaquetado (ver `whitepaper.md`, sección 7, paso 1-2): esquema
interno canónico y adaptador para StarEmbed. Probado con pruebas de humo
end-to-end (validación de curvas, limpieza/orden, extracción de features,
selección de banda principal) — no solo verificado en sintaxis.

| Archivo | Descripción | Estado |
|---|---|---|
| `anomaly_detector/__init__.py` | Punto de entrada del paquete, expone `LightCurve`, `Dataset`, `extraer_features`. | ✅ Probado |
| `anomaly_detector/schema.py` | Esquema interno canónico: `LightCurve` (una curva, una banda) y `Dataset` (colección, con `banda_principal()` para reproducir el criterio r>g>i de 20-24b). Incluye validación explícita (`validar()`) y limpieza/ordenado (`limpia()`) separadas del cálculo de features. | ✅ Probado |
| `anomaly_detector/features.py` | Las 12 variables validadas, refactorizadas para operar sobre `LightCurve` en vez de sobre filas de parquet. Mismas fórmulas que 20-24b, sin cambios numéricos. | ✅ Probado |
| `anomaly_detector/adapters/__init__.py` | Expone los adaptadores disponibles. | ✅ Probado |
| `anomaly_detector/adapters/base.py` | Interfaz `AdaptadorDataset` que debe implementar cualquier adaptador nuevo. | ✅ Probado |
| `anomaly_detector/adapters/starembed.py` | Adaptador concreto para StarEmbed. Incluye `cargar_train_test_starembed()` como atajo equivalente a la carga usada en 20-24b. **Registrado en el sistema de fuentes** (`@registrar_fuente`, id=`starembed`). | ✅ Probado y registrado |
| `verificar_paquete.py` | Compara `anomaly_detector/features.py` contra el método original de 20-24b, objeto a objeto, sobre datos reales. Incluye allowlist de diferencias documentadas y aceptadas (`EXCEPCIONES_CONOCIDAS`). Resultado: TEST 8.000/8.000 exacto; TRAIN 24.999/25.000 exacto + 1 duplicado exacto (mismo objeto repetido) + 1 excepción documentada (banda degenerada de 1 punto, ver `whitepaper.md` sección 4.6). **Paquete verificado.** | ✅ Auditado |
| `anomaly_detector/deteccion.py` | Empaquetado de 24 (`detectar_global`) + 24b (`detectar_por_clase`) + control supervisado, unificados en `detectar_anomalias()` — una función, una tabla de salida, con `candidato_robusto` y `candidato_interesante` ya calculados. Probado con datos sintéticos: reproduce fielmente el sesgo de clase pequeña en el método global (documentado, no es un bug) y detecta correctamente anomalías inyectadas vía LOF. | ✅ Probado |
| `anomaly_detector/adapters/registro.py` | Sistema de registro de fuentes de datos: combina fuentes declarativas (YAML, vía `fuentes_declarativas()`) y fuentes Python (decorador `@registrar_fuente`, vía `fuentes_python_registradas()`) en una única lista `fuentes_disponibles()` — lo que alimenta el selector de "Fuente de datos" de la futura interfaz. Cada `FuenteDatos` trae sus `ParametroConfig` para que la UI genere el formulario dinámicamente, sin lógica específica de cada fuente hardcodeada. | ✅ Probado |
| `anomaly_detector/adapters/tabular_generico.py` | Adaptador de Nivel 1: interpreta un manifiesto YAML y traduce cualquier CSV/parquet en formato "largo" (una fila por objeto-época) al esquema canónico, sin escribir Python. Soporta mapeo de columnas, conversión de unidades de tiempo (offset configurable, p. ej. JD→MJD), y valores nulos personalizados. | ✅ Probado |
| `fuentes/EJEMPLO_generico.yaml` | Plantilla de manifiesto comentada — punto de partida para que cualquiera (incluida la comunidad, si el repo es público) añada una fuente declarativa copiando y editando este fichero. | ✅ Probado como plantilla funcional |
| `anomaly_detector/adapters/gaia.py` | **Segundo adaptador real del proyecto — Nivel 2, respaldado por red** (TAP + DataLink, no fichero local). Consulta `gaiadr3.vari_classifier_result` y descarga fotometría por época por lotes. Formato de tabla confirmado con datos reales (`diagnostico_gaia.py`): estructura "ancha", G/BP/RP como columnas separadas por fila, distinto del diseño inicial (que asumía una columna `band`). Dos bugs reales corregidos durante el desarrollo (nombres de columna SIMBAD-style incorrectos; conversión `TableElement`→`Table` necesaria antes de `.colnames`, con lógica defensiva probada en los tres casos posibles). Instrumentado con `ids_solicitados`/`lotes_fallidos` para diagnosticar pérdidas de red a escala. **Probado con 5 fuentes (features físicamente coherentes) y con 300 fuentes reales (`probar_gaia_escala.py`): 100% de éxito, 0 lotes fallidos, 0 pérdidas silenciosas, 0 NaN, ~0.90 s/objeto.** Registrado en el sistema de fuentes (`@registrar_fuente`, id=`gaia_dr3`). | ✅ Validado a escala, registrado — **fase cerrada** |
| `probar_gaia_escala.py` | Prueba a 300 objetos (multi-lote): confirma paginación, mide rendimiento real, detecta diversidad de clases (incluida `AGN`, un núcleo galáctico activo — no una estrella — procesado sin error, la prueba de generalización más fuerte de la sesión). | ✅ Ejecutado, resultado real documentado |
| `diagnostico_gaia.py` / `probar_gaia_pequeno.py` | Herramientas de reconocimiento y prueba a pequeña escala usadas para construir `gaia.py` con certeza en vez de suposición — mismo patrón que `diagnostico_simbad.py` en la fase 07. No forman parte del pipeline final. | ✅ Cumplida su función |

**Estado: `anomaly_detector` completo, verificado END-TO-END, y con la
generalización de arquitectura CONFIRMADA con un segundo dataset real**
(esquema canónico + adaptadores StarEmbed y Gaia DR3 + features + detección
de anomalías, ver `whitepaper.md` secciones 3, 4.6 y 4.7). `probar_deteccion.py`
confirmó 100% de recall sobre los candidatos ya conocidos de StarEmbed;
`probar_gaia_pequeno.py` confirmó que el mismo pipeline de features funciona
sin cambios sobre un dataset con arquitectura de adaptador completamente
distinta (red vs. fichero local).

**Sistema de fuentes plegable: ambos adaptadores de Nivel 2 registrados**
(`starembed`, `gaia_dr3`) junto al adaptador tabular genérico de ejemplo
(`ejemplo_generico`) — `fuentes_disponibles()` devuelve las tres, sin import
circular. El pendiente histórico de registrar StarEmbed formalmente queda
cerrado.

Gaia DR3 validado también a escala (300 objetos, 100% éxito, ver fila de
`gaia.py` arriba) — fase de integración cerrada. Siguiente paso: la interfaz
PySide6, con dos fuentes de datos reales y funcionales de base.

## 7. Interfaz PySide6 (app/) — primera versión, generada en esta sesión

**Estado global: código completo, sintaxis verificada en los 8 ficheros, pero
solo parcialmente probado en ejecución real** — no hay PySide6 ni pantalla en
el entorno donde se escribió. Ver desglose por fichero.

| Archivo | Descripcion | Estado |
|---|---|---|
| `app/controlador.py` | Toda la lógica de negocio (cargar fuente, extraer features, ejecutar detección, preparar tabla, obtener curva, exportar), **sin ningún import de PySide6**. Probado de extremo a extremo con datos sintéticos reales a través del adaptador tabular genérico: el objeto anómalo inyectado (amplitud 10× mayor) queda correctamente primero en la tabla por `score_anomalia`. | ✅ Probado de extremo a extremo |
| `app/main.py` | Punto de entrada (`python app\main.py`, ejecutar desde la raíz del proyecto). | ⚠️ Sintaxis verificada, no ejecutado |
| `app/ventana_principal.py` | Ventana principal: selector de fuente, formulario dinámico, tabla de resultados, gráfico de curva, exportación. Ejecuta el análisis en un `QThread` aparte (`HiloAnalisis`) para no congelar la interfaz — importante con Gaia DR3 (minutos de carga, whitepaper.md sección 3). | ⚠️ Sintaxis verificada, coherencia de nombres entre ficheros comprobada a mano, no ejecutado |
| `app/modelo_resultados.py` | `QAbstractTableModel` que envuelve el DataFrame de resultados. | ⚠️ Sintaxis verificada, no ejecutado |
| `app/widgets/formulario_fuente.py` | Genera los campos del formulario dinámicamente a partir de `ParametroConfig` de cada fuente — sin ningún campo hardcodeado por fuente concreta, es el mecanismo que motivó el sistema de fuentes plegable (whitepaper.md sección 5.0). | ⚠️ Sintaxis verificada, no ejecutado |
| `app/widgets/grafico_curva.py` | Curva de luz embebida con matplotlib (`backend_qtagg`). | ⚠️ Sintaxis verificada, no ejecutado |

**Simplificaciones deliberadas de esta primera versión (MVP), no descuidos:**
- **Detección global, no por clase.** Usa `detectar_global()` sobre el propio
  conjunto cargado como referencia y evaluación a la vez (autodetección) —
  no hay todavía en la interfaz un concepto de "conjunto de referencia"
  separado. `detectar_por_clase()`/`detectar_ood_multiclase()` (más potentes,
  ya construidos y verificados) quedan para una iteración futura.
- **Una tabla única ordenable** con columna `candidato_fuerte`, no las dos
  secciones "seguros"/"ambiguos" con comportamiento adaptativo del diseño
  completo (whitepaper.md sección 5.1).

**Primer paso al probarlo:** `pip install -r docs/requirements.txt` (añade
`PySide6` y `matplotlib`, antes ausentes) y `python app\main.py` desde la
raíz. Como con Gaia, es razonable que aparezca algún ajuste en la primera
ejecución real — pega el traceback completo.

## 8. Validación externa con catálogos (auditorias/07_validacion_externa) — generado en esta sesión — CERRADA

| Archivo | Descripcion | Estado |
|---|---|---|
| `cruzar_candidatos_catalogos.py` | Cruza los 8 candidatos `candidato_robusto_interesante` de `24c_candidatos_interseccion.csv` contra SIMBAD y VSX/AAVSO (vía VizieR), usando coordenadas parseadas del identificador `CSS_J...` (convención Catalina Surveys). Bug inicial de columnas SIMBAD (`MAIN_ID`/`OTYPE` vs los reales `main_id`/`otype`) corregido tras diagnóstico real del autor y verificado contra la fila real devuelta por SIMBAD antes de reentregar. | ✅ Corregido y verificado |
| `cruce_catalogos_resultado.csv` | **Resultado final, con SIMBAD funcionando:** las 8/8 tienen contrapartida en SIMBAD (`EB*`, o `SB*`/`V* MX Her` en el caso con nombre propio) y en VSX (7×`EW`, 1×`EA/SD`) — validación cruzada en tres fuentes independientes (SIMBAD, VSX, RF propio), consistente en los 8 casos. Ninguno resultó genuinamente no catalogado ni de tipo taxonómico nuevo. Interpretación: el detector encuentra objetos reales y correctamente clasificados (no ruido) — casos estadísticamente extremos dentro de clases conocidas, coherente con los límites documentados en whitepaper.md sección 4.9. | ✅ Cerrado, documentado en whitepaper.md |
| `diagnostico_simbad.py` | Script de reconocimiento (no forma parte del pipeline final): consulta una sola coordenada real contra SIMBAD e imprime `resultado.colnames` — es el que reveló que las columnas reales son `main_id`/`otype` en minúsculas, no `MAIN_ID`/`OTYPE`. Creado por el autor a partir del snippet dado en el chat. | ✅ Cumplida su función |
| `diagnostico_gaia_dr3.py` | Renombrado desde `diagnostico_gaia.py`. Mismo propósito de reconocimiento para el adaptador de Gaia — confirmó el formato real "ancho" de la tabla de fotometría por época (ver `whitepaper.md` sección 3). No forma parte del pipeline final. | ✅ Cumplida su función |
| `probar_gaia_pequeno.py`, `probar_gaia_escala.py` | Reubicados aquí desde la raíz (ubicación más coherente: son pruebas de validación externa/red, igual categoría que el resto de esta fase). Descripción completa y resultados en la sección 6 de este documento (paquete `anomaly_detector`, fila de `gaia.py`) — no se duplica aquí. | ✅ Ejecutados, ver sección 6 |

## 9. Documentos de proyecto

| Archivo | Descripcion | Estado |
|---|---|---|
| `whitepaper.md` | Fundamento cientifico y metodologico, resumen de validacion, arquitectura propuesta para generalizar a cualquier dataset. | v1.9 |
| `FILES.md` | Este inventario. | v7 |
| `readme.md` | Punto de entrada practico al proyecto: que es, como se usa, estado y hoja de ruta. | actualizado |
| `docs/requirements.txt` | Dependencias del proyecto, repasadas contra todo lo usado en la sesion (pandas, numpy, pyarrow, scikit-learn, scipy, pyyaml, astropy, astroquery). PySide6 anotado como pendiente, no instalado todavia. | ✅ Creado |

## 10. Convenciones de salida de datos (CSV)

Los scripts a partir del 16b siguen el patron `<numero_script>_<descripcion>.csv`
guardado en el mismo `BASE_DIR` que los parquet de entrada. No se han centralizado
en una carpeta de resultados separada — pendiente de decidir si conviene mover a
`outputs/` o similar al construir la version empaquetada del pipeline.

## 11. Pendiente de la proxima actualizacion de este documento

- ~~Confirmar/corregir las descripciones ⚪ de los scripts 01-19~~ — **hecho**:
  auditoria completa de 01-19 incorporada, incluyendo el cierre de la rama de
  investigacion del dataset ZTF DR3 original (M31/DEEP/DISK, ver
  `whitepaper.md` seccion 8) y los resultados reales de 11-19 sobre StarEmbed.
- Registrar StarEmbed formalmente en el sistema de fuentes (`@registrar_fuente`).
- Añadir aquí Gaia DR3 (adaptador de Nivel 2) cuando se construya.
- Añadir el codigo de la aplicacion PySide6 cuando exista.
