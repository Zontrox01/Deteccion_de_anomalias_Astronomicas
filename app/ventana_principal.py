# -*- coding: utf-8 -*-

r"""
======================================================================
ventana_principal.py - VENTANA PRINCIPAL DE LA APP
======================================================================

NO PROBADA EN EJECUCION (ver aviso en modelo_resultados.py).

Toda la logica real (cargar_y_analizar, obtener_curva_para_grafico,
exportar_csv) vive en controlador.py y SI esta probada de extremo a
extremo. Este fichero es "pegamento" de Qt: conecta señales y
widgets, no toma ninguna decision de negocio por su cuenta.

El analisis se ejecuta en un QThread aparte (HiloAnalisis) para no
congelar la interfaz -- imprescindible con Gaia DR3, donde cargar
puede tardar varios minutos (whitepaper.md, seccion 3).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton,
    QTableView, QSplitter, QLabel, QMessageBox, QFileDialog, QProgressBar,
    QHeaderView,
)

from controlador import (
    listar_fuentes, cargar_y_analizar, obtener_curva_para_grafico,
    exportar_csv, ResultadoAnalisis,
)
from modelo_resultados import ModeloResultados
from widgets.formulario_fuente import FormularioFuente
from widgets.grafico_curva import GraficoCurva


class HiloAnalisis(QThread):
    """Ejecuta cargar_y_analizar() fuera del hilo de la interfaz."""

    terminado = Signal(object)
    fallo = Signal(str)

    def __init__(self, fuente, valores_parametros, parent=None):
        super().__init__(parent)
        self.fuente = fuente
        self.valores_parametros = valores_parametros

    def run(self):
        try:
            resultado = cargar_y_analizar(self.fuente, self.valores_parametros)
            self.terminado.emit(resultado)
        except Exception as e:
            self.fallo.emit(f"{type(e).__name__}: {e}")


class VentanaPrincipal(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Detector de Anomalías Astronómicas")
        self.resize(1150, 700)

        self.resultado_actual: ResultadoAnalisis | None = None
        self.hilo: HiloAnalisis | None = None
        self.fuentes = []

        self._construir_ui()
        self._cargar_fuentes()

    # ------------------------------------------------------------------
    # CONSTRUCCION DE LA INTERFAZ
    # ------------------------------------------------------------------

    def _construir_ui(self):

        central = QWidget()
        self.setCentralWidget(central)
        layout_principal = QVBoxLayout(central)

        # --- fila superior: fuente + acciones ---
        fila_superior = QHBoxLayout()

        fila_superior.addWidget(QLabel("Fuente de datos:"))

        self.combo_fuentes = QComboBox()
        self.combo_fuentes.currentIndexChanged.connect(self._al_cambiar_fuente)
        fila_superior.addWidget(self.combo_fuentes, stretch=1)

        self.boton_analizar = QPushButton("Cargar y analizar")
        self.boton_analizar.clicked.connect(self._al_pulsar_analizar)
        fila_superior.addWidget(self.boton_analizar)

        self.boton_exportar = QPushButton("Exportar CSV")
        self.boton_exportar.setEnabled(False)
        self.boton_exportar.clicked.connect(self._al_pulsar_exportar)
        fila_superior.addWidget(self.boton_exportar)

        layout_principal.addLayout(fila_superior)

        # --- formulario dinamico de parametros de la fuente ---
        self.formulario_fuente = FormularioFuente()
        layout_principal.addWidget(self.formulario_fuente)

        # --- barra de progreso (indeterminada, se muestra durante el analisis) ---
        self.barra_progreso = QProgressBar()
        self.barra_progreso.setRange(0, 0)
        self.barra_progreso.hide()
        layout_principal.addWidget(self.barra_progreso)

        # --- zona central: tabla + grafico, redimensionable ---
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.tabla = QTableView()
        self.modelo_tabla = ModeloResultados()
        self.tabla.setModel(self.modelo_tabla)
        self.tabla.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.tabla.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.tabla.setSortingEnabled(True)
        self.tabla.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        splitter.addWidget(self.tabla)

        self.grafico = GraficoCurva()
        splitter.addWidget(self.grafico)

        splitter.setSizes([700, 450])
        layout_principal.addWidget(splitter, stretch=1)

        # --- barra de estado ---
        self.etiqueta_estado = QLabel("Selecciona una fuente de datos para empezar.")
        layout_principal.addWidget(self.etiqueta_estado)

    # ------------------------------------------------------------------
    # FUENTES
    # ------------------------------------------------------------------

    def _cargar_fuentes(self):

        self.fuentes = listar_fuentes()

        self.combo_fuentes.clear()
        for fuente in self.fuentes:
            self.combo_fuentes.addItem(fuente.nombre, fuente.id)

        if not self.fuentes:
            self.etiqueta_estado.setText(
                "No se encontro ninguna fuente de datos disponible. "
                "Revisa la carpeta fuentes\\ y los adaptadores registrados."
            )
            self.boton_analizar.setEnabled(False)
            return

        self._al_cambiar_fuente(0)

    def _al_cambiar_fuente(self, indice: int):
        if indice < 0 or indice >= len(self.fuentes):
            return
        fuente = self.fuentes[indice]
        self.formulario_fuente.construir(fuente.parametros)

    def _fuente_actual(self):
        indice = self.combo_fuentes.currentIndex()
        if indice < 0 or indice >= len(self.fuentes):
            return None
        return self.fuentes[indice]

    # ------------------------------------------------------------------
    # ANALISIS
    # ------------------------------------------------------------------

    def _al_pulsar_analizar(self):

        fuente = self._fuente_actual()
        if fuente is None:
            return

        valores_parametros = self.formulario_fuente.obtener_valores()

        self.boton_analizar.setEnabled(False)
        self.boton_exportar.setEnabled(False)
        self.barra_progreso.show()
        self.etiqueta_estado.setText(
            f"Cargando y analizando '{fuente.nombre}'... "
            f"esto puede tardar desde segundos hasta varios minutos "
            f"segun la fuente."
        )

        self.hilo = HiloAnalisis(fuente, valores_parametros, parent=self)
        self.hilo.terminado.connect(self._al_terminar_analisis)
        self.hilo.fallo.connect(self._al_fallar_analisis)
        self.hilo.start()

    def _al_terminar_analisis(self, resultado: ResultadoAnalisis):

        self.resultado_actual = resultado
        self.modelo_tabla.actualizar(resultado.tabla)

        # reconectar la seleccion cada vez (el modelo se resetea al
        # actualizar, lo que invalida la conexion anterior)
        self.tabla.selectionModel().selectionChanged.connect(self._al_seleccionar_fila)

        self.barra_progreso.hide()
        self.boton_analizar.setEnabled(True)
        self.boton_exportar.setEnabled(True)

        n_candidatos = int(resultado.tabla["candidato_fuerte"].sum())

        self.etiqueta_estado.setText(
            f"{len(resultado.tabla)} objetos analizados con '{resultado.fuente_nombre}' "
            f"({len(resultado.variables_usadas)} variables). "
            f"{n_candidatos} candidatos fuertes. "
            f"Selecciona una fila para ver su curva de luz."
        )

    def _al_fallar_analisis(self, mensaje_error: str):

        self.barra_progreso.hide()
        self.boton_analizar.setEnabled(True)
        self.etiqueta_estado.setText("Error durante el analisis -- ver detalle.")

        QMessageBox.critical(self, "Error durante el análisis", mensaje_error)

    # ------------------------------------------------------------------
    # SELECCION DE FILA -> GRAFICO
    # ------------------------------------------------------------------

    def _al_seleccionar_fila(self, seleccionado, deseleccionado):

        indices = self.tabla.selectionModel().selectedRows()
        if not indices or self.resultado_actual is None:
            return

        fila = indices[0].row()
        id_objeto = self.modelo_tabla.id_objeto_en_fila(fila)

        if id_objeto is None:
            return

        try:
            tiempo, magnitud, banda = obtener_curva_para_grafico(
                self.resultado_actual, id_objeto
            )
            self.grafico.dibujar_curva(tiempo, magnitud, banda, id_objeto)
        except Exception as e:
            self.etiqueta_estado.setText(f"No se pudo dibujar la curva de {id_objeto}: {e}")

    # ------------------------------------------------------------------
    # EXPORTAR
    # ------------------------------------------------------------------

    def _al_pulsar_exportar(self):

        if self.resultado_actual is None:
            return

        ruta, _ = QFileDialog.getSaveFileName(
            self, "Exportar resultados", "resultados_anomalias.csv", "CSV (*.csv)"
        )

        if not ruta:
            return

        try:
            exportar_csv(self.resultado_actual, ruta)
            self.etiqueta_estado.setText(f"Exportado a: {ruta}")
        except Exception as e:
            QMessageBox.critical(self, "Error al exportar", str(e))
