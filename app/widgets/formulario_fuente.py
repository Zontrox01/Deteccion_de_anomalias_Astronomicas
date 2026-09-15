# -*- coding: utf-8 -*-

r"""
======================================================================
widgets/formulario_fuente.py - FORMULARIO DINAMICO POR FUENTE
======================================================================

NO PROBADO EN EJECUCION (ver aviso en modelo_resultados.py).

Genera los campos del formulario a partir de la lista de
ParametroConfig de la fuente seleccionada (anomaly_detector.adapters.
registro.ParametroConfig) -- ningun campo esta hardcodeado por
fuente concreta, es literalmente el mecanismo que motivo construir
el sistema de fuentes plegable (whitepaper.md, seccion 5.0).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QFormLayout, QLineEdit, QPushButton, QHBoxLayout, QFileDialog,
)


class FormularioFuente(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QFormLayout(self)
        self._campos: dict[str, QWidget] = {}

    def construir(self, parametros):
        """Reconstruye el formulario entero para una fuente nueva."""

        while self._layout.rowCount() > 0:
            self._layout.removeRow(0)

        self._campos = {}

        for parametro in parametros:

            if parametro.tipo in ("ruta_archivo", "ruta_carpeta"):
                campo = self._crear_campo_ruta(parametro)
            else:
                campo = QLineEdit()
                if parametro.valor_por_defecto is not None:
                    campo.setText(str(parametro.valor_por_defecto))
                if parametro.ayuda:
                    campo.setToolTip(parametro.ayuda)

            self._campos[parametro.nombre] = campo

            etiqueta = parametro.etiqueta + (" *" if parametro.requerido else "")
            self._layout.addRow(etiqueta, campo)

    def _crear_campo_ruta(self, parametro):
        """
        Campo de texto + boton 'Examinar...'. Si el parametro es
        'ruta_archivo' y su ayuda menciona multiples ficheros (caso de
        StarEmbed, que necesita varios parquet a la vez), se permite
        seleccion multiple -- las rutas se guardan separadas por ';'
        y controlador._convertir_parametros() ya sabe partirlas.
        """

        contenedor = QWidget()
        h = QHBoxLayout(contenedor)
        h.setContentsMargins(0, 0, 0, 0)

        campo_texto = QLineEdit()
        if parametro.ayuda:
            campo_texto.setToolTip(parametro.ayuda)

        boton = QPushButton("Examinar...")

        permite_multiple = (
            parametro.tipo == "ruta_archivo"
            and parametro.ayuda
            and "varios" in parametro.ayuda.lower()
        )

        def elegir():
            if parametro.tipo == "ruta_carpeta":
                ruta = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta")
                if ruta:
                    campo_texto.setText(ruta)
            elif permite_multiple:
                rutas, _ = QFileDialog.getOpenFileNames(self, "Seleccionar ficheros")
                if rutas:
                    campo_texto.setText(";".join(rutas))
            else:
                ruta, _ = QFileDialog.getOpenFileName(self, "Seleccionar fichero")
                if ruta:
                    campo_texto.setText(ruta)

        boton.clicked.connect(elegir)

        h.addWidget(campo_texto)
        h.addWidget(boton)

        # guardamos una referencia directa para poder leer el valor
        # despues sin tener que buscar dentro del layout
        contenedor.campo_texto = campo_texto

        return contenedor

    def obtener_valores(self) -> dict:
        valores = {}
        for nombre, campo in self._campos.items():
            if hasattr(campo, "campo_texto"):
                valores[nombre] = campo.campo_texto.text()
            else:
                valores[nombre] = campo.text()
        return valores
