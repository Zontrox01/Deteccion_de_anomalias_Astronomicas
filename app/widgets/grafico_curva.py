# -*- coding: utf-8 -*-

r"""
======================================================================
widgets/grafico_curva.py - CURVA DE LUZ EMBEBIDA (matplotlib + Qt)
======================================================================

NO PROBADO EN EJECUCION (ver aviso en modelo_resultados.py). Ademas
del riesgo habitual, este fichero depende de que matplotlib tenga
disponible el backend Qt (backend_qtagg, matplotlib >= 3.5, funciona
tanto con PyQt como con PySide) -- si `pip install matplotlib` no
trae ese backend activo, esta es la primera pieza donde mirar.
"""

from __future__ import annotations

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class GraficoCurva(FigureCanvasQTAgg):

    def __init__(self, parent=None):
        self.figura = Figure(figsize=(5, 3.5))
        super().__init__(self.figura)
        if parent is not None:
            self.setParent(parent)
        self.eje = self.figura.add_subplot(111)
        self._dibujar_vacio()

    def _dibujar_vacio(self):
        self.eje.clear()
        self.eje.text(
            0.5, 0.5, "Selecciona un objeto en la tabla",
            ha="center", va="center", transform=self.eje.transAxes,
            color="gray",
        )
        self.eje.set_xticks([])
        self.eje.set_yticks([])
        self.draw()

    def dibujar_curva(
        self,
        tiempo: np.ndarray,
        magnitud: np.ndarray,
        banda: str | None,
        id_objeto: str,
    ):
        self.eje.clear()

        self.eje.scatter(tiempo, magnitud, s=10, alpha=0.7)

        # el eje Y de magnitud va invertido: menor magnitud = mas brillo
        self.eje.invert_yaxis()

        self.eje.set_xlabel("Tiempo (MJD)")
        self.eje.set_ylabel("Magnitud")

        titulo = str(id_objeto)
        if banda:
            titulo += f"  (banda {banda})"
        self.eje.set_title(titulo, fontsize=10)

        self.eje.grid(True, alpha=0.3)

        self.figura.tight_layout()
        self.draw()
