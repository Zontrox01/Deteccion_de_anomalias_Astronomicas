# -*- coding: utf-8 -*-

r"""
======================================================================
modelo_resultados.py - QAbstractTableModel sobre el DataFrame de resultados
======================================================================

NO PROBADO EN EJECUCION (sin PySide6/pantalla en el entorno donde se
escribio -- ver controlador.py para lo que si se probo de verdad).
Revisado con cuidado contra la documentacion de Qt, pero la primera
vez que lo ejecutes es razonable que aparezca algun ajuste -- pega el
traceback completo si pasa, igual que con Gaia.
"""

from __future__ import annotations

import pandas as pd

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex


class ModeloResultados(QAbstractTableModel):

    def __init__(self, df: pd.DataFrame | None = None, parent=None):
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()

    def actualizar(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df.reset_index(drop=True)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._df.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):

        if not index.isValid():
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            valor = self._df.iat[index.row(), index.column()]
            if isinstance(valor, float):
                return f"{valor:.4f}"
            return str(valor)

        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):

        if role != Qt.ItemDataRole.DisplayRole:
            return None

        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])

        return str(section + 1)

    def id_objeto_en_fila(self, fila: int):
        if "id_objeto" not in self._df.columns or fila < 0 or fila >= len(self._df):
            return None
        return self._df.iloc[fila]["id_objeto"]
