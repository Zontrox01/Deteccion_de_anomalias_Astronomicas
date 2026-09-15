# -*- coding: utf-8 -*-

r"""
======================================================================
main.py - PUNTO DE ENTRADA DE LA APLICACION
======================================================================

Ejecutar desde la raiz del proyecto:

    cd C:\Users\Usuario\Documents\Astronomia
    python app\main.py

(no desde dentro de app\, porque anomaly_detector\ y fuentes\ estan
en la raiz -- ver whitepaper.md seccion 3 y FILES.md seccion 0)
"""

import sys
import os

# permite "python app\main.py" desde la raiz sin instalar el proyecto
# como paquete, y que los imports "from controlador import ..." dentro
# de app\ funcionen
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication

from ventana_principal import VentanaPrincipal


def main():
    app = QApplication(sys.argv)
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
