from qtpy.QtCore import Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QApplication, QLabel
from pyqt_vertical_tab_widget.verticalTabWidget import VerticalTabWidget
import sys
from cellpose.gui import gui
from pipeline.gui import tab_pre_process
import warnings


def start_gui():
    warnings.filterwarnings("ignore")

    app = QApplication(sys.argv)

    myWindow = VerticalTabWidget()
    tab1 = gui.MainW()
    myWindow.addTab(tab1, 'Cellpose')

    tab2 = tab_pre_process.preprocess_tab()
    myWindow.addTab(tab2, 'Preprocess')

    myWindow.show()
    sys.exit(app.exec_())