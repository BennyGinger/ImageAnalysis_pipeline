from qtpy.QtCore import Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QApplication
from pyqt_vertical_tab_widget.verticalTabWidget import VerticalTabWidget
import sys
from cellpose.gui import gui
from pipeline.gui import tab_pre_process, tab_wound_mask
import warnings


def start_gui():
    warnings.filterwarnings("ignore")

    app = QApplication(sys.argv)

    myWindow = VerticalTabWidget()
    
    tab1 = tab_pre_process.preprocess_tab()
    myWindow.addTab(tab1, 'Preprocess')
    
    tab2 = gui.MainW()
    myWindow.addTab(tab2, 'Segmentation')
    
    tab3 = tab_pre_process.preprocess_tab()
    myWindow.addTab(tab3, 'Tracking')
    
    # tab4 = tab_wound_mask.woundmask_tab()
    # myWindow.addTab(tab4, 'Wound Mask')
   
    tab5 = tab_pre_process.preprocess_tab()
    myWindow.addTab(tab5, 'Analysis')
    
    tab6 = gui.MainW()
    myWindow.addTab(tab6, 'Cellpose')



    myWindow.show()
    sys.exit(app.exec_())