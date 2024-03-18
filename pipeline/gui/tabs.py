from qtpy.QtCore import Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QApplication
from pyqt_vertical_tab_widget.verticalTabWidget import VerticalTabWidget
from PySide6_VerticalQTabWidget import VerticalQTabWidget
import sys
from cellpose.gui import gui
from pipeline.gui import tab_pre_process, tab_wound_mask, tab_test
import warnings



# either use https://github.com/mauriliogenovese/PySide6_VerticalQTabWidget or https://github.com/yjg30737/pyqt-vertical-tab-widget

from qtpy.QtWidgets import (QTabBar, QStylePainter, QStyle, QStyleOptionTab, QTabWidget, QStyleOptionTabWidgetFrame, QApplication)
from qtpy import QtCore


class VerticalQTabWidget(QTabWidget):
    def __init__(self, force_top_valign=False):
        super(VerticalQTabWidget, self).__init__()
        self.setTabBar(VerticalQTabBar())
        self.setTabPosition(QTabWidget.West)
        if force_top_valign:
            self.setStyleSheet("QTabWidget::tab-bar {left : 0;}")  # using stylesheet on initializing

    def paintEvent(self, event):
        painter = QStylePainter(self)
        option = QStyleOptionTabWidgetFrame()
        self.initStyleOption(option)
        option.rect = QtCore.QRect(QtCore.QPoint(self.tabBar().geometry().width(), 0),
                                   QtCore.QSize(option.rect.width(), option.rect.height()))
        painter.drawPrimitive(QStyle.PE_FrameTabWidget, option)


class VerticalQTabBar(QTabBar):
    def __init__(self, *args, **kwargs):
        super(VerticalQTabBar, self).__init__(*args, **kwargs)
        self.setElideMode(QtCore.Qt.ElideNone)

    def tabSizeHint(self, index):
        size_hint = super(VerticalQTabBar, self).tabSizeHint(index)
        size_hint.transpose()
        return size_hint

    def paintEvent(self, event):
        painter = QStylePainter(self)
        option = QStyleOptionTab()
        for index in range(self.count()):
            self.initStyleOption(option, index)
            if QApplication.style().objectName() == "macos":
                option.shape = QTabBar.RoundedNorth
                option.position = QStyleOptionTab.Beginning
            else:
                option.shape = QTabBar.RoundedWest
            painter.drawControl(QStyle.CE_TabBarTabShape, option)
            option.shape = QTabBar.RoundedNorth
            painter.drawControl(QStyle.CE_TabBarTabLabel, option)






def start_gui():
    
    
    warnings.filterwarnings("ignore")

    app = QApplication(sys.argv)
    
    # vertical_tab_widget = VerticalQTabWidget()
    # widget1 = tab_pre_process.preprocess_tab()
    # widget2 = gui.MainW()
    # widget3 = tab_pre_process.preprocess_tab()
    # widget4 = tab_wound_mask.woundmask_tab()
    # widget5 = tab_test.woundmask_tab()
    # widget6 = gui.MainW()
    
    # vertical_tab_widget.addTab(widget1, "Preprocess")
    # vertical_tab_widget.addTab(widget2, "Segmentation")
    # vertical_tab_widget.addTab(widget3, "Tracking")
    # vertical_tab_widget.addTab(widget4, "Wound Mask")
    # vertical_tab_widget.addTab(widget5, "Analysis")
    # vertical_tab_widget.addTab(widget6, "Cellpose")
    
    
    # vertical_tab_widget.show()
    # sys.exit(app.exec_())
    


    myWindow = VerticalTabWidget()
    
    # tab1 = tab_pre_process.preprocess_tab()
    # myWindow.addTab(tab1, 'Preprocess')
    
    tab2 = gui.MainW()
    myWindow.addTab(tab2, 'Segmentation')
    
    # tab3 = tab_pre_process.preprocess_tab()
    # myWindow.addTab(tab3, 'Tracking')
    
    # tab4 = tab_wound_mask.woundmask_tab()
    # myWindow.addTab(tab4, 'Wound Mask')
   
    tab5 = tab_test.woundmask_tab()
    myWindow.addTab(tab5, 'Analysis')
    
    tab6 = gui.MainW()
    myWindow.addTab(tab6, 'Cellpose')



    myWindow.show()
    sys.exit(app.exec_())