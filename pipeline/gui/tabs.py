import sys
from qtpy.QtWidgets import QMainWindow, QApplication
from qtpy import QtGui, QtWidgets, QtCore
import tabwidget

class TabWidget(QtWidgets.QTabWidget):
    def __init__(self, *args, **kwargs):
        QtWidgets.QTabWidget.__init__(self, *args, **kwargs)
        self.setTabBar(TabBar(self))
        self.setTabPosition(QtWidgets.QTabWidget.West)

# new additions
class TabBar(QtWidgets.QTabBar):
    def tabSizeHint(self, index):
        s = QtWidgets.QTabBar.tabSizeHint(self, index)
        s.transpose()
        return s

    def paintEvent(self, event):
        painter = QtWidgets.QStylePainter(self)
        opt = QtWidgets.QStyleOptionTab()

        for i in range(self.count()):
            self.initStyleOption(opt, i)
            painter.drawControl(QtWidgets.QStyle.CE_TabBarTabShape, opt)
            painter.save()

            s = opt.rect.size()
            s.transpose()
            r = QtCore.QRect(QtCore.QPoint(), s)
            r.moveCenter(opt.rect.center())
            opt.rect = r

            c = self.tabRect(i).center()
            painter.translate(c)
            painter.rotate(90)
            painter.translate(-c)
            painter.drawControl(QtWidgets.QStyle.CE_TabBarTabLabel, opt)
            painter.restore()

class app_window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = tabwidget.Ui_MainWindow()
        self.ui.setupUi(self)
        # self.ui.tabWidget = TabWidget()
        self.ui.tabWidget.setTabBar(TabBar(self.ui.tabWidget))
        self.ui.tabWidget.setTabPosition(self.ui.tabWidget.West)
        self.ui.tabWidget.insertTab(0, self.ui.tab, "My tab")
        self.ui.pushButton_reach.clicked.connect(self.display)
        self.show()

    def display(self):
        print("reached")
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = app_window()
    w.show()
    sys.exit(app.exec_())