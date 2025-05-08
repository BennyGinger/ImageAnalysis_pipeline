from cellpose.gui.gui import MainW
import pyqtgraph as pg
from qtpy.QtWidgets import QMainWindow, QWidget, QGridLayout, QLabel, QAction, QGroupBox, QComboBox, QCheckBox, QPushButton
from qtpy import QtGui, QtCore
import os, pathlib
from cellpose.gui import guiparts, gui, io
import numpy as np
import matplotlib.pyplot as plt

class woundmask_tab(MainW):
    def __init__(self, image=None):
        super(woundmask_tab, self).__init__()

        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(50, 50, 1200, 1000)
        self.setWindowTitle("Image preprocessing")
        self.cp_path = os.path.dirname(os.path.realpath(__file__))
        app_icon = QtGui.QIcon()
        icon_path = pathlib.Path.home().joinpath('zebrafish.ico')
        icon_path = str(icon_path.resolve())
        app_icon.addFile(icon_path, QtCore.QSize(16, 16))
        app_icon.addFile(icon_path, QtCore.QSize(24, 24))
        app_icon.addFile(icon_path, QtCore.QSize(32, 32))
        app_icon.addFile(icon_path, QtCore.QSize(48, 48))
        app_icon.addFile(icon_path, QtCore.QSize(64, 64))
        app_icon.addFile(icon_path, QtCore.QSize(256, 256))
        self.setWindowIcon(app_icon)

        # menus.mainmenu(self)
        # menus.editmenu(self)
        # menus.modelmenu(self)
        # menus.helpmenu(self)

        self.setStyleSheet("QMainWindow {background: 'black';}")
        self.stylePressed = ("QPushButton {Text-align: left; "
                             "background-color: rgb(150,50,150); "
                             "border-color: white;"
                             "color:white;}")
        self.styleUnpressed = ("QPushButton {Text-align: left; "
                               "background-color: rgb(50,50,50); "
                                "border-color: white;"
                               "color:white;}")
        self.styleInactive = ("QPushButton {Text-align: left; "
                              "background-color: rgb(30,30,30); "
                             "border-color: white;"
                              "color:rgb(80,80,80);}")
        self.loaded = False

        # ---- MAIN WIDGET LAYOUT ---- #
        self.cwidget = QWidget(self)
        self.l0 = QGridLayout()
        self.cwidget.setLayout(self.l0)
        self.setCentralWidget(self.cwidget)
        self.l0.setVerticalSpacing(6)

        self.imask = 0

        b = self.make_buttons()
        # ---- drawing area ---- #
        self.win = pg.GraphicsLayoutWidget()
        
        self.l0.addWidget(self.win, 0, 9, 1, 30) 
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        self.win.scene().sigMouseMoved.connect(self.mouse_moved)
        self.make_viewbox()
        self.make_orthoviews()
        self.l0.setColumnStretch(10, 1)
        bwrmap = gui.make_bwr()
        self.bwr = bwrmap.getLookupTable(start=0.0, stop=255.0, alpha=False)
        self.cmap = []
        # spectral colormap
        self.cmap.append(gui.make_spectral().getLookupTable(start=0.0, stop=255.0, alpha=False))
        # single channel colormaps
        for i in range(3):
            self.cmap.append(gui.make_cmap(i).getLookupTable(start=0.0, stop=255.0, alpha=False))


        self.colormap = (plt.get_cmap('gist_ncar')(np.linspace(0.0,.9,1000000)) * 255).astype(np.uint8)
        np.random.seed(42) # make colors stable
        self.colormap = self.colormap[np.random.permutation(1000000)]

        self.reset()

        self.is_stack = True # always loading images of same FOV
        
        self.autoloadMasks = QAction("Autoload masks from _masks.tif file", self, checkable=True)
        self.autoloadMasks.setChecked(True)
        self.restore = None
        self.filename = '/home/Fabian/ImageData/mfap4-mpx_isohypo_2h_WT-MaxIP_s1/Images_Registered/GFP_s01_f0001_z0001.tif'
        io._load_image(self, self.filename)
        
        

        # file_menu.addAction(parent.autoloadMasks)
        
        # if called with image, load it
        # if image is not None:
        #     self.filename = image
        #     io._load_image(self, self.filename)

        # # training settings
        # d = datetime.datetime.now()
        # self.training_params = {'model_index': 0,
        #                         'learning_rate': 0.1, 
        #                         'weight_decay': 0.0001, 
        #                         'n_epochs': 100,
        #                         'model_name': 'CP' + d.strftime("_%Y%m%d_%H%M%S")
        #                        }

        self.setAcceptDrops(True)
        self.win.show()
        self.show()
        
        
    def make_buttons(self):
        self.boldfont = QtGui.QFont("Arial", 11, QtGui.QFont.Bold)
        self.boldmedfont = QtGui.QFont("Arial", 9, QtGui.QFont.Bold)
        self.medfont = QtGui.QFont("Arial", 9)
        self.smallfont = QtGui.QFont("Arial", 8)
        
        b=0
        self.satBox = QGroupBox('Views')
        self.satBox.setFont(self.boldfont)
        self.satBoxG = QGridLayout()
        self.satBox.setLayout(self.satBoxG)
        self.l0.addWidget(self.satBox, b, 0, 1, 9)

        b0=0
        self.view = 0 # 0=image, 1=flowsXY, 2=flowsZ, 3=cellprob
        self.color = 0 # 0=RGB, 1=gray, 2=R, 3=G, 4=B
        self.RGBDropDown = QComboBox()
        self.RGBDropDown.addItems(["RGB","red=R","green=G","blue=B","gray","spectral"])
        self.RGBDropDown.setFont(self.medfont)
        self.RGBDropDown.currentIndexChanged.connect(self.color_choose)
        self.satBoxG.addWidget(self.RGBDropDown, b0,0,1,3)
        
        label = QLabel('<p>[&uarr; / &darr; or W/S]</p>'); label.setFont(self.smallfont)
        self.satBoxG.addWidget(label, b0,3,1,3)
        label = QLabel('[R / G / B \n toggles color ]'); label.setFont(self.smallfont)
        self.satBoxG.addWidget(label, b0,6,1,3)

        b0+=1
        self.ViewDropDown = QComboBox()
        self.ViewDropDown.addItems(["image", "gradXY", "cellprob", "restored"])
        self.ViewDropDown.setFont(self.medfont)
        self.ViewDropDown.model().item(3).setEnabled(False)
        self.ViewDropDown.currentIndexChanged.connect(self.update_plot)
        self.satBoxG.addWidget(self.ViewDropDown, b0,0,2,3)

        label = QLabel('[pageup / pagedown]')
        label.setFont(self.smallfont)
        self.satBoxG.addWidget(label, b0,3,1,5)

        b0+=2
        label = QLabel('')
        label.setToolTip('NOTE: manually changing the saturation bars does not affect normalization in segmentation')
        self.satBoxG.addWidget(label, b0,0,1,5)

        self.autobtn = QCheckBox('auto-adjust saturation')
        self.autobtn.setToolTip('sets scale-bars as normalized for segmentation')
        self.autobtn.setFont(self.medfont)
        self.autobtn.setChecked(True)
        self.satBoxG.addWidget(self.autobtn, b0,1,1,8)

        b0+=1
        self.sliders = []
        colors = [[255,0,0], [0,255,0], [0,0,255], [100,100,100]]
        colornames = ['red', 'Chartreuse', 'DodgerBlue']
        names = ['red', 'green', 'blue']
        for r in range(3):
            b0+=1
            if r==0:
                label = QLabel('<font color="gray">gray/</font><br>red')
            else:
                label = QLabel(names[r] + ':')
            label.setStyleSheet(f"color: {colornames[r]}")
            label.setFont(self.boldmedfont)
            self.satBoxG.addWidget(label, b0, 0, 1, 2)
            self.sliders.append(gui.Slider(self, names[r], colors[r]))
            self.sliders[-1].setMinimum(-.1)
            self.sliders[-1].setMaximum(255.1)
            self.sliders[-1].setValue([0, 255])
            self.sliders[-1].setToolTip('NOTE: manually changing the saturation bars does not affect normalization in segmentation')
            #self.sliders[-1].setTickPosition(QSlider.TicksRight)
            self.satBoxG.addWidget(self.sliders[-1], b0, 2,1,7)
        
        b+=1
        self.drawBox = QGroupBox('Drawing')
        self.drawBox.setFont(self.boldfont)
        self.drawBoxG = QGridLayout()
        self.drawBox.setLayout(self.drawBoxG)
        self.l0.addWidget(self.drawBox, b, 0, 1, 9)
        self.autosave = True

        b0 = 0
        self.brush_size = 3
        self.BrushChoose = QComboBox()
        self.BrushChoose.addItems(["1","3","5","7","9"])
        self.BrushChoose.currentIndexChanged.connect(self.brush_choose)
        self.BrushChoose.setFixedWidth(40)
        self.BrushChoose.setFont(self.medfont)
        self.drawBoxG.addWidget(self.BrushChoose, b0, 3,1,2)
        label = QLabel('brush size:')
        label.setFont(self.medfont)
        self.drawBoxG.addWidget(label, b0,0,1,3)
        
        b0+=1
        # turn off masks
        self.layer_off = False
        self.masksOn = True
        self.MCheckBox = QCheckBox('MASKS ON [X]')
        self.MCheckBox.setFont(self.medfont)
        self.MCheckBox.setChecked(True)
        self.MCheckBox.toggled.connect(self.toggle_masks)
        self.drawBoxG.addWidget(self.MCheckBox, b0,0,1,5)

        b0+=1
        # turn off outlines
        self.outlinesOn = False # turn off by default
        self.OCheckBox = QCheckBox('outlines on [Z]')
        self.OCheckBox.setFont(self.medfont)
        self.drawBoxG.addWidget(self.OCheckBox, b0,0,1,5)
        self.OCheckBox.setChecked(False)
        self.OCheckBox.toggled.connect(self.toggle_masks) 

        b0+=1
        self.SCheckBox = QCheckBox('single stroke')
        self.SCheckBox.setFont(self.medfont)
        self.SCheckBox.setChecked(True)
        self.SCheckBox.toggled.connect(self.autosave_on)
        self.SCheckBox.setEnabled(True)
        self.drawBoxG.addWidget(self.SCheckBox, b0,0,1,5)


        # buttons for deleting multiple cells
        self.deleteBox = QGroupBox('delete multiple ROIs')
        self.deleteBox.setStyleSheet('color: rgb(200, 200, 200)')
        self.deleteBox.setFont(self.medfont)
        self.deleteBoxG = QGridLayout()
        self.deleteBox.setLayout(self.deleteBoxG)
        self.drawBoxG.addWidget(self.deleteBox, 0, 5, 4, 4)
        self.MakeDeletionRegionButton = QPushButton('region-select')
        self.MakeDeletionRegionButton.clicked.connect(self.remove_region_cells)
        self.deleteBoxG.addWidget(self.MakeDeletionRegionButton, 0, 0, 1, 4)
        self.MakeDeletionRegionButton.setFont(self.smallfont)
        self.MakeDeletionRegionButton.setFixedWidth(70)
        self.DeleteMultipleROIButton = QPushButton('click-select')
        self.DeleteMultipleROIButton.clicked.connect(self.delete_multiple_cells)
        self.deleteBoxG.addWidget(self.DeleteMultipleROIButton, 1, 0, 1, 4)
        self.DeleteMultipleROIButton.setFont(self.smallfont)
        self.DeleteMultipleROIButton.setFixedWidth(70)
        self.DoneDeleteMultipleROIButton = QPushButton('done')
        self.DoneDeleteMultipleROIButton.clicked.connect(self.done_remove_multiple_cells)
        self.deleteBoxG.addWidget(self.DoneDeleteMultipleROIButton, 2, 0, 1, 2)
        self.DoneDeleteMultipleROIButton.setFont(self.smallfont)
        self.DoneDeleteMultipleROIButton.setFixedWidth(35)
        self.CancelDeleteMultipleROIButton = QPushButton('cancel')
        self.CancelDeleteMultipleROIButton.clicked.connect(self.cancel_remove_multiple)
        self.deleteBoxG.addWidget(self.CancelDeleteMultipleROIButton, 2, 2, 1, 2)
        self.CancelDeleteMultipleROIButton.setFont(self.smallfont)
        self.CancelDeleteMultipleROIButton.setFixedWidth(35)