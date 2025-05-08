import pyqtgraph as pg
from qtpy.QtWidgets import QMainWindow, QWidget, QGridLayout, QLabel, QAction, QGroupBox, QComboBox, QCheckBox
from qtpy import QtGui, QtCore
import os, pathlib
from cellpose.gui import guiparts, gui, io
import numpy as np
import matplotlib.pyplot as plt



# ----------------------main Window function ----------------adapted from cellpose



class woundmask_tab(QWidget):
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
        
# -------------------- functions from cellpose for the drawing area ----------------------------ű
    
    
    def make_viewbox(self):
        self.p0 = guiparts.ViewBoxNoRightDrag(
            parent=self,
            lockAspect=True,
            name="plot1",
            border=[100, 100, 100],
            invertY=True
        )
        self.p0.setCursor(QtCore.Qt.CrossCursor)
        self.brush_size=3
        self.win.addItem(self.p0, 0, 0, rowspan=1, colspan=1)
        self.p0.setMenuEnabled(False)
        self.p0.setMouseEnabled(x=True, y=True)
        self.img = pg.ImageItem(viewbox=self.p0, parent=self)
        self.img.autoDownsample = False
        self.layer = guiparts.ImageDraw(viewbox=self.p0, parent=self)
        self.layer.setLevels([0,255])
        self.scale = pg.ImageItem(viewbox=self.p0, parent=self)
        self.scale.setLevels([0,255])
        self.p0.scene().contextMenuItem = self.p0
        #self.p0.setMouseEnabled(x=False,y=False)
        self.Ly,self.Lx = 512,512
        self.p0.addItem(self.img)
        self.p0.addItem(self.layer)
        self.p0.addItem(self.scale)
        
    def make_orthoviews(self):
        self.pOrtho, self.imgOrtho, self.layerOrtho = [], [], []
        for j in range(2):
            self.pOrtho.append(pg.ViewBox(
                                lockAspect=True,
                                name=f'plotOrtho{j}',
                                border=[100, 100, 100],
                                invertY=True,
                                enableMouse=False
                            ))
            self.pOrtho[j].setMenuEnabled(False)

            self.imgOrtho.append(pg.ImageItem(viewbox=self.pOrtho[j], parent=self))
            self.imgOrtho[j].autoDownsample = False

            self.layerOrtho.append(pg.ImageItem(viewbox=self.pOrtho[j], parent=self))
            self.layerOrtho[j].setLevels([0,255])

            #self.pOrtho[j].scene().contextMenuItem = self.pOrtho[j]
            self.pOrtho[j].addItem(self.imgOrtho[j])
            self.pOrtho[j].addItem(self.layerOrtho[j])
            self.pOrtho[j].addItem(self.vLineOrtho[j], ignoreBounds=False)
            self.pOrtho[j].addItem(self.hLineOrtho[j], ignoreBounds=False)
        
        self.pOrtho[0].linkView(self.pOrtho[0].YAxis, self.p0)
        self.pOrtho[1].linkView(self.pOrtho[1].XAxis, self.p0)
    
    def color_choose(self):
        self.color = self.RGBDropDown.currentIndex()
        self.view = 0
        self.ViewDropDown.setCurrentIndex(self.view)
        self.update_plot()
    
    def update_plot(self):
        self.view = self.ViewDropDown.currentIndex()
        self.Ly, self.Lx, _ = self.stack[self.currentZ].shape
        
        if self.restore and "upsample" in self.restore:
            if self.view!=0:
                if self.view==3:
                    self.resize = True 
                elif len(self.flows[0]) > 0 and self.flows[0].shape[1]==self.Lyr:
                    self.resize = True 
                else:
                    self.resize = False
            else:
                self.resize = False
            self.draw_layer()
            self.update_scale()
            self.update_layer()
        
        if self.view==0 or self.view==3:
            image = self.stack[self.currentZ] if self.view==0 else self.stack_filtered[self.currentZ]
            if self.nchan==1:
                # show single channel
                image = image[...,0]
            if self.color==0:
                self.img.setImage(image, autoLevels=False, lut=None)
                if self.nchan > 1: 
                    levels = np.array([self.saturation[0][self.currentZ], 
                                       self.saturation[1][self.currentZ], 
                                       self.saturation[2][self.currentZ]])
                    self.img.setLevels(levels)
                else:
                    self.img.setLevels(self.saturation[0][self.currentZ])
            elif self.color>0 and self.color<4:
                if self.nchan > 1:
                    image = image[:,:,self.color-1]
                self.img.setImage(image, autoLevels=False, lut=self.cmap[self.color])
                if self.nchan > 1:
                    self.img.setLevels(self.saturation[self.color-1][self.currentZ])
                else:
                    self.img.setLevels(self.saturation[0][self.currentZ])
            elif self.color==4:
                if self.nchan > 1:
                    image = image.mean(axis=-1)
                self.img.setImage(image, autoLevels=False, lut=None)
                self.img.setLevels(self.saturation[0][self.currentZ])
            elif self.color==5:
                if self.nchan > 1:
                    image = image.mean(axis=-1)
                self.img.setImage(image, autoLevels=False, lut=self.cmap[0])
                self.img.setLevels(self.saturation[0][self.currentZ])
        else:
            image = np.zeros((self.Ly,self.Lx), np.uint8)
            if len(self.flows)>=self.view-1 and len(self.flows[self.view-1])>0:
                image = self.flows[self.view-1][self.currentZ]
            if self.view>1:
                self.img.setImage(image, autoLevels=False, lut=self.bwr)
            else:
                self.img.setImage(image, autoLevels=False, lut=None)
            self.img.setLevels([0.0, 255.0])
        
        for r in range(3):
            self.sliders[r].setValue([self.saturation[r][self.currentZ][0], 
                                      self.saturation[r][self.currentZ][1]])
        self.win.show()
        self.show()
    
    
    def plot_clicked(self, event):
        if event.button()==QtCore.Qt.LeftButton and not event.modifiers() & (QtCore.Qt.ShiftModifier | QtCore.Qt.AltModifier):
            if event.double():
                try:
                    self.p0.setYRange(0,self.Ly+self.pr)
                except:
                    self.p0.setYRange(0,self.Ly)
                self.p0.setXRange(0,self.Lx)
            elif self.loaded and not self.in_stroke:
                if self.orthobtn.isChecked():
                    items = self.win.scene().items(event.scenePos())
                    for x in items:
                        if x==self.p0:
                            pos = self.p0.mapSceneToView(event.scenePos())
                            x = int(pos.x())
                            y = int(pos.y())
                            if y>=0 and y<self.Ly and x>=0 and x<self.Lx:
                                self.yortho = y 
                                self.xortho = x
                                self.update_ortho()

    def mouse_moved(self, pos):
        items = self.win.scene().items(pos)
        #for x in items:
        #    if not x==self.p0:
        #        QtWidgets.QApplication.restoreOverrideCursor()
        #        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.DefaultCursor)
        
    # def reset(self):
    #     # ---- start sets of points ---- #
    #     self.selected = 0
    #     self.X2 = 0
    #     self.resize = -1
    #     self.onechan = False
    #     self.loaded = False
    #     self.channel = [0,1]
    #     self.current_point_set = []
    #     self.in_stroke = False
    #     self.strokes = []
    #     self.stroke_appended = True
    #     self.ncells = 0
    #     self.zdraw = []
    #     self.removed_cell = []
    #     self.cellcolors = np.array([255,255,255])[np.newaxis,:]
    #     # -- set menus to default -- #
    #     self.color = 0
    #     # self.RGBDropDown.setCurrentIndex(self.color)
    #     self.view = 0
    #     # self.RGBChoose.button(self.view).setChecked(True)
    #     # self.BrushChoose.setCurrentIndex(1)
    #     # self.SCheckBox.setChecked(True)
    #     # self.SCheckBox.setEnabled(False)

    #     # -- zero out image stack -- #
    #     self.opacity = 128 # how opaque masks should be
    #     self.outcolor = [200,200,255,200]
    #     self.NZ, self.Ly, self.Lx = 1,512,512
    #     self.saturation = [[0,255] for n in range(self.NZ)]
    #     # self.slider.setValue([0,255])
    #     #self.slider.setHigh(255)
    #     # self.slider.show()
    #     self.currentZ = 0
    #     self.flows = [[],[],[],[],[[]]]
    #     self.stack = np.zeros((1,self.Ly,self.Lx,3))
    #     # masks matrix
    #     self.layerz = 0*np.ones((self.Ly,self.Lx,4), np.uint8)
    #     # image matrix with a scale disk
    #     self.radii = 0*np.ones((self.Ly,self.Lx,4), np.uint8)
    #     self.cellpix = np.zeros((1,self.Ly,self.Lx), np.uint32)
    #     self.outpix = np.zeros((1,self.Ly,self.Lx), np.uint32)
    #     self.ismanual = np.zeros(0, 'bool')
    #     # self.update_plot()
    #     # self.orthobtn.setChecked(False)
    #     self.filename = []
    #     self.loaded = False
    #     self.recompute_masks = False
        
        
    def reset(self):
        # ---- start sets of points ---- #
        self.selected = 0
        self.nchan = 3
        self.loaded = False
        self.channel = [0,1]
        self.current_point_set = []
        self.in_stroke = False
        self.strokes = []
        self.stroke_appended = True
        self.resize = False
        self.ncells = 0
        self.zdraw = []
        self.removed_cell = []
        self.cellcolors = np.array([255,255,255])[np.newaxis,:]
        

        # -- zero out image stack -- #
        self.opacity = 128 # how opaque masks should be
        self.outcolor = [200,200,255,200]
        self.NZ, self.Ly, self.Lx = 1,224,224
        self.saturation = []
        for r in range(3):
            self.saturation.append([[0,255] for n in range(self.NZ)])
            self.sliders[r].setValue([0,255])
            self.sliders[r].setEnabled(False)
            self.sliders[r].show()
        self.currentZ = 0
        self.flows = [[],[],[],[],[[]]]
        # masks matrix
        # image matrix with a scale disk
        self.stack = np.zeros((1,self.Ly,self.Lx,3))
        self.Lyr, self.Lxr = self.Ly, self.Lx
        self.Ly0, self.Lx0 = self.Ly, self.Lx
        self.radii = 0*np.ones((self.Ly,self.Lx,4), np.uint8)
        self.layerz = 0*np.ones((self.Ly,self.Lx,4), np.uint8)
        self.cellpix = np.zeros((1,self.Ly,self.Lx), np.uint16)
        self.outpix = np.zeros((1,self.Ly,self.Lx), np.uint16)
        if self.restore and "upsample" in self.restore:
            self.cellpix_resize = self.cellpix
            self.cellpix_orig = self.cellpix
            self.outpix_resize = self.cellpix
            self.outpix_orig = self.cellpix
        self.ismanual = np.zeros(0, 'bool')
        
        # -- set menus to default -- #
        self.color = 0
        # self.RGBDropDown.setCurrentIndex(self.color)
        self.view = 0
        # self.ViewDropDown.setCurrentIndex(0)
        # self.ViewDropDown.model().item(3).setEnabled(False)
        self.delete_restore()

        # self.BrushChoose.setCurrentIndex(1)
        self.clear_all()

        #self.update_plot()
        self.filename = []
        self.loaded = False
        self.recompute_masks = False

        self.deleting_multiple = False
        self.removing_cells_list = []
        self.removing_region = False
        self.remove_roi_obj = None        
        
    def clear_all(self):
        self.prev_selected = 0
        self.selected = 0
        if self.restore and "upsample" in self.restore: 
            self.layerz = 0*np.ones((self.Lyr,self.Lxr,4), np.uint8)
            self.cellpix = np.zeros((self.NZ,self.Lyr,self.Lxr), np.uint16)
            self.outpix = np.zeros((self.NZ,self.Lyr,self.Lxr), np.uint16)
            self.cellpix_resize = self.cellpix.copy()
            self.outpix_resize = self.outpix.copy()
            self.cellpix_orig = np.zeros((self.NZ,self.Ly0,self.Lx0), np.uint16)
            self.outpix_orig = np.zeros((self.NZ,self.Ly0,self.Lx0), np.uint16)
        else:
            self.layerz = 0*np.ones((self.Ly,self.Lx,4), np.uint8)
            self.cellpix = np.zeros((self.NZ,self.Ly,self.Lx), np.uint16)
            self.outpix = np.zeros((self.NZ,self.Ly,self.Lx), np.uint16)
        
        self.cellcolors = np.array([255,255,255])[np.newaxis,:]
        self.ncells = 0
        self.toggle_removals()
        self.update_scale()
        self.update_layer()
        
        
        
        
    # def update_crosshairs(self):
    #     self.yortho = min(self.Ly-1, max(0, int(self.yortho)))
    #     self.xortho = min(self.Lx-1, max(0, int(self.xortho)))
    #     self.vLine.setPos(self.xortho)
    #     self.hLine.setPos(self.yortho)
    #     self.vLineOrtho[1].setPos(self.xortho)
    #     self.hLineOrtho[1].setPos(self.dz)
    #     self.vLineOrtho[0].setPos(self.dz)
    #     self.hLineOrtho[0].setPos(self.yortho)
        
    # def update_ortho(self):
    #     if self.NZ>1 and self.orthobtn.isChecked():
    #         dzcurrent = self.dz
    #         self.dz = min(100, max(3,int(self.dzedit.text() )))
    #         self.zaspect = max(0.01, min(100., float(self.zaspectedit.text())))
    #         self.dzedit.setText(str(self.dz))
    #         self.zaspectedit.setText(str(self.zaspect))
    #         self.update_crosshairs()
    #         if self.dz != dzcurrent:
    #             self.pOrtho[0].setXRange(-self.dz/3,self.dz*2 + self.dz/3)
    #             self.pOrtho[1].setYRange(-self.dz/3,self.dz*2 + self.dz/3)

    #         y = self.yortho
    #         x = self.xortho
    #         z = self.currentZ
    #         zmin, zmax = max(0, z-self.dz), min(self.NZ, z+self.dz)
    #         if self.view==0:
    #             for j in range(2):
    #                 if j==0:
    #                     image = self.stack[zmin:zmax, :, x].transpose(1,0,2)
    #                 else:
    #                     image = self.stack[zmin:zmax, y, :]
    #                 if self.color==0:
    #                     if self.onechan:
    #                         # show single channel
    #                         image = image[...,0]
    #                     self.imgOrtho[j].setImage(image, autoLevels=False, lut=None)
    #                 elif self.color>0 and self.color<4:
    #                     image = image[...,self.color-1]
    #                     self.imgOrtho[j].setImage(image, autoLevels=False, lut=self.cmap[self.color])
    #                 elif self.color==4:
    #                     image = image.astype(np.float32).mean(axis=-1).astype(np.uint8)
    #                     self.imgOrtho[j].setImage(image, autoLevels=False, lut=None)
    #                 elif self.color==5:
    #                     image = image.astype(np.float32).mean(axis=-1).astype(np.uint8)
    #                     self.imgOrtho[j].setImage(image, autoLevels=False, lut=self.cmap[0])
    #                 self.imgOrtho[j].setLevels(self.saturation[self.currentZ])
    #             self.pOrtho[0].setAspectLocked(lock=True, ratio=self.zaspect)
    #             self.pOrtho[1].setAspectLocked(lock=True, ratio=1./self.zaspect)

    #         else:
    #             image = np.zeros((10,10), np.uint8)
    #             self.img.setImage(image, autoLevels=False, lut=None)
    #             self.img.setLevels([0.0, 255.0])        
    #     self.win.show()
    #     self.show()

    # def add_orthoviews(self):
    #     self.yortho = self.Ly//2
    #     self.xortho = self.Lx//2
    #     if self.NZ > 1:
    #         self.update_ortho()        
