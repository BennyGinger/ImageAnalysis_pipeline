import pyqtgraph as pg
from qtpy.QtWidgets import QMainWindow, QWidget, QGridLayout, QLabel
from qtpy import QtGui, QtCore
import os, pathlib
from cellpose.gui import guiparts, gui
import numpy as np
import matplotlib.pyplot as plt


# def make_bwr():
#     # make a bwr colormap
#     b = np.append(255*np.ones(128), np.linspace(0, 255, 128)[::-1])[:,np.newaxis]
#     r = np.append(np.linspace(0, 255, 128), 255*np.ones(128))[:,np.newaxis]
#     g = np.append(np.linspace(0, 255, 128), np.linspace(0, 255, 128)[::-1])[:,np.newaxis]
#     color = np.concatenate((r,g,b), axis=-1).astype(np.uint8)
#     bwr = pg.ColorMap(pos=np.linspace(0.0,255,256), color=color)
#     return bwr

# def make_spectral():
#     # make spectral colormap
#     r = np.array([0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63,67,71,75,79,83,87,91,95,99,103,107,111,115,119,123,127,131,135,139,143,147,151,155,159,163,167,171,175,179,183,187,191,195,199,203,207,211,215,219,223,227,231,235,239,243,247,251,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255])
#     g = np.array([0,1,2,3,4,5,6,7,8,9,10,9,9,8,8,7,7,6,6,5,5,5,4,4,3,3,2,2,1,1,0,0,0,7,15,23,31,39,47,55,63,71,79,87,95,103,111,119,127,135,143,151,159,167,175,183,191,199,207,215,223,231,239,247,255,247,239,231,223,215,207,199,191,183,175,167,159,151,143,135,128,129,131,132,134,135,137,139,140,142,143,145,147,148,150,151,153,154,156,158,159,161,162,164,166,167,169,170,172,174,175,177,178,180,181,183,185,186,188,189,191,193,194,196,197,199,201,202,204,205,207,208,210,212,213,215,216,218,220,221,223,224,226,228,229,231,232,234,235,237,239,240,242,243,245,247,248,250,251,253,255,251,247,243,239,235,231,227,223,219,215,211,207,203,199,195,191,187,183,179,175,171,167,163,159,155,151,147,143,139,135,131,127,123,119,115,111,107,103,99,95,91,87,83,79,75,71,67,63,59,55,51,47,43,39,35,31,27,23,19,15,11,7,3,0,8,16,24,32,41,49,57,65,74,82,90,98,106,115,123,131,139,148,156,164,172,180,189,197,205,213,222,230,238,246,254])
#     b = np.array([0,7,15,23,31,39,47,55,63,71,79,87,95,103,111,119,127,135,143,151,159,167,175,183,191,199,207,215,223,231,239,247,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,251,247,243,239,235,231,227,223,219,215,211,207,203,199,195,191,187,183,179,175,171,167,163,159,155,151,147,143,139,135,131,128,126,124,122,120,118,116,114,112,110,108,106,104,102,100,98,96,94,92,90,88,86,84,82,80,78,76,74,72,70,68,66,64,62,60,58,56,54,52,50,48,46,44,42,40,38,36,34,32,30,28,26,24,22,20,18,16,14,12,10,8,6,4,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,16,24,32,41,49,57,65,74,82,90,98,106,115,123,131,139,148,156,164,172,180,189,197,205,213,222,230,238,246,254])
#     color = (np.vstack((r,g,b)).T).astype(np.uint8)
#     spectral = pg.ColorMap(pos=np.linspace(0.0,255,256), color=color)
#     return spectral

# def make_cmap(cm=0):
#     # make a single channel colormap
#     r = np.arange(0,256)
#     color = np.zeros((256,3))
#     color[:,cm] = r
#     color = color.astype(np.uint8)
#     cmap = pg.ColorMap(pos=np.linspace(0.0,255,256), color=color)
#     return cmap


# ----------------------main Window function ----------------adapted from cellpose

class preprocess_tab(QMainWindow):
    def __init__(self, image=None):
        super(preprocess_tab, self).__init__()

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
        label_style = """QLabel{
                            color: white
                            } 
                         QToolTip { 
                           background-color: black; 
                           color: white; 
                           border: black solid 1px
                           }"""
        self.boldfont = QtGui.QFont("Arial", 12, QtGui.QFont.Bold)
        self.medfont = QtGui.QFont("Arial", 10)
        self.smallfont = QtGui.QFont("Arial", 8)
        self.headings = ('color: rgb(150,255,150);')
        self.dropdowns = ("color: white;"
                        "background-color: rgb(40,40,40);"
                        "selection-color: white;"
                        "selection-background-color: rgb(50,100,50);")
        self.checkstyle = "color: rgb(190,190,190);"

        label = QLabel('Views:')#[\u2191 \u2193]')
        label.setStyleSheet(self.headings)
        label.setFont(self.boldfont)
        self.l0.addWidget(label, 0,0,1,4)
        
                # cross-hair
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.vLineOrtho = [pg.InfiniteLine(angle=90, movable=False), pg.InfiniteLine(angle=90, movable=False)]
        self.hLineOrtho = [pg.InfiniteLine(angle=0, movable=False), pg.InfiniteLine(angle=0, movable=False)]
        
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
        
        
    def reset(self):
        # ---- start sets of points ---- #
        self.selected = 0
        self.X2 = 0
        self.resize = -1
        self.onechan = False
        self.loaded = False
        self.channel = [0,1]
        self.current_point_set = []
        self.in_stroke = False
        self.strokes = []
        self.stroke_appended = True
        self.ncells = 0
        self.zdraw = []
        self.removed_cell = []
        self.cellcolors = np.array([255,255,255])[np.newaxis,:]
        # -- set menus to default -- #
        self.color = 0
        # self.RGBDropDown.setCurrentIndex(self.color)
        self.view = 0
        # self.RGBChoose.button(self.view).setChecked(True)
        # self.BrushChoose.setCurrentIndex(1)
        # self.SCheckBox.setChecked(True)
        # self.SCheckBox.setEnabled(False)

        # -- zero out image stack -- #
        self.opacity = 128 # how opaque masks should be
        self.outcolor = [200,200,255,200]
        self.NZ, self.Ly, self.Lx = 1,512,512
        self.saturation = [[0,255] for n in range(self.NZ)]
        # self.slider.setValue([0,255])
        #self.slider.setHigh(255)
        # self.slider.show()
        self.currentZ = 0
        self.flows = [[],[],[],[],[[]]]
        self.stack = np.zeros((1,self.Ly,self.Lx,3))
        # masks matrix
        self.layerz = 0*np.ones((self.Ly,self.Lx,4), np.uint8)
        # image matrix with a scale disk
        self.radii = 0*np.ones((self.Ly,self.Lx,4), np.uint8)
        self.cellpix = np.zeros((1,self.Ly,self.Lx), np.uint32)
        self.outpix = np.zeros((1,self.Ly,self.Lx), np.uint32)
        self.ismanual = np.zeros(0, 'bool')
        # self.update_plot()
        # self.orthobtn.setChecked(False)
        self.filename = []
        self.loaded = False
        self.recompute_masks = False
        
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
