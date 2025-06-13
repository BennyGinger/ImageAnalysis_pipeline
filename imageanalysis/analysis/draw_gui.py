from __future__ import annotations
import sys

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsScene, QGraphicsView,
    QGraphicsPixmapItem, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QGroupBox
)
from PyQt6.QtGui import QImage, QPixmap, QPainterPath, QPen
from PyQt6.QtCore import Qt, QRectF
import numpy as np
from skimage.draw import polygon as skimage_polygon


def draw_polygons(image_series: np.ndarray) -> dict[int, np.ndarray]:
    """
    Launch the annotator GUI for a given 16-bit image series.
    Returns a dict mapping frame indices to boolean masks.
    """
    app = QApplication(sys.argv)
    window = ImageAnnotator(image_series)
    window.show()
    app.exec()
    return window.masks


class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.mask_item = QGraphicsPixmapItem()
        self.scene.addItem(self.mask_item)
        self._init_drawing()
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

    def _init_drawing(self):
        if hasattr(self, 'path_item'):
            try:
                self.scene.removeItem(self.path_item)
            except:
                pass
        self.drawing = False
        self.path = QPainterPath()
        self.path_item = self.scene.addPath(self.path, QPen(Qt.GlobalColor.red, 2))
        self.points = []

    def set_image(self, qimage):
        pixmap = QPixmap.fromImage(qimage)
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))
        self._init_drawing()
        self.mask_item.setPixmap(QPixmap())
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def set_mask(self, mask: np.ndarray):
        if mask is None:
            self.mask_item.setPixmap(QPixmap())
            return
        h, w = mask.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[..., 0] = 255
        overlay[..., 3] = (mask.astype(np.uint8) * 100)
        height, width, _ = overlay.shape
        img = QImage(overlay.data, width, height, width*4, QImage.Format.Format_RGBA8888)
        self.mask_item.setPixmap(QPixmap.fromImage(img))
        self.mask_item.setZValue(1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._init_drawing()
            self.drawing = True
            pos = self.mapToScene(event.position().toPoint())
            self.points = [pos]
            self.path.moveTo(pos)
            self.path_item.setPath(self.path)

    def mouseMoveEvent(self, event):
        if self.drawing:
            pos = self.mapToScene(event.position().toPoint())
            self.points.append(pos)
            self.path.lineTo(pos)
            self.path_item.setPath(self.path)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False
            self.path.closeSubpath()
            self.path_item.setPath(self.path)


class ImageAnnotator(QMainWindow):
    def __init__(self, image_series):
        super().__init__()
        self.setWindowTitle("Image Annotator")
        self.image_series = image_series.astype(np.float32)
        self.n_frames, self.h, self.w = self.image_series.shape
        self.current_frame = 0
        self.masks = {}

        # Viewer and navigation
        self.viewer = ImageViewer()
        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")

        # Action buttons
        self.auto_scale_btn = QPushButton("Auto Scale")
        # Make Auto Scale narrower
        self.auto_scale_btn.setMaximumWidth(100)
        self.save_btn = QPushButton("Save Annotation")
        self.process_btn = QPushButton("Process")

        # LUT controls
        self.min_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_slider.setRange(0, np.iinfo(np.uint16).max)
        self.min_slider.setValue(0)
        self.max_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_slider.setRange(0, np.iinfo(np.uint16).max)
        self.max_slider.setValue(np.iinfo(np.uint16).max)
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(1, 300)
        self.contrast_slider.setValue(100)

        # Group LUT + Auto Scale
        self.lut_group = QGroupBox("LUT Controls")
        lut_layout = QVBoxLayout()
        lut_layout.addWidget(QLabel("Min"))
        lut_layout.addWidget(self.min_slider)
        lut_layout.addWidget(QLabel("Max"))
        lut_layout.addWidget(self.max_slider)
        lut_layout.addWidget(QLabel("Brightness %"))
        lut_layout.addWidget(self.brightness_slider)
        lut_layout.addWidget(QLabel("Contrast %"))
        lut_layout.addWidget(self.contrast_slider)
        lut_layout.addWidget(self.auto_scale_btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.lut_group.setLayout(lut_layout)

        # Arrange all controls
        controls = QWidget()
        ctrl_layout = QVBoxLayout(controls)
        ctrl_layout.addWidget(self.prev_btn)
        ctrl_layout.addWidget(self.next_btn)
        ctrl_layout.addWidget(self.lut_group)
        ctrl_layout.addSpacing(12)
        ctrl_layout.addWidget(self.save_btn)
        ctrl_layout.addWidget(self.process_btn)

        # Main container
        container = QWidget()
        main_layout = QHBoxLayout(container)
        main_layout.addWidget(self.viewer)
        main_layout.addWidget(controls)
        self.setCentralWidget(container)

        # Connect signals
        self.prev_btn.clicked.connect(self.load_prev)
        self.next_btn.clicked.connect(self.load_next)
        self.auto_scale_btn.clicked.connect(self.auto_scale)
        self.save_btn.clicked.connect(self.save_annotation)
        self.process_btn.clicked.connect(self.process)
        self.min_slider.valueChanged.connect(self.update_image)
        self.max_slider.valueChanged.connect(self.update_image)
        self.brightness_slider.valueChanged.connect(self.update_image)
        self.contrast_slider.valueChanged.connect(self.update_image)

        self.update_image()
        self._update_nav_buttons()

    def auto_scale(self):
        arr = self.image_series[self.current_frame]
        self.min_slider.setValue(int(np.min(arr)))
        self.max_slider.setValue(int(np.max(arr)))
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)
        self.update_image()

    def update_image(self):
        arr = self.image_series[self.current_frame]
        min_val, max_val = self.min_slider.value(), self.max_slider.value()
        if max_val <= min_val:
            max_val = min_val + 1
        norm = (np.clip(arr, min_val, max_val) - min_val) / (max_val - min_val)
        norm = np.clip(norm * (self.contrast_slider.value()/100.0) + (self.brightness_slider.value()/100.0), 0.0, 1.0)
        img16 = (norm * np.iinfo(np.uint16).max).astype(np.uint16)
        qimage = QImage(img16.data, self.w, self.h, self.w*2, QImage.Format.Format_Grayscale16)
        self.viewer.set_image(qimage)
        self.viewer.set_mask(self.masks.get(self.current_frame))
        self._update_nav_buttons()

    def _update_nav_buttons(self):
        self.prev_btn.setEnabled(self.current_frame > 0)
        self.next_btn.setEnabled(self.current_frame < self.n_frames-1)

    def load_prev(self):
        self.current_frame = max(0, self.current_frame-1)
        self.update_image()

    def load_next(self):
        self.current_frame = min(self.n_frames-1, self.current_frame+1)
        self.update_image()

    def save_annotation(self):
        pts = [(int(p.x()),int(p.y())) for p in self.viewer.points]
        if len(pts) <3:
            print("Draw a polygon first.")
            return
        xs, ys = zip(*pts)
        rr, cc = skimage_polygon(ys, xs, shape=(self.h,self.w))
        mask = np.zeros((self.h,self.w), bool)
        mask[rr, cc] = True
        self.masks[self.current_frame] = mask
        self.viewer.set_mask(mask)
        if self.current_frame < self.n_frames-1:
            self.current_frame += 1
            self.update_image()

    def process(self):
        self.close()


def polygon_into_mask(poly_dict: dict, img_shape: tuple) -> np.ndarray:
    mask_stack = np.zeros(img_shape, "uint8")
    for frame, polygon in poly_dict.items():
        mask_stack[frame] = polygon
    return mask_stack

if __name__ == "__main__":
    from pathlib import Path
    from tifffile import imwrite

    from imageanalysis.utilities.data_utility import load_stack

    folder = Path("/media/ben/Analysis/Python/Docker_mount/Test_images/Dia_annoying/brillouin_LifeActGFP_hypocontr@5frame_002_Brill+Fluor_s1/Images_Registered")
    files = sorted(folder.glob("*.tif"))
    data = load_stack(files, channels='GFP', return_2D=True)
    masks_dict = draw_polygons(data)
    
    # Save the masks to a file or process them as needed
    masks = polygon_into_mask(masks_dict, data.shape)
    output_file = folder.parent.joinpath("masks.tif")
    imwrite(output_file, masks.astype(np.uint8))
