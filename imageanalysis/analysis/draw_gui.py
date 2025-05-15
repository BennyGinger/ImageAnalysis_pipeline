from __future__ import annotations
import sys

from PyQt6.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel
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

        self.viewer = ImageViewer()
        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.save_btn = QPushButton("Save Annotation")
        self.process_btn = QPushButton("Process")

        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)

        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(1, 300)
        self.contrast_slider.setValue(100)

        controls = QWidget()
        ctrl_layout = QVBoxLayout()
        ctrl_layout.addWidget(self.prev_btn)
        ctrl_layout.addWidget(self.next_btn)
        ctrl_layout.addWidget(QLabel("Brightness"))
        ctrl_layout.addWidget(self.brightness_slider)
        ctrl_layout.addWidget(QLabel("Contrast"))
        ctrl_layout.addWidget(self.contrast_slider)
        ctrl_layout.addWidget(self.save_btn)
        ctrl_layout.addWidget(self.process_btn)
        controls.setLayout(ctrl_layout)

        container = QWidget()
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.viewer)
        main_layout.addWidget(controls)
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.prev_btn.clicked.connect(self.load_prev)
        self.next_btn.clicked.connect(self.load_next)
        self.save_btn.clicked.connect(self.save_annotation)
        self.process_btn.clicked.connect(self.process)
        self.brightness_slider.valueChanged.connect(self.update_image)
        self.contrast_slider.valueChanged.connect(self.update_image)

        self.update_image()
        self._update_nav_buttons()

    def update_image(self):
        arr = self.image_series[self.current_frame]
        img16 = np.clip(
            arr * (self.contrast_slider.value() / 100.0)
            + self.brightness_slider.value(),
            0,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16)
        img8 = (img16 >> 8).astype(np.uint8)
        qimage = QImage(img8.data, self.w, self.h, self.w, QImage.Format.Format_Grayscale8)
        self.viewer.set_image(qimage)
        mask = self.masks.get(self.current_frame, None)
        self.viewer.set_mask(mask)
        self._update_nav_buttons()

    def _update_nav_buttons(self):
        self.prev_btn.setEnabled(self.current_frame > 0)
        self.next_btn.setEnabled(self.current_frame < self.n_frames - 1)

    def load_prev(self):
        self.current_frame = max(0, self.current_frame - 1)
        self.update_image()

    def load_next(self):
        self.current_frame = min(self.n_frames - 1, self.current_frame + 1)
        self.update_image()

    def save_annotation(self):
        # convert points to (x, y) ints
        pts = [(int(p.x()), int(p.y())) for p in self.viewer.points]
        if len(pts) >= 3:
            xs, ys = zip(*pts)  # unpack x/y
            rr, cc = skimage_polygon(ys, xs, shape=(self.h, self.w))
            mask = np.zeros((self.h, self.w), dtype=bool)
            mask[rr, cc] = True
            self.masks[self.current_frame] = mask
            self.viewer.set_mask(mask)
            # auto-advance
            if self.current_frame < self.n_frames - 1:
                self.current_frame += 1
                self.update_image()
        else:
            print("Draw a polygon first.")

    def process(self):
        self.close()

def polygon_into_mask(poly_dict: dict, img_shape: tuple)->np.array:
    mask_stack = np.zeros(shape=img_shape, dtype=('uint8'))
    for frame, polygon in poly_dict.items():
        mask_stack[frame] = polygon
    return mask_stack

if __name__ == "__main__":
    from pathlib import Path
    from tifffile import imwrite

    from utilities.data_utility import load_stack

    folder = Path("/home/Test_images/Dia_annoying/brillouin_LifeActGFP_hypocontr@5frame_002_Brill+Fluor_s1/Images_Registered")
    files = sorted(folder.glob("*.tif"))
    data = load_stack(files, channels='GFP', return_2D=True)
    masks_dict = draw_polygons(data)
    
    # Save the masks to a file or process them as needed
    masks = polygon_into_mask(masks_dict, data.shape)
    output_file = folder.parent.joinpath("masks.tif")
    imwrite(output_file, masks.astype(np.uint8))
