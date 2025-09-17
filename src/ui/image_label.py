from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QRect, QPoint

class ImageLabel(QtWidgets.QLabel):
    roiDragFinished = QtCore.Signal(QRect)   # display coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundRole(QtGui.QPalette.Base)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.setScaledContents(False)
        self._pixmap = None

        self._rubberBand = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        self._origin = QPoint()
        self._selecting = False

        self._img_size = None
        self._scaled_size = None
        self._offset = None

        # overlay: list of (QRect img_rect, QColor, label:int)
        self._rois = []

    def setImage(self, qpix: QtGui.QPixmap):
        self._pixmap = qpix
        self._img_size = (qpix.width(), qpix.height())
        self._rois = []
        self.update()

    def setRois(self, rois):
        self._rois = rois
        self.update()

    def hasImage(self):
        return self._pixmap is not None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._pixmap:
            return
        lbl_w = self.width()
        lbl_h = self.height()
        img_w, img_h = self._img_size
        scale = min(lbl_w / img_w, lbl_h / img_h)
        w_s = int(img_w * scale)
        h_s = int(img_h * scale)
        dx = (lbl_w - w_s) // 2
        dy = (lbl_h - h_s) // 2
        self._scaled_size = (w_s, h_s)
        self._offset = (dx, dy)

        painter = QtGui.QPainter(self)
        target = QtCore.QRect(dx, dy, w_s, h_s)
        painter.drawPixmap(target, self._pixmap)

        if self._rois:
            for img_rect, color, label in self._rois:
                disp_rect = self.imageRectToDisplayRect(img_rect)
                if not disp_rect.isEmpty():
                    painter.setPen(QtGui.QPen(color, 2, Qt.SolidLine))
                    painter.drawRect(disp_rect)
                    bg = QtGui.QColor(color); bg.setAlpha(180)
                    text = f"{label}"
                    metrics = painter.fontMetrics()
                    tw = metrics.horizontalAdvance(text) + 8
                    th = metrics.height() + 4
                    tb = QtCore.QRect(disp_rect.x(), max(0, disp_rect.y() - th), tw, th)
                    painter.fillRect(tb, bg)
                    painter.setPen(Qt.white)
                    painter.drawText(tb.adjusted(4, 2, -4, -2), Qt.AlignLeft | Qt.AlignVCenter, text)

        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.hasImage():
            self._origin = event.pos()
            self._rubberBand.setGeometry(QtCore.QRect(self._origin, QtCore.QSize()))
            self._rubberBand.show()
            self._selecting = True
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._selecting:
            rect = QtCore.QRect(self._origin, event.pos()).normalized()
            self._rubberBand.setGeometry(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._selecting:
            self._selecting = False
            self._rubberBand.hide()
            rect = self._rubberBand.geometry().intersected(self.rect())
            if rect.width() > 5 and rect.height() > 5:
                self.roiDragFinished.emit(rect)
        super().mouseReleaseEvent(event)

    def displayRectToImageRect(self, disp_rect: QtCore.QRect) -> QtCore.QRect:
        if not self._pixmap or not self._scaled_size or not self._offset:
            return QtCore.QRect()
        img_w, img_h = self._img_size
        w_s, h_s = self._scaled_size
        dx, dy = self._offset
        image_area = QtCore.QRect(dx, dy, w_s, h_s)
        rect = disp_rect.intersected(image_area)
        if rect.isEmpty():
            return QtCore.QRect()
        rx = rect.x() - dx
        ry = rect.y() - dy
        sx = img_w / w_s
        sy = img_h / h_s
        x = int(round(rx * sx))
        y = int(round(ry * sy))
        w = int(round(rect.width() * sx))
        h = int(round(rect.height() * sy))
        x = max(0, min(img_w - 1, x))
        y = max(0, min(img_h - 1, y))
        if x + w > img_w:
            w = img_w - x
        if y + h > img_h:
            h = img_h - y
        return QtCore.QRect(x, y, w, h)

    def imageRectToDisplayRect(self, img_rect: QtCore.QRect) -> QtCore.QRect:
        if not self._pixmap or not self._scaled_size or not self._offset:
            return QtCore.QRect()
        img_w, img_h = self._img_size
        w_s, h_s = self._scaled_size
        dx, dy = self._offset
        sx = w_s / img_w
        sy = h_s / img_h
        x = int(round(img_rect.x() * sx + dx))
        y = int(round(img_rect.y() * sy + dy))
        w = int(round(img_rect.width() * sx))
        h = int(round(img_rect.height() * sy))
        return QtCore.QRect(x, y, w, h)
