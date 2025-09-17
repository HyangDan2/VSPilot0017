from PySide6 import QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class PlotDialog(QtWidgets.QDialog):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.canvas = FigureCanvas(Figure(figsize=(5, 4), tight_layout=True))
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.ax = None

    def clear(self):
        self.canvas.figure.clf()
        self.ax = None
        self.canvas.draw()

class ESFDialog(QtWidgets.QDialog):
    """ESF dialog: top = strip image, bottom = ESF curves."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ESF")
        self.canvas = FigureCanvas(Figure(figsize=(6, 6), tight_layout=True))
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.ax_img = None
        self.ax_esf = None

    def prepare(self):
        self.canvas.figure.clf()
        gs = self.canvas.figure.add_gridspec(nrows=3, ncols=1,
                                             height_ratios=[2.0, 0.15, 3.2], hspace=0.5)
        self.ax_img = self.canvas.figure.add_subplot(gs[0, 0])
        spacer = self.canvas.figure.add_subplot(gs[1, 0]); spacer.axis('off')
        self.ax_esf = self.canvas.figure.add_subplot(gs[2, 0])
        self.ax_esf.set_title("ESF (oversampled & smoothed)")
        self.ax_esf.set_xlabel("Position (pixels, along normal; oversampled)")
        self.ax_esf.set_ylabel("Intensity")
        self.ax_esf.grid(True, alpha=0.3)

    def clear(self):
        self.canvas.figure.clf()
        self.ax_img = None
        self.ax_esf = None
        self.canvas.draw()
