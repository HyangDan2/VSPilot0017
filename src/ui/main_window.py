import os
import numpy as np
import cv2

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.slanted_edge import slanted_edge_esf_lsf_mtf
from ui.image_label import ImageLabel
from ui.dialogs import PlotDialog, ESFDialog
from utils.colors import qcolor_for_index, qcolor_to_rgba

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Slanted Edge – Multi-ROI ESF/LSF/MTF (sfrmat3-alignment MVP + Save=C)")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # Left: image
        left = QtWidgets.QVBoxLayout()
        layout.addLayout(left, 2)

        self.imageLabel = ImageLabel()
        self.imageLabel.setStyleSheet("QLabel { background: #202020; }")
        self.imageLabel.setAlignment(Qt.AlignCenter)
        left.addWidget(self.imageLabel, stretch=1)

        ctrl_row = QtWidgets.QHBoxLayout()
        left.addLayout(ctrl_row)
        self.btnLoad = QtWidgets.QPushButton("Load Image")
        self.btnAddFromDrag = QtWidgets.QPushButton("Add ROI (from drag)")
        self.btnAddFromDrag.setEnabled(False)
        self.btnClearRois = QtWidgets.QPushButton("Clear ROIs")
        self.btnSave = QtWidgets.QPushButton("Save (C)")
        ctrl_row.addWidget(self.btnLoad)
        ctrl_row.addWidget(self.btnAddFromDrag)
        ctrl_row.addWidget(self.btnClearRois)
        ctrl_row.addWidget(self.btnSave)

        # Right: controls
        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right, 1)

        grid = QtWidgets.QGridLayout()
        right.addLayout(grid)

        # ROI numeric input
        grid.addWidget(QtWidgets.QLabel("ROI x"), 0, 0)
        self.spinX = QtWidgets.QSpinBox(); self.spinX.setRange(0, 100000)
        grid.addWidget(self.spinX, 0, 1)

        grid.addWidget(QtWidgets.QLabel("ROI y"), 1, 0)
        self.spinY = QtWidgets.QSpinBox(); self.spinY.setRange(0, 100000)
        grid.addWidget(self.spinY, 1, 1)

        grid.addWidget(QtWidgets.QLabel("ROI w"), 2, 0)
        self.spinW = QtWidgets.QSpinBox(); self.spinW.setRange(1, 100000)
        grid.addWidget(self.spinW, 2, 1)

        grid.addWidget(QtWidgets.QLabel("ROI h"), 3, 0)
        self.spinH = QtWidgets.QSpinBox(); self.spinH.setRange(1, 100000)
        grid.addWidget(self.spinH, 3, 1)

        self.btnAddNumeric = QtWidgets.QPushButton("Add ROI (numeric)")
        grid.addWidget(self.btnAddNumeric, 4, 0, 1, 2)

        # Pixel pitch
        grid.addWidget(QtWidgets.QLabel("Pixel pitch (mm/pixel)"), 5, 0)
        self.spinPitch = QtWidgets.QDoubleSpinBox()
        self.spinPitch.setDecimals(6); self.spinPitch.setRange(1e-6, 10.0)
        self.spinPitch.setSingleStep(0.001); self.spinPitch.setValue(0.005)
        grid.addWidget(self.spinPitch, 5, 1)

        # ESF smoothing sigma
        grid.addWidget(QtWidgets.QLabel("ESF smoothing σ"), 6, 0)
        self.spinSigma = QtWidgets.QDoubleSpinBox()
        self.spinSigma.setDecimals(3); self.spinSigma.setRange(0.1, 5.0)
        self.spinSigma.setSingleStep(0.1); self.spinSigma.setValue(1.0)
        grid.addWidget(self.spinSigma, 6, 1)

        # ESF tail drop N
        grid.addWidget(QtWidgets.QLabel("ESF: drop last N points"), 7, 0)
        self.spinEsfDropN = QtWidgets.QSpinBox()
        self.spinEsfDropN.setRange(0, 10000); self.spinEsfDropN.setValue(0)
        grid.addWidget(self.spinEsfDropN, 7, 1)

        # Channel
        grid.addWidget(QtWidgets.QLabel("Channel"), 8, 0)
        self.comboChannel = QtWidgets.QComboBox()
        self.comboChannel.addItems(["Y", "G", "GRAY"])
        grid.addWidget(self.comboChannel, 8, 1)

        # Window type
        grid.addWidget(QtWidgets.QLabel("Window"), 9, 0)
        self.comboWindow = QtWidgets.QComboBox()
        self.comboWindow.addItems(["Hann", "Hamming"])
        grid.addWidget(self.comboWindow, 9, 1)

        # Derivative
        grid.addWidget(QtWidgets.QLabel("Derivative"), 10, 0)
        self.comboDeriv = QtWidgets.QComboBox()
        self.comboDeriv.addItems(["Central", "FirstDiff"])
        grid.addWidget(self.comboDeriv, 10, 1)

        # Pixel box correction
        self.chkBoxCorr = QtWidgets.QCheckBox("Pixel box correction (sinc)")
        self.chkBoxCorr.setChecked(False)
        grid.addWidget(self.chkBoxCorr, 11, 0, 1, 2)

        # Nyquist guide
        self.chkNyquist = QtWidgets.QCheckBox("Show Nyquist limit")
        self.chkNyquist.setChecked(True)
        grid.addWidget(self.chkNyquist, 12, 0, 1, 2)

        # MTF unit / x limits / x scale
        grid.addWidget(QtWidgets.QLabel("MTF X unit"), 13, 0)
        self.comboMtfUnit = QtWidgets.QComboBox()
        self.comboMtfUnit.addItems(["cycles/mm", "cycles/pixel"])
        grid.addWidget(self.comboMtfUnit, 13, 1)

        grid.addWidget(QtWidgets.QLabel("MTF X min (display)"), 14, 0)
        self.spinMtfXMin = QtWidgets.QDoubleSpinBox()
        self.spinMtfXMin.setDecimals(3); self.spinMtfXMin.setRange(0.0, 1e6)
        self.spinMtfXMin.setSingleStep(1.0); self.spinMtfXMin.setValue(0.0)
        grid.addWidget(self.spinMtfXMin, 14, 1)

        grid.addWidget(QtWidgets.QLabel("MTF X max (display)"), 15, 0)
        self.spinMtfXMax = QtWidgets.QDoubleSpinBox()
        self.spinMtfXMax.setDecimals(3); self.spinMtfXMax.setRange(0.0, 1e6)
        self.spinMtfXMax.setSingleStep(1.0); self.spinMtfXMax.setValue(0.0)
        grid.addWidget(self.spinMtfXMax, 15, 1)

        grid.addWidget(QtWidgets.QLabel("MTF X scale (×)"), 16, 0)
        self.spinMtfXScale = QtWidgets.QDoubleSpinBox()
        self.spinMtfXScale.setDecimals(6); self.spinMtfXScale.setRange(1e-6, 1e6)
        self.spinMtfXScale.setSingleStep(0.01); self.spinMtfXScale.setValue(1.0)
        grid.addWidget(self.spinMtfXScale, 16, 1)

        # Normalize & gamma
        self.chkNormalize = QtWidgets.QCheckBox("Normalize ESF (percentile 1–99 → 0..1)")
        self.chkNormalize.setChecked(True)
        grid.addWidget(self.chkNormalize, 17, 0, 1, 2)

        self.chkInvGamma = QtWidgets.QCheckBox("Apply inverse gamma (linearize)")
        self.chkInvGamma.setChecked(False)
        grid.addWidget(self.chkInvGamma, 18, 0, 1, 2)

        grid.addWidget(QtWidgets.QLabel("Gamma exponent"), 19, 0)
        self.spinGamma = QtWidgets.QDoubleSpinBox()
        self.spinGamma.setDecimals(3); self.spinGamma.setRange(0.1, 5.0)
        self.spinGamma.setSingleStep(0.1); self.spinGamma.setValue(2.2)
        grid.addWidget(self.spinGamma, 19, 1)

        # Compute
        self.btnCompute = QtWidgets.QPushButton("Compute ESF/LSF/MTF (All ROIs)")
        right.addWidget(self.btnCompute)

        # ROI list
        self.listWidget = QtWidgets.QListWidget()
        self.listWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.listWidget.installEventFilter(self)  # Delete key
        right.addWidget(self.listWidget, 1)

        self.statusBar().showMessage("Load an image, add ROIs (drag or numeric), then Compute.")

        # State
        self._img_bgr = None
        self._rois = []  # (QRect, QColor, label)
        self._roi_counter = 0
        self._last_results = []  # [(label, result_dict), ...]

        # Plot dialogs
        self.esfDlg = ESFDialog(self)
        self.lsfDlg = PlotDialog("LSF")
        self.mtfDlg = PlotDialog("MTF")

        # Signals
        self.btnLoad.clicked.connect(self.onLoad)
        self.imageLabel.roiDragFinished.connect(self.onDragFinished)
        self.btnAddFromDrag.clicked.connect(self.onAddFromDrag)
        self.btnAddNumeric.clicked.connect(self.onAddNumeric)
        self.btnClearRois.clicked.connect(self.onClearRois)
        self.btnCompute.clicked.connect(self.onComputeAll)
        self.btnSave.clicked.connect(self.onSaveAll)

        # Keyboard shortcut: C to Save
        self.shortcutSave = QtGui.QShortcut(QtGui.QKeySequence(Qt.Key_C), self)
        self.shortcutSave.activated.connect(self.onSaveAll)

        # auto-refresh plots on option changes
        for w in [
            self.spinEsfDropN, self.comboMtfUnit, self.spinMtfXMin, self.spinMtfXMax,
            self.spinMtfXScale, self.spinPitch, self.spinSigma, self.comboWindow,
            self.comboDeriv, self.chkBoxCorr, self.comboChannel, self.chkNormalize,
            self.chkInvGamma, self.spinGamma, self.chkNyquist
        ]:
            if isinstance(w, QtWidgets.QDoubleSpinBox):
                w.valueChanged.connect(self.onComputeAll)
            elif isinstance(w, QtWidgets.QSpinBox):
                w.valueChanged.connect(self.onComputeAll)
            elif isinstance(w, QtWidgets.QComboBox):
                w.currentIndexChanged.connect(self.onComputeAll)
            elif isinstance(w, QtWidgets.QCheckBox):
                w.stateChanged.connect(self.onComputeAll)

    # --------- Delete key on ROI list ---------
    def eventFilter(self, obj, event):
        if obj is self.listWidget and event.type() == QtCore.QEvent.KeyPress:
            if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                self.deleteSelectedRois()
                return True
        return super().eventFilter(obj, event)

    def deleteSelectedRois(self):
        rows = sorted({idx.row() for idx in self.listWidget.selectedIndexes()}, reverse=True)
        if not rows:
            return
        for r in rows:
            if 0 <= r < len(self._rois):
                self._rois.pop(r)
                self.listWidget.takeItem(r)
        self.updateOverlay()
        self.statusBar().showMessage("Removed selected ROI(s).")

    # ---------- Image & ROI ----------
    def onLoad(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not fn:
            return
        bgr = cv2.imread(fn, cv2.IMREAD_COLOR)
        if bgr is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Failed to load image.")
            return
        self._img_bgr = bgr
        h, w = bgr.shape[:2]
        qimg = QtGui.QImage(bgr.data, w, h, bgr.strides[0], QtGui.QImage.Format_BGR888)
        self.imageLabel.setImage(QtGui.QPixmap.fromImage(qimg))
        self._rois.clear()
        self._roi_counter = 0
        self._last_results.clear()
        self.updateOverlay()
        self.btnAddFromDrag.setEnabled(True)
        self.spinX.setRange(0, max(0, w - 1))
        self.spinY.setRange(0, max(0, h - 1))
        self.spinW.setRange(1, w)
        self.spinH.setRange(1, h)
        self.listWidget.clear()
        self.statusBar().showMessage("Image loaded. Add ROIs.")

    def onDragFinished(self, disp_rect: QtCore.QRect):
        if self._img_bgr is None:
            return
        img_rect = self.imageLabel.displayRectToImageRect(disp_rect)
        if img_rect.isEmpty():
            self.statusBar().showMessage("Dragged ROI invalid.")
            return
        self.spinX.setValue(img_rect.x())
        self.spinY.setValue(img_rect.y())
        self.spinW.setValue(img_rect.width())
        self.spinH.setValue(img_rect.height())
        self.statusBar().showMessage(
            f"Dragged ROI: x={img_rect.x()}, y={img_rect.y()}, w={img_rect.width()}, h={img_rect.height()} → 'Add ROI (from drag)'"
        )

    def _add_roi(self, img_rect: QtCore.QRect):
        if self._img_bgr is None:
            return
        H, W = self._img_bgr.shape[:2]
        r = QtCore.QRect(
            max(0, min(W-1, img_rect.x())),
            max(0, min(H-1, img_rect.y())),
            min(img_rect.width(), W - img_rect.x()),
            min(img_rect.height(), H - img_rect.y())
        )
        if r.width() < 6 or r.height() < 6:
            QtWidgets.QMessageBox.warning(self, "ROI too small", "Use at least 6×6 pixels.")
            return
        self._roi_counter += 1
        color = qcolor_for_index(self._roi_counter - 1)
        self._rois.append((r, color, self._roi_counter))
        self.updateOverlay()
        self.listWidget.addItem(f"ROI {self._roi_counter}: x={r.x()}, y={r.y()}, w={r.width()}, h={r.height()}")
        self.statusBar().showMessage(f"ROI {self._roi_counter} added.")

    def onAddFromDrag(self):
        img_rect = QtCore.QRect(self.spinX.value(), self.spinY.value(), self.spinW.value(), self.spinH.value())
        self._add_roi(img_rect)

    def onAddNumeric(self):
        img_rect = QtCore.QRect(self.spinX.value(), self.spinY.value(), self.spinW.value(), self.spinH.value())
        self._add_roi(img_rect)

    def onClearRois(self):
        self._rois.clear()
        self._roi_counter = 0
        self._last_results.clear()
        self.updateOverlay()
        self.listWidget.clear()
        self.statusBar().showMessage("Cleared all ROIs.")

    def updateOverlay(self):
        self.imageLabel.setRois(self._rois)

    # ---------- Compute & Plot ----------
    def ensurePlotWindows(self):
        if not self.esfDlg.isVisible():
            self.esfDlg.show()
        if not self.lsfDlg.isVisible():
            self.lsfDlg.show()
        if not self.mtfDlg.isVisible():
            self.mtfDlg.show()
        self.esfDlg.prepare()
        self.lsfDlg.clear()
        self.mtfDlg.clear()
        self.lsfDlg.ax = self.lsfDlg.canvas.figure.add_subplot(111)
        self.lsfDlg.ax.set_title("LSF (windowed)")
        self.lsfDlg.ax.set_xlabel("Position (mm)")
        self.lsfDlg.ax.set_ylabel("Amplitude (a.u.)")
        self.lsfDlg.ax.grid(True, alpha=0.3)
        self.mtfDlg.ax = self.mtfDlg.canvas.figure.add_subplot(111)
        self.mtfDlg.ax.set_title("MTF")
        self.mtfDlg.ax.set_xlabel("Spatial frequency")
        self.mtfDlg.ax.set_ylabel("MTF")
        self.mtfDlg.ax.set_ylim(0, 1.05)
        self.mtfDlg.ax.grid(True, alpha=0.3)

    def onComputeAll(self):
        if self._img_bgr is None or not self._rois:
            return

        pixel_pitch = float(self.spinPitch.value())
        sigma = float(self.spinSigma.value())
        drop_n = int(self.spinEsfDropN.value())
        channel_mode = self.comboChannel.currentText()
        window_type = self.comboWindow.currentText()
        deriv_mode = self.comboDeriv.currentText()
        pixel_box = self.chkBoxCorr.isChecked()
        unit = self.comboMtfUnit.currentText()
        x_scale = float(self.spinMtfXScale.value())
        x_min_user = float(self.spinMtfXMin.value())
        x_max_user = float(self.spinMtfXMax.value())
        normalize_esf = self.chkNormalize.isChecked()
        inv_gamma = self.chkInvGamma.isChecked()
        gamma = float(self.spinGamma.value())

        self.ensurePlotWindows()

        # reset axes
        self.esfDlg.ax_esf.cla()
        self.esfDlg.ax_esf.set_title("ESF (oversampled & smoothed)")
        self.esfDlg.ax_esf.set_xlabel("Position (pixels, along normal; oversampled)")
        self.esfDlg.ax_esf.set_ylabel("Intensity")
        self.esfDlg.ax_esf.grid(True, alpha=0.3)

        self.lsfDlg.ax.cla()
        self.lsfDlg.ax.set_title(f"LSF ({window_type} window)")
        self.lsfDlg.ax.set_xlabel("Position (mm)")
        self.lsfDlg.ax.set_ylabel("Amplitude (a.u.)")
        self.lsfDlg.ax.grid(True, alpha=0.3)

        self.mtfDlg.ax.cla()
        self.mtfDlg.ax.set_title("MTF")
        self.mtfDlg.ax.set_ylabel("MTF")
        self.mtfDlg.ax.set_ylim(0, 1.05)
        self.mtfDlg.ax.grid(True, alpha=0.3)
        self.mtfDlg.ax.set_xlabel(f"Spatial frequency ({unit})")

        self._last_results = []

        last_strip = None
        for (r, color, label) in self._rois:
            roi = self._img_bgr[r.y():r.y()+r.height(), r.x():r.x()+r.width()].copy()
            try:
                res = slanted_edge_esf_lsf_mtf(
                    roi,
                    pixel_pitch_mm=pixel_pitch,
                    oversample=4,
                    smooth_sigma=sigma,
                    normalize_esf=normalize_esf,
                    inv_gamma=inv_gamma,
                    gamma=gamma,
                    channel_mode=channel_mode,
                    window_type=window_type,
                    derivative_mode=deriv_mode,
                    pixel_box_correction=pixel_box
                )
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, f"ROI {label} Error", str(e))
                continue

            self._last_results.append((label, res))

            # ESF (optional tail drop for display)
            x_pix = res["esf_x_pix"]; esf = res["esf"]
            if drop_n > 0 and x_pix.size > drop_n:
                x_pix = x_pix[:-drop_n]
                esf = esf[:-drop_n]
            line_esf, = self.esfDlg.ax_esf.plot(x_pix, esf, label=f"ROI {label}", linewidth=1.6)
            line_esf.set_color(qcolor_to_rgba(color))

            # LSF
            x_mm = res["lsf_x_mm"]; lsf = res["lsf"]
            line_lsf, = self.lsfDlg.ax.plot(x_mm, lsf, label=f"ROI {label}", linewidth=1.6)
            line_lsf.set_color(qcolor_to_rgba(color))

            # MTF (unit & scale)
            if unit == "cycles/mm":
                f = res["freq_cyc_per_mm"] * x_scale
            else:
                f = res["freq_cyc_per_pix"] * x_scale
            mtf = res["mtf"]
            line_mtf, = self.mtfDlg.ax.plot(f, mtf, label=f"ROI {label}", linewidth=1.6)
            line_mtf.set_color(qcolor_to_rgba(color))

            # --- MTF50 / MTF20: 0.50/0.20 교차 X 찾기 + 표시 ---
            f50 = self._first_crossing_x(f, mtf, 0.5)
            f20 = self._first_crossing_x(f, mtf, 0.2)

            # --- Nyquist limit (sensor-level, not per ROI) ---
            if self.chkNyquist.isChecked():
                nyq = self._nyquist_value(unit, pixel_pitch, x_scale)
                if nyq is not None and np.isfinite(nyq):
                    ax = self.mtfDlg.ax
                    ax.axvline(nyq, ymin=0.0, ymax=1.0, linestyle="-.", linewidth=1.6, color="#cc3333", alpha=0.9)
                    ax.text(nyq, 1.02, f"Nyquist = {nyq:.3g}",
                            transform=ax.get_xaxis_transform(), rotation=90,
                            va="bottom", ha="center", fontsize=8, color="#cc3333", clip_on=False)

            ax = self.mtfDlg.ax
            roi_color = qcolor_to_rgba(color)

            # 세로선은 플롯 전체 높이로
            if f50 is not None:
                ax.axvline(f50, ymin=0.0, ymax=1.0, linestyle=":", linewidth=1.4, color=roi_color, alpha=0.95)
                # X는 데이터 좌표, Y는 축좌표(위 살짝)로 텍스트 배치
                ax.text(f50, 1.02, f"R{label} MTF50 = {f50:.3g}",
                        transform=ax.get_xaxis_transform(), rotation=90,
                        va="bottom", ha="center", fontsize=8, color=roi_color, clip_on=False)

            if f20 is not None:
                ax.axvline(f20, ymin=0.0, ymax=1.0, linestyle="--", linewidth=1.2, color=roi_color, alpha=0.95)
                ax.text(f20, 1.02, f"R{label} MTF20 = {f20:.3g}",
                        transform=ax.get_xaxis_transform(), rotation=90,
                        va="bottom", ha="center", fontsize=8, color=roi_color, clip_on=False)


            last_strip = res["strip"]

        # strip image
        self.esfDlg.ax_img.clear()
        if last_strip is not None:
            self.esfDlg.ax_img.imshow(last_strip, aspect='auto', origin='upper', interpolation='nearest')
            self.esfDlg.ax_img.set_title("Resampled strip along normal")
            self.esfDlg.ax_img.set_xticks([]); self.esfDlg.ax_img.set_yticks([])
        else:
            self.esfDlg.ax_img.text(0.5, 0.5, "No valid strip", ha='center', va='center')
            self.esfDlg.ax_img.axis('off')

        # legends
        self.esfDlg.ax_esf.legend(loc="best", fontsize=8)
        self.lsfDlg.ax.legend(loc="best", fontsize=8)
        self.mtfDlg.ax.legend(loc="best", fontsize=8)

        # MTF x-limits (0 → auto)
        if x_max_user > 0 or x_min_user > 0:
            xmin = x_min_user if x_min_user > 0 else 0.0
            xmax = x_max_user if x_max_user > 0 else None
            if xmax is not None and xmax > xmin:
                self.mtfDlg.ax.set_xlim(xmin, xmax)
            else:
                self.mtfDlg.ax.set_xlim(left=xmin)
        else:
            self.mtfDlg.ax.set_xlim(auto=True)

        # draw
        self.esfDlg.canvas.draw()
        self.lsfDlg.canvas.draw()
        self.mtfDlg.canvas.draw()

    # ---------- Save (button + 'C') ----------
    def onSaveAll(self):
        if not self._last_results:
            QtWidgets.QMessageBox.information(self, "Nothing to save", "Please compute first.")
            return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder")
        if not out_dir:
            return
        try:
            for (label, res) in self._last_results:
                np.savetxt(os.path.join(out_dir, f"esf_ROI{label}.csv"),
                           np.column_stack([res["esf_x_pix"], res["esf"]]),
                           delimiter=",", header="x_pix,esf", comments="")
                np.savetxt(os.path.join(out_dir, f"lsf_ROI{label}.csv"),
                           np.column_stack([res["lsf_x_mm"], res["lsf"]]),
                           delimiter=",", header="x_mm,lsf", comments="")
                np.savetxt(os.path.join(out_dir, f"mtf_mm_ROI{label}.csv"),
                           np.column_stack([res["freq_cyc_per_mm"], res["mtf"]]),
                           delimiter=",", header="freq_cyc_per_mm,mtf", comments="")
                np.savetxt(os.path.join(out_dir, f"mtf_pix_ROI{label}.csv"),
                           np.column_stack([res["freq_cyc_per_pix"], res["mtf"]]),
                           delimiter=",", header="freq_cyc_per_pix,mtf", comments="")

            self.esfDlg.canvas.figure.savefig(os.path.join(out_dir, "ESF.png"), dpi=150)
            self.lsfDlg.canvas.figure.savefig(os.path.join(out_dir, "LSF.png"), dpi=150)
            self.mtfDlg.canvas.figure.savefig(os.path.join(out_dir, "MTF.png"), dpi=150)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save error", str(e))
            return

        self.statusBar().showMessage(f"Saved CSVs and PNGs to: {out_dir}")

    @staticmethod
    def _first_crossing_x(x: np.ndarray, y: np.ndarray, thresh: float):
        """
        y(x)가 대체로 단조감소라 가정하고 y가 thresh를 '처음'으로 내리는 x 위치를
        선형보간으로 추정. 교차 없으면 None.
        """
        if x.size < 2 or y.size != x.size:
            return None
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]; y = y[m]
        if x.size < 2:
            return None

        below = np.where(y <= thresh)[0]
        if below.size == 0:
            return None

        k = below[0]
        if k == 0:
            return x[0]

        x0, x1 = x[k-1], x[k]
        y0, y1 = y[k-1], y[k]
        if y1 == y0:
            return x1
        t = (thresh - y0) / (y1 - y0)
        return x0 + t * (x1 - x0)

    @staticmethod
    def _nyquist_value(unit: str, pixel_pitch_mm: float, x_scale: float) -> float:
        """
        Nyquist frequency with unit & x-scale applied.
        - cycles/pixel: 0.5 * x_scale
        - cycles/mm   : (1 / (2 * pixel_pitch_mm)) * x_scale
        """
        if pixel_pitch_mm <= 0:
            return None
        if unit == "cycles/pixel":
            return 0.5 * x_scale
        # cycles/mm
        return (1.0 / (2.0 * pixel_pitch_mm)) * x_scale
