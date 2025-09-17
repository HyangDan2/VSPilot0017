# --- add this at the very top of src/app.py ---
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent          # ...\VSPilot0017\src
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# --- end patch ---
from PySide6 import QtWidgets
from ui.main_window import MainWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1560, 940)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
