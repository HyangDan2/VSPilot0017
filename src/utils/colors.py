from PySide6 import QtGui

def qcolor_to_rgba(color: QtGui.QColor):
    return (color.redF(), color.greenF(), color.blueF(), color.alphaF())

def qcolor_for_index(i: int) -> QtGui.QColor:
    palette = [
        QtGui.QColor("#ff3b30"),  # red
        QtGui.QColor("#34c759"),  # green
        QtGui.QColor("#007aff"),  # blue
        QtGui.QColor("#ff9500"),  # orange
        QtGui.QColor("#af52de"),  # purple
        QtGui.QColor("#5ac8fa"),  # teal
        QtGui.QColor("#8e8e93"),  # gray
    ]
    return palette[i % len(palette)]
