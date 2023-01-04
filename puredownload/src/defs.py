from PySide6.QtWidgets import (QApplication, QGridLayout, QLabel, QMainWindow,
                               QWidget)


class NewTaskWindow(QWidget):

  def __init__(self):
    super().__init__()
    self.main_layout = QGridLayout(self)
    self.setGeometry(100,100,300,600)

    test_label = QLabel("test")
    test1_label = QLabel("test")
    self.main_layout.setRowStretch(0, 0)
    self.main_layout.setRowStretch(1, 1)
    self.main_layout.addWidget(test_label, 0, 0)
    self.main_layout.addWidget(test1_label, 1, 0)
