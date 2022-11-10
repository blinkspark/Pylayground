from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel
from PySide6.QtCore import Qt
import asyncio, sys


class MyWindow(QMainWindow):

  def __init__(self, app: QApplication):
    super().__init__()
    self.app = app
    self.setWindowTitle("test")
    self.main_widget = QWidget()
    self.main_layout = QGridLayout()
    self.main_widget.setLayout(self.main_layout)
    self.label = QLabel(text="Hello", alignment=Qt.AlignmentFlag.AlignBottom)
    self.main_layout.addWidget(self.label, 0, 0)
    self.setCentralWidget(self.main_widget)


async def main():
  app = QApplication(sys.argv)
  win = MyWindow(app)
  cb = app.clipboard()
  cb.dataChanged.connect(lambda : print('m'))
  win.show()
  app.exec()


if __name__ == '__main__':
  asyncio.run(main())