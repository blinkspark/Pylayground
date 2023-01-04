import asyncio
import sys

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QApplication, QGridLayout, QLabel, QMainWindow,
                               QWidget)
from defs import NewTaskWindow


class MyWindow(QMainWindow):

  def __init__(self, app: QApplication):
    super().__init__()
    self.app = app
    self.setWindowTitle("test")
    self.set_init_size(800, 600)
    # main_widget
    self.main_widget = QWidget()
    self.main_layout = QGridLayout()
    self.main_widget.setLayout(self.main_layout)
    self.setCentralWidget(self.main_widget)

    self.w: QWidget = None

    self.init_menu()

  def init_menu(self):
    # file_menu
    self.file_menu = self.menuBar().addMenu("&File")
    self.file_menu.addAction("&New", self.file_new_action)
    self.file_menu.addAction("&Quit", lambda: self.app.quit())

  def set_init_size(self, w: int, h: int):
    screen_size = QGuiApplication.primaryScreen().size()
    x = (screen_size.width() - w) // 2
    y = (screen_size.height() - h) // 2
    self.setGeometry(x, y, w, h)

  ############## actions #########################
  def file_new_action(self):
    if self.w is None:
      self.new_task_window()
    else:
      self.w.show()

  ############## logics ##########################
  def new_task_window(self):
    self.w = NewTaskWindow()
    self.w.show()


async def main():
  app = QApplication(sys.argv)
  win = MyWindow(app)
  cb = app.clipboard()
  cb.dataChanged.connect(lambda: print('m'))
  win.show()
  app.exec()


if __name__ == '__main__':
  asyncio.run(main())