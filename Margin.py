# -*- coding: utf-8 -*-
import sys
import os
import time
from PyQt5.QtWidgets import QApplication, QDesktopWidget, QWidget, QFrame
from PyQt5.QtCore import Qt, QEventLoop, QTimer


def sleep(s):
    loop = QEventLoop()
    QTimer.singleShot(int(s * 1000), loop.quit)
    loop.exec()


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.left = 0
        self.top = 0  # if os.name == "nt" else 30
        screenRect = QDesktopWidget.screenGeometry(QApplication.desktop())
        self.width = screenRect.width()
        self.height = screenRect.height()
        if os.name == 'nt':
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.SubWindow | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.show()

    def getDisplayArea(self):
        return self.width, self.height


class Marquee(QFrame):
    def __init__(self, window):
        super().__init__(window)
        self.screenWidth, self.screenHeight = window.getDisplayArea()
        self.setStyleSheet("border:25px solid blue;")
        self.setGeometry(0,
                         0,
                         self.screenWidth,
                         self.screenHeight)
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MyMainWindow = App()
    MyMarquee = Marquee(MyMainWindow)
    while True:
        # loop every 100ms
        if os.name == 'nt':
            MyMainWindow.raise_()
        sleep(0.1)
        time.sleep(1)
