# -*-coding:utf-8 -*-
import math

from PyQt5.QtWidgets import QLabel, QGraphicsDropShadowEffect, QFrame
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import QPropertyAnimation, QRect, Qt


# marquee label
class Marquee(QFrame):

    def __init__(self, window):
        super().__init__(window)
        self.screenWidth, self.screenHeight = window.getDisplayArea()
        self.initUi(self.screenWidth, self.screenHeight)

    def initUi(self, screenWidth, screenHeight):
        self.setStyleSheet("border:5px solid red;")
        self.setGeometry(0,
                         0,
                         self.screenWidth,
                         self.screenHeight)
        self.show()
