# -*- coding: utf-8 -*-
import argparse
import sys
import os
import time
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QDesktopWidget
from PyQt5.QtCore import Qt

from Danmu.App import App
from Danmu.Marquee import Marquee
from Danmu.utils import *

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MyMainWindow = App()
    MyMarquee = Marquee(MyMainWindow)
    while True:
        # loop every 100ms
        if os.name == 'nt':
            MyMainWindow.raise_()
        sleep(0.1)
        time.sleep(0.001)
