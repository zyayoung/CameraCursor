# CameraCursor (Archery Simulator)

## Intro

> A python script that calls many classes to finish a hard task in an efficient way.
>
> Proudly Powered By OpenCV with classes called by [zyayoung](https://github.com/zyayoung).
>
> \- [BoYanZh](https://github.com/BoYanZh)

This is a python script that can move your cursor to the place where your camera aims at.

## Requirements

- opencv
- pynput
- pyqt

## Our Poster for Archery Simulator

![](https://raw.githubusercontent.com/zyayoung/CameraCursor/master/Group12_poster.jpg)

## Symposium Q&A

- How do you find where the player is aiming at?
  - We first transmit the frames caputured by the camera to the computer and do some image-processing.
    - Display a red border on the screen
    - Detect the red border on the screen
    - Perform a perspective transformation
    
- How do you detect a trigger?
  - When the arrow is released, there will be a huge change in acceleration on the bow. And the smartphone is installed on the bow, so we made an Android app to detect a shoot.

## Setup process

- Phone
    - Open Portable hotspot
    - Open IP Camera
    - Open AccClick

- Computer
    - Connect Portable hotspot
    - Run Findposition.py
    - Run Game
## Credit

- [opencv](https://github.com/opencv/opencv)
- [pynput](https://github.com/moses-palmer/pynput)
- [pyqt](https://pypi.org/project/PyQt5/)
- [animal-tracking](https://github.com/colinlaney/animal-tracking)
