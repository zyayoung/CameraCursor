import cv2
import time
import numpy as np
import sys
from pynput.mouse import Controller
from PyQt5.QtWidgets import QApplication
from Margin import App, Marquee, sleep

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))


def counter_clockwise_sort(tetragon):
    tetragon = sorted(tetragon, key=lambda e: e[0])
    tetragon[0:2] = sorted(tetragon[0:2], key=lambda e: e[1])
    tetragon[2:4] = sorted(tetragon[2:4], key=lambda e: e[1], reverse=True)
    return tetragon

DEBUG = False

MouseSmoothness = 0.5
ScreenWidth = 1920
ScreenHeight = 1080
ScreenOverlap = 250
CalibrateInterval = 0  # s

perspectiveMatrix = np.zeros((3, 3))
point = np.zeros((2,))
tetragonVertices = np.zeros((4, 2), dtype=np.float32)
tetragonVerticesUpd = np.float32([[0, 0], [0, 720], [1280, 720], [1280, 0]])

# Parameters for lucas kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

if __name__ == "__main__":
    # Margin
    app = QApplication(sys.argv)
    MyMainWindow = App()
    MyMarquee = Marquee(MyMainWindow)

    mouse = Controller()
    cap = cv2.VideoCapture("http://192.168.43.1:8080/video")
    # cap = cv2.VideoCapture("/Users/zya/Downloads/VID_20181025_234734.mp4")
    points_old = None
    while not cap.read()[0]:
        pass
    old_gray = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
    time_start = cv2.getTickCount()
    # wri = cv2.VideoWriter(
    #     "/Users/zya/Downloads/VID_20181025_235333_.mp4",
    #     cv2.VideoWriter_fourcc('F', 'M', 'P', '4'),
    #     30,
    #     (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    # )
    calibrate_timer = cv2.getTickCount()
    try:
        while True:
            ret, frame = cap.read()
            sleep(0.00001)
            if not ret:
                raise KeyboardInterrupt
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            margin_binary = np.logical_and(
                np.logical_or(frame_hsv[:, :, 0] < 10, frame_hsv[:, :, 0] > 170),
                frame_hsv[:, :, 1] > 120
            )
            _, contours, hierarchy = cv2.findContours(
                np.uint8(margin_binary)*255,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE
            )
            area_max = 0
            if points_old is None or (cv2.getTickCount() - calibrate_timer) / cv2.getTickFrequency() > CalibrateInterval:
                for contour in contours:
                    contourPerimeter = cv2.arcLength(contour, True)
                    hull = cv2.convexHull(contour)
                    contour = cv2.approxPolyDP(hull, 0.02 * contourPerimeter, True)
                    area = cv2.contourArea(contour)
                    if len(contour) == 4 and area > frame.shape[0]*frame.shape[1] / 16:
                        contour = contour.reshape(-1, 2)
                        max_cos = np.max(
                            [angle_cos(contour[i], contour[(i + 1) % 4], contour[(i + 2) % 4]) for i in range(4)]
                        )
                        if max_cos < 0.26 and area > area_max:
                            # cv2.drawContours(frame, [contour], 0, (0, 0, 255), 2)
                            area_max = area
                            tetragonVertices = np.float32(counter_clockwise_sort(contour))
                if area_max > 0:
                    perspectiveMatrix = cv2.getPerspectiveTransform(tetragonVertices, tetragonVerticesUpd)
                    calibrate_timer = cv2.getTickCount()
                    if DEBUG:
                        for inx in range(4):
                            frame = cv2.circle(
                                frame,
                                (int(tetragonVertices[inx][0]), int(tetragonVertices[inx][1])),
                                1,
                                (0, 0, 255),
                                10
                            )
                    # warped = cv2.warpPerspective(frame, perspectiveMatrix, (1280, 720))
                    point = np.dot(perspectiveMatrix, np.array([[frame.shape[1]/2], [frame.shape[0]/2], [1]]))
                    point = (point[:, 0]/point[2, 0])[:2]
                    if -ScreenOverlap < point[0] < 1280 + ScreenOverlap\
                            and -ScreenOverlap < point[1] < 720 + ScreenOverlap:
                        mouse.position = (
                            int(mouse.position[0] * (1 - MouseSmoothness) + point[0] / 720 * ScreenHeight * MouseSmoothness),
                            int(mouse.position[1] * (1 - MouseSmoothness) + point[1] / 1280 * ScreenWidth * MouseSmoothness)
                        )
                        points_old = tetragonVertices.reshape(4, 1, 2)
                    else:
                        area_max = 0
            if area_max == 0 and points_old is not None:
                # calcOpticalFlowPyrLK
                points_new, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, points_old, None, **lk_params)
                if DEBUG:
                    for inx in range(4):
                        frame = cv2.circle(
                            frame,
                            (int(points_new[inx][0][0]), int(points_new[inx][0][1])),
                            1,
                            (0, 255, 0),
                            10
                        )
                st = np.logical_and(st, np.abs(points_new - points_old).reshape(4, 2).max(axis=1) < 100)
                # print(np.abs(points_new - points_old).reshape(4, 2).max(axis=1))
                if False not in st:
                    # print("opt works")
                    tetragonVertices = np.float32(counter_clockwise_sort(points_new.reshape(-1, 2)))
                    perspectiveMatrix = cv2.getPerspectiveTransform(tetragonVertices, tetragonVerticesUpd)
                    # warped = cv2.warpPerspective(frame, perspectiveMatrix, (1280, 720))
                    point = np.dot(perspectiveMatrix, np.array([[frame.shape[1]/2], [frame.shape[0]/2], [1]]))
                    point = (point[:, 0]/point[2, 0])[:2]
                    # print(point)
                    if -ScreenOverlap < point[0] < 1280 + ScreenOverlap \
                            and -ScreenOverlap < point[1] < 720 + ScreenOverlap:
                        mouse.position = (
                            int(mouse.position[0] * (1 - MouseSmoothness) + point[0] / 720 * ScreenHeight * MouseSmoothness),
                            int(mouse.position[1] * (1 - MouseSmoothness) + point[1] / 1280 * ScreenWidth * MouseSmoothness)
                        )
                    points_old = tetragonVertices.reshape(4, 1, 2)
                else:
                    # print("opt fail")
                    points_old = None
            old_gray = frame_gray

            # wri.write(frame)

            if DEBUG:
                cv2.imshow('pos', cv2.resize(frame, (720, 480)))
                k = cv2.waitKey(1)
                if k == 27:
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("exit")
        # wri.release()
        print(cap.get(cv2.CAP_PROP_FRAME_COUNT)/(-time_start + cv2.getTickCount())*cv2.getTickFrequency())
        cv2.destroyAllWindows()
