import cv2
import time
import numpy as np
import sys
from pynput.mouse import Controller
from PyQt5.QtWidgets import QApplication
from Margin import App, Marquee, sleep
import imageio
import os


class GifAgent:
    def __init__(self):
        self.storage = []
        self.max_score = 0
        self.max_storage = []

    def store(self, img):
        self.storage.append(img)

    def commit(self, score, auto_output=False):
        if score > self.max_score:
            self.max_score = score
            self.max_storage = self.storage.copy()
            if auto_output:
                self.output()

        self.storage = []

    def output(self, name='max_score.gif'):
        if 'gif' not in os.listdir(os.getcwd()):
            os.mkdir('./gif')
        imageio.mimsave('./gif/'+name, self.max_storage, 'GIF', duration=0.1)

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))


def counter_clockwise_sort(tetragon):
    tetragon = sorted(tetragon, key=lambda e: e[0])
    tetragon[0:2] = sorted(tetragon[0:2], key=lambda e: e[1])
    tetragon[2:4] = sorted(tetragon[2:4], key=lambda e: e[1], reverse=True)
    return tetragon

DEBUG = False

MouseSmoothness = 0.25
ScreenWidth = 1920
ScreenHeight = 1080
ScreenOverlap = 250
CalibrateInterval = 1e-100  # s

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
    # app = QApplication(sys.argv)
    # MyMainWindow = App()
    # MyMarquee = Marquee(MyMainWindow)

    # mouse = Controller()
    # cap = cv2.VideoCapture("http://192.168.43.1:8080/video")
    # cap = cv2.VideoCapture("/Users/zya/Downloads/VID_20181025_234734.mp4")
    # points_old = None
    # while not cap.read()[0]:
    #     pass
    # old_gray = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
    time_start = cv2.getTickCount()
    # wri = cv2.VideoWriter(
    #     "/Users/zya/Downloads/VID_20181025_235333_.mp4",
    #     cv2.VideoWriter_fourcc('F', 'M', 'P', '4'),
    #     30,
    #     (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    # )
    calibrate_timer = cv2.getTickCount()
    try:
        # while True:
        #     sleep(0.00001)
        #     ret, frame = cap.read()
        #     if not ret:
        #         raise KeyboardInterrupt
            frame = cv2.resize(cv2.imread('0original.jpg'), (1920, 1080))
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            margin_binary = np.logical_and(
                np.logical_and(frame_hsv[:, :, 0] < 130, frame_hsv[:, :, 0] > 110),
                frame_hsv[:, :, 1] > 127
            )
            cv2.imwrite('1threshold.jpg', np.uint8(margin_binary)*255)
            _, contours, hierarchy = cv2.findContours(
                np.uint8(margin_binary)*255,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE
            )
            area_max = 0
            frame_gray_bgr = cv2.cvtColor(np.uint8(margin_binary)*255, cv2.COLOR_GRAY2BGR)
            frame_labeled = frame.copy()
            cv2.circle(frame_labeled, (1920//2,1080//2), 2, (127,8,8), 10)

            if (cv2.getTickCount() - calibrate_timer) / cv2.getTickFrequency() > CalibrateInterval:
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
                            frame_gray_bgr = cv2.cvtColor(np.uint8(margin_binary)*255, cv2.COLOR_GRAY2BGR)
                            cv2.drawContours(frame_gray_bgr, [contour], 0, (255, 0, 0), 10)
                            # frame_labeled = frame.copy()
                            # cv2.drawContours(frame_labeled, [contour], 0, (9, 0, 255), 10)
                            area_max = area
                            tetragonVertices = np.float32(counter_clockwise_sort(contour))
                
                
                cv2.imwrite('2find_contours.jpg', frame_gray_bgr)
                frame_gray_bgr = cv2.cvtColor(np.uint8(margin_binary)*255, cv2.COLOR_GRAY2BGR)
                if area_max > 0:
                    ori = np.array([[0,0],[0,1080],[1920,1080],[1920,0]], dtype=np.float32)
                    gitagent = GifAgent()
                    for i in np.linspace(0,1,16):
                        # print(tetragonVertices, ori)
                        print(tetragonVertices)
                        print((1-i)*ori+i*tetragonVertices)
                        # perspectiveMatrix = cv2.getPerspectiveTransform(((1-i)*ori+i*tetragonVertices), tetragonVerticesUpd)
                        perspectiveMatrix = cv2.getPerspectiveTransform(np.float32((1-i)*ori+i*tetragonVertices), tetragonVerticesUpd)
                    # calibrate_timer = cv2.getTickCount()
                    # for inx in range(4):
                    #     frame = cv2.circle(
                    #         frame,
                    #         (int(tetragonVertices[inx][0]), int(tetragonVertices[inx][1])),
                    #         1,
                    #         (0, 0, 255),
                    #         10
                    #     )
                    
                        warped = cv2.warpPerspective(frame_labeled, perspectiveMatrix, (1280, 720))
                        gitagent.store(cv2.cvtColor(cv2.resize(warped, (1920//2, 1080//2)), cv2.COLOR_BGR2RGB))
                        # cv2.imwrite("3warpped%d.jpg"%(i*10), warped)
                    gitagent.commit(10, True)

                    
                    # point = np.dot(perspectiveMatrix, np.array([[frame.shape[1]/2], [frame.shape[0]/2], [1]]))
                    # point = (point[:, 0]/point[2, 0])[:2]
            
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
