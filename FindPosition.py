import cv2
import sys
import numpy as np
from pynput.mouse import Button, Controller


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))


def counterclockwiseSort(tetragon):
    tetragon = sorted(tetragon, key=lambda e: e[0])
    tetragon[0:2] = sorted(tetragon[0:2], key=lambda e: e[1])
    tetragon[2:4] = sorted(tetragon[2:4], key=lambda e: e[1], reverse=True)
    return tetragon

smoothness = 0.5
ScreenWidth = 1280
ScreenHeight = 800
ScreenOverLap = 100

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
    mouse = Controller()
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("/Users/zya/Downloads/VID_20181025_234734.mp4")
    points_old = None
    while not cap.read()[0]:
        pass
    old_gray = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
    # print(old_gray.shape)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    time_start = cv2.getTickCount()
    # wri = cv2.VideoWriter(
    #     "/Users/zya/Downloads/VID_20181025_235333_.mp4",
    #     cv2.VideoWriter_fourcc('F', 'M', 'P', '4'),
    #     30,
    #     (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    # )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise KeyboardInterrupt
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            margin_binary = np.logical_and(
                np.logical_or(frame_hsv[:, :, 0] < 10, frame_hsv[:, :, 0] > 170),
                frame_hsv[:, :, 1] > 120
            )
            _, contours, hierarchy = cv2.findContours(np.uint8(margin_binary)*255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            area_max = 0
            for contour in contours:
                contourPerimeter = cv2.arcLength(contour, True)
                hull = cv2.convexHull(contour)
                contour = cv2.approxPolyDP(hull, 0.02 * contourPerimeter, True)
                area = cv2.contourArea(contour)
                if len(contour) == 4 and area > frame.shape[0]*frame.shape[1] / 16:
                    contour = contour.reshape(-1, 2)
                    max_cos = np.max([angle_cos(contour[i], contour[(i + 1) % 4], contour[(i + 2) % 4]) for i in range(4)])
                    if max_cos < 0.26 and area > area_max:
                        # cv2.drawContours(frame, [contour], 0, (0, 0, 255), 2)
                        area_max = area
                        tetragonVertices = np.float32(counterclockwiseSort(contour))
            if area_max > 0:
                perspectiveMatrix = cv2.getPerspectiveTransform(tetragonVertices, tetragonVerticesUpd)

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
                if -ScreenOverLap < point[0] < 1280 + ScreenOverLap and -ScreenOverLap < point[1] < 720 + ScreenOverLap:
                    mouse.position = (
                        int(mouse.position[0] * (1 - smoothness) + point[0] / 720 * ScreenHeight * smoothness),
                        int(mouse.position[1] * (1 - smoothness) + point[1] / 1280 * ScreenWidth * smoothness)
                    )
                    points_old = tetragonVertices.reshape(4, 1, 2)
                else:
                    area_max = 0
            if area_max == 0:
                if points_old is not None:
                    print("opt works")
                    # calcOpticalFlowPyrLK
                    points_new, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, points_old, None, **lk_params)

                    for inx in range(4):
                        frame = cv2.circle(
                            frame,
                            (int(points_new[inx][0][0]), int(points_new[inx][0][1])),
                            1,
                            (0, 255, 0),
                            10
                        )

                    if False not in st:
                        tetragonVertices = np.float32(counterclockwiseSort(points_new.reshape(-1, 2)))
                        perspectiveMatrix = cv2.getPerspectiveTransform(tetragonVertices, tetragonVerticesUpd)
                        # warped = cv2.warpPerspective(frame, perspectiveMatrix, (1280, 720))
                        point = np.dot(perspectiveMatrix, np.array([[frame.shape[1]/2], [frame.shape[0]/2], [1]]))
                        point = (point[:, 0]/point[2, 0])[:2]
                        print(point)
                        if -ScreenOverLap < point[0] < 1280 + ScreenOverLap and -ScreenOverLap < point[1] < 720 + ScreenOverLap:
                            mouse.position = (
                                int(mouse.position[0] * (1 - smoothness) + point[0] / 720 * ScreenHeight * smoothness),
                                int(mouse.position[1] * (1 - smoothness) + point[1] / 1280 * ScreenWidth * smoothness)
                            )
                        points_old = tetragonVertices.reshape(4, 1, 2)
                    else:
                        print("opt fail")
                        points_old = None
            old_gray = frame_gray

            # wri.write(frame)

            cv2.imshow('pos', cv2.resize(frame, (720, 480)))
            k = cv2.waitKey(1)
            if k == 27:
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("exit")
        # wri.release()
        print(cap.get(cv2.CAP_PROP_FRAME_COUNT)/(-time_start + cv2.getTickCount())*cv2.getTickFrequency())
        cv2.destroyAllWindows()
