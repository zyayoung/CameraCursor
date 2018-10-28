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


perspectiveMatrix = np.zeros((3, 3))
point = np.zeros((2,))
tetragonVertices = np.zeros((4, 2), dtype=np.float32)
tetragonVerticesUpd = np.float32([[0, 0], [0, 720], [1280, 720], [1280, 0]])


if __name__ == "__main__":
    mouse = Controller()
    cap = cv2.VideoCapture(0)
    time_start = cv2.getTickCount()
    # wri = cv2.VideoWriter(
    #     "/Users/zya/Downloads/VID_20181025_235333_.mp4",
    #     cv2.VideoWriter_fourcc('F', 'M', 'P', '4'),
    #     30,
    #     (1280, 720),
    # )

    while True:
        ret, frame = cap.read()
        # if not ret:
        #     break
        framehsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        margin_binary = np.logical_and(
            np.logical_or(framehsv[:, :, 0] < 10, framehsv[:, :, 0] > 170),
            framehsv[:, :, 1] > 120
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
                if max_cos < 0.3 and area > area_max:
                    cv2.drawContours(frame, [contour], 0, (0, 0, 255), 2)
                    area_max = area
                    tetragonVertices = np.float32(counterclockwiseSort(contour))
        if area_max > 0:
            perspectiveMatrix = cv2.getPerspectiveTransform(tetragonVertices, tetragonVerticesUpd)
            # warped = cv2.warpPerspective(frame, perspectiveMatrix, (1280, 720))
            point = np.dot(perspectiveMatrix, np.array([[frame.shape[1]/2], [frame.shape[0]/2], [1]]))
            point = (point[:, 0]/point[2, 0])[:2]
            if 0 < point[0] < 1280 and 0 < point[1] < 720:
                mouse.position = (
                    int(mouse.position[0] * 0.8 + point[0] * 0.2),
                    int(mouse.position[1] * 0.8 + point[1] * 0.2)
                )
        # frame = cv2.circle(frame, (int(point[0]), int(point[1])), 1, (0, 255, 0), 10)
        # wri.write(frame)
        cv2.imshow('pos', cv2.resize(frame, (720, 480)))
        k = cv2.waitKey(1)
        if k == 27:
            sys.exit()
    # wri.release()
    print(cap.get(cv2.CAP_PROP_FRAME_COUNT)/(-time_start + cv2.getTickCount())*cv2.getTickFrequency())
cv2.destroyAllWindows()
