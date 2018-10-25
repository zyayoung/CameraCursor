import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt


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
tetragonVerticesUpd = np.float32([[0, 0], [0, 720], [1280, 720], [1280, 0]])


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 300
    cap = cv2.VideoCapture("/Users/zya/Downloads/VID_20181025_234734.mp4")

    wri = cv2.VideoWriter(
        "/Users/zya/Downloads/VID_20181025_234734_.mp4",
        cv2.VideoWriter_fourcc('F', 'M', 'P', '4'),
        30,
        (1280, 720),
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        framehsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        margin_binary = np.logical_and(
            np.logical_or(framehsv[:, :, 0] < 10, framehsv[:, :, 0] > 170),
            framehsv[:, :, 1] > 120
        )
        # plt.imshow(margin_binary, cmap='gray')
        # plt.show()
        # break
        _, contours, hierarchy = cv2.findContours(np.uint8(margin_binary)*255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
        # plt.imshow(frame)
        # plt.show()
        # break
        for contour in contours:
            # print(cv2.contourArea(contour))
            # if cv2.contourArea(contour) < frame.shape[0]*frame.shape[1] / 16:
            #     continue

            # print(c ontour)
            contourPerimeter = cv2.arcLength(contour, True)
            hull = cv2.convexHull(contour)
            contour = cv2.approxPolyDP(hull, 0.02 * contourPerimeter, True)

            if len(contour) == 4 and cv2.contourArea(contour) > frame.shape[0]*frame.shape[1] / 16:
                contour = contour.reshape(-1, 2)
                max_cos = np.max([angle_cos(contour[i], contour[(i + 1) % 4], contour[(i + 2) % 4]) for i in range(4)])
                if max_cos < 0.3:
                    tetragonVertices = counterclockwiseSort(contour)
                    tetragonVertices = np.float32(tetragonVertices)
                    perspectiveMatrix = cv2.getPerspectiveTransform(tetragonVertices, tetragonVerticesUpd)
                    warpped = cv2.warpPerspective(frame, perspectiveMatrix, (1280, 720))
                    point = np.dot(perspectiveMatrix, np.array([[frame.shape[1]/2], [frame.shape[0]/2], [1]]))
                    point = (point[:, 0]/point[2, 0])[:2]
                    break
        print((int(point[0]), int(point[1])))
        frame = cv2.circle(frame, (int(point[0]), int(point[1])), 1, (0, 255, 0), 10)
        wri.write(frame)
        cv2.imshow('pos', frame)
        k = cv2.waitKey(1)
        if k == 27:
            sys.exit()
    wri.release()
cv2.destroyAllWindows()
