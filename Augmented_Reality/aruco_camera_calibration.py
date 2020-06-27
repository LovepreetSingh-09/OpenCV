import time
import cv2
import numpy as np
import pickle

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
board = cv2.aruco.CharucoBoard_create(3, 3, .025, .0125, dictionary)

image_board = board.draw((200 * 3, 200 * 3))

cv2.imwrite('charuco.png', image_board)

cap = cv2.VideoCapture(0)

all_corners = []
all_ids = []
counter = 0
for i in range(300):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.aruco.detectMarkers(gray, dictionary)
    if len(res[0]) > 0:
        res2 = cv2.aruco.interpolateCornersCharuco(res[0], res[1], gray, board)
        if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and counter % 3 == 0:
            all_corners.append(res2[1])
            all_ids.append(res2[2])

        cv2.aruco.drawDetectedMarkers(gray, res[0], res[1])

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    counter += 1

try:
    cal = cv2.aruco.calibrateCameraCharuco(all_corners, all_ids, board, gray.shape, None, None)
except:
    cap.release()
    print("Calibration could not be done ...")

retval, cameraMatrix, distCoeffs, rvecs, tvecs = cal

f = open('calibration2.pckl', 'wb')
pickle.dump((cameraMatrix, distCoeffs), f)
f.close()

cap.release()
cv2.destroyAllWindows()