import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyepair_cascade = cv2.CascadeClassifier("haarcascade_mcs_eyepair_big.xml")

# Load glasses image. The parameter -1 reads also de alpha channel (if exists)
# Therefore, the loaded image has four channels (Blue, Green, Red, Alpha):
img_glasses = cv2.imread('glasses.png', -1)

img_glasses_mask = img_glasses[:, :, 3]
cv2.imshow("img glasses mask", img_glasses_mask)
img_glasses = img_glasses[:, :, 0:3]

test_face = cv2.imread("face_test.png")
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    # frame = test_face.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyepairs = eyepair_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyepairs:
            #cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 255), 2)
            x1 = int(ex - ew / 10)
            x2 = int((ex + ew) + ew / 10)
            y1 = int(ey)
            y2 = int(ey + eh + eh / 2)
            if x1 < 0 or x2 < 0 or x2 > w or y2 > h:
                continue
            #cv2.rectangle(roi_color, (x1, y1), (x2, y2), (0, 255, 255), 2)
            img_glasses_res_width = int(x2 - x1)
            img_glasses_res_height = int(y2 - y1)
            mask = cv2.resize(img_glasses_mask, (img_glasses_res_width, img_glasses_res_height))
            mask_inv = cv2.bitwise_not(mask)
            img = cv2.resize(img_glasses, (img_glasses_res_width, img_glasses_res_height))
            roi = roi_color[y1:y2, x1:x2]
            roi_bakground = cv2.bitwise_and(roi, roi, mask=mask_inv)
            roi_foreground = cv2.bitwise_and(img, img, mask=mask)
            cv2.imshow('roi_bakground', roi_bakground)
            cv2.imshow('roi_foreground', roi_foreground)
            res = cv2.add(roi_bakground, roi_foreground)
            roi_color[y1:y2, x1:x2] = res
            break
    cv2.imshow('Snapchat-based OpenCV glasses filter', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()