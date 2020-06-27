import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")

# Load moustache image. The parameter -1 reads also de alpha channel
# Therefore, the loaded image has four channels (Blue, Green, Red, Alpha):
img_moustache = cv2.imread('moustache.png', -1)

img_moustache_mask = img_moustache[:, :, 3]
test_face = cv2.imread("face_test.png")
cv2.imshow('roi_bakground', img_moustache_mask)

img_moustache = img_moustache[:, :, 0:3]
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    # frame = test_face.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        noses = nose_cascade.detectMultiScale(roi_gray)
        for (nx, ny, nw, nh) in noses:
            # Draw a rectangle to see the detected nose (debugging purposes):
            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 255), 2)
            x1 = int(nx - nw / 2)
            x2 = int(nx + nw / 2 + nw)
            y1 = int(ny + nh / 2 + nh / 8)
            y2 = int(ny + nh + nh / 4 + nh / 6)
            if x1 < 0 or x2 < 0 or x2 > w or y2 > h:
                continue
            cv2.rectangle(roi_color, (x1, y1), (x2, y2), (255, 0, 0), 2)
            img_moustache_res_width = int(x2 - x1)
            img_moustache_res_height = int(y2 - y1)
            mask = cv2.resize(img_moustache_mask, (img_moustache_res_width, img_moustache_res_height))
            mask_inv = cv2.bitwise_not(mask)
            img = cv2.resize(img_moustache, (img_moustache_res_width, img_moustache_res_height))
            roi = roi_color[y1:y2, x1:x2]
            roi_bakground = cv2.bitwise_and(roi, roi, mask=mask_inv)
            roi_foreground = cv2.bitwise_and(img, img, mask=mask)
            cv2.imshow('roi_bakground', roi_bakground)
            cv2.imshow('roi_foreground', roi_foreground)
            res = cv2.add(roi_bakground, roi_foreground)
            roi_color[y1:y2, x1:x2] = res
            break
    cv2.imshow('Snapchat-based OpenCV moustache overlay', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
