import cv2
import dlib


def draw_text_info():
    menu_pos_1 = (10, 20)
    menu_pos_2 = (10, 40)
    cv2.putText(frame, "Use '1' to re-initialize tracking", menu_pos_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    if tracking_face:
        cv2.putText(frame, "tracking the face", menu_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    else:
        cv2.putText(frame, "detecting a face to initialize tracking...", menu_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255))


capture = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()

tracker = dlib.correlation_tracker()

tracking_face = False

while True:
    ret, frame = capture.read()
    draw_text_info()
    if tracking_face is False:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        if len(rects) > 0:
            tracker.start_track(frame, rects[0])
            tracking_face = True

    if tracking_face is True:
        print(tracker.update(frame))
        pos = tracker.get_position()
        cv2.rectangle(frame, (int(pos.left()), int(pos.top())), (int(pos.right()), int(pos.bottom())), (0, 255, 0), 3)

    key = 0xFF & cv2.waitKey(1)
    if key == ord("1"):
        tracking_face = False

    if key == ord('q'):
        break

    cv2.imshow("Face tracking using dlib frontal face detector and correlation filters for tracking", frame)

capture.release()
cv2.destroyAllWindows()
