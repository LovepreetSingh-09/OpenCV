import cv2
import dlib


def draw_text_info():
    menu_pos = (10, 20)
    menu_pos_2 = (10, 40)
    menu_pos_3 = (10, 60)
    info_1 = "Use left click of the mouse to select the object to track"
    info_2 = "Use '1' to start tracking, '2' to reset tracking and 'q' to exit"
    cv2.putText(frame, info_1, menu_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    cv2.putText(frame, info_2, menu_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    if tracking_state:
        cv2.putText(frame, "tracking", menu_pos_3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    else:
        cv2.putText(frame, "not tracking", menu_pos_3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


points = []

def mouse_event_handler(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        points.append((x, y))


capture = cv2.VideoCapture(0)

window_name = "Object tracking using dlib correlation filter algorithm"

cv2.namedWindow(window_name)

cv2.setMouseCallback(window_name, mouse_event_handler)

tracker = dlib.correlation_tracker()

tracking_state = False

while True:
    ret, frame = capture.read()
    draw_text_info()
    if len(points) == 2:
        cv2.rectangle(frame, points[0], points[1], (0, 0, 255), 3)
        dlib_rectangle = dlib.rectangle(points[0][0], points[0][1], points[1][0], points[1][1])

    if tracking_state == True:
        tracker.update(frame)
        pos = tracker.get_position()
        cv2.rectangle(frame, (int(pos.left()), int(pos.top())), (int(pos.right()), int(pos.bottom())), (0, 255, 0), 3)

    key = 0xFF & cv2.waitKey(1)
    if key == ord("1"):
        if len(points) == 2:
            tracker.start_track(frame, dlib_rectangle)
            tracking_state = True
            points = []

    if key == ord("2"):
        points = []
        tracking_state = False

    if key == ord('q'):
        break

    cv2.imshow(window_name, frame)

capture.release()
cv2.destroyAllWindows()
