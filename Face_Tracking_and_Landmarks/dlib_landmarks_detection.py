import cv2
import dlib
import numpy as np

JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_BRIDGE_POINTS = list(range(27, 31))
LOWER_NOSE_POINTS = list(range(31, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))
ALL_POINTS = list(range(0, 68))


def draw_shape_lines_all(np_shape, image):
    draw_shape_lines_range(np_shape, image, JAWLINE_POINTS)
    draw_shape_lines_range(np_shape, image, RIGHT_EYEBROW_POINTS)
    draw_shape_lines_range(np_shape, image, LEFT_EYEBROW_POINTS)
    draw_shape_lines_range(np_shape, image, NOSE_BRIDGE_POINTS)
    draw_shape_lines_range(np_shape, image, LOWER_NOSE_POINTS)
    draw_shape_lines_range(np_shape, image, RIGHT_EYE_POINTS, True)
    draw_shape_lines_range(np_shape, image, LEFT_EYE_POINTS, True)
    draw_shape_lines_range(np_shape, image, MOUTH_OUTLINE_POINTS, True)
    draw_shape_lines_range(np_shape, image, MOUTH_INNER_POINTS, True)


def draw_shape_lines_range(np_shape, image, range_points, is_closed=False):
    np_shape_display = np_shape[range_points]
    points = np.array(np_shape_display, dtype=np.int32)
    cv2.polylines(image, [points], is_closed, (255, 255, 0), thickness=1, lineType=cv2.LINE_8)


def draw_shape_points_pos_range(np_shape, image, points):
    np_shape_display = np_shape[points]
    draw_shape_points_pos(np_shape_display, image)


def draw_shape_points_pos(np_shape, image):
    for idx, (x, y) in enumerate(np_shape):
        # Draw the positions for every detected landmark:
        cv2.putText(image, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
        # Draw a point on every landmark position:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)


def draw_shape_points_range(np_shape, image, points):
    """Draws the shape using points for every landmark filtering by points parameter"""
    np_shape_display = np_shape[points]
    draw_shape_points(np_shape_display, image)


def draw_shape_points(np_shape, image):
    # Draw a point on every landmark position:
    for (x, y) in np_shape:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)


def shape_to_np(dlib_shape, dtype="int"):
    # Initialize the list of (x,y) coordinates
    coordinates = np.zeros((dlib_shape.num_parts, 2), dtype=dtype)
    # Loop over all facial landmarks and convert them to a tuple with (x,y) coordinates:
    for i in range(0, dlib_shape.num_parts):
        coordinates[i] = (dlib_shape.part(i).x, dlib_shape.part(i).y)
    return coordinates


# Name of the two shape predictors:
p = "shape_predictor_68_face_landmarks.dat"
# p = "shape_predictor_5_face_landmarks.dat"

# Initialize frontal face detector and shape predictor:
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

video_capture = cv2.VideoCapture(0)

test_face = cv2.imread("face_test.png")

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 1)
        # Get the shape using the predictor:
        shape = predictor(gray, rect)
        # Convert the shape to numpy array:
        shape = shape_to_np(shape)

        # Draw all lines connecting the different face parts:
        # draw_shape_lines_all(shape, frame)

        # Draw jaw line:
        # draw_shape_lines_range(shape, frame, JAWLINE_POINTS)

        # Draw all points and their position:
        # draw_shape_points_pos(shape, frame)
        # You can also use:
        # draw_shape_points_pos_range(shape, frame, ALL_POINTS)

        # Draw all shape points:
        draw_shape_points(shape, frame)

        # Draw left eye, right eye and bridge shape points and positions
        # draw_shape_points_pos_range(shape, frame, LEFT_EYE_POINTS + RIGHT_EYE_POINTS + NOSE_BRIDGE_POINTS)

    cv2.imshow("Landmarks detection using dlib", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()

print(shape.shape, shape)