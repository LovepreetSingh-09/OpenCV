import cv2
import dlib
from matplotlib import pyplot as plt

def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def show_detection(image, faces):
    for face in faces:
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 10)
    return image

def show_detection_cnn(image, faces):
    # faces contains a list of mmod_rectangle objects
    # The mmod_rectangle object has two member variables, a dlib.rectangle object, and a confidence score
    for face in faces:
        cv2.rectangle(image, (face.rect.left(), face.rect.top()), (face.rect.right(), face.rect.bottom()), (255, 0, 0),
                      10)
    return image


img = cv2.imread("test_face_detection.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

rects_1 = detector(gray, 1)
rects_2 = cnn_face_detector(img.copy(), 0)

img_faces_1 = show_detection(img.copy(), rects_1)
img_faces_2 = show_detection_cnn(img.copy(), rects_2)

fig = plt.figure(figsize=(10, 4))
plt.suptitle("Face detection using dlib frontal face detector", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

show_img_with_matplotlib(img_faces_2, "detector(gray, 1): " + str(len(rects_1)), 1)
show_img_with_matplotlib(img_faces_1, "cnn_detector: " + str(len(rects_2)), 2)

plt.show()
