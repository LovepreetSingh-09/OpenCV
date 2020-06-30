import cv2
import cvlib as cv
from matplotlib import pyplot as plt


def show_detection(image, faces):
    for (startX, startY, endX, endY) in faces:
        cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)
    return image


def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


image = cv2.imread("test_face_detection.jpg")

faces, confidences = cv.detect_face(image)

img_result = show_detection(image.copy(), faces)

fig = plt.figure(figsize=(10, 5))
plt.suptitle("Face detection using cvlib face detector", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

show_img_with_matplotlib(img_result, "cvlib face detector: " + str(len(faces)), 1)

plt.show()
