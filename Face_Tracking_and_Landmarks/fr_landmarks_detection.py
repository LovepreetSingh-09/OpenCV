import cv2
import face_recognition
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


image = cv2.imread("face_test.png")

image_68 = image.copy()
image_5 = image.copy()

rgb = image[:, :, ::-1]

face_landmarks_list_68 = face_recognition.face_landmarks(rgb)

print(face_landmarks_list_68)

for face_landmarks in face_landmarks_list_68:
    for facial_feature in face_landmarks.keys():
        for p in face_landmarks[facial_feature]:
            cv2.circle(image_68, p, 2, (0, 255, 0), -1)

face_landmarks_list_5 = face_recognition.face_landmarks(rgb, None, "small")

print(face_landmarks_list_5)

for face_landmarks in face_landmarks_list_5:
    for facial_feature in face_landmarks.keys():
        for p in face_landmarks[facial_feature]:
            cv2.circle(image_5, p, 2, (0, 255, 0), -1)

fig = plt.figure(figsize=(10, 5))
plt.suptitle("Facial landmarks detection using face_recognition", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

show_img_with_matplotlib(image_68, "68 facial landmarks", 1)
show_img_with_matplotlib(image_5, "5 facial landmarks", 2)

plt.show()
