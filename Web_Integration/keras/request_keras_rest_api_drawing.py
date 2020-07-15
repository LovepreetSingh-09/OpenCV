import cv2
import numpy as np
import requests
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "car.jpg"

image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

r = requests.post(KERAS_REST_API_URL, files=payload).json()

image_array = np.asarray(bytearray(image), dtype=np.uint8)
img_opencv = cv2.imdecode(image_array, -1)
img_opencv = cv2.resize(img_opencv, (500, 500))

y_pos = 40
if r["success"]:
    for (i, result) in enumerate(r["predictions"]):
        print("{}. {}: {:.4f}".format(i + 1, result["label"], result["probability"]))
        cv2.putText(img_opencv, "{}. {}: {:.4f}".format(i + 1, result["label"], result["probability"]),
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        y_pos += 30
else:
    print("Request failed")

fig = plt.figure(figsize=(8, 6))
plt.suptitle("Using Keras Deep Learning REST API", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

show_img_with_matplotlib(img_opencv, "Classification results (NASNetMobile)", 1)

plt.show()