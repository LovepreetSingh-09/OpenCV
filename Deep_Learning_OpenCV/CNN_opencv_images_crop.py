import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def get_cropped_imgs(imgs):
    imgs_cropped = []
    for img in imgs:
        img_copy = img.copy()
        size = min(img_copy.shape[1], img_copy.shape[0])
        x1 = int(0.5 * (img_copy.shape[1] - size))
        y1 = int(0.5 * (img_copy.shape[0] - size))
        imgs_cropped.append(img_copy[y1:(y1 + size), x1:(x1 + size)])
    return imgs_cropped

net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

image = cv2.imread("face_test.jpg")
image2 = cv2.imread("face_test_2.jpg")
images = [image, image2]

images_cropped = get_cropped_imgs(images)
blob_cropped = cv2.dnn.blobFromImages(images, 1.0, (300, 300), [104., 117., 123.], False, True)

net.setInput(blob_cropped)
detections = net.forward()

for i in range(0, detections.shape[2]):
    img_id = int(detections[0, 0, i, 0])
    confidence = detections[0, 0, i, 2]
    if confidence > 0.25:
        (h, w) = images_cropped[img_id].shape[:2]
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(images_cropped[img_id], (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(images_cropped[img_id], text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

fig = plt.figure(figsize=(16, 8))
plt.suptitle("OpenCV DNN face detector when feeding several images and cropping", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

show_img_with_matplotlib(image, "input img 1", 1)
show_img_with_matplotlib(image2, "input img 2", 2)
show_img_with_matplotlib(images_cropped[0], "output cropped img 1", 3)
show_img_with_matplotlib(images_cropped[1], "output cropped img 2", 4)

plt.show()