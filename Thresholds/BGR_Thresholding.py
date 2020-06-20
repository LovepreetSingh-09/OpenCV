import cv2
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(1, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


fig = plt.figure(figsize=(12, 4))
plt.suptitle("Thresholding BGR images", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image = cv2.imread('cat.jpg')

show_img_with_matplotlib(image, "image", 1)

ret1, thresh1 = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)

(b, g, r) = cv2.split(image)
ret2, thresh2 = cv2.threshold(b, 120, 255, cv2.THRESH_BINARY)
ret3, thresh3 = cv2.threshold(g, 120, 255, cv2.THRESH_BINARY)
ret4, thresh4 = cv2.threshold(r, 120, 255, cv2.THRESH_BINARY)
bgr_thresh = cv2.merge((thresh2, thresh3, thresh4))

show_img_with_matplotlib(thresh1, "threshold (120) BGR image", 2)
show_img_with_matplotlib(bgr_thresh, "threshold (120) each channel and merge", 3)

plt.show()