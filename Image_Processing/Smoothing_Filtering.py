import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(3, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


plt.figure(figsize=(12, 6))
plt.suptitle("Smoothing techniques", fontsize=14, fontweight='bold')

image = cv2.imread('cat-face.png')
kernel_averaging_10_10 = np.ones((10, 10), np.float32) / 100
kernel_averaging_5_5 = np.ones((5, 5), np.float32) / 25

print("kernel: {}".format(kernel_averaging_5_5))

smooth_image_f2D_5_5 = cv2.filter2D(image, -1, kernel_averaging_5_5)
smooth_image_f2D_10_10 = cv2.filter2D(image, -1, kernel_averaging_10_10)

smooth_image_b = cv2.blur(image, (10, 10))

# If normalize (by default True) of cv2.boxFilter() is equals to True, then it is equal to cv2.boxFilter()
smooth_image_bfi = cv2.boxFilter(image, -1, (10, 10), normalize=True)

smooth_image_gb = cv2.GaussianBlur(image, (9, 9), 0)

smooth_image_mb = cv2.medianBlur(image, 9)

smooth_image_bf = cv2.bilateralFilter(image, 5, 10, 10)
smooth_image_bf_2 = cv2.bilateralFilter(image, 9, 200, 200)

show_with_matplotlib(image, "original", 1)
show_with_matplotlib(smooth_image_f2D_5_5, "cv2.filter2D() (5,5) kernel", 2)
show_with_matplotlib(smooth_image_f2D_10_10, "cv2.filter2D() (10,10) kernel", 3)
show_with_matplotlib(smooth_image_b, "cv2.blur()", 4)
show_with_matplotlib(smooth_image_bfi, "cv2.boxFilter()", 5)
show_with_matplotlib(smooth_image_gb, "cv2.GaussianBlur()", 6)
show_with_matplotlib(smooth_image_mb, "cv2.medianBlur()", 7)
show_with_matplotlib(smooth_image_bf, "cv2.bilateralFilter() - small values", 8)
show_with_matplotlib(smooth_image_bf_2, "cv2.bilateralFilter() - big values", 9)

plt.show()
