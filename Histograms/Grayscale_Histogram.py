import numpy as np
import cv2
from matplotlib import pyplot as plt

def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def show_hist_with_matplotlib_gray(hist, title, pos, color):
    ax = plt.subplot(2, 3, pos)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)


plt.figure(figsize=(15, 6))
plt.suptitle("Grayscale histograms", fontsize=14, fontweight='bold')

image = cv2.imread('lenna.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist(image, [0], None, [256], [0, 256])

show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray", 1)
show_hist_with_matplotlib_gray(hist, "grayscale histogram", 4, 'm')

M = np.ones(gray_image.shape, dtype="uint8") * 35
added_image = cv2.add(gray_image, M)
hist_added_image = cv2.calcHist([added_image], [0], None, [256], [0, 256])

mask = np.zeros(gray_image.shape[:2], np.uint8)
mask[30:190, 30:190] = 255
subtracted_image = cv2.subtract(gray_image, M)
hist_subtracted_masked_image = cv2.calcHist([subtracted_image], [0], mask, [256], [0, 256])
masked_img = cv2.bitwise_and(subtracted_image, subtracted_image, mask=mask)

show_img_with_matplotlib(cv2.cvtColor(added_image, cv2.COLOR_GRAY2BGR), "gray lighter", 2)
show_hist_with_matplotlib_gray(hist_added_image, "grayscale histogram", 5, 'm')
show_img_with_matplotlib(cv2.cvtColor(masked_img, cv2.COLOR_GRAY2BGR), "masked gray darker", 3)
show_hist_with_matplotlib_gray(hist_subtracted_masked_image, "grayscale histogram", 6, 'm')

plt.show()


def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(3, 4, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def show_hist_with_matplotlib_gray(hist, title, pos, color):
    ax = plt.subplot(3, 4, pos)
    # plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)


plt.figure(figsize=(18, 14))
plt.suptitle("Grayscale histogram equalization with cv2.equalizeHist()", fontsize=16, fontweight='bold')
image = cv2.imread('lenna.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# It normalizes brightness and increases contrast
gray_image_eq = cv2.equalizeHist(gray_image)
hist_eq = cv2.calcHist([gray_image_eq], [0], None, [256], [0, 256])

M = np.ones(gray_image.shape, dtype="uint8") * 35
added_image = cv2.add(gray_image, M)
hist_added_image = cv2.calcHist([added_image], [0], None, [256], [0, 256])

added_image_eq = cv2.equalizeHist(added_image)
hist_eq_added_image = cv2.calcHist([added_image_eq], [0], None, [256], [0, 256])

subtracted_image = cv2.subtract(gray_image, M)
hist_subtracted_image = cv2.calcHist([subtracted_image], [0], None, [256], [0, 256])

subtracted_image_eq = cv2.equalizeHist(subtracted_image)
hist_eq_subtracted_image = cv2.calcHist([subtracted_image_eq], [0], None, [256], [0, 256])

show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray", 1)
show_hist_with_matplotlib_gray(hist, "grayscale histogram", 2, 'm')
show_img_with_matplotlib(cv2.cvtColor(added_image, cv2.COLOR_GRAY2BGR), "gray lighter", 5)
show_hist_with_matplotlib_gray(hist_added_image, "grayscale histogram", 6, 'm')
show_img_with_matplotlib(cv2.cvtColor(subtracted_image, cv2.COLOR_GRAY2BGR), "gray darker", 9)
show_hist_with_matplotlib_gray(hist_subtracted_image, "grayscale histogram", 10, 'm')

show_img_with_matplotlib(cv2.cvtColor(gray_image_eq, cv2.COLOR_GRAY2BGR), "grayscale equalized", 3)
show_hist_with_matplotlib_gray(hist_eq, "grayscale equalized histogram", 4, 'm')
show_img_with_matplotlib(cv2.cvtColor(added_image_eq, cv2.COLOR_GRAY2BGR), "gray lighter equalized", 7)
show_hist_with_matplotlib_gray(hist_eq_added_image, "grayscale equalized histogram", 8, 'm')
show_img_with_matplotlib(cv2.cvtColor(subtracted_image_eq, cv2.COLOR_GRAY2BGR), "gray darker equalized", 11)
show_hist_with_matplotlib_gray(hist_eq_subtracted_image, "grayscale equalized histogram", 12, 'm')

plt.show()
