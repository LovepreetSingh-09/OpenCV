import numpy as np
import cv2
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def show_hist_with_matplotlib_rgb(hist, title, pos, color):
    ax = plt.subplot(2, 3, pos)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    for (h, c) in zip(hist, color):
        plt.plot(h, color=c)


def hist_color_img(img):
    histr = []
    histr.append(cv2.calcHist([img], [0], None, [256], [0, 256]))
    histr.append(cv2.calcHist([img], [1], None, [256], [0, 256]))
    histr.append(cv2.calcHist([img], [2], None, [256], [0, 256]))
    return histr


plt.figure(figsize=(15, 6))
plt.suptitle("Color histograms", fontsize=14, fontweight='bold')

image = cv2.imread('lenna.png')

hist_color = hist_color_img(image)

show_img_with_matplotlib(image, "image", 1)

show_hist_with_matplotlib_rgb(hist_color, "color histogram", 4, ['b', 'g', 'r'])

M = np.ones(image.shape, dtype="uint8") * 15
added_image = cv2.add(image, M)
hist_color_added_image = hist_color_img(added_image)

subtracted_image = cv2.subtract(image, M)
hist_color_subtracted_image = hist_color_img(subtracted_image)

show_img_with_matplotlib(added_image, "image lighter", 2)
show_hist_with_matplotlib_rgb(hist_color_added_image, "color histogram", 5, ['b', 'g', 'r'])
show_img_with_matplotlib(subtracted_image, "image darker", 3)
show_hist_with_matplotlib_rgb(hist_color_subtracted_image, "color histogram", 6, ['b', 'g', 'r'])

plt.show()


def equalize_hist_color(img):
    channels = cv2.split(img)
    eq_channels = []
    for ch in channels:
        eq_channels.append(cv2.equalizeHist(ch))
    eq_image = cv2.merge(eq_channels)
    return eq_image


def equalize_hist_color_hsv(img):
    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    eq_V = cv2.equalizeHist(V)
    eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2BGR)
    return eq_image