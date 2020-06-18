import cv2
from matplotlib import pyplot as plt

def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(2, 5, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def equalize_clahe_color_hsv(img):
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    eq_V = clahe.apply(V)
    eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2BGR)
    return eq_image


def equalize_clahe_color_lab(img):
    cla = cv2.createCLAHE(clipLimit=4.0)
    L, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2Lab))
    eq_L = cla.apply(L)
    eq_image = cv2.cvtColor(cv2.merge([eq_L, a, b]), cv2.COLOR_Lab2BGR)
    return eq_image


def equalize_clahe_color_yuv(img):
    cla = cv2.createCLAHE(clipLimit=4.0)
    Y, U, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YUV))
    eq_Y = cla.apply(Y)
    eq_image = cv2.cvtColor(cv2.merge([eq_Y, U, V]), cv2.COLOR_YUV2BGR)
    return eq_image


def equalize_clahe_color(img):
    cla = cv2.createCLAHE(clipLimit=4.0)
    channels = cv2.split(img)
    eq_channels = []
    for ch in channels:
        eq_channels.append(cla.apply(ch))
    eq_image = cv2.merge(eq_channels)
    return eq_image


plt.figure(figsize=(18, 14))
plt.suptitle("Histogram equalization using CLAHE", fontsize=16, fontweight='bold')

image = cv2.imread('lenna.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0)
gray_image_clahe = clahe.apply(gray_image)
clahe.setClipLimit(5.0)
gray_image_clahe_2 = clahe.apply(gray_image)
clahe.setClipLimit(10.0)
gray_image_clahe_3 = clahe.apply(gray_image)
clahe.setClipLimit(20.0)
gray_image_clahe_4 = clahe.apply(gray_image)

image_clahe_color = equalize_clahe_color(image)
image_clahe_color_lab = equalize_clahe_color_lab(image)
image_clahe_color_hsv = equalize_clahe_color_hsv(image)
image_clahe_color_yuv = equalize_clahe_color_yuv(image)

show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray", 1)
show_img_with_matplotlib(cv2.cvtColor(gray_image_clahe, cv2.COLOR_GRAY2BGR), "gray CLAHE clipLimit=2.0", 2)
show_img_with_matplotlib(cv2.cvtColor(gray_image_clahe_2, cv2.COLOR_GRAY2BGR), "gray CLAHE clipLimit=5.0", 3)
show_img_with_matplotlib(cv2.cvtColor(gray_image_clahe_3, cv2.COLOR_GRAY2BGR), "gray CLAHE clipLimit=10.0", 4)
show_img_with_matplotlib(cv2.cvtColor(gray_image_clahe_4, cv2.COLOR_GRAY2BGR), "gray CLAHE clipLimit=20.0", 5)
show_img_with_matplotlib(image, "color", 6)
show_img_with_matplotlib(image_clahe_color, "clahe on each channel (BGR)", 7)
show_img_with_matplotlib(image_clahe_color_lab, "clahe on L channel (LAB)", 8)
show_img_with_matplotlib(image_clahe_color_hsv, "clahe on V channel (HSV)", 9)
show_img_with_matplotlib(image_clahe_color_yuv, "clahe on Y channel (YUV)", 10)

plt.show()


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

plt.figure(figsize=(14, 8))
plt.suptitle("Grayscale histogram equalization with cv2.calcHist() and CLAHE", fontsize=16, fontweight='bold')

image = cv2.imread('lenna.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

gray_image_eq = cv2.equalizeHist(gray_image)
hist_eq = cv2.calcHist([gray_image_eq], [0], None, [256], [0, 256])

clahe = cv2.createCLAHE(clipLimit=4.0)
gray_image_clahe = clahe.apply(gray_image)
hist_clahe = cv2.calcHist([gray_image_clahe], [0], None, [256], [0, 256])

show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray", 1)
show_hist_with_matplotlib_gray(hist, "grayscale histogram", 4, 'm')
show_img_with_matplotlib(cv2.cvtColor(gray_image_eq, cv2.COLOR_GRAY2BGR), "grayscale equalized", 2)
show_hist_with_matplotlib_gray(hist_eq, "grayscale equalized histogram", 5, 'm')
show_img_with_matplotlib(cv2.cvtColor(gray_image_clahe, cv2.COLOR_GRAY2BGR), "grayscale CLAHE", 3)
show_hist_with_matplotlib_gray(hist_clahe, "grayscale clahe histogram", 6, 'm')

plt.show()