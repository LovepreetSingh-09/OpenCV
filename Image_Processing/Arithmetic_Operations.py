import numpy as np
import cv2
import matplotlib.pyplot as plt

x = np.uint8([250])
y = np.uint8([50])

result_opencv = cv2.add(x, y)  # == 255  Values are clipped
print("cv2.add(x:'{}' , y:'{}') = '{}'".format(x, y, result_opencv))
result_numpy = x + y  # 300 % 256 = 44 becoz of unsigned 8  bit integer
print("x:'{}' + y:'{}' = '{}'".format(x, y, result_numpy))


def show_with_matplotlib_1(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(1, 4, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


plt.figure(figsize=(10, 4))
plt.suptitle("Sobel operator and cv2.addWeighted() to show the output", fontsize=14, fontweight='bold')

image = cv2.imread('lenna.png')

image_filtered = cv2.GaussianBlur(image, (3, 3), 0)
gray_image = cv2.cvtColor(image_filtered, cv2.COLOR_BGR2GRAY)

# CV_16S = one channel of 2-byte signed integers (16-bit signed integers)
gradient_x = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0, 3)
gradient_y = cv2.Sobel(gray_image, cv2.CV_16S, 0, 1, 3)

abs_gradient_x = cv2.convertScaleAbs(gradient_x)
abs_gradient_y = cv2.convertScaleAbs(gradient_y)
sobel_image = cv2.addWeighted(abs_gradient_x, 0.5, abs_gradient_y, 0.5, 0)

show_with_matplotlib_1(image, "Image", 1)
show_with_matplotlib_1(cv2.cvtColor(abs_gradient_x, cv2.COLOR_GRAY2BGR), "Gradient x", 2)
show_with_matplotlib_1(cv2.cvtColor(abs_gradient_y, cv2.COLOR_GRAY2BGR), "Gradient y", 3)
show_with_matplotlib_1(cv2.cvtColor(sobel_image, cv2.COLOR_GRAY2BGR), "Sobel output", 4)
plt.show()



def show_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(3, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


plt.figure(figsize=(12, 6))
plt.suptitle("Bitwise operations (AND, OR, XOR, NOT)", fontsize=14, fontweight='bold')

img_1 = np.zeros((300, 300), dtype="uint8")
cv2.rectangle(img_1, (10, 10), (110, 110), (255, 255, 255), -1)
cv2.circle(img_1, (200, 200), 50, (255, 255, 255), -1)

img_2 = np.zeros((300, 300), dtype="uint8")
cv2.rectangle(img_2, (50, 50), (150, 150), (255, 255, 255), -1)
cv2.circle(img_2, (225, 200), 50, (255, 255, 255), -1)

bitwise_or = cv2.bitwise_or(img_1, img_2)

bitwise_and = cv2.bitwise_and(img_1, img_2)

bitwise_xor = cv2.bitwise_xor(img_1, img_2)

bitwise_not_1 = cv2.bitwise_not(img_1)

bitwise_not_2 = cv2.bitwise_not(img_2)

show_with_matplotlib(cv2.cvtColor(img_1, cv2.COLOR_GRAY2BGR), "image 1", 1)
show_with_matplotlib(cv2.cvtColor(img_2, cv2.COLOR_GRAY2BGR), "image 2", 2)
show_with_matplotlib(cv2.cvtColor(bitwise_or, cv2.COLOR_GRAY2BGR), "image 1 OR image 2", 3)
show_with_matplotlib(cv2.cvtColor(bitwise_and, cv2.COLOR_GRAY2BGR), "image 1 AND image 2", 4)
show_with_matplotlib(cv2.cvtColor(bitwise_xor, cv2.COLOR_GRAY2BGR), "image 1 XOR image 2", 5)
show_with_matplotlib(cv2.cvtColor(bitwise_not_1, cv2.COLOR_GRAY2BGR), "NOT (image 1)", 6)
show_with_matplotlib(cv2.cvtColor(bitwise_not_2, cv2.COLOR_GRAY2BGR), "NOT (image 2)", 7)

image = cv2.imread('cat.jpg')
img_3 = np.zeros((300, 300), dtype="uint8")
img_3 = cv2.resize(img_3, image.shape[1::-1])   #then do not use mask instead pass it as 2nd argument
print(image.shape[1::-1], image.shape, img_3.shape)
cv2.circle(img_3, (430, 525), 400, (255, 255, 255), -1)
bitwise_and_example = cv2.bitwise_and(image, image, mask=img_3)

show_with_matplotlib(cv2.cvtColor(img_3, cv2.COLOR_GRAY2BGR), "image 3", 8)
show_with_matplotlib(bitwise_and_example, "image 3 AND a loaded image", 9)

plt.show()


def show_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


plt.figure(figsize=(6, 5))
plt.suptitle("Bitwise AND/OR between two images", fontsize=14, fontweight='bold')

image = cv2.imread('lenna_250.png')
binary_image = cv2.imread('opencv_binary_logo_250.png')

print(image.shape, binary_image.shape)

bitwise_and = cv2.bitwise_and(image, binary_image)

bitwise_or = cv2.bitwise_or(image, binary_image)

show_with_matplotlib(image, "image", 1)
show_with_matplotlib(binary_image, "binary logo", 2)
show_with_matplotlib(bitwise_and, "AND operation", 3)
show_with_matplotlib(bitwise_or, "OR operation", 4)

plt.show()