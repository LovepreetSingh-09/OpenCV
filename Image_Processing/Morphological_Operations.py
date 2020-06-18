import cv2
import matplotlib.pyplot as plt

image_names = ['test1.png', 'test2.png', 'test3.png']

kernel_size_3_3 = (3, 3)
kernel_size_5_5 = (5, 5)

def load_all_test_images():
    test_morph_images = []
    for name_image in (image_names):
        test_morph_images.append(cv2.imread(name_image))
    return test_morph_images


def show_images(array_img, title, pos):
    for index_image, image in enumerate(array_img):
        show_with_matplotlib(image, title + "_" + str(index_image + 1),
                             pos + index_image * (len(morphological_operations) + 1))


def show_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(len(image_names), len(morphological_operations) + 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def build_kernel(kernel_type, kernel_size):
    if kernel_type == cv2.MORPH_ELLIPSE:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    elif kernel_type == cv2.MORPH_CROSS:
        return cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    else:  # cv2.MORPH_RECT
        return cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)


def erode(image, kernel_type, kernel_size):
    kernel = build_kernel(kernel_type, kernel_size)
    erosion = cv2.erode(image, kernel, iterations=1)
    return erosion


def dilate(image, kernel_type, kernel_size):
    kernel = build_kernel(kernel_type, kernel_size)
    dilation = cv2.dilate(image, kernel, iterations=1)
    return dilation


# This function closes the image
# Closing = dilation + erosion
def closing(image, kernel_type, kernel_size):
    kernel = build_kernel(kernel_type, kernel_size)
    clos = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return clos


# This function opens the image
# Opening = erosion + dilation
def opening(image, kernel_type, kernel_size):
    """Opens the image with the specified kernel type and size"""

    kernel = build_kernel(kernel_type, kernel_size)
    ope = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return ope


# This function applies the morphological gradient to the image
def morphological_gradient(image, kernel_type, kernel_size):
    kernel = build_kernel(kernel_type, kernel_size)
    morph_gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    return morph_gradient


# This function closes and opens the image
def closing_and_opening(image, kernel_type, kernel_size):
    closing_img = closing(image, kernel_type, kernel_size)
    opening_img = opening(closing_img, kernel_type, kernel_size)
    return opening_img


# This function opens and closes the image
def opening_and_closing(image, kernel_type, kernel_size):
    opening_img = opening(image, kernel_type, kernel_size)
    closing_img = closing(opening_img, kernel_type, kernel_size)
    return closing_img


morphological_operations = {
    'erode': erode,
    'dilate': dilate,
    'closing': closing,
    'opening': opening,
    'gradient': morphological_gradient,
    'closing|opening': closing_and_opening,
    'opening|closing': opening_and_closing
}


# Apply the 'morphological_operation' (e.g. 'erode', 'dilate', 'closing') to all the images in the array
def apply_morphological_operation(array_img, morphological_operation, kernel_type, kernel_size):
    morphological_operation_result = []
    for index_image, image in enumerate(array_img):
        result = morphological_operations[morphological_operation](image, kernel_type, kernel_size)
        morphological_operation_result.append(result)
    return morphological_operation_result


# Show the morphological_operations dictionary
# This is only for debugging purposes
for i, (k, v) in enumerate(morphological_operations.items()):
    print("index: '{}', key: '{}', value: '{}'".format(i, k, v))

test_images = load_all_test_images()

plt.figure(figsize=(16, 8))
plt.suptitle("Morpho operations - kernel_type='cv2.MORPH_RECT', kernel_size='(3,3)'", fontsize=14, fontweight='bold')

show_images(test_images, "test img", 1)

for i, (k, v) in enumerate(morphological_operations.items()):
    show_images(apply_morphological_operation(test_images, k, cv2.MORPH_RECT, kernel_size_3_3), k, i + 2)

plt.show()

plt.figure(figsize=(16, 8))
plt.suptitle("Morpho operations - kernel_type='cv2.MORPH_RECT', kernel_size='(5,5)'", fontsize=14, fontweight='bold')

show_images(test_images, "test img", 1)

for i, (k, v) in enumerate(morphological_operations.items()):
    show_images(apply_morphological_operation(test_images, k, cv2.MORPH_RECT, kernel_size_5_5), k, i + 2)

plt.show()

plt.figure(figsize=(16, 8))
plt.suptitle("Morpho operations - kernel_type='cv2.MORPH_CROSS', kernel_size='(3,3)'", fontsize=14, fontweight='bold')

show_images(test_images, "test img", 1)

for i, (k, v) in enumerate(morphological_operations.items()):
    show_images(apply_morphological_operation(test_images, k, cv2.MORPH_CROSS, kernel_size_3_3), k, i + 2)

plt.show()

plt.figure(figsize=(16, 8))
plt.suptitle("Morpho operations - kernel_type='cv2.MORPH_CROSS', kernel_size='(5,5)'", fontsize=14, fontweight='bold')

show_images(test_images, "test img", 1)

for i, (k, v) in enumerate(morphological_operations.items()):
    show_images(apply_morphological_operation(test_images, k, cv2.MORPH_CROSS, kernel_size_5_5), k, i + 2)

plt.show()
