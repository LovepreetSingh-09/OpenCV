import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_with_matplotlib(img, title):
    img_RGB = img[:, :, ::-1]
    plt.imshow(img_RGB)
    plt.title(title)
    plt.show()

image = cv2.imread('lena_image.png')
show_with_matplotlib(image, 'Original image')

# 1.  Resizing Image
dst_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
height, width = image.shape[:2]
dst_image_2 = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)
show_with_matplotlib(dst_image, 'Resized image')
show_with_matplotlib(dst_image_2, 'Resized image 2')

# 2. Translating Image
M = np.float32([[1, 0, 200], [0, 1, -30]])
dst_image = cv2.warpAffine(image, M, (width, height))
show_with_matplotlib(dst_image, 'Translated image (positive values)')

# 3. Rotating Image
M = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 100, 1)
dst_image = cv2.warpAffine(image, M, (width, height))
cv2.circle(dst_image, (round(width / 2.0), round(height / 2.0)), 5, (255, 0, 0), -1)
show_with_matplotlib(dst_image, 'Image rotated 180 degrees')
# Changing the center of rotation
M = cv2.getRotationMatrix2D((width / 1.5, height / 1.5), 30, 1)
dst_image = cv2.warpAffine(image, M, (width, height))
cv2.circle(dst_image, (round(width / 1.5), round(height / 1.5)), 5, (255, 0, 0), -1)
show_with_matplotlib(dst_image, 'Image rotated 30 degrees')

# 4. Affine Transformation
image_points = image.copy()
cv2.circle(image_points, (135, 45), 5, (255, 0, 255), -1)
cv2.circle(image_points, (385, 45), 5, (255, 0, 255), -1)
cv2.circle(image_points, (135, 230), 5, (255, 0, 255), -1)
show_with_matplotlib(image_points, 'before affine transformation')
# We create the arrays with the aforementioned three points and the desired positions in the output image:
pts_1 = np.float32([[135, 45], [385, 45], [135, 230]])
pts_2 = np.float32([[135, 45], [385, 45], [150, 230]])
M = cv2.getAffineTransform(pts_1, pts_2)
dst_image = cv2.warpAffine(image_points, M, (width, height))
show_with_matplotlib(dst_image, 'Affine transformation')

# 5. Perspective Transformation
image_points = image.copy()
cv2.circle(image_points, (450, 65), 5, (255, 0, 255), -1)
cv2.circle(image_points, (517, 65), 5, (255, 0, 255), -1)
cv2.circle(image_points, (431, 164), 5, (255, 0, 255), -1)
cv2.circle(image_points, (552, 164), 5, (255, 0, 255), -1)
show_with_matplotlib(image_points, 'before perspective transformation')
pts_1 = np.float32([[450, 65], [517, 65], [431, 164], [552, 164]])
pts_2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
M = cv2.getPerspectiveTransform(pts_1, pts_2)
dst_image = cv2.warpPerspective(image, M, (300, 300))
show_with_matplotlib(dst_image, 'perspective transformation')

# 6. Cropping
image_points = image.copy()
cv2.circle(image_points, (230, 80), 5, (0, 0, 255), -1)
cv2.circle(image_points, (330, 80), 5, (0, 0, 255), -1)
cv2.circle(image_points, (230, 200), 5, (0, 0, 255), -1)
cv2.circle(image_points, (330, 200), 5, (0, 0, 255), -1)
cv2.line(image_points, (230, 80), (330, 80), (0, 0, 255))
cv2.line(image_points, (230, 200), (330, 200), (0, 0, 255))
cv2.line(image_points, (230, 80), (230, 200), (0, 0, 255))
cv2.line(image_points, (330, 200), (330, 80), (0, 0, 255))
show_with_matplotlib(image_points, 'Before cropping')
dst_image = image[80:200, 230:330]
show_with_matplotlib(dst_image, 'Cropping image')
