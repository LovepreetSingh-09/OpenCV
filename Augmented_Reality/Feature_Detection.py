import cv2
from matplotlib import pyplot as plt

def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


fig = plt.figure(figsize=(12, 5))
plt.suptitle("ORB keypoint detector", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image = cv2.imread('opencv_logo_with_text.png')

orb = cv2.ORB_create()

keypoints = orb.detect(image, None)

keypoints, descriptors = orb.compute(image, keypoints)
print(len(keypoints), descriptors.shape)
print("First extracted descriptor: {}".format(descriptors[0]))

image_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(255, 0, 255), flags=0)

show_img_with_matplotlib(image, "image", 1)
show_img_with_matplotlib(image_keypoints, "detected keypoints", 2)

plt.show()
