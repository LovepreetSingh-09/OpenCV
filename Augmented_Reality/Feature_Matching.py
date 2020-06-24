import cv2
from matplotlib import pyplot as plt

def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


fig = plt.figure(figsize=(8, 6))
plt.suptitle("ORB descriptors and Brute-Force (BF) matcher", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image_query = cv2.imread('opencv_logo_with_text.png')
image_scene = cv2.imread('opencv_logo_with_text_scene.png')

orb = cv2.ORB_create()

keypoints_1, descriptors_1 = orb.detectAndCompute(image_query, None)
keypoints_2, descriptors_2 = orb.detectAndCompute(image_scene, None)

bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

bf_matches = bf_matcher.match(descriptors_1, descriptors_2)

bf_matches = sorted(bf_matches, key=lambda x: x.distance)
print('bf_matches : {}'.format(bf_matches[0]))

result = cv2.drawMatches(image_query, keypoints_1, image_scene, keypoints_2, bf_matches, None,
                         matchColor=(255, 255, 0), singlePointColor=(255, 0, 255), flags=0)

show_img_with_matplotlib(result, "matches between the two images", 1)

plt.show()