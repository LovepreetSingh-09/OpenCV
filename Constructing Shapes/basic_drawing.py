import cv2
import matplotlib.pyplot as plt
import numpy as np

def show(image, title='Plot_Shape'):
    img = image[:, :, ::-1]
    plt.imshow(img)
    plt.title(title)
    plt.show()

colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255),
          'magenta': (255, 0, 255), 'cyan': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0),
          'gray': (125, 125, 125), 'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
          'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}

image = np.zeros((400, 400, 3), dtype="uint8")
image[:] = colors['light_gray']

show(image, 'original_image')

cv2.line(image, (0, 0), (400, 400), color=colors['green'], thickness=2, lineType=8, shift=0)
cv2.line(image, (0, 400), (400, 0), colors['blue'], 3)
cv2.line(image, (200, 0), (200, 400), colors['red'], 10)
cv2.line(image, (0, 200), (400, 200), colors['yellow'], 10)

show(image, 'lined_image')

image[:] = colors['light_gray']
cv2.rectangle(image, (0, 0), (50, 100), color=colors['green'], thickness=2, lineType=cv2.LINE_AA)
cv2.rectangle(image, (80, 50), (130, 300), colors['blue'], -1)
cv2.rectangle(image, (150, 50), (350, 100), colors['red'], -1)
cv2.rectangle(image, (150, 150), (350, 300), colors['cyan'], 10)

show(image, 'Rectangled_Image')

image[:] = colors['light_gray']
cv2.circle(image, (10, 10), 10, color=colors['green'], thickness=5)
cv2.circle(image, (100, 100), 30, colors['blue'], -1)
cv2.circle(image, (200, 200), 40, colors['magenta'], 10)
cv2.circle(image, (300, 300), 40, colors['cyan'], -1)

show(image, 'Circled_Image')

image[:] = colors['light_gray']
cv2.line(image, (0, 0), (400, 400), color=colors['green'], thickness=3, lineType=8)
cv2.rectangle(image, (0, 0), (50, 50), color=colors['blue'], thickness=2)

ret, p1, p2 = cv2.clipLine((0, 0, 50, 50), (0, 0), (400, 400))
if ret:
    cv2.line(image, p1, p2, color=colors['yellow'], thickness=3)

show(image, 'Clipped Line')

image[:] = colors['light_gray']

cv2.arrowedLine(image, (40, 40), (250, 40), color=colors['red'], thickness=3, line_type=8, shift=0, tipLength=0.05)
cv2.arrowedLine(image, (50, 120), (200, 120), colors['green'], 3, cv2.LINE_AA, 0, 0.3)
cv2.arrowedLine(image, (50, 200), (200, 200), colors['blue'], 3, 8, 0, 0.3)

show(image, 'Arrowed Lines')

image[:] = colors['light_gray']

cv2.ellipse(image, (80,80), (40, 70), 30, 0, 360, colors['green'], 5, 8)
cv2.ellipse(image, (80, 200), (10, 40), 0, 0, 360, colors['blue'], 3)
cv2.ellipse(image, (200, 200), (10, 40), 0, 0, 180, colors['yellow'], 3)
cv2.ellipse(image, (200, 100), (10, 40), 0, 0, 270, colors['cyan'], 3)
cv2.ellipse(image, (250, 250), (30, 30), 0, 0, 360, colors['magenta'], 3)
cv2.ellipse(image, (250, 100), (20, 40), 45, 0, 360, colors['gray'], 3)

show(image)


image[:] = colors['light_gray']

pts = np.array([[250, 5], [220, 80], [280, 80]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(image, [pts], True, colors['green'], 3)

pts = np.array([[250, 105], [220, 180], [280, 180]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(image, [pts], False, colors['green'], 3)

pts = np.array([[20, 90], [60, 60], [100, 90], [80, 130], [40, 130]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(image, [pts], True, colors['blue'], 3)

pts = np.array([[20, 180], [60, 150], [100, 180], [80, 220], [40, 220]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(image, [pts], False, colors['blue'], 3)

pts = np.array([[150, 100], [200, 100], [200, 150], [150, 150]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(image, [pts], True, colors['yellow'], 3)

pts = np.array([[150, 200], [200, 200], [200, 250], [150, 250]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(image, [pts], False, colors['yellow'], 3)

show(image, 'cv2.polylines()')
