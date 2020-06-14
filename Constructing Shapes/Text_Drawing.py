import cv2
import matplotlib.pyplot as plt
import numpy as np


def show(image, title='Plot_Shape'):
    print(image.shape)
    img = image[:, :, ::-1]
    plt.imshow(img)
    plt.title(title)
    plt.show()


colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255),
          'magenta': (255, 0, 255), 'cyan': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0),
          'gray': (125, 125, 125), 'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
          'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}

image = np.zeros((700, 700, 3), dtype="uint8")
image.fill(255)

cv2.putText(image, 'Hello World 1 ', (50, 30), cv2.FONT_HERSHEY_PLAIN, 2, colors['green'], 2, 4, True)
cv2.putText(image, 'Hello World 2 ', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors['red'], 2, cv2.LINE_8)
cv2.putText(image, 'Hello World 3 ', (10, 110), 4, 0.9, colors['blue'], 2, cv2.LINE_AA)

show(image, 'cv2.putText')

image.fill(255)
fonts = {0: "FONT HERSHEY SIMPLEX", 1: "FONT HERSHEY PLAIN", 2: "FONT HERSHEY DUPLEX", 3: "FONT HERSHEY COMPLEX",
         4: "FONT HERSHEY TRIPLEX", 5: "FONT HERSHEY COMPLEX SMALL ", 6: "FONT HERSHEY SCRIPT SIMPLEX",
         7: "FONT HERSHEY SCRIPT COMPLEX"}

index_colors = {0: 'blue', 1: 'green', 2: 'red', 3: 'yellow', 4: 'magenta', 5: 'cyan', 6: 'black', 7: 'dark_gray'}

position = (10, 30)

for i in range(8):
    print("i index value: '{}' text: '{}' + color: '{}' = '{}'".format(i, fonts[i].lower(), index_colors[i],
                                                                       colors[index_colors[i]]))
    cv2.putText(image, fonts[i], position, i, 1.1, colors[index_colors[i]], 2, 8)
    position = (position[0], position[1]+40)
    cv2.putText(image, fonts[i].lower(), position, i, 1.1, colors[index_colors[i]], 2, 8)
    position = (position[0], position[1]+40)

show(image, 'Different Fonts')
image = np.zeros((400, 1200, 3), dtype="uint8")

image.fill(220)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2.5
thickness = 5
text = 'abcdefghijklmnopqrstuvwxyz'
circle_radius = 10

ret, baseline = cv2.getTextSize(text, font, font_scale, thickness)

text_width, text_height = ret

text_x = int(round((image.shape[1] - text_width) / 2))
text_y = int(round((image.shape[0] + text_height) / 2))

cv2.circle(image, (text_x, text_y), 5, colors['yellow'], 3, 8)
cv2.line(image, (text_x, text_y + int(round(thickness / 2))), (text_x + text_width, text_y + int(round(thickness / 2))),
         colors['green'], 5, 8)

cv2.rectangle(image, (text_x, text_y + baseline), (text_x + text_width - thickness, text_y - text_height),
              colors['blue'], 5, 8)

cv2.putText(image, text, (text_x, text_y), font, font_scale, colors['red'], 2)
show(image)
