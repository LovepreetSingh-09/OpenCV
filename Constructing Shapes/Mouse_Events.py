import cv2
import matplotlib.pyplot as plt
import numpy as np

colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255),
          'magenta': (255, 0, 255), 'cyan': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0),
          'gray': (125, 125, 125), 'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
          'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}

image = np.zeros((700, 700, 3), dtype="uint8")
image.fill(255)


def draw_rectangles(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("event: EVENT_LBUTTONDBLCLK")
        cv2.rectangle(image, (x, y), (x + 20, y + 40), colors['green'], 7, 8)

    if event == cv2.EVENT_MOUSEMOVE:
        print("event: EVENT_MOUSEMOVE")

    if event == cv2.EVENT_LBUTTONUP:
        print("event: EVENT_LBUTTONUP")

    if event == cv2.EVENT_LBUTTONDOWN:
        print("event: EVENT_LBUTTONDOWN")


cv2.namedWindow('img_window')

cv2.setMouseCallback('img_window', draw_rectangles)

while True:
    cv2.imshow('img_window', image)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

image = np.zeros((700, 700, 3), dtype="uint8")
image.fill(255)


def update_img_with_matplotlib():
    img_RGB = image[:, :, ::-1]
    plt.imshow(img_RGB)
    # Redraw the updated image
    figure.canvas.draw()


def click_mouse_event(event):
    cv2.circle(image, (int(event.xdata), int(event.ydata)), 10, colors['green'], 5, 8)
    update_img_with_matplotlib()


figure = plt.figure()

update_img_with_matplotlib()

figure.canvas.mpl_connect('button_press_event', click_mouse_event)

plt.show()
cv2.destroyAllWindows()

image = np.zeros((700, 700, 3), dtype="uint8")
image.fill(125)


def draw_text():
    # We set the position to be used for drawing text:
    menu_pos = (10, 500)
    menu_pos2 = (10, 525)
    menu_pos3 = (10, 550)
    menu_pos4 = (10, 575)
    cv2.putText(image, 'Double left click: add a circle', menu_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))
    cv2.putText(image, 'Simple right click: delete last circle', menu_pos2, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255))
    cv2.putText(image, 'Double right click: delete all circle', menu_pos3, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255))
    cv2.putText(image, 'Press \'q\' to exit', menu_pos4, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))


def draw_circles(event, x, y, flag, params):
    global circles
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # Add the circle with coordinates x,y
        print("event: EVENT_LBUTTONDBLCLK")
        circles.append((x, y))
    if event == cv2.EVENT_RBUTTONDBLCLK:
        # Delete all circles (clean the screen)
        print("event: EVENT_RBUTTONDBLCLK")
        circles[:] = []
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Delete last added circle
        print("event: EVENT_RBUTTONDOWN")
        try:
            circles.pop()
        except (IndexError):
            print("no circles to delete")
    if event == cv2.EVENT_MOUSEMOVE:
        print("event: EVENT_MOUSEMOVE")
    if event == cv2.EVENT_LBUTTONUP:
        print("event: EVENT_LBUTTONUP")
    if event == cv2.EVENT_LBUTTONDOWN:
        print("event: EVENT_LBUTTONDOWN")


circles = []
cv2.namedWindow('image_window')
cv2.setMouseCallback('image_window', draw_circles)
draw_text()

clone = image.copy()

while True:
    image = clone.copy()
    for pos in circles:
        cv2.circle(image, pos, 10, colors['green'], 5, 8)

    cv2.imshow('image_window', image)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
