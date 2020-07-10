import cv2
import numpy as np
import os

class ImageProcessing(object):
    def __init__(self):
        self.file_cascade = os.path.join(os.path.join(os.path.dirname(__file__), "data"),
                                         "haarcascade_frontalcatface_extended.xml")
        self.file_prototxt = os.path.join(os.path.join(os.path.dirname(__file__), "data"),
                                          "MobileNetSSD_deploy.prototxt.txt")
        self.file_caffemodel = os.path.join(os.path.join(os.path.dirname(__file__), "data"),
                                            "MobileNetSSD_deploy.caffemodel")
        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                        "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        self.cat_cascade = cv2.CascadeClassifier(self.file_cascade)
        self.net = cv2.dnn.readNetFromCaffe(self.file_prototxt, self.file_caffemodel)

    def cat_face_detection(self, image):
        image_array = np.asarray(bytearray(image), dtype=np.uint8)
        img_opencv = cv2.imdecode(image_array, -1)
        output = []
        gray = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2GRAY)
        cats = self.cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
        for cat in cats:
            x, y, w, h = cat.tolist()
            face = {"box": [x, y, x + w, y + h]}
            output.append(face)
        return output

    def cat_detection(self, image):
        image_array = np.asarray(bytearray(image), dtype=np.uint8)
        img_opencv = cv2.imdecode(image_array, -1)
        blob = cv2.dnn.blobFromImage(img_opencv, 0.007843, (300, 300), (127.5, 127.5, 127.5))
        self.net.setInput(blob)
        detections = self.net.forward()
        dim = 300
        output = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.1:
                class_id = int(detections[0, 0, i, 1])
                left = int(detections[0, 0, i, 3] * dim)
                top = int(detections[0, 0, i, 4] * dim)
                right = int(detections[0, 0, i, 5] * dim)
                bottom = int(detections[0, 0, i, 6] * dim)
                heightFactor = img_opencv.shape[0] / dim
                widthFactor = img_opencv.shape[1] / dim
                left = int(widthFactor * left)
                top = int(heightFactor * top)
                right = int(widthFactor * right)
                bottom = int(heightFactor * bottom)
                if self.classes[class_id] == 'cat':
                    cat = {"box": [left, top, right, bottom]}
                    output.append(cat)
        return output
