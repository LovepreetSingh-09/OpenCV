import cv2
import numpy as np
import os

class FaceProcessing(object):
    def __init__(self):
        self.file = os.path.join("haarcascade_frontalface_alt.xml")
        self.face_cascade = cv2.CascadeClassifier(self.file)

    def face_detection(self, image):
        image_array = np.asarray(bytearray(image), dtype=np.uint8)
        img_opencv = cv2.imdecode(image_array, -1)
        output = []
        gray = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
        for face in faces:
            x, y, w, h = face.tolist()
            face = {"box": [x, y, x + w, y + h]}
            output.append(face)
        return output
