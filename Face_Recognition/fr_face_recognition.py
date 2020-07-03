import face_recognition
import cv2

image = cv2.imread("jared_1.jpg")
image = image[:, :, ::-1]

encodings = face_recognition.face_encodings(image)
print(encodings[0])

known_image_1 = face_recognition.load_image_file("jared_1.jpg")
known_image_2 = face_recognition.load_image_file("jared_2.jpg")
known_image_3 = face_recognition.load_image_file("jared_3.jpg")
known_image_4 = face_recognition.load_image_file("obama.jpg")

names = ["jared_1.jpg", "jared_2.jpg", "jared_3.jpg", "obama.jpg"]

unknown_image = face_recognition.load_image_file("jared_4.jpg")

known_image_1_encoding = face_recognition.face_encodings(known_image_1)[0]
known_image_2_encoding = face_recognition.face_encodings(known_image_2)[0]
known_image_3_encoding = face_recognition.face_encodings(known_image_3)[0]
known_image_4_encoding = face_recognition.face_encodings(known_image_4)[0]
known_encodings = [known_image_1_encoding, known_image_2_encoding, known_image_3_encoding, known_image_4_encoding]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# uses the threshold/tolerance of 0.6
results = face_recognition.compare_faces(known_encodings, unknown_encoding)
print(results)