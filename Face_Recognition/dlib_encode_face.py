import cv2
import dlib
import numpy as np

pose_predictor_5_point = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
detector = dlib.get_frontal_face_detector()

def face_encodings(face_image, number_of_times_to_upsample=1, num_jitters=1):
    face_locations = detector(face_image, number_of_times_to_upsample)
    raw_landmarks = [pose_predictor_5_point(face_image, face_location) for face_location in face_locations]
    # Calculate the face encoding for every detected face using the detected landmarks for each one:
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for
            raw_landmark_set in raw_landmarks]


image = cv2.imread("jared_1.jpg")
rgb = image[:, :, ::-1]

encodings = face_encodings(rgb)
print(encodings[0])
