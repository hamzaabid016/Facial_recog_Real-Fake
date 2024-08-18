import os
import cv2
import numpy as np
from sklearn.preprocessing import Normalizer
from face_detection import RetinaFace
from architecture import InceptionResNetV2

# Initialize face encoder
required_shape = (160, 160)
face_encoder = InceptionResNetV2()
package_file = os.path.abspath(os.path.dirname(__file__))
path = "facenet_keras_weights.h5"
face_encoder.load_weights(os.path.join(package_file, path))
l2_normalizer = Normalizer('l2')
face_detector = RetinaFace()

# Helper functions
def change_box_value(bbox):
    return int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def get_encoding(image):
    encodes = []
    img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_detector(img_RGB)
    if faces:
        x1, y1, width, height = change_box_value(faces[0][0])
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = img_RGB[y1:y2, x1:x2]

        face = normalize(face)
        face = cv2.resize(face, required_shape)
        face_d = np.expand_dims(face, axis=0)
        encode = face_encoder.predict(face_d)[0]
        encodes.append(encode)

    if encodes:
        encode = np.sum(encodes, axis=0)
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
        return encode
    return None
