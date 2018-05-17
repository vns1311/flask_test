import cv2
from PIL import Image
from io import BytesIO
import numpy as np


def check_for_face(img_data, haar_face_cascade_path, app):
    img_pil = Image.open(BytesIO(img_data))
    img_GRAY = cv2.cvtColor(np.array(img_pil), cv2.COLOR_BGR2GRAY)
    haar_face_cascade = cv2.CascadeClassifier(haar_face_cascade_path)
    faces = haar_face_cascade.detectMultiScale(img_GRAY, scaleFactor=1.1, minNeighbors=5)
    face_count = len(faces)
    app.logger.info("%s faces found in image", face_count)
    return face_count
