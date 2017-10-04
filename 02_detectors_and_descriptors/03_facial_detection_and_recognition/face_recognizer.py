import cv2
import numpy as np
import glob
from sklearn.decomposition import PCA
from face_detector import FaceDetector

face_data = None
for file in glob.glob("./assets/vu*.jpg"):
    face = cv2.imread(file, 0)
    face_flipped = cv2.flip(face, 1)
    face = face.reshape((1, -1))
    face_flipped = face_flipped.reshape((1, -1))
    face_data = face if face_data is None else np.vstack([face_data, face])
    face_data = np.vstack([face_data, face_flipped])

pca = PCA(svd_solver="randomized", n_components=150, whiten=True)
train_face = pca.fit_transform(face_data)

image = cv2.imread("./images/photo.jpg")
gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detector = FaceDetector()
faces = detector.detect(image)

for (x, y, w, h) in faces:
    face_image = cv2.resize(gray[y:y+h, x:x+w], (64, 64))
    test_face = pca.score(face_image.reshape((1, -1)))
    print(test_face)


