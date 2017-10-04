import cv2
import numpy as np

# change this to False when you're running on your local machine
USING_ONLINE_VM = True

class FaceDetector:
    def __init__(self):
        self.detector = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")

    def detect(self, image, draw=False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.3, 5)
        if draw:
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        return faces

if __name__ == "__main__":
    image = cv2.imread("./images/photo.jpg")
    detector = FaceDetector()
    detector.detect(image, draw=True)
    if USING_ONLINE_VM:
        cv2.imwrite("detected_faces.jpg", image)
    else:
        cv2.imshow("Faces", image)
        cv2.waitKey(0)
