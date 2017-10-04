import numpy as np
import cv2

# create a white canvas
image = np.ones((480, 640, 3), dtype="uint8") * 255

# NumPy drawing: fill top left with green
image[:240,:320] = (0, 255, 0)

# OpenCV drawing: fill bottom left with yellow
cv2.rectangle(image, (0, 240), (319, 479), (0, 255, 255), -1)

# if you're running on the online VM, run this:
cv2.imwrite("canvas.png", image)

# if you're running on your own local machine, you can view it directly
# by uncommenting these lines:
# cv2.imshow("Canvas", image)
# cv2.waitKey(0)
