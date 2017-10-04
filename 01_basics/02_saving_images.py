import numpy as np
import cv2

# create a completely black image, of width 640, height 480, with 3 color
# channels and data type of 8-bit unsigned integer
image = np.zeros((480, 640, 3), dtype="uint8")

# save the image to a file
cv2.imwrite("black.png", image)
