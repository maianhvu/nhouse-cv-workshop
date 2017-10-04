import cv2
import numpy as np

# read NUS logo as a grayscale image
image = cv2.imread("../images/nus.png", 0)

# threshold the logo to have a black and white image first
_, thresh = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY_INV)

# create a circular kernel of size 3x3
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# EROSION
eroded = np.empty((0, image.shape[1]), dtype="uint8")
for i in xrange(5):
  # stack the current vertical arrangement of eroded images with a new
  # eroded image
  eroded = np.vstack([
    eroded,
    cv2.erode(thresh, kernel, iterations=i)
    ])

# DILATION
dilated = np.empty((0, image.shape[1]), dtype="uint8")
for i in xrange(5):
  dilated = np.vstack([
    dilated,
    cv2.dilate(thresh, kernel, iterations=i)
    ])

# OPENING
opened = np.empty((0, image.shape[1]), dtype="uint8")
for i in xrange(5):
  opened = np.vstack([
    opened,
    cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=i)
    ])

# CLOSING
closed = np.empty((0, image.shape[1]), dtype="uint8")
for i in xrange(5):
  closed = np.vstack([
    closed,
    cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=i)
    ])

# if you're running on the online VM, run this:
cv2.imwrite("nus_erosion.png", eroded)
cv2.imwrite("nus_dilation.png", dilated)
cv2.imwrite("nus_opening.png", opened)
cv2.imwrite("nus_closing.png", closed)

# if you're running on your own local machine, you can view it directly
# by uncommenting these lines:
# cv2.imshow("Erosion", eroded)
# cv2.imshow("Dilation", dilated)
# cv2.imshow("Opening", opened)
# cv2.imshow("Closing", closed)
# cv2.waitKey(0)

