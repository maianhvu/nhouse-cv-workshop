from __future__ import print_function
import cv2

# read the image from disk
image = cv2.imread("../images/puppy.jpg")

# print out the size of the image
print(image.shape)
