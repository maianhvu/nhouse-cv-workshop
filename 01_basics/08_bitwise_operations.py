import cv2
import numpy as np

# create a square image
square = np.zeros((250, 250), dtype="uint8")
cv2.rectangle(square, (25, 25), (224, 224), 255, -1)

# create a circle image
circle = np.zeros((250, 250), dtype="uint8")
cv2.circle(circle, (124, 124), 125, 255, -1)

# calculate all the bitwise operations
and_result = cv2.bitwise_and(square, circle)
or_result  = cv2.bitwise_or (square, circle)
xor_result = cv2.bitwise_xor(square, circle)
not_result = cv2.bitwise_not(circle)

# if you're running on the online VM, run this:
cv2.imwrite("bitwise_square.png", square)
cv2.imwrite("bitwise_circle.png", circle)
cv2.imwrite("bitwise_and.png", and_result)
cv2.imwrite("bitwise_or.png", or_result)
cv2.imwrite("bitwise_xor.png", xor_result)
cv2.imwrite("bitwise_not.png", not_result)

# if you're running on your own local machine, you can view it directly
# by uncommenting these lines:
# cv2.imshow("Square", square)
# cv2.imshow("Circle", circle)
# cv2.imshow("AND", and_result)
# cv2.imshow("OR", or_result)
# cv2.imshow("XOR", xor_result)
# cv2.imshow("NOT", not_result)
# cv2.waitKey(0)

