import cv2

image = cv2.imread("../images/coins.png")
gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# threshold the grayscale image with t = 180 and max = 255, and then invert
# the output image with the THRESH_BINARY_INV option (as oppsed to just
# THRESH_BINARY)
_, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

# close the thresholded image with a 3x3 circular kernel 4 times
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)

# mask the original image using the thresholded image
masked = cv2.bitwise_and(image, image, mask=closed)

# if you're running on the online VM, run this:
cv2.imwrite("coins_closed_mask.png", closed)
cv2.imwrite("coins_full_masked.png", masked)

# if you're running on your own local machine, you can view it directly
# by uncommenting these lines:
# cv2.imshow("Closed Threshold", closed)
# cv2.imshow("Masked", masked)
# cv2.waitKey(0)
