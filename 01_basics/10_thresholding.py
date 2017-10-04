import cv2

image = cv2.imread("../images/coins.png")
gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# threshold the grayscale image with t = 180 and max = 255, and then invert
# the output image with the THRESH_BINARY_INV option (as oppsed to just
# THRESH_BINARY)
_, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

# mask the original image using the thresholded image
masked = cv2.bitwise_and(image, image, mask=thresh)

# if you're running on the online VM, run this:
cv2.imwrite("coins_thresholded.png", thresh)
cv2.imwrite("coins_masked.png", masked)

# if you're running on your own local machine, you can view it directly
# by uncommenting these lines:
# cv2.imshow("Threshold", thresh)
# cv2.imshow("Masked", masked)
# cv2.waitKey(0)
