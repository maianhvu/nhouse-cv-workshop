import cv2

image = cv2.imread("../images/puppy.jpg")
mask  = cv2.imread("../images/puppy_mask.png", 0) # read as grayscale

masked = cv2.bitwise_and(image, image, mask=mask)

# if you're running on the online VM, run this:
cv2.imwrite("puppy_masked.png", masked)

# if you're running on your own local machine, you can view it directly
# by uncommenting these lines:
# cv2.imshow("Masked", masked)
# cv2.waitKey(0)
