import cv2
import imutils

image = cv2.imread("../images/puppy.jpg")

# resize to half using OpenCV (have to specify both width and height)
half = cv2.resize(image, (180, 135))

# resize to double using imutils (specify only either width or height, the
# other side will be scaled according to aspect ratio)
double = imutils.resize(image, width=720)

# if you're running on the online VM, run this:
cv2.imwrite("puppy_half.png", half)
cv2.imwrite("puppy_double.png", double)

# if you're running on your own local machine, you can view it directly
# by uncommenting these lines:
# cv2.imshow("Half", half)
# cv2.imshow("Double", double)
# cv2.waitKey(0)
