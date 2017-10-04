import cv2

image = cv2.imread("../images/puppy.jpg")

# crop a region starting from the 50th row and 100th column to
# the 150th row and 250th column (end-exclusive)
region = image[50:150, 100:250]

# extract the width and height of the image
(height, width) = image.shape[:2]

# get 4 corners of the image
top_left  = image[:height/2, :width/2]
top_right = image[:height/2, width/2:]
bottom_left  = image[height/2:, :width/2]
bottom_right = image[height/2:, width/2:]

# if you're running on the online VM, run this:
cv2.imwrite("puppy_small_region.jpg", region)
cv2.imwrite("puppy_top_left.jpg",  top_left)
cv2.imwrite("puppy_top_right.jpg", top_right)
cv2.imwrite("puppy_bottom_left.jpg",  bottom_left)
cv2.imwrite("puppy_bottom_right.jpg", bottom_right)

# if you're running on your own local machine, you can view it directly
# by uncommenting these lines:
# cv2.imshow("Region", region)
# cv2.imshow("Top left",  top_left)
# cv2.imshow("Top right", top_right)
# cv2.imshow("Bottom left",  bottom_left)
# cv2.imshow("Bottom right", bottom_right)
# cv2.waitKey(0)
