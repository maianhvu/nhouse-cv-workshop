import cv2
import numpy as np

image = cv2.imread("../images/shapes.png")
gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# find external contours within the image
ext_contours = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

# draw the contours with a yellow outline
externals_image = image.copy()
cv2.drawContours(externals_image, ext_contours, -1, (0, 255, 255), 3)

# find all contours within the image
all_contours = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]

# draw the contours with a yellow outline
all_image = image.copy()
cv2.drawContours(all_image, all_contours, -1, (0, 255, 255), 3)

# if you're running on the online VM, run this:
cv2.imwrite("shapes_contours_external.png", externals_image)
cv2.imwrite("shapes_contours_all.png", all_image)

# if you're running on your own local machine, you can view it directly
# by uncommenting these lines:
# cv2.imshow("External Contours", externals_image)
# cv2.imshow("All Contours", all_image)
# cv2.waitKey(0)
