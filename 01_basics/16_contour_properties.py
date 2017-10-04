import cv2
import numpy as np

image = cv2.imread("../images/shapes2.png")
gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

contours = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

# find the bounding boxes of the contours and draw them
bb_image = image.copy()
for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(bb_image, (x, y), (x + w, y + h), (0, 255, 255), 2)

# find the rotated bounding boxes
rotated_bb_image = image.copy()
for c in contours:
    rotated_bb = cv2.minAreaRect(c)
    # the `points` variable holds the coordinates of the vertices of the
    # rotated bounding box, which is also considered a contour itself
    points = cv2.boxPoints(rotated_bb).astype("int64")
    cv2.drawContours(rotated_bb_image, [points], -1, (0, 255, 255), 2)

# if you're running on the online VM, run this:
cv2.imwrite("shapes2_bb.png", bb_image)
cv2.imwrite("shapes2_rotated_bb.png", rotated_bb_image)

# if you're running on your own local machine, you can view it directly
# by uncommenting these lines:
# cv2.imshow("Bounding Boxes", bb_image)
# cv2.imshow("Rotated Bounding Boxes", rotated_bb_image)
# cv2.waitKey(0)
