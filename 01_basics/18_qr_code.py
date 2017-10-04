import cv2
import numpy as np

image = cv2.imread("../images/qr.png")
gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# find all the contours and their hierarchical relationships using the
# cv2.RETR_TREE (retrieve a tree-like hierarchy) option
_, contours, hierarchy = cv2.findContours(gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# hierarchy is an array of matrices, we should only get the first one
hierarchy = hierarchy[0]

# define a few helper function for us to ascertain a contour's level
def approx_contour(contour, peri_ratio=0.01):
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, peri_ratio * peri, True)

def first_child_index(h):
    return h[2]

def level(contour_idx, hierarchy, memory):
    if contour_idx in memory:
        return memory[contour_idx]

    lvl = None
    child_id = first_child_index(hierarchy[contour_idx])
    if child_id == -1:
        lvl = 0
    else:
        lvl = 1 + level(child_id, hierarchy, memory)

    # memoizate result
    memory[contour_idx] = lvl
    return lvl

# define the dynamic programming memoizating dictionary
memory = {}

# find the contours indicating positional squares
# these squares will be at level 2 (having a grandchild) and approximates to
# 4 vertices
qr_pos_squares = filter(
    lambda c_id: level(c_id, hierarchy, memory) == 2 and len(approx_contour(contours[c_id])) == 4,
    xrange(len(contours))
    )

for c_id in qr_pos_squares:
    cv2.drawContours(image, contours, c_id, (0, 0, 255), 2)

# if you're running on the online VM, run this:
cv2.imwrite("qr_pos_squares.png", image)

# if you're running on your own local machine, you can view it directly
# by uncommenting these lines:
# cv2.imshow("QR Code", image)
# cv2.waitKey(0)
