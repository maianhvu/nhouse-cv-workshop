import cv2
image = cv2.imread("../images/puppy.jpg")

# flip vertically
v_flip = cv2.flip(image, 0)

# flip horizontally
h_flip = cv2.flip(image, 1)

# flip in both directions
flip = cv2.flip(image, -1)

# if you're running on the online VM, run this:
cv2.imwrite("puppy_flipped_v.jpg", v_flip)
cv2.imwrite("puppy_flipped_h.jpg", h_flip)
cv2.imwrite("puppy_flipped.jpg", flip)

# if you're running on your own local machine, you can view it directly
# by uncommenting these lines:
# cv2.imshow("Vertically Flipped", v_flip)
# cv2.imshow("Horizontally Flipped", h_flip)
# cv2.imshow("Flipped in Both Directions", flip)
# cv2.waitKey(0)
