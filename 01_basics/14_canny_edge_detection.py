import cv2
import numpy as np

image = cv2.imread("../images/puppy.jpg")
gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# blur the image using a Gaussian Blur to remove noises
# here we are blurring with a 5x5 kernel and sigmaX = sigmaY = 0
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# find the edges within the blurred image
edges = cv2.Canny(blurred, 100, 140)

# try to overlay the edges on top of the image
# but we have to dilate the edges a little bit so that it's clearer to see
dilated = cv2.dilate(edges, None, iterations=1)
# create an image of the dilated edges, in color yellow
edge_image = np.zeros(image.shape, dtype="uint8")
edge_image[:] = (0, 255, 255) # fill the image in yellow
edge_image = cv2.bitwise_and(edge_image, edge_image, mask=dilated)

# create the edge overlaid image
edge_overlaid = cv2.bitwise_or(edge_image, image)

# if you're running on the online VM, run this:
cv2.imwrite("puppy_edges.png", edges)
cv2.imwrite("puppy_edge_overlaid.jpg", edge_overlaid)

# if you're running on your own local machine, you can view it directly
# by uncommenting these lines:
# cv2.imshow("Edges", edges)
# cv2.imshow("Overlay", edge_overlaid)
# cv2.waitKey(0)
