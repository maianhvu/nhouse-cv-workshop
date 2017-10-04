import cv2
import numpy as np

image = cv2.imread("../images/puppy.jpg", 0) # read the grayscale image
kernel = np.array([
  [1, 4, 6, 4, 1],
  [4, 16, 24, 16, 4],
  [6, 24, 36, 24, 6],
  [4, 16, 24, 16, 4],
  [1, 4, 6, 4, 1]
  ], dtype="float32") / 256.0

# convolve the kernel with the image
convolved = np.zeros(image.shape, dtype="float32")
for y in xrange(kernel.shape[0] / 2, image.shape[0] - kernel.shape[0] / 2):
  for x in xrange(kernel.shape[1] / 2, image.shape[1] - kernel.shape[1] / 2):
    for j in xrange(kernel.shape[0]):
      for i in xrange(kernel.shape[1]):
        convolved[y, x] += kernel[j, i] * image[y - kernel.shape[0] / 2 + j, x - kernel.shape[1] / 2]

# normalize the output image
convolved = (convolved - convolved.min()) / (convolved.max() - convolved.min())
convolved = np.round(convolved * 255, 0).astype("uint8")

# if you're running on the online VM, run this:
cv2.imwrite("puppy_convolved.jpg", convolved)

# if you're running on your own local machine, you can view it directly
# by uncommenting these lines:
# cv2.imshow("Original", image)
# cv2.imshow("Convolved", convolved)
# cv2.waitKey(0)
