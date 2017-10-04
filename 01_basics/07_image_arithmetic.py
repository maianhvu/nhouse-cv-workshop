import cv2
import numpy as np

image1 = cv2.imread("../images/puppy.jpg")
image2 = cv2.imread("../images/puppy2.jpg")

# calculate the sum of all pixels from the two images, clipping the maximum
# value at 255
image_sum = cv2.add(image1, image2)

# calculate the absolute sum of the two images, then normalize the values
# to be in the range [0, 255]
normalized_sum = image1.astype("float32") + image2.astype("float32")
normalized_sum = ((normalized_sum - np.min(normalized_sum)) /
  (np.max(normalized_sum) - np.min(normalized_sum)))
normalized_sum = (normalized_sum * 255).astype("uint8")

# add some brightness
brighter = cv2.add(image1, np.ones(image1.shape, dtype="uint8") * 100)
darker = cv2.subtract(image1, np.ones(image1.shape, dtype="uint8") * 100)

# if you're running on the online VM, run this:
cv2.imwrite("puppies_sum.jpg", image_sum)
cv2.imwrite("puppies_sum_normalized.jpg", normalized_sum)
cv2.imwrite("puppy_brighter.jpg", brighter)
cv2.imwrite("puppy_darker.jpg", darker)

# if you're running on your own local machine, you can view it directly
# by uncommenting these lines:
# cv2.imshow("Sum", image_sum)
# cv2.imshow("Normalized Sum", normalized_sum)
# cv2.imshow("Brighter", brighter)
# cv2.imshow("Darker", darker)
# cv2.waitKey(0)
