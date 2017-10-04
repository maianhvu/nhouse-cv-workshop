import cv2

image = cv2.imread("../images/shapes3.png")
gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

contours = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

# define a function that approximates a contour based on a certain ratio of
# its perimeter
def approx(contour, peri_ratio=0.01):
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, peri_ratio * peri, True)

# filter out only contours that approximate to 4 vertices
rectangles = filter(lambda c: len(approx(c)) == 4, contours)
# draw all of these contours on the image with a yellow outline
cv2.drawContours(image, rectangles, -1, (0, 255, 255), 2)


# if you're running on the online VM, run this:
cv2.imwrite("shapes3_rectangles.png", image)

# if you're running on your own local machine, you can view it directly
# by uncommenting these lines:
# cv2.imshow("Rectangle Contours", image)
# cv2.waitKey(0)
