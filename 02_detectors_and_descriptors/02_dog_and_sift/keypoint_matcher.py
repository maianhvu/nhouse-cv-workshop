import cv2
import numpy as np

# change this to False when you're running on your local machine
USING_ONLINE_VM = True

image1 = cv2.imread("./images/smashingbook.jpg")
image2 = cv2.imread("./images/books.jpg")

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kps1, descs1 = sift.detectAndCompute(gray1, None)
kps2, descs2 = sift.detectAndCompute(gray2, None)

# match the keypoints
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(index_params, search_params)
matches = matcher.knnMatch(descs1, descs2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.8 * n.distance:
        good_matches.append(m)

# find a consistent homography using RANSAC
src_pts = np.float32([kps1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kps2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5, 0)
matches_mask = mask.ravel().tolist()

draw_params = dict(
    matchColor=(0, 255, 0),
    singlePointColor=None,
    matchesMask=matches_mask,
    flags=2
    )

# draw the first 10 matches
match_image = cv2.drawMatches(
    image1, kps1,
    image2, kps2,
    good_matches,
    outImg=None,
    **draw_params
    )

# draw the keypoints in the two images
for kp in kps1:
    radius = int(kp.size / 2)
    (x, y) = np.int64(kp.pt)
    cv2.circle(image1, (x, y), radius, (0, 255, 255), 2)

for kp in kps2:
    radius = int(kp.size / 2)
    (x, y) = np.int64(kp.pt)
    cv2.circle(image2, (x, y), radius, (0, 255, 255), 2)


if USING_ONLINE_VM:
    cv2.imwrite("keypoints1.jpg", image1)
    cv2.imwrite("keypoints2.jpg", image2)
    cv2.imwrite("matched_keypoints.jpg", match_image)
else:
    cv2.imshow("Keypoints 1", image1)
    cv2.imshow("Keypoints 2", image2)
    cv2.imshow("Matches", match_image)
    cv2.waitKey(0)
