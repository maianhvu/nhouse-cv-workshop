from __future__ import print_function
import cv2
import glob
import numpy as np
import imutils

# change this to False when you're running on your local machine
USING_ONLINE_VM = False

def calculate_histogram(image, bins=[4] * 3):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    normalized_hist = cv2.normalize(hist, None)
    return normalized_hist.flatten()


images = [cv2.imread(file_name) for file_name in glob.glob("./images/*.jpg")]
features = [calculate_histogram(image) for image in images]
features = np.reshape(features, (len(images), -2))

# define number of clusters
num_clusters = 3

# perform k-means clustering
num_points = len(features)
centroids = features[np.random.choice(num_points, num_clusters, replace=False)]
cluster_ids = -np.ones((num_points, ), dtype="int32")

while True:
    new_cluster_ids = np.zeros((num_points,), dtype="int32")
    for (i, feature) in enumerate(features):
        new_cluster_ids[i] = np.argmin(np.linalg.norm(centroids - feature, axis=1))

    if np.all(new_cluster_ids == cluster_ids):
        break

    cluster_ids = new_cluster_ids
    for i in xrange(num_clusters):
        centroids[i] = features[np.where(cluster_ids == i)].mean(axis=0)

# now that we have finished our k-means clustering, we should construct
# a montage to show our results
def join_horizontally(image1, image2):
    if image1 is None:
        return image2
    elif image2 is None:
        return image1

    if image1.shape[0] > image2.shape[0]:
        new_image2 = np.zeros((image1.shape[0], ) + image2.shape[1:], dtype="uint8")
        new_image2[:image2.shape[0],:,:] = image2
        image2 = new_image2
    elif image2.shape[0] > image1.shape[0]:
        new_image1 = np.zeros((image2.shape[0], ) + image1.shape[1:], dtype="uint8")
        new_image1[:image1.shape[0],:,:] = image1
        image1 = new_image1

    return np.hstack([image1, image2])


cluster_images = [None] * num_clusters
for (i, cluster_id) in enumerate(cluster_ids):
    cluster_images[cluster_id] = join_horizontally(
        cluster_images[cluster_id],
        imutils.resize(images[i], width=240)
        )

for cluster_id in cluster_ids:
    if USING_ONLINE_VM:
        cv2.imwrite("cluster_{}.jpg".format(cluster_id), cluster_images[cluster_id])
    else:
        cv2.imshow("Cluster {}".format(cluster_id), cluster_images[cluster_id])

if not USING_ONLINE_VM:
    cv2.waitKey(0)
