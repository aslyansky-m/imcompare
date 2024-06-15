import numpy as np
import matplotlib.pyplot as plt
import cv2

im1_path, im2_path = "input/im2.png", "input/im1.png"

im1 = cv2.cvtColor(cv2.imread(im1_path), cv2.COLOR_BGR2RGB)
im2 = cv2.cvtColor(cv2.imread(im2_path), cv2.COLOR_BGR2RGB)


gray1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
gray2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
sift = cv2.SIFT_create(nfeatures=200)
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# use bf matcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)
good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
mask = mask.squeeze().astype(bool)

# plot matches
# im_matches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, good_matches, None)
# plt.imshow(im_matches)
# plt.show()

gray_warped = cv2.warpPerspective(gray1, H, (gray2.shape[1], gray2.shape[0]))

# plot warped image
im = np.stack([gray2, gray_warped, gray2], axis=-1)
plt.imshow(im)
plt.show()