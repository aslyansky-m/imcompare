import cv2
import numpy as np
import matplotlib.pyplot as plt

def calc_transform(shape, scale, rotation, x_offset, y_offset):
    cols, rows = shape[:2]
    M1 = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, scale)
    M2 = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
    M1 = np.vstack([M1, [0, 0, 1]])
    M2 = np.vstack([M2, [0, 0, 1]])
    M = np.dot(M2,M1)
    return M
    

def decompose_homography(H, image_size):
    h, w = image_size[:2]
    corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    M, _ = cv2.estimateAffinePartial2D(corners.reshape(-1, 2), transformed_corners)

    scale = np.sqrt(np.linalg.det(M[:2,:2]))
    rotation = -np.rad2deg(np.arctan2(M[1, 0], M[0, 0]))

    M2 = calc_transform((w,h),scale,rotation,0,0)
    M1 = np.vstack([M, [0, 0, 1]])
    M3 = M1@np.linalg.inv(M2)
    translation = M3[:2, 2]

    H1 = calc_transform((w,h), scale, rotation, translation[0], translation[1])
    H2 = H@np.linalg.inv(H1)

    return translation, rotation, scale, H2


def sift_matching_with_homography(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    if len(good_matches) > 10:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H
    else:
        return None
    



img1_path, img2_path = "output/simulated/image_00.jpg", "output/simulated/image_01.jpg"
img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)

H = sift_matching_with_homography(img1, img2)
translation, rotation, scale, H2 = decompose_homography(H, img2.shape)
# H2 = calc_transform((img2.shape[1],img2.shape[0]), scale, rotation, translation[0], translation[1])
print(H)
print(H2)
print(H-H2)
img3 = cv2.warpPerspective(img1,H,(img2.shape[1],img2.shape[0]))
img3 = np.stack([img2.mean(axis=-1), img3.mean(axis=-1), img2.mean(axis=-1)], axis=-1)/255.0
img4 = cv2.warpPerspective(img1,H2,(img2.shape[1],img2.shape[0]))


fig, ax = plt.subplots(1,4)
ax[0].imshow(img1)
ax[1].imshow(img2)
ax[2].imshow(img3)
ax[3].imshow(img4)

plt.show()