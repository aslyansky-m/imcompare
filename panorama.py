import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from glob import glob
from tqdm import tqdm, trange
import networkx as nx

sift = cv2.SIFT_create(nfeatures=1000)
def get_sift_features(image):
    return sift.detectAndCompute(image, None)

def match_sift_features(feat1, feat2, use_knn=False):
    keypoints1, descriptors1 = feat1
    keypoints2, descriptors2 = feat2
    if use_knn:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    else:
        bf = cv2.BFMatcher()
        matches1 = bf.knnMatch(descriptors1, descriptors2, 2)
        matches2 = bf.knnMatch(descriptors2, descriptors1, 2)

        # Apply ratio test and cross-checking
        good_matches = []
        for m, n in matches1:
            if m.distance < 0.75 * n.distance:
                for n_match in matches2[m.trainIdx]:
                    if n_match.trainIdx == m.queryIdx:
                        good_matches.append(m)

    if len(good_matches) > 10:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H, np.mean(mask), np.sum(mask)
    else:
        return np.eye(3), 0, 0
    
def lightest_path(adj_matrix, source, target):
    # Create a graph from the adjacency matrix
    G = nx.Graph()
    num_nodes = len(adj_matrix)
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i][j] != 0:
                G.add_edge(i, j, weight=adj_matrix[i][j])
    
    # Find the lightest path
    lightest_path = nx.shortest_path(G, source=source, target=target, weight='weight')
    
    # Calculate the total weight of the lightest path
    total_weight = nx.shortest_path_length(G, source=source, target=target, weight='weight')
    
    return lightest_path, total_weight

# Function to find the corners of an image
def get_image_corners(image):
    h, w = image.shape[:2]
    return np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1]
    ]).T

# Function to transform the corners with a given homography
def transform_corners(corners, H):
    transformed_corners = H @ corners
    transformed_corners /= transformed_corners[2]  # Normalize by the last row
    return transformed_corners[:2]

    
def stitch_images(images):
    features = []

    for image in images:
        features.append(get_sift_features(image))

    N = len(images)

    inliers = np.zeros([N,N])
    ratios = np.zeros([N,N])
    Hs = []

    for i in range(N):
        for j in range(i):
            H, score, inlier = match_sift_features(features[i], features[j])
            ratios[i,j] = score
            inliers[i,j] = inlier
    
    temp = np.minimum(ratios,inliers/100)
    scores = -np.log(temp + temp.T)
    
    # Precompute homographies and transformed corners
    homographies = []
    transformed_corners_list = []
    upscale = 3.0
    new_shape = (int(images[0].shape[1] * upscale), int(images[0].shape[0] * upscale))
    H_upscale = np.diag([upscale, upscale, 1])

    for i in range(0, 16):
        H = np.eye(3)
        if i > 0:
            path, weight = lightest_path(scores + 0.1, 0, i)
            for j in range(1, len(path)):
                H_rel, _, _ = match_sift_features(features[path[j]], features[path[j - 1]])
                H = H @ H_rel
        
        H_final = H_upscale @ H
        homographies.append(H_final)
        
        corners = get_image_corners(images[i])
        transformed_corners = transform_corners(corners, H_final)
        transformed_corners_list.append(transformed_corners)

    # Determine the bounds of the panorama
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    for transformed_corners in transformed_corners_list:
        min_x = min(min_x, np.min(transformed_corners[0]))
        max_x = max(max_x, np.max(transformed_corners[0]))
        min_y = min(min_y, np.min(transformed_corners[1]))
        max_y = max(max_y, np.max(transformed_corners[1]))

    # Calculate the size of the final panorama
    panorama_width = int(np.ceil(max_x - min_x))
    panorama_height = int(np.ceil(max_y - min_y))
    offset_x = -min_x
    offset_y = -min_y
    H_translation = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ])

    # Stitch the images into the final panorama
    aligned_images = []
    for i in range(0, 16):
        H_final = H_translation @ homographies[i]
        im = cv2.warpPerspective(images[i], H_final, (panorama_width, panorama_height))
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        aligned_images.append(im)

    sum_image = aligned_images[0].copy()
    for image in aligned_images:
        mask = ((image > 0) * 255).astype(np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8))[:, :, 0]
        sum_image[mask > 0] = image[mask > 0]
        
    return sum_image
