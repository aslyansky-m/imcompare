import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from glob import glob
import networkx as nx
import tkinter as tk
from tkinter import ttk

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

def update_progress_bar(progress_bar, value):
    if progress_bar is None:
        return
    progress_bar['value'] = value
    progress_bar.update()

def create_cache(images, with_gui=True, step_penalty = 0.1, matching_dist=10):

    if with_gui:
        root = tk.Tk()
        root.title("Cache Initialization Progress")

        step_label = tk.Label(root, text="Initializing...")
        step_label.pack(padx=10,pady=10)
        
        current_progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
        current_progress.pack(padx=10,pady=10)
        
        total_progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
        total_progress.pack(padx=10,pady=10)
    else:
        current_progress = None
        total_progress = None

    N = len(images)

    def update_step_label(text):
        if with_gui:
            step_label.config(text=text)
            step_label.update()

    time_ratio = 0.8
    first_steps = N
    second_steps = 0
    for i in range(N):
        for j in range(i):
            second_steps += (abs(i-j) < matching_dist)
    total_steps = time_ratio*first_steps + (1-time_ratio)*second_steps
    # Step 1: Extracting features
    features = []
    update_step_label("Extracting features")
    for i, image in enumerate(images):
        features.append(get_sift_features(image))
        cur_ratio = i / N
        global_ratio = time_ratio*i/total_steps
        update_progress_bar(total_progress, global_ratio * 100)
        update_progress_bar(current_progress, cur_ratio * 100)

    inliers = np.zeros([N, N])
    ratios = np.zeros([N, N])
    # 2d array of None homographies
    matrices = [[None for _ in range(N)] for _ in range(N)]

    # Step 2: Matching features
    update_step_label("Matching features")
    step = 0
    for i in range(N):
        for j in range(i):
            if abs(i-j) < matching_dist:
                H, score, inlier = match_sift_features(features[i], features[j])
                ratios[i, j] = score
                inliers[i, j] = inlier
                matrices[i][j] = H
                matrices[j][i] = np.linalg.inv(H)
            step += 1
            cur_ratio = step / second_steps
            global_ratio = (time_ratio*first_steps + (1-time_ratio)*step)/total_steps
            update_progress_bar(current_progress, cur_ratio * 100)
            update_progress_bar(total_progress, global_ratio * 100)

    metric = np.minimum(ratios, inliers / 100)
    
    np.seterr(divide = 'ignore')
    scores = -np.log(metric + metric.T) + step_penalty
    np.seterr(divide = 'warn') 
    
    cache = dict(N=N, scores=scores, matrices=matrices)

    if with_gui:
        root.destroy()
    return cache
    
def stitch_images(images, cache, main_image=0, weight_threshold=1.0, target_size = [1920, 1080], with_gui=True):

    if with_gui:
        root = tk.Tk()
        root.title("Panorama Stitching Progress")

        step_label = tk.Label(root, text="Initializing...")
        step_label.pack(padx=10,pady=10)
        
        current_progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
        current_progress.pack(padx=10,pady=10)
        
        total_progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
        total_progress.pack(padx=10,pady=10)
    else:
        current_progress = None
        total_progress = None

    N = len(images)
    total_steps = 2

    def update_step_label(text):
        if with_gui:
            step_label.config(text=text)
            step_label.update()
            
    N2 = cache['N']
    if N != N2:
        raise ValueError(f"Number of images in cache ({N2}) does not match number of images provided ({N})")
    scores = cache['scores']
    matrices = cache['matrices'] 


    # Step 1: Computing homographies
    update_step_label("Computing homographies")
    homographies = []
    min_x, min_y = np.inf, np.inf
    max_x, max_y = -np.inf, -np.inf
    for i in range(N):
        src = i
        H = np.eye(3)
        if src != main_image:
            path, weight = lightest_path(scores, main_image, src)
            if weight > weight_threshold:
                H = None
            else:
                for j in range(1, len(path)):
                    H_rel = matrices[path[j]][path[j - 1]]
                    H = H @ H_rel

        homographies.append(H)
        
        if H is not None:
            corners = get_image_corners(images[i])
            transformed_corners = transform_corners(corners, H)
            min_x = min(min_x, np.min(transformed_corners[0]))
            max_x = max(max_x, np.max(transformed_corners[0]))
            min_y = min(min_y, np.min(transformed_corners[1]))
            max_y = max(max_y, np.max(transformed_corners[1]))
        
        cur_ratio = i / N
        global_ratio = (2+cur_ratio)/total_steps
        update_progress_bar(total_progress, global_ratio * 100)
        update_progress_bar(current_progress, cur_ratio * 100)

    
    offset_x = -min_x
    offset_y = -min_y
    H_translation = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ])
    scale_x = target_size[0] / (max_x - min_x)
    scale_y = target_size[1] / (max_y - min_y)
    scale = min(scale_x, scale_y)
    H_scale = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ])
    panorama_width = int(np.ceil(max_x - min_x) * scale)
    panorama_height = int(np.ceil(max_y - min_y) * scale)

    final_transforms = []
    aligned_images = []

    # Step 2: Warping images
    scales = []
    update_step_label("Warping images")
    for i in range(N):
        if homographies[i] is None:
            final_transforms.append(None)
            continue
        scale = 1/np.linalg.norm(homographies[i][:2, :2])
        scales.append(scale)
        H_final = H_scale @ H_translation @ homographies[i]
        im = cv2.warpPerspective(images[i], H_final, (panorama_width, panorama_height))
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        final_transforms.append(H_final)
        aligned_images.append(im)
        
        cur_ratio = i / N
        global_ratio = (0+cur_ratio)/total_steps
        update_progress_bar(current_progress, cur_ratio * 100)
        update_progress_bar(total_progress, global_ratio * 100)

    sum_image = aligned_images[0].copy()
    
    # Step 3: Stitching images
    update_step_label("Stitching images")
    order = np.argsort(scales)
    for i, ind in enumerate(order):
        image = aligned_images[ind]
        mask = ((image > 0) * 255).astype(np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8))[:, :, 0]
        sum_image[mask > 0] = image[mask > 0]
        cur_ratio = i / len(aligned_images)
        global_ratio = (1+cur_ratio)/total_steps
        update_progress_bar(current_progress, cur_ratio * 100)
        update_progress_bar(total_progress, global_ratio * 100)

    if with_gui:
        root.destroy()
    return sum_image, final_transforms


# Example usage:
if __name__ == "__main__":
    image_files = sorted(glob('output/simulated2/*'))
    images = [cv2.imread(image_file) for image_file in image_files]
    cache = create_cache(images)
    panorama, transforms = stitch_images(images, cache)
    plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    plt.show()
