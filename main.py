# credit goes to Maksym Aslianskyi, ChatGPT, GitHub Copilot and StackOverflow

import os
import cv2
import numpy as np
import types
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from cpw import SmoothWarp
# from nn_warp import SmoothWarp

screen_size = (1080, 700)
window_size = screen_size
SCREEN_FACTOR = 0.7
DEBUG = True


import cv2
import numpy as np

def track_frames(rgb0, rgb1):
    detection_params = dict(maxCorners=1000,
                            qualityLevel=0.0001,
                            minDistance=30,
                            blockSize=11)

    tracking_params = dict(winSize=(8, 8),
                           maxLevel=4,
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.03),
                           minEigThreshold=5e-5)

    match_params = dict(bidirectional_thresh=2.0, nfeatures=2000, fastThreshold=15)

    ransac_params = dict(ransacReprojThreshold=2,
                         maxIters=2000,
                         confidence=0.999,
                         refineIters=10)
    
    im0 = cv2.cvtColor(rgb0, cv2.COLOR_RGB2GRAY)
    im1 = cv2.cvtColor(rgb1, cv2.COLOR_RGB2GRAY)

    p0 = cv2.goodFeaturesToTrack(im0, mask=None, **detection_params)

    p1, st1, err1 = cv2.calcOpticalFlowPyrLK(im0, im1, p0, None, **tracking_params)
    
    if match_params['bidirectional_thresh'] > 0:
        p2, st2, err2 = cv2.calcOpticalFlowPyrLK(im1, im0, p1, None, **tracking_params)
        proj_err = np.linalg.norm(np.squeeze(p2 - p0), axis=1)
        st = np.squeeze(st1 * st2) * (proj_err < match_params['bidirectional_thresh']).astype(np.uint8)
    else:
        st = np.squeeze(st1)

    p0 = np.squeeze(p0)[st == 1]
    p1 = np.squeeze(p1)[st == 1]
    
    H, mask = cv2.findHomography(p0, p1, cv2.RANSAC, 150.0)
    mask = mask.flatten().astype(bool)
    return p0[mask], p1[mask]

    return p0, p1



def match_sift_features(im1, im2, use_knn=False):
    sift = cv2.SIFT_create(nfeatures=1000)
    keypoints1, descriptors1 = sift.detectAndCompute(im1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(im2, None)
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
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 50.0)
        mask = mask.flatten().astype(bool)
        return src_pts[mask], dst_pts[mask]
    else:
        return None, None

class Anchor:
    def __init__(self, x, y, original=False):
        self.pos0 = (x, y)
        self.pos = (x, y)
        self.original = original
        self.moved = False
    
    def move(self, x, y):
        self.pos = (x, y)
        self.moved = True
    
    def reset(self):  
        self.moved = False
        self.pos0 = self.pos
        
    def plot(self, canvas, M_global, last=False):
        if self.original:
            last = False
        r = 4
        pos_t = apply_homography(M_global, self.pos).astype(int)
        pos0_t = apply_homography(M_global, self.pos0).astype(int)
        if self.original:
            canvas.create_oval(pos_t[0] - r, pos_t[1] - r, pos_t[0] + r, pos_t[1] + r, fill='green')
        else:
            canvas.create_line(pos0_t[0], pos0_t[1], pos_t[0], pos_t[1], fill='green', width=3)
            r0 = 3
            canvas.create_oval(pos0_t[0] - r0, pos0_t[1] - r0, pos0_t[0] + r0, pos0_t[1] + r0, fill='yellow')
            color = 'red' if self.moved else 'blue'
            if last:
                color = 'violet'
                r = 5
            canvas.create_oval(pos_t[0] - r, pos_t[1] - r, pos_t[0] + r, pos_t[1] + r, fill=color)



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

def calc_transform(shape, scale, rotation, x_offset, y_offset):
    cols, rows = shape[:2]
    M1 = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, scale)
    M2 = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
    M1 = np.vstack([M1, [0, 0, 1]])
    M2 = np.vstack([M2, [0, 0, 1]])
    M = np.dot(M2,M1)
    return M

def calc_homography(anchors):
    if len(anchors) < 4:
        return np.eye(3)
    pts0 = np.array([anchor.pos0 for anchor in anchors], dtype=np.float32)
    pts1 = np.array([anchor.pos for anchor in anchors], dtype=np.float32)
    H, _ = cv2.findHomography(pts0, pts1)
    return H

def decompose_homography(H, image_size, scale_ratio):
    h, w = image_size[:2]
    corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    M, _ = cv2.estimateAffinePartial2D(corners.reshape(-1, 2), transformed_corners)

    scale = np.sqrt(np.linalg.det(M[:2,:2]))
    rotation = -np.rad2deg(np.arctan2(M[1, 0], M[0, 0]))

    M2 = calc_transform((w*scale_ratio,h*scale_ratio),scale,rotation,0,0)
    M1 = np.vstack([M, [0, 0, 1]])
    M3 = M1@np.linalg.inv(M2)
    translation = M3[:2, 2]

    H1 = calc_transform((w*scale_ratio,h*scale_ratio), scale, rotation, translation[0], translation[1])
    H2 = H@np.linalg.inv(H1)
    
    return translation, rotation, scale, H2

def apply_homography(H, point):
    pt = np.array([point[0], point[1], 1])
    pt = H @ pt
    pt = pt[:2]/pt[2]
    return pt

def draw_grid(image, grid_spacing, color=(192, 192, 192), thickness=1):
    height, width = image.shape[:2]
    
    for x in range(0, width, grid_spacing):
        cv2.line(image, (x, 0), (x, height), color, thickness)
    
    for y in range(0, height, grid_spacing):
        cv2.line(image, (0, y), (width, y), color, thickness)
    
    return image

class ImagePair:
    def __init__(self, img1_path, img2_path, M_anchors = np.eye(3)):
        self.img1_path = img1_path
        self.img2_path = img2_path
        self.valid = True
        self.scale = 1.0
        self.rotation = 0
        self.x_offset = 0
        self.y_offset = 0
        self.anchors = []
        self.scale_ratio = 1.0
        self.M_anchors = M_anchors
        self.M_original = np.eye(3)
        self.error_message = ''
        self.img1 = None
        self.img2 = None

        self.state_stack = []
        self.current_state_index = -1

        try:
            self.img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
        except:
            pass
        try:
            self.img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)
        except:
            pass

        if self.img1 is None or self.img2 is None:
            bad_images = []
            if self.img1 is None:
                bad_images.append(img1_path)
            if self.img2 is None:
                bad_images.append(img2_path)

            self.error_message = f"Could not load image(s): {', '.join(bad_images)}"
            self.valid = False
            return

        self.scale_ratio = 1.0 #min(window_size[0] / self.img1.shape[1], window_size[1] / self.img1.shape[0])
        # self.M_original = np.diag([self.scale_ratio, self.scale_ratio, 1])

        self.initialize_from_homography(M_anchors)
        self.save_state()

    def save_state(self):
        current_state = (self.scale, self.rotation, self.x_offset, self.y_offset, [(a.pos, a.original) for a in self.anchors], self.M_anchors)
        if len(self.state_stack) > 0:
            previous_state = self.state_stack[self.current_state_index]
            identical = (np.linalg.norm(np.array(previous_state[:4]) - np.array(current_state[:4]) ) < 1e-3) and (np.linalg.norm(previous_state[5]-current_state[5]) < 1e-3)
            if identical:
                return
        self.state_stack = self.state_stack[:self.current_state_index + 1]
        self.state_stack.append(current_state)
        self.current_state_index += 1

    def undo(self):
        if self.current_state_index > 0:
            self.current_state_index -= 1
            self.load_state()

    def redo(self):
        if self.current_state_index < len(self.state_stack) - 1:
            self.current_state_index += 1
            self.load_state()
    
    def is_identity(self):
        return self.x_offset == 0 and self.y_offset == 0 and self.scale == 1.0 and self.rotation == 0 \
                and np.linalg.norm(self.M_anchors-np.eye(3)) < 1e-6

    def load_state(self):
        if 0 <= self.current_state_index < len(self.state_stack):
            state = self.state_stack[self.current_state_index]
            self.scale, self.rotation, self.x_offset, self.y_offset, anchor_states, self.M_anchors = state
            self.anchors = [Anchor(pos[0], pos[1], original) for pos, original in anchor_states]

    def reset_anchors(self):
        if self.img2 is not None:
            m = 30
            w = self.img2.shape[1]
            h = self.img2.shape[0]
            anchors_pos = [(m, m), (m, h - m), (w - m, m), (w - m, h - m)]
            M = calc_transform((self.img2.shape[1] * self.scale_ratio, self.img2.shape[0] * self.scale_ratio), self.scale, self.rotation, self.x_offset, self.y_offset)
            anchors_pos = [apply_homography(self.M_anchors @ M @ self.M_original, pos) for pos in anchors_pos]
            self.anchors = [Anchor(np.clip(x, m, window_size[0]-m), np.clip(y, m, window_size[1]-m), original=True) for x, y in anchors_pos]
        else:
            m = 100
            w = window_size[0]
            h = window_size[1]
            anchors_pos = [(m, m), (m, h - m), (w - m, m), (w - m, h - m)]
            self.anchors = [Anchor(x, y, original=True) for x, y in anchors_pos]
    
    def initialize_from_homography(self, H):
        translation, rotation, scale, H2 = decompose_homography(H, self.img2.shape, self.scale_ratio)

        self.scale = scale
        self.rotation = rotation
        self.x_offset = translation[0]
        self.y_offset = translation[1]
        self.M_anchors = H2
        self.reset_anchors()
        return 

    def run_matching(self):
        # src, dst = match_sift_features(self.img2, self.img1)
        src, dst = track_frames(self.img2, self.img1)
        self.anchors = []
        for s, d in zip(src, dst):
            anchor = Anchor(d[0], d[1])
            anchor.pos0 = (s[0], s[1])
            self.anchors.append(anchor)
        return True

    def render(self, app):
        if not self.valid:
            return np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)

        M = calc_transform([self.img2.shape[1] * self.scale_ratio, self.img2.shape[0] * self.scale_ratio], self.scale, self.rotation, self.x_offset, self.y_offset)
        src = np.array([anchor.pos for anchor in self.anchors])
        dst = np.array([anchor.pos0 for anchor in self.anchors])
        
        # H = calc_homography(self.anchors)
        # M_global = app.M_global()

        im1 = self.img1 #cv2.warpPerspective(self.img1, M_global @ self.M_original, window_size)
        # im2 = cv2.warpPerspective(self.img2, M_global @ H @ self.M_anchors @ M @ self.M_original, window_size)
        
        sw = SmoothWarp(self.img2.shape,20,20,0.0001,100)
        sw.solve(src, dst)
        im2 = sw.warp(self.img2)
        
        # sw = SmoothWarp(self.img2.shape)
        # im2 = sw.warp(self.img2, src, dst)

        if app.toggle:
            im1, im2 = im2, im1

        if app.contrast_mode:
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
            blend_image = np.stack([im1, im2, im1], axis=-1)
        else:
            blend_image = cv2.addWeighted(im1, 1 - app.alpha, im2, app.alpha, 0)

        if app.viewport_mode:
            blend_image = (blend_image * 0.7).astype(np.uint8)

        return blend_image

    def push_anchor(self, pt):
        min_dist = 200
        closest_anchor = None
        for anchor in self.anchors:
            dist = np.sqrt((anchor.pos[0] - pt[0]) ** 2 + (anchor.pos[1] - pt[1]) ** 2)
            if min_dist < 0 or dist < min_dist:
                min_dist = dist
                closest_anchor = anchor
        new_anchor = Anchor(pt[0], pt[1])
        if closest_anchor:
            self.anchors.remove(closest_anchor)
        self.anchors.append(new_anchor)
        self.save_state()

    def relative_transform(self):
        M = calc_transform([self.img2.shape[1] * self.scale_ratio, self.img2.shape[0] * self.scale_ratio], self.scale, self.rotation, self.x_offset, self.y_offset)
        H = calc_homography(self.anchors)
        T = H @ self.M_anchors @ M
        return T

    def __str__(self):
        T = self.relative_transform()
        return f"{self.img1_path}, {self.img2_path}, {','.join([str(x) for x in T.flatten().tolist()])}"
       
class DebugInfo:
    def __init__(self, root, app):
        self.debug_window = None
        self.debug_frame = None
        self.label_widgets = []
        self.root = root
        self.app = app

    def show_debug_info(self):
        if self.debug_window is None:
            self.debug_window = tk.Toplevel()
            self.debug_window.title("Debug Information")
        
        if self.debug_frame is None:
            self.debug_frame = tk.Frame(self.debug_window, bg='grey')
            self.debug_frame.pack(side="left", fill="y")
            tk.Label(self.debug_frame, text="Debug Information", font=('Helvetica', 16, 'bold'), bg='grey').pack(side="top", pady=10)

        # Remove existing labels
        for label in self.label_widgets:
            label.destroy()
        self.label_widgets.clear()
        
        # Display all global variables
        app_vars = vars(self.app)
        image_vars = vars(self.app.images)
        for var_name, var_value in {**app_vars, **image_vars}.items():
            if isinstance(var_value,np.ndarray) or var_value == self.app.images:
                continue
            try:
                label = tk.Label(self.debug_frame, text=f"{var_name}: {var_value}", bg='grey')
                label.pack(side="top", anchor="w", padx=10, pady=5)
                self.label_widgets.append(label)
            except:
                self.debug_window = None
                self.debug_frame = None
                self.app.toggle_debug_mode()
                return

class ButtonPanel:
    def __init__(self, root, app):
        self.root = root
        self.app = app
        self.frame = tk.Frame(self.root, bg='grey')
        self.frame.pack(side="left", fill="y")
        self.create_widgets()
        self.current_index = 0
        self.image_pairs = []

    def create_widgets(self):
        self.alpha_slider = tk.Scale(self.frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Alpha Blending", command=self.app.update_alpha,bg='white')
        self.alpha_slider.set(self.app.alpha)
        
        self.rotation_slider = tk.Scale(self.frame, from_=-180, to=180, resolution=0.01, orient=tk.HORIZONTAL, label="Rotation", command=self.app.update_rotation,bg='white')
        self.rotation_slider.set(self.app.images.rotation)

        self.scale_slider = tk.Scale(self.frame, from_=-1, to=1, resolution=0.001, orient=tk.HORIZONTAL, label="Scale Factor", command=lambda x: self.app.update_scale(np.power(10,float(x))),bg='white')
        self.scale_slider.set(np.log10(self.app.images.scale))

        self.homography_button = tk.Button(self.frame, text="Homography Mode: OFF", command=self.app.toggle_homography_mode,bg='white')
        self.homography_reset_button = tk.Button(self.frame, text="Reset Homography", command=self.app.reset_homography,bg='white')
        self.homography_calculate_button = tk.Button(self.frame, text="Automatic Homography", command=self.app.run_matching,bg='white')
        self.viewport_button = tk.Button(self.frame, text="Viewport Mode: OFF", command=self.app.toggle_viewport_mode,bg='white')
        self.contrast_button = tk.Button(self.frame, text="Contrast Mode: OFF", command=self.app.toggle_contrast_mode,bg='white')
        self.switch_button = tk.Button(self.frame, text="Switch Images: OFF", command=self.app.toggle_switch,bg='white')
        self.grid_button = tk.Button(self.frame, text="Show Grid: OFF", command=self.app.toggle_grid,bg='white')
        self.debug_button = tk.Button(self.frame, text="Debug Mode: OFF", command=self.app.toggle_debug_mode,bg='white')
        self.help_button = tk.Button(self.frame, text="Help: OFF", command=self.app.toggle_help_mode,bg='white')
        self.help_frame = tk.Frame(self.frame)
        self.help_text_box = tk.Text(self.help_frame, height=7, width=30, wrap="word",bg='white')
        self.help_text_box.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.help_scrollbar = tk.Scrollbar(self.help_frame, command=self.help_text_box.yview)
        self.help_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.help_text_box['yscrollcommand'] = self.help_scrollbar.set
        
        self.upload_button = tk.Button(self.frame, text="Upload Images", command=self.app.upload_images,bg='white')
        self.load_csv_button = tk.Button(self.frame, text="Load CSV", command=self.load_csv,bg='white')
        self.save_results_button = tk.Button(self.frame, text="Save Results", command=self.save_results,bg='white')
        self.image_list_frame = tk.Frame(self.frame,bg='white')
        self.image_list_scrollbar = tk.Scrollbar(self.image_list_frame)
        self.image_list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_listbox = tk.Listbox(self.image_list_frame, yscrollcommand=self.image_list_scrollbar.set,bg='white')
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_listbox.bind('<<ListboxSelect>>', self.select_image)
        self.image_list_scrollbar.config(command=self.image_listbox.yview)

        buttons = [
            self.alpha_slider,
            self.rotation_slider,
            self.scale_slider,
            self.homography_button,
            self.homography_reset_button,
            self.homography_calculate_button,
            self.viewport_button,
            self.contrast_button,
            self.switch_button,
            self.grid_button,
            self.debug_button,
            self.help_button,
            self.help_frame,
            self.upload_button,
            self.load_csv_button,
            self.save_results_button,
            self.image_list_frame
        ]
        for i, b in enumerate(buttons):
            b.grid(row=i, column=0, sticky='ew')

        self.frame.grid_columnconfigure(0, weight=1)
        
    def load_csv(self):
        self.app.clear_messages()
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not os.path.exists(file_path):
            self.app.display_message(f"ERROR: {file_path} does not exist")
            return
        new_pairs = []
        with open(file_path, 'r') as file:
            for line in file:
                line_parsed = line.strip().split(',')
                if len(line_parsed) != 2 and len(line_parsed) != 11:
                    self.app.display_message(f"ERROR: {line} is not a valid image pair")
                    continue
                image_path1, image_path2 = line_parsed[0], line_parsed[1]
                image_path1, image_path2 = image_path1.replace(' ',''), image_path2.replace(' ','')
                H = np.eye(3)
                if len(line_parsed) == 11:
                    H = np.array([float(line_parsed[i]) for i in range(2,11)]).reshape(3,3)
                new_pair = ImagePair(image_path1, image_path2, H)
                if new_pair.valid:
                    new_pairs.append(new_pair)
                else:
                    self.app.display_message(f"ERROR: {new_pair.error_message}")
        if len(new_pairs) > 0:
            self.image_listbox.delete(0,tk.END)
            self.app.display_message(f"Loaded {len(new_pairs)} image pairs from {file_path}")
            for pair in new_pairs:
                self.image_listbox.insert(tk.END, pair.img2_path.split('/')[-1])
            self.image_pairs = new_pairs
        else:
            self.app.display_message("ERROR: No valid image pairs found in CSV")
        self.current_index = -1
        #if not self.app.images.valid and len(self.image_pairs) > 0:
        self.next_image()
        self.app.sync_sliders()

    def save_results(self):
        self.app.clear_messages()
        output_folder = filedialog.askdirectory(title='Select output folder...')
        if not output_folder:
            self.app.display_message("ERROR: Please select an output folder")
            return
        output_file = f"{output_folder}/results.csv"
        self.app.display_message(f"Saved results to {output_file}")
        with open(output_file, 'w') as file:
            if len(self.image_pairs) == 0:
                file.write(f"{self.app.images}\n")
                image = self.app.images.render(self.app)
                cv2.imwrite(f"{output_folder}/result_1.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            else:
                for i, image_pair in enumerate(self.image_pairs):
                    file.write(f"{image_pair}\n")
                    image = image_pair.render(self.app)
                    cv2.imwrite(f"{output_folder}/result_{i+1}.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    
            
    
    def step_image(self, sign):
        if len(self.image_pairs) == 0:
            self.app.clear_messages()
            self.app.display_message("ERROR: No image pairs loaded")
            return
        prev = self.app.images
        self.current_index = (self.current_index + sign) % len(self.image_pairs)
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(self.current_index)
        self.image_listbox.see(self.current_index)
        self.app.images = self.image_pairs[self.current_index]
        cur = self.app.images
        if cur.is_identity():
            cur.scale, cur.rotation, cur.x_offset, cur.y_offset, cur.M_anchors = prev.scale, prev.rotation, prev.x_offset, prev.y_offset, prev.M_anchors
            cur.reset_anchors()
        self.app.render()
        self.app.sync_sliders()
        
    def next_image(self):
        self.step_image(+1)
        
    def previous_image(self):
        self.step_image(-1)

    def select_image(self, event):
        if len(self.image_pairs) == 0:
            self.app.clear_messages()
            self.app.display_message("ERROR: No image pairs loaded")
            return
        self.current_index = self.image_listbox.curselection()[0]
        self.app.images = self.image_pairs[self.current_index]
        self.app.render()
        self.app.sync_sliders()

class ImageAlignerApp:
    def __init__(self, root, image_pair=None):
        if not 'root' in dir(self):
            self.root = root
        
        if image_pair is None:
            image_pair = ImagePair("", "")

        self.images = image_pair
        self.alpha = 0.75
        
        self.global_x_offset = 0
        self.global_y_offset = 0
        self.global_scale = 1.0
        
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0

        self.viewport_mode = False
        self.toggle = False
        self.debug_mode = False
        self.help_mode = False
        self.contrast_mode = False
        self.homography_mode = False
        self.rotation_mode = False
        self.draw_grid = False

        if not 'debug_info' in dir(self):
            self.debug_info = DebugInfo(root, self)
            self.button_panel = ButtonPanel(root, self)
            self.canvas = tk.Canvas(self.root, width=window_size[0], height=window_size[1])
            self.canvas.pack()
            self.setup_bindings()
        
        if not image_pair.valid:
            self.display_message('ERROR: ' + image_pair.error_message)
        self.render()

    def setup_bindings(self):
        self.root.bind('<KeyPress>', self.on_key_press)
        self.root.bind('<KeyRelease>', self.on_key_release)
        self.root.bind('<MouseWheel>', self.on_mouse_wheel)
        self.root.bind("<Button-2>", self.on_middle_click)
        self.root.bind("<Button-3>", self.on_right_click)
        self.root.bind("<Button-4>", self.on_mouse_wheel)
        self.root.bind("<Button-5>", self.on_mouse_wheel)
        self.root.bind('<ButtonPress-1>', self.on_mouse_press)
        self.root.bind('<B1-Motion>', self.on_mouse_drag)
        self.root.bind('<ButtonRelease-1>', self.on_mouse_release)
        self.root.bind('<Control-z>', self.undo)
        self.root.bind('<Control-y>', self.redo)
        self.root.bind("<Delete>", self.on_delete)
        self.root.bind("<Escape>", self.exit)
    
    def upload_images(self):
        self.clear_messages()
        image_paths = filedialog.askopenfilenames(title='Select images to align...')
        if len(image_paths) != 2:
            self.display_message('ERROR: Please select exactly 2 images')
            return
        self.images = ImagePair(image_paths[0], image_paths[1]) 
        if not self.images.valid:
            self.display_message('ERROR: ' + self.images.error_message)
            return
        self.display_message('Images uploaded successfully')
        self.render()
        
    def M_global(self):
        return calc_transform(window_size, self.global_scale, 0, self.global_x_offset, self.global_y_offset)
    
    def render(self, update_state=True):
        if not self.dragging and update_state:
            self.images.save_state()
        rendered_image = self.images.render(self)
        if self.draw_grid:
            rendered_image = draw_grid(rendered_image,100)

        img_pil = Image.fromarray(rendered_image)
        self.tk_image = ImageTk.PhotoImage(img_pil)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        if self.homography_mode:
            M_global = self.M_global()
            for anchor in self.images.anchors[:-1]:
                anchor.plot(self.canvas, M_global)
        
            self.images.anchors[-1].plot(self.canvas, M_global, last=True)

        if self.debug_mode:
            self.debug_info.show_debug_info()

    # Event Handlers
    def update_alpha(self, val):
        self.alpha = float(val)
        self.button_panel.alpha_slider.set(self.alpha)
        self.render()
    
    def update_rotation(self, val):
        self.move_anchors()
        val = float(val)
        self.images.rotation = ((val + 180)%360 -180) if val != 180 else 180
        self.button_panel.rotation_slider.set(self.images.rotation)
        self.render()
        
    def update_scale(self, val):
        self.move_anchors()
        self.images.scale = np.clip(float(val),0.1,10)
        self.button_panel.scale_slider.set(np.log10(self.images.scale))
        self.render()
    
    def sync_sliders(self):
        self.button_panel.rotation_slider.set(self.images.rotation)
        self.button_panel.scale_slider.set(np.log10(self.images.scale))
        
    def toggle_help_mode(self):
        self.help_mode = not self.help_mode
        self.button_panel.help_button.config(text="Help:  ON" if self.help_mode else "Help: OFF", bg=('grey' if self.help_mode else 'white'))
        descriptions = [('r', "Rotate by 90 degrees"),
                        ('+', "Zoom in 10%"),
                        ('-', "Zoom out 10%"),
                        ('d', "Debug mode"),
                        ('c', "Contrast mode"),
                        ('t/right click', "Toggle images"),
                        ('ctrl', "Homography mode"),
                        ('a', "Automatic homography"),
                        ('o', "Reset homography"),
                        ('space', "Change field of view"),
                        ('mouse wheel', "Zoom in/out"),
                        ('middle click', "Toggle rotation mode"),
                        ('<-/->', "Previous/Next image")]
        self.button_panel.help_text_box.delete('1.0', tk.END)
        if self.help_mode:
            for key, description in descriptions:
                self.button_panel.help_text_box.insert(tk.END, f"{key}: {description}\n")
            self.button_panel.help_text_box.config(state=tk.NORMAL)
    
    def toggle_homography_mode(self):
        self.homography_mode = not self.homography_mode
        if len(self.images.anchors) == 0:
            self.images.reset_anchors()
        self.button_panel.homography_button.config(text="Homography Mode:  ON" if self.homography_mode else "Homography Mode: OFF", bg=('grey' if self.homography_mode else 'white'))
        self.render()

    def toggle_contrast_mode(self):
        self.contrast_mode = not self.contrast_mode
        self.button_panel.contrast_button.config(text="Contrast Mode:  ON" if self.contrast_mode else "Contrast Mode: OFF", bg=('grey' if self.contrast_mode else 'white'))
        self.render()

    def toggle_switch(self):
        self.toggle = not self.toggle
        self.button_panel.switch_button.config(text="Switch Images:  ON" if self.toggle else "Switch Images: OFF", bg=('grey' if self.toggle else 'white'))
        self.render()

    def toggle_grid(self):
        self.draw_grid = not self.draw_grid
        self.button_panel.grid_button.config(text="Show Grid:  ON" if self.draw_grid else "Show Grid: OFF", bg=('grey' if self.draw_grid else 'white'))
        self.render()

    def reset_homography(self):
        self.images.M_anchors = np.eye(3)
        self.images.reset_anchors()
        self.render()

    def reset(self):
        self.update_scale(1.0)
        self.update_rotation(0)
        self.images.x_offset = 0
        self.images.y_offset = 0
        self.images.M_anchors = np.eye(3)
        self.images.reset_anchors()
        self.render()
        
    def run_matching(self):
        ret = self.images.run_matching()
        self.sync_sliders()
        if not ret:
            self.display_message('ERROR: Could not calculate homography')
        self.render()

    def toggle_viewport_mode(self):
        self.viewport_mode = not self.viewport_mode
        self.button_panel.viewport_button.config(text="Viewport Mode:  ON" if self.viewport_mode else "Viewport Mode: OFF", bg=('grey' if self.viewport_mode else 'white'))
        self.render()
        
    def toggle_debug_mode(self):
        self.debug_mode = not self.debug_mode
        self.button_panel.debug_button.config(text="Debug Mode:  ON" if self.debug_mode else "Debug Mode: OFF", bg=('grey' if self.debug_mode else 'white'))
        self.render()
    
    def undo(self, event):
        self.images.undo()
        self.render(False)

    def redo(self, event):
        self.images.redo()
        self.render(False)
        
    def on_delete(self, event):
        if self.homography_mode and not self.viewport_mode:
            if len(self.images.anchors) > 0 and not self.images.anchors[-1].original:
                self.images.anchors.pop()
                self.render()
        else:
            self.reset()
    
    def on_key_press(self, event):
        if event.char == 'r':
            self.update_rotation(self.images.rotation + 90)
        elif event.char == '-':
            self.global_scale *= 0.9
        elif event.char == '=':
            self.global_scale *= 1.1
        elif event.char == 'd':
            self.toggle_debug_mode()
        elif event.char == 'c':
            self.toggle_contrast_mode()
        elif event.char == 't':
            self.toggle_switch()
        elif event.char == 'h':
            self.toggle_homography_mode()
        elif event.char == 'o':
            self.reset_homography()
        elif event.char == 'q':
            self.reset()
        elif event.char == 'g':
            self.toggle_grid()
        elif event.char == 'a':
            self.run_matching()
        elif event.char == ' ':
            self.toggle_viewport_mode()
        elif event.keysym == 'Right':
            self.button_panel.next_image()
        elif event.keysym == 'Left':
            self.button_panel.previous_image()
        self.render()

    def on_key_release(self, event):
        self.render()
        
    def check_relevancy(self, event):
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        if widget == self.canvas:
            return True
        return False
    
    def on_middle_click(self, event):
        self.rotation_mode = not self.rotation_mode

    def on_right_click(self, event):
        self.toggle_switch()

    def on_mouse_wheel(self, event):
        if not self.check_relevancy(event):
            return
        step_size = 120
        step = event.delta / step_size
        if event.num == 4:
            step = 3
        elif event.num == 5:
            step = -3
        if self.viewport_mode:
            self.global_scale = max(self.global_scale * (1.05 ** step), 0.1)
        else:
            if self.rotation_mode:
                self.update_rotation(self.images.rotation - step)
            elif not self.homography_mode:
                self.update_scale(self.images.scale * (1.02 ** step))
        self.render()

    def on_mouse_press(self, event):
        if not self.check_relevancy(event):
            return
        self.dragging = True
        pt0 = [event.x, event.y]
        pt = apply_homography(np.linalg.inv(self.M_global()), pt0)
        if self.homography_mode and not self.viewport_mode:
            self.images.push_anchor(pt)
            self.render()
            return

        self.move_anchors()
        if self.viewport_mode:
            self.drag_start_x = pt0[0]
            self.drag_start_y = pt0[1]
        else:
            self.drag_start_x = pt[0]
            self.drag_start_y = pt[1]

    def on_mouse_drag(self, event):
        if not self.dragging:
            return
        
        pt0 = [event.x, event.y]
        pt = apply_homography(np.linalg.inv(self.M_global()), pt0)
        
        if self.homography_mode and not self.viewport_mode:
            self.images.anchors[-1].move(*pt)
            self.render()
            return

        self.move_anchors()
        if self.viewport_mode:
            self.global_x_offset += pt0[0] - self.drag_start_x
            self.global_y_offset += pt0[1] - self.drag_start_y
            self.drag_start_x = pt0[0]
            self.drag_start_y = pt0[1]
        else:
            self.images.x_offset += pt[0] - self.drag_start_x
            self.images.y_offset += pt[1] - self.drag_start_y
            self.drag_start_x = pt[0]
            self.drag_start_y = pt[1]
        self.render()

    def on_mouse_release(self, event):
        self.dragging = False
        # if self.homography_mode and not self.viewport_mode:
            # self.images.M_anchors = calc_homography(self.images.anchors) @ self.images.M_anchors
            # for anchor in self.images.anchors:
            #     anchor.reset()
        self.render()
        return
        
    def move_anchors(self):
        self.images.M_anchors = calc_homography(self.images.anchors) @ self.images.M_anchors
        self.images.reset_anchors()

    def clear_messages(self):
        self.button_panel.help_text_box.delete('1.0', tk.END)
        self.button_panel.help_text_box.config(state=tk.NORMAL)
        
    def display_message(self, message):
        if self.help_mode:
            self.toggle_help_mode()
        self.button_panel.help_text_box.insert(tk.END, message + '\n')
        self.button_panel.help_text_box.config(state=tk.NORMAL)

    def exit(self, event=None):
        self.root.quit()
        self.root.destroy()
        

def main():
    root = tk.Tk()
    global window_size, screen_size
    screen_size = (int(root.winfo_screenwidth()*SCREEN_FACTOR), int(root.winfo_screenheight()*SCREEN_FACTOR))
    window_size = (int(screen_size[0]*0.8), screen_size[1])
    root.title("TagIm Aligning App")
    root.geometry(f"{screen_size[0]}x{screen_size[1]}")
    photo = ImageTk.PhotoImage(Image.open('resources/logo.jpg'))
    root.wm_iconphoto(False, photo)

    image_pair = ImagePair("input/test1.png", "input/test2.png")
    
    app = ImageAlignerApp(root, image_pair if DEBUG else None)

    root.mainloop()

if __name__ == "__main__":
    main()
