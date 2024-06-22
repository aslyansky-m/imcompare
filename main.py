# credit goes to Maksym Aslianskyi, ChatGPT, GitHub Copilot and StackOverflow

# data is here: https://ufile.io/8hcsrlo1

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import font
from PIL import Image, ImageTk
import pandas as pd
import time
from pathlib import Path
from common import *
from enum import Enum
from panorama import sift_matching_with_homography, create_cache, stitch_images

screen_size = (1080, 700)
window_size = screen_size
SCREEN_FACTOR = 0.9



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
        
    def plot(self, canvas, M_global):
        r = 4
        pos_t = apply_homography(M_global, self.pos).astype(int)
        pos0_t = apply_homography(M_global, self.pos0).astype(int)
        if self.original:
            canvas.create_oval(pos_t[0] - r, pos_t[1] - r, pos_t[0] + r, pos_t[1] + r, fill='green')
        else:
            r0 = 3
            canvas.create_oval(pos0_t[0] - r0, pos0_t[1] - r0, pos0_t[0] + r0, pos0_t[1] + r0, fill='yellow')
            color = 'red' if self.moved else 'blue'
            canvas.create_oval(pos_t[0] - r, pos_t[1] - r, pos_t[0] + r, pos_t[1] + r, fill=color)


def calc_transform(shape, scale, rotation, x_offset, y_offset):
    w, h = shape[:2]
    M1 = cv2.getRotationMatrix2D((w/2, h/2), rotation, scale)
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
    if H is None:
        return np.eye(3)
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

def draw_grid(image, H, grid_spacing=100, color=(192, 192, 192), thickness=1):
    height, width = image.shape[:2]
    
    new_grid_spacing_x = H[0, 0] * grid_spacing
    new_grid_spacing_y = H[1, 1] * grid_spacing
    
    max_diff = 5.0
    factor = np.ceil(np.log(grid_spacing/new_grid_spacing_x)/np.log(max_diff))
    new_grid_spacing_x *= max_diff**factor
    new_grid_spacing_y *= max_diff**factor
    new_start_x = H[0, 2]
    new_start_y = H[1, 2]
    new_start_x -= np.round(new_start_x / new_grid_spacing_x)*new_grid_spacing_x
    new_start_y -= np.round(new_start_y / new_grid_spacing_y)*new_grid_spacing_y

    x = new_start_x
    while x < width:
        cv2.line(image, (int(x), 0), (int(x), height), color, thickness)
        x += new_grid_spacing_x
    
    y = new_start_y
    while y < height:
        cv2.line(image, (0, int(y)), (width, int(y)), color, thickness)
        y += new_grid_spacing_y
    
    return image

def edge_detection(image, blur=5, low_threshold=80, high_threshold=150):
    blur = max(1, blur)
    blur = blur + 1 if blur % 2 == 0 else blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur, blur), 1.4)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    edges = (cv2.boxFilter(edges, -1, (3, 3)) > 0).astype(np.uint8) * 255
    output = (np.clip(edges + gray*0.6,0,255)).astype(np.uint8)
    return output

# enum for state
class ImageState(Enum):
    NOT_VALID = 0
    NOT_LOADED = 1
    INITIALIZED = 2
    LOADED = 3
    MOVED = 4
    MATCHED = 5
    LOCKED = 6
    PANORAMA = 7
    
    @staticmethod
    def to_color(state):
        colors = {
            ImageState.NOT_VALID: 'red',
            ImageState.NOT_LOADED: 'white',
            ImageState.INITIALIZED: 'Slategray1',
            ImageState.LOADED: 'ivory',
            ImageState.MOVED: 'SeaGreen2',
            ImageState.MATCHED: 'RosyBrown1',
            ImageState.LOCKED: 'saddle brown',
            ImageState.PANORAMA: 'violet'
        }
        return colors[state]
    
class ImageObject:
    def __init__(self, image_path, M_anchors = None):
        self.image_path = image_path
        self.state = ImageState.NOT_VALID
        self.scale = 1.0
        self.rotation = 0
        self.x_offset = 0
        self.y_offset = 0
        self.anchors = []
        self.scale_ratio = 1.0
        self.M_anchors = M_anchors
        self.M_original = np.eye(3)
        self.error_message = ''
        self.image = None

        self.state_stack = []
        self.current_state_index = -1
        
        self.derived_images = []
        self.is_panorama = False
        
        if not os.path.exists(image_path):
            self.error_message = f"Could not find image: {image_path}"
            return
        
        if M_anchors is None:
            self.state = ImageState.NOT_LOADED
        else:
            self.state = ImageState.INITIALIZED

        
    def get_image(self):
        if self.state == ImageState.NOT_VALID:
            return None
        if self.state == ImageState.NOT_LOADED or self.image is None:
            try:
                self.image = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)
                self.scale_ratio = min(window_size[0] / self.image.shape[1], window_size[1] / self.image.shape[0])
                self.M_original = np.diag([self.scale_ratio, self.scale_ratio, 1])
                if self.M_anchors is not None:
                    H = self.M_anchors @ np.linalg.inv(self.M_original)
                    self.initialize_from_homography(H)
                    self.state = ImageState.MOVED
                else:
                    self.M_anchors = np.eye(3)
                    self.state = ImageState.LOADED
                    
                self.save_state()
            except Exception as e:
                print(e)
                return None
            
        return self.image

    def save_state(self):
        if self.image is None:
            return
        current_state = (self.scale, self.rotation, self.x_offset, self.y_offset, [(a.pos, a.original) for a in self.anchors], self.M_anchors, self.state)
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
        if self.M_anchors is not None and np.linalg.norm(self.M_anchors-np.eye(3)) > 1e-6:
            return False
        return self.x_offset == 0 and self.y_offset == 0 and self.scale == 1.0 and self.rotation == 0

    def load_state(self):
        if 0 <= self.current_state_index < len(self.state_stack):
            state = self.state_stack[self.current_state_index]
            self.scale, self.rotation, self.x_offset, self.y_offset, anchor_states, self.M_anchors, self.state = state
            self.anchors = [Anchor(pos[0], pos[1], original) for pos, original in anchor_states]

    def reset_anchors(self):
        image = self.get_image()
        if image is not None:
            m = 30
            w = image.shape[1]
            h = image.shape[0]
            anchors_pos = [(m, m), (m, h - m), (w - m, m), (w - m, h - m)]
            M = calc_transform((image.shape[1] * self.scale_ratio, image.shape[0] * self.scale_ratio), self.scale, self.rotation, self.x_offset, self.y_offset)
            anchors_pos = [apply_homography(self.M_anchors @ M @ self.M_original, pos) for pos in anchors_pos]
            self.anchors = [Anchor(x, y, original=True) for x, y in anchors_pos]
        else:
            m = 100
            w = window_size[0]
            h = window_size[1]
            anchors_pos = [(m, m), (m, h - m), (w - m, m), (w - m, h - m)]
            self.anchors = [Anchor(x, y, original=True) for x, y in anchors_pos]
    
    def initialize_from_homography(self, H):
        translation, rotation, scale, H_residual = decompose_homography(H, self.image.shape, self.scale_ratio)

        self.scale = scale
        self.rotation = rotation
        self.x_offset = translation[0]
        self.y_offset = translation[1]
        self.M_anchors = H_residual
        self.reset_anchors()
        return 

    def render(self, M_global):
        if self.state == ImageState.NOT_VALID:
            return None
        image = self.get_image()
        if image is None:
            return None
        M = calc_transform([image.shape[1] * self.scale_ratio, image.shape[0] * self.scale_ratio], self.scale, self.rotation, self.x_offset, self.y_offset)
        H = calc_homography(self.anchors)
        
        result = cv2.warpPerspective(image, M_global @ H @ self.M_anchors @ M @ self.M_original, window_size)
        self.update_derived_images()
        return result
    
    def update_derived_images(self):
        H_prev = self.M_anchors @ calc_transform([self.image.shape[1] * self.scale_ratio, self.image.shape[0] * self.scale_ratio], self.scale, self.rotation, self.x_offset, self.y_offset) @ self.M_original
        for cur, H_rel in self.derived_images:
            if cur is None:
                continue
            if not (cur.state == ImageState.MATCHED or cur.state == ImageState.LOADED):
                continue
            cur_H = H_prev @ H_rel @ np.linalg.inv(cur.M_original)
            cur.initialize_from_homography(cur_H)
            cur.reset_anchors()
            cur.state = ImageState.MATCHED

    def push_anchor(self, pt):
        min_dist = -1
        closest_anchor = None
        for anchor in self.anchors:
            dist = np.linalg.norm(np.array(anchor.pos) - np.array(pt))
            if min_dist < 0 or dist < min_dist:
                min_dist = dist
                closest_anchor = anchor
        new_anchor = Anchor(pt[0], pt[1])
        self.anchors.remove(closest_anchor)
        self.anchors.append(new_anchor)
        self.save_state()

    def relative_transform(self):
        if self.image is None:
            return np.eye(3)
        M = calc_transform([self.image.shape[1] * self.scale_ratio, self.image.shape[0] * self.scale_ratio], self.scale, self.rotation, self.x_offset, self.y_offset)
        H = calc_homography(self.anchors)
        T = H @ self.M_anchors @ M @ self.M_original
        return T

    def __str__(self):
        T = self.relative_transform()
        return f"{self.image_path}, {','.join([str(x) for x in T.flatten().tolist()])}"
    
    @staticmethod
    def create_panorama(image, H = None, name="panorama"):
        o = ImageObject(name)
        o.is_panorama = True
        o.state = ImageState.PANORAMA
        o.image = image
        o.scale_ratio = min(window_size[0] / o.image.shape[1], window_size[1] / o.image.shape[0])
        o.M_original = np.diag([o.scale_ratio, o.scale_ratio, 1])
        if H is not None:
            cur_H = H @ np.linalg.inv(o.M_original)
            o.initialize_from_homography(cur_H)
        else:
            o.M_anchors = np.eye(3)
        return o
       
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

        for label in self.label_widgets:
            label.destroy()
        self.label_widgets.clear()
    
        app_vars = vars(self.app)
        image_vars = vars(self.app.image)
        for var_name, var_value in {**app_vars, **image_vars}.items():
            if isinstance(var_value,np.ndarray) or var_value == self.app.image:
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

class InfoPanel:
    def __init__(self, root, app):
        self.root = root
        self.app = app
        self.frame = tk.Frame(self.root, bg='grey')
        self.frame.pack(side="bottom", fill="y")
        self.create_widgets()

    def create_widgets(self):
        self.position_box = tk.Text(self.frame, height=1, width=40,bg='white')
        self.position_box.pack(side="left", padx=2, pady=2)
        self.scale_box = tk.Text(self.frame, height=1, width=30,bg='white')
        self.scale_box.pack(side="left", padx=2, pady=2)
        self.fps_box = tk.Text(self.frame, height=1, width=30,bg='white')
        self.fps_box.pack(side="left", padx=2, pady=2)
        
        self.update_position([0,0])
        self.update_scale(1.0)
        self.update_fps(0) 

    def update_position(self, position):
        self.position_box.config(state=tk.NORMAL)
        self.position_box.delete('1.0', tk.END)
        self.position_box.insert(tk.END, f"Coordinate: [{position[0]:.5f}°, {position[1]:.5f}°]")
        self.position_box.config(state=tk.DISABLED)
        
    def update_scale(self, scale):
        self.scale_box.config(state=tk.NORMAL)
        self.scale_box.delete('1.0', tk.END)
        self.scale_box.insert(tk.END, f"Scale: [1:{1/scale:.2f}]")
        self.scale_box.config(state=tk.DISABLED)
    
    def update_fps(self, fps):
        self.fps_box.config(state=tk.NORMAL)
        self.fps_box.delete('1.0', tk.END)
        self.fps_box.insert(tk.END, f"FPS: {fps:.2f}")
        self.fps_box.config(state=tk.DISABLED)
        
class ButtonPanel:
    def __init__(self, root, app):
        self.root = root
        self.app = app
        self.frame = tk.Frame(self.root, bg='grey')
        self.frame.pack(side="left", fill="y")
        self.create_widgets()
        self.current_index = 0
        self.images = []
        self.output_folder = None

    def create_widgets(self):
        self.alpha_slider = tk.Scale(self.frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Alpha Blending", command=self.app.update_alpha,bg='white')
        self.alpha_slider.set(self.app.alpha)
        
        self.rotation_slider = tk.Scale(self.frame, from_=-180, to=180, resolution=0.01, orient=tk.HORIZONTAL, label="Rotation", command=self.app.update_rotation,bg='white')
        self.rotation_slider.set(self.app.image.rotation)

        self.scale_slider = tk.Scale(self.frame, from_=-1, to=1, resolution=0.001, orient=tk.HORIZONTAL, label="Scale Factor", command=lambda x: self.app.update_scale(np.power(10,float(x))),bg='white')
        self.scale_slider.set(np.log10(self.app.image.scale))

        self.homography_button = tk.Button(self.frame, text="Homography Mode: OFF", command=self.app.toggle_homography_mode,bg='white')
        self.homography_reset_button = tk.Button(self.frame, text="Reset Homography", command=self.app.reset_homography,bg='white')
        self.panorama_button = tk.Button(self.frame, text="Create Panorama", command=self.app.create_panorama,bg='white')
        self.homography_calculate_button = tk.Button(self.frame, text="Calculate Homography", command=self.app.run_matching,bg='white')
        self.automatic_matching_button = tk.Button(self.frame, text="Automatic Matching: OFF", command=self.app.toggle_automatic_matching,bg='white')
        self.viewport_button = tk.Button(self.frame, text="Viewport Mode: OFF", command=self.app.toggle_viewport_mode,bg='white')
        self.contrast_button = tk.Button(self.frame, text="Contrast Mode: OFF", command=self.app.toggle_contrast_mode,bg='white')
        self.switch_button = tk.Button(self.frame, text="Switch Images: OFF", command=self.app.toggle_images,bg='white')
        self.grid_button = tk.Button(self.frame, text="Show Grid: OFF", command=self.app.toggle_grid,bg='white')
        self.borders_button = tk.Button(self.frame, text="Show Borders: OFF", command=self.app.toggle_borders,bg='white')
        #self.debug_button = tk.Button(self.frame, text="Debug Mode: OFF", command=self.app.toggle_debug_mode,bg='white')
        self.help_button = tk.Button(self.frame, text="Help: OFF", command=self.app.toggle_help_mode,bg='white')
        self.help_frame = tk.Frame(self.frame)
        self.help_text_box = tk.Text(self.help_frame, height=7, width=30, wrap="word",bg='white')
        self.help_text_box.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.help_scrollbar = tk.Scrollbar(self.help_frame, command=self.help_text_box.yview)
        self.help_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.help_text_box['yscrollcommand'] = self.help_scrollbar.set
        
        self.upload_button = tk.Button(self.frame, text="Upload Image", command=self.app.upload_image,bg='white')
        self.load_csv_button = tk.Button(self.frame, text="Load CSV", command=self.load_csv,bg='white')
        self.save_image_button = tk.Button(self.frame, text="Save Image", command=self.on_save_image,bg='white')
        self.save_results_button = tk.Button(self.frame, text="Save Results", command=self.save_results,bg='white')
        self.image_list_frame = tk.Frame(self.frame,bg='white')
        self.image_list_scrollbar = tk.Scrollbar(self.image_list_frame)
        self.image_list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_listbox = tk.Listbox(self.image_list_frame, yscrollcommand=self.image_list_scrollbar.set,bg='white')
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_selection)
        self.image_list_scrollbar.config(command=self.image_listbox.yview)

        buttons = [
            self.alpha_slider,
            self.rotation_slider,
            self.scale_slider,
            self.homography_button,
            self.homography_reset_button,
            self.panorama_button,
            self.homography_calculate_button,
            self.automatic_matching_button,
            self.viewport_button,
            self.contrast_button,
            self.switch_button,
            self.grid_button,
            self.borders_button,
            #self.debug_button,
            self.help_button,
            self.help_frame,
            self.upload_button,
            self.load_csv_button,
            self.save_image_button,
            self.save_results_button,
            self.image_list_frame
        ]
        for i, b in enumerate(buttons):
            b.grid(row=i, column=0, sticky='ew')

        self.frame.grid_columnconfigure(0, weight=1)
        
    def load_csv(self, file_path=None):
        self.app.clear_messages()
        if not file_path:
            file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not os.path.exists(file_path):
            self.app.display_message(f"ERROR: {file_path} does not exist")
            return
        file_dir = os.path.dirname(file_path)
        df = pd.read_csv(file_path)
        df["images"] = df["images"].apply(lambda x: x if os.path.isabs(x) else os.path.abspath(os.path.join(file_dir, x)))
    
        if "map_path" in df:
            df["map_path"] = df["map_path"].apply(lambda x: x if os.path.isabs(x) else os.path.abspath(os.path.join(file_dir, x)))
            count = df["map_path"].nunique()
            map_path = df["map_path"][0]
            if count > 1:
                self.app.display_message(f"WARNING: Multiple maps found in CSV, slecting: {map_path}")
                df = df[df["map_path"] == map_path]
            if not os.path.exists(map_path):
                self.app.display_message(f"ERROR: {map_path} does not exist")
                return
            self.app.map = PyramidMap(map_path)
            self.app.last_map = None
            self.app.display_message(f"Loaded map from: \n{map_path}")
        elif not self.app.map:
            self.app.display_message("ERROR: No map loaded")
            return
        new_objects = []
        for index, row in df.iterrows():
            image_path = row["images"]
            H = None
            if "homography" in row:
                parsed_H = row["homography"].replace("[","").replace("]","").replace("\n","").split()
                parsed_H = [float(x) for x in parsed_H if x]
                H = np.array(parsed_H).reshape(3,3)
                if np.linalg.norm(H - np.eye(3)) < 1e-6:
                    H = None
            new_object = ImageObject(image_path, H)
            if not new_object.state == ImageState.NOT_VALID:
                new_objects.append(new_object)
            else:
                self.app.display_message(f"ERROR: {new_object.error_message}")
        if len(new_objects) > 0:
            self.image_listbox.delete(0,tk.END)
            self.app.display_message(f"Loaded {len(new_objects)} image(s) from: \n{file_path}")
            for new_object in new_objects:
                normalized_path = os.path.normpath(new_object.image_path)
                parts = normalized_path.split(os.sep)
                if len(parts) > 3:
                    parts = parts[-3:]
                result = os.path.join(*parts)
                self.image_listbox.insert(tk.END, result)
            self.images = new_objects
        else:
            self.app.display_message(f"ERROR: No valid image pairs found in CSV: \n{file_path}")
        self.app.image = None
        self.app.panorama_cache = None
        self.select_image(0)
        self.app.sync_sliders()

    def save_results(self):
        self.app.clear_messages()
        if True or not self.output_folder:
            self.output_folder = filedialog.askdirectory(title='Select output folder...')
        if not self.output_folder:
            self.app.display_message("ERROR: Please select an output folder")
            return
        output_file = f"{self.output_folder}/results.csv"
        images_to_save = [self.app.image]
        if len(self.images) > 0:
            images_to_save = self.images
            
        data_to_write = []

        for i, image_object in enumerate(images_to_save):
            data_to_write.append(dict(images=image_object.image_path, homography=image_object.relative_transform().flatten(),map_path=self.app.map.map_file))
        
        pd.DataFrame(data_to_write).to_csv(output_file, index=False)
        self.app.display_message(f"Saved results to {output_file}")
        
    def on_save_image(self, event=None):
        self.app.clear_messages()
        if True or not self.output_folder:
            self.output_folder = filedialog.askdirectory(title='Select output folder...')
        if not self.output_folder:
            self.app.display_message("ERROR: Please select an output folder")
            return
        
        selected_image = self.app.image
        output_file = f"{self.output_folder}/{Path(selected_image.image_path).stem}.tif"
        H_tot = np.linalg.inv(selected_image.relative_transform())
        query_aligned, bbox_gps = align_image(selected_image.get_image(), self.app.map, H_tot, target_size = max(selected_image.get_image().shape))
        dump_geotif(query_aligned, bbox_gps, output_file)
        
        self.app.display_message(f"Saved results to {output_file}")
        
    def on_delete(self, event):
        
        if self.app.homography_mode:
            return
        
        if len(self.images) == 0:
            return
        
        self.app.clear_messages()
        self.app.display_message(f"Deleted image pair: {self.images[self.current_index].image_path.split('/')[-1]}")
        del self.images[self.current_index]
        
        if len(self.images) == 0:
            return
        self.image_listbox.delete(self.current_index)
        self.select_image(self.current_index)
            
    def select_image(self, new_index):
        if len(self.images) == 0:
            self.app.clear_messages()
            self.app.display_message("ERROR: No image pairs loaded")
            return
        prev = self.app.image
        self.current_index = new_index % len(self.images)
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(self.current_index)
        self.image_listbox.see(self.current_index)
        self.app.image = self.images[self.current_index]
        cur = self.app.image
        if self.app.automatic_matching and \
            prev is not None and (cur.state == ImageState.NOT_LOADED or cur.state == ImageState.LOADED or cur.state == ImageState.MATCHED) and \
            not (prev.state == ImageState.MOVED and cur.state == ImageState.MOVED): 
 
            H_rel = sift_matching_with_homography(cur.get_image(), prev.get_image())
            if H_rel is None:
                cur.scale, cur.rotation, cur.x_offset, cur.y_offset, cur.M_anchors = prev.scale, prev.rotation, prev.x_offset, prev.y_offset, prev.M_anchors
            else:
                prev.derived_images.append((cur, H_rel))
                H_prev = prev.M_anchors @ calc_transform([prev.image.shape[1] * prev.scale_ratio, prev.image.shape[0] * prev.scale_ratio], prev.scale, prev.rotation, prev.x_offset, prev.y_offset) @ prev.M_original
                cur_H =  H_prev @ H_rel @ np.linalg.inv(cur.M_original)
                cur.initialize_from_homography(cur_H)
            cur.reset_anchors()
            cur.state = ImageState.MATCHED
        self.app.render()
        self.app.sync_sliders()
        
    def next_image(self):
        self.select_image(self.current_index+1)
        
    def previous_image(self):
        self.select_image(self.current_index-1)

    def on_selection(self, event):
        if len(self.image_listbox.curselection())==0:
            return
        selected_index = self.image_listbox.curselection()[0]
        self.select_image(selected_index)
        
    def update_listbox(self):
        for i, image in enumerate(self.images):
            self.image_listbox.itemconfig(i, bg=ImageState.to_color(image.state), fg='black')
        cur_image = self.app.image
        if cur_image is None:
            return
        for image, _ in cur_image.derived_images:
            if image is None or image.state == ImageState.MOVED:
                continue
            index = self.images.index(image)
            self.image_listbox.itemconfig(index, bg=ImageState.to_color(image.state), fg='navy')


class ImageAlignerApp:
    def __init__(self, root, image_object=None, map_object=None):
        if not 'root' in dir(self):
            self.root = root
        
        if image_object is None:
            image_object = ImageObject("")

        self.image = image_object
        self.map = map_object
        self.alpha = 0.75
        self.last_map = None
        self.last_state = None
        self.zoom_center = None
        self.last_global_scale = 1.0
        
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
        self.edge_mode = False
        self.homography_mode = False
        self.rotation_mode = False
        self.draw_grid = False
        self.automatic_matching = False
        self.show_borders = False
        
        self.panorama_cache = None
        self.num_panoramas = 0

        if not 'debug_info' in dir(self):
            self.debug_info = DebugInfo(root, self)
            self.info_panel = InfoPanel(root, self)
            self.button_panel = ButtonPanel(root, self)
            self.canvas = tk.Canvas(self.root, width=window_size[0], height=window_size[1])
            self.canvas.pack(fill='both', expand=True)
            self.setup_bindings()
        
        if self.image.state == ImageState.NOT_VALID:
            self.display_message('ERROR: ' + self.image.error_message)
        self.render()

    def setup_bindings(self):
        self.root.bind('<KeyPress>', self.on_key_press)
        self.root.bind('<KeyRelease>', self.on_key_release)
        self.root.bind('<MouseWheel>', self.on_mouse_wheel)
        self.root.bind("<Button-2>", self.on_middle_click)
        self.root.bind("<Button-3>", self.toggle_images)
        self.root.bind("<Button-4>", self.on_mouse_wheel)
        self.root.bind("<Button-5>", self.on_mouse_wheel)
        self.root.bind('<ButtonPress-1>', self.on_mouse_press)
        self.root.bind('<B1-Motion>', self.on_mouse_drag)
        self.root.bind('<ButtonRelease-1>', self.on_mouse_release)
        self.root.bind('<Control-z>', self.undo)
        self.root.bind('<Control-Z>', self.redo)
        self.root.bind('<Control-s>', self.button_panel.on_save_image)
        self.root.bind('<Control-l>', self.toggle_lock_image)
        self.root.bind('<Delete>', self.button_panel.on_delete)
        self.root.bind("<Escape>", self.exit)
        self.root.bind("<Motion>", self.on_mouse_position)
        self.root.bind("<Configure>", self.on_root_resize)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def on_root_resize(self, event):
        self.canvas.config(width=event.width, height=event.height)
    
    def on_canvas_resize(self, event):
        global window_size
        window_size = [event.width, event.height]
        self.last_map = None
        self.render(update_state=False)
    
    def upload_image(self):
        self.clear_messages()
        image_path = filedialog.askopenfilename(title='Select image to align...')
        self.image = ImageObject(image_path) 
        if self.image.state == ImageState.NOT_VALID:
            self.display_message('ERROR: ' + self.image.error_message)
            return
        self.display_message('Image uploaded successfully')
        self.render()
        
    def M_global(self):
        center = np.array(window_size)/2
        
        if self.zoom_center is not None:
            prev_matrix = translation_matrix([self.global_x_offset, self.global_y_offset]) @ translation_matrix(center)@scale_matrix(self.last_global_scale)@translation_matrix(-center)
            desired_change = translation_matrix(self.zoom_center)@scale_matrix(self.global_scale/self.last_global_scale)@translation_matrix(-self.zoom_center)
            desired_matrix = desired_change@prev_matrix
            cur_scale = translation_matrix(center)@scale_matrix(self.global_scale)@translation_matrix(-center)
            T = desired_matrix @ np.linalg.inv(cur_scale)
            self.global_x_offset = T[0, 2]
            self.global_y_offset = T[1, 2]
            self.zoom_center = None
            
        self.last_global_scale = self.global_scale
            
        M = translation_matrix([self.global_x_offset, self.global_y_offset]) @ translation_matrix(center)@scale_matrix(self.global_scale)@translation_matrix(-center)
        return M
    
    def blend_images(self):
        if self.map is None:
            return np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)
        
        M_global = self.M_global()
        im2 = self.image.render(M_global)
        
        if im2 is None:
            return np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)
        
        M_global = self.M_global()
        if self.last_map is None or not (M_global==self.last_state).all():
            self.last_map = self.map.warp_map(M_global, window_size)
            if self.last_map is None:
                self.last_map = np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)
            self.last_state = M_global
        
        im1 = self.last_map.copy()
        
        if im1 is None:
            im1 = np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)

        if self.contrast_mode or self.edge_mode:
            if self.contrast_mode:
                im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
                im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
            else:
                im1 = edge_detection(im1,blur=3)
                im2 = edge_detection(im2,blur=9)
            if self.toggle:
                im1, im2 = im2, im1
            blend_image = np.stack([im1, im2, im1], axis=-1)
        else:
            alpha = self.alpha
            if self.toggle:
                alpha = 1 - alpha
            blend_image = cv2.addWeighted(im1, 1 - alpha, im2, alpha, 0)
            if self.show_borders:
                mask = (im2[:,:,0] == 0).astype(np.uint8)
                im1_new = cv2.multiply(im1,alpha)
                blend_image = cv2.add(blend_image, cv2.bitwise_and(im1_new, im1_new, mask=mask))
            
        return blend_image
        
    
    def render(self, update_state=True):
        t = time.time()
        
        if not self.dragging and update_state:
            self.image.save_state()
            
        rendered_image = self.blend_images()
        
        if self.draw_grid:
            rendered_image = draw_grid(rendered_image, self.M_global())

        if self.viewport_mode:
            rendered_image = cv2.rectangle(rendered_image, (0, 0), (window_size[0], window_size[1]), (0, 0, 255), 10)

        img_pil = Image.fromarray(rendered_image)
        self.tk_image = ImageTk.PhotoImage(img_pil)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        if self.homography_mode:
            M_global = self.M_global()
            for anchor in self.image.anchors:
                anchor.plot(self.canvas, M_global)
                
        fps = 1/(time.time()-t)
        self.info_panel.update_fps(fps)
        self.info_panel.update_scale(self.global_scale)
        
        self.button_panel.update_listbox()

        if self.debug_mode:
            self.debug_info.show_debug_info()
            
    def create_panorama(self):
        #create panorama
        selected_index = self.button_panel.current_index
        good_indices = []
        good_images = []
        good_files = []
        dst = None
        i = 0
        for num, image in enumerate(self.button_panel.images):
            if image.is_panorama:
                continue
            if num == selected_index:
                dst = i
            good_indices.append(i)
            i += 1
            good_images.append(image.get_image())
            good_files.append(image.image_path)
        if self.panorama_cache is None:
            self.panorama_cache = create_cache(good_images)
            self.panorama_cache["files"] = good_files
        else:
            if self.panorama_cache["files"] != good_files:
                # TODO: if some images is deleted you can re-use cache
                self.panorama_cache = create_cache(good_images)
                self.panorama_cache["files"] = good_files
        panorama_image, final_transforms = stitch_images(good_images, self.panorama_cache, dst)
        num_good = len([x for x in final_transforms if x is not None])
        if num_good < 2:
            self.clear_messages()
            self.display_message("ERROR: Failed to create panorama")
            return
        #calculate it's homography based on selected image
        H_rel = np.linalg.inv(final_transforms[dst])
        selected = self.image
        selected.state = ImageState.MATCHED
        M_selected = calc_transform([selected.image.shape[1] * selected.scale_ratio, selected.image.shape[0] * selected.scale_ratio], selected.scale, selected.rotation, selected.x_offset, selected.y_offset)
        cur_H = selected.M_anchors @ M_selected @ selected.M_original @ H_rel 
        self.num_panoramas += 1
        new_pair = ImageObject.create_panorama(panorama_image, cur_H, name=f"panorama_{self.num_panoramas}")
        #add to the list
        self.button_panel.images.append(new_pair)
        self.button_panel.image_listbox.delete(0,tk.END)
        for new_object in self.button_panel.images:
            normalized_path = os.path.normpath(new_object.image_path)
            parts = normalized_path.split(os.sep)
            if len(parts) > 3:
                parts = parts[-3:]
            result = os.path.join(*parts)
            self.button_panel.image_listbox.insert(tk.END, result)
        self.button_panel.select_image(len(self.button_panel.images)-1)
        #update relative transforms
        if self.automatic_matching:
            self.toggle_automatic_matching()
        prev = new_pair 
        H_prev = prev.M_anchors @ calc_transform([prev.image.shape[1] * prev.scale_ratio, prev.image.shape[0] * prev.scale_ratio], prev.scale, prev.rotation, prev.x_offset, prev.y_offset) @ prev.M_original
        for cur_index, ind in enumerate(good_indices):
            cur = self.button_panel.images[ind]
            if final_transforms[cur_index] is None or cur.is_panorama:
                continue
            H_rel = final_transforms[cur_index]
            prev.derived_images.append([cur, H_rel])
            if cur.state == ImageState.MOVED or cur.state == ImageState.LOCKED:
                continue
            cur_H = H_prev @ H_rel @ np.linalg.inv(cur.M_original)
            cur.initialize_from_homography(cur_H)
            cur.reset_anchors()
            cur.state = ImageState.MATCHED
        #success
        self.button_panel.update_listbox()
        self.clear_messages()
        self.display_message(f"Successfully created panorama from {num_good} images")
        

    # Event Handlers
    def update_alpha(self, val):
        self.alpha = float(val)
        self.button_panel.alpha_slider.set(self.alpha)
        self.render()
    
    def update_rotation(self, val):
        self.move_anchors()
        val = float(val)
        self.image.rotation = ((val + 180)%360 -180) if val != 180 else 180
        self.button_panel.rotation_slider.set(self.image.rotation)
        self.render()
        
    def update_scale(self, val):
        self.move_anchors()
        self.image.scale = np.clip(float(val),0.1,10)
        self.button_panel.scale_slider.set(np.log10(self.image.scale))
        self.render()
    
    def sync_sliders(self):
        self.button_panel.rotation_slider.set(self.image.rotation)
        self.button_panel.scale_slider.set(np.log10(self.image.scale))
        
    def toggle_help_mode(self):
        self.help_mode = not self.help_mode
        self.button_panel.help_button.config(text="Help:  ON" if self.help_mode else "Help: OFF", bg=('grey' if self.help_mode else 'white'))
        descriptions = [('r', "Rotate by 90 degrees"),
                        ('+', "Zoom in 10%"),
                        ('-', "Zoom out 10%"),
                        ('d', "Debug mode"),
                        ('c', "Contrast mode"),
                        ('t/right click', "Toggle images"),
                        ('h', "Homography mode"),
                        ('a', "Automatic homography"),
                        ('o', "Reset homography"),
                        ('space', "Change field of view"),
                        ('mouse wheel', "Zoom in/out"),
                        ('middle click', "Toggle rotation mode"),
                        ('<-/->', "Previous/Next image")]
        self.button_panel.help_text_box.delete('1.0', tk.END)
        if self.help_mode:
            self.clear_messages()
            self.button_panel.help_text_box.config(state=tk.NORMAL)
            for key, description in descriptions:
                self.button_panel.help_text_box.insert(tk.END, f"{key}: {description}\n")
            self.button_panel.help_text_box.config(state=tk.DISABLED)
    
    def toggle_homography_mode(self):
        self.homography_mode = not self.homography_mode
        if len(self.image.anchors) == 0:
            self.image.reset_anchors()
        self.button_panel.homography_button.config(text="Homography Mode:  ON" if self.homography_mode else "Homography Mode: OFF", bg=('grey' if self.homography_mode else 'white'))
        self.render()

    def toggle_contrast_mode(self):
        self.contrast_mode = not self.contrast_mode
        if self.contrast_mode and self.edge_mode:
            self.toggle_edge_mode()
        self.button_panel.contrast_button.config(text="Contrast Mode:  ON" if self.contrast_mode else "Contrast Mode: OFF", bg=('grey' if self.contrast_mode else 'white'))
        self.render()
    
    def toggle_edge_mode(self):
        self.edge_mode = not self.edge_mode
        if self.edge_mode and self.contrast_mode:
            self.toggle_contrast_mode()
        self.render()

    def toggle_images(self, _ = None):
        self.toggle = not self.toggle
        self.button_panel.switch_button.config(text="Switch Images:  ON" if self.toggle else "Switch Images: OFF", bg=('grey' if self.toggle else 'white'))
        self.render()

    def toggle_grid(self):
        self.draw_grid = not self.draw_grid
        self.button_panel.grid_button.config(text="Show Grid:  ON" if self.draw_grid else "Show Grid: OFF", bg=('grey' if self.draw_grid else 'white'))
        self.render()

    def reset_homography(self):
        self.image.M_anchors = np.eye(3)
        self.image.reset_anchors()
        self.render()

    def reset(self):
        self.update_scale(1.0)
        self.update_rotation(0)
        self.image.x_offset = 0
        self.image.y_offset = 0
        self.image.M_anchors = np.eye(3)
        self.image.reset_anchors()
        self.image.state = ImageState.LOADED
        self.render()
        
    def run_matching(self):
        M_global = self.M_global()
        im1 = self.map.warp_map(M_global, window_size)
        im2 = self.image.get_image()
        H = sift_matching_with_homography(im2, im1)
        if H is None:
            self.display_message('ERROR: Could not calculate homography')
            return
        
        self.image.initialize_from_homography( np.linalg.inv(M_global) @ H @ np.linalg.inv(self.image.M_original))
        self.image.state = ImageState.MOVED
        self.sync_sliders()
        self.render() 
    
    def toggle_lock_image(self, _ = None):
        if self.image.state == ImageState.LOCKED:
            self.image.state = ImageState.MOVED
        else:
            self.image.state = ImageState.LOCKED
        self.button_panel.update_listbox()

    def toggle_viewport_mode(self, _ = None):
        self.viewport_mode = not self.viewport_mode
        self.button_panel.viewport_button.config(text="Viewport Mode:  ON" if self.viewport_mode else "Viewport Mode: OFF", bg=('grey' if self.viewport_mode else 'white'))
        self.root.config(cursor=('fleur' if self.viewport_mode else 'arrow'))
        self.render()
    
    def toggle_automatic_matching(self):
        self.automatic_matching = not self.automatic_matching
        self.button_panel.automatic_matching_button.config(text="Automatic Matching:  ON" if self.automatic_matching else "Automatic Matching: OFF", bg=('grey' if self.automatic_matching else 'white'))
        
    def toggle_debug_mode(self):
        self.debug_mode = not self.debug_mode
        self.button_panel.debug_button.config(text="Debug Mode:  ON" if self.debug_mode else "Debug Mode: OFF", bg=('grey' if self.debug_mode else 'white'))
        self.render()
        
    def toggle_borders(self):
        self.show_borders = not self.show_borders
        self.button_panel.borders_button.config(text="Show Borders:  ON" if self.show_borders else "Show Borders: OFF", bg=('grey' if self.show_borders else 'white'))
        self.render()
    
    def undo(self, event):
        self.image.undo()
        self.render(False)

    def redo(self, event):
        self.image.redo()
        self.render(False)
        
    def on_mouse_position(self, event):
        relevant = self.check_relevancy(event)
        self.root.config(cursor=('fleur' if self.viewport_mode and relevant else 'arrow'))
        if not relevant:
            return

        pos = [event.x, event.y]
        if self.map is None:
            self.info_panel.update_position(pos)
            return
        pos = apply_homography(np.linalg.inv(self.M_global()), pos)
        pos = self.map.pix2gps(pos)
        self.info_panel.update_position(pos)
    
    def on_key_press(self, event):
        if event.char == 'r':
            self.update_rotation(self.image.rotation + 90)
        elif event.char == '-':
            self.global_scale *= 0.9
        elif event.char == '=':
            self.global_scale *= 1.1
        elif event.char == 'd':
            self.toggle_debug_mode()
        elif event.char == 'c':
            self.toggle_contrast_mode()
        elif event.char == 'e':
            self.toggle_edge_mode()
        elif event.char == ' ':
            self.toggle_viewport_mode()
        elif event.char == 'h':
            self.toggle_homography_mode()
        elif event.char == 'o':
            self.reset_homography()
        elif event.char == 'p':
            self.create_panorama()
        elif event.char == 'q':
            self.reset()
        elif event.char == 'm':
            self.toggle_automatic_matching()
        elif event.char == 'g':
            self.toggle_grid()
        elif event.char == 'b':
            self.toggle_borders()
        elif event.char == 'a':
            self.run_matching()
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
            self.zoom_center = np.array([event.x, event.y])
        else:
            if self.image.state != ImageState.LOCKED:
                if self.rotation_mode:
                    self.update_rotation(self.image.rotation - step)
                elif not self.homography_mode:
                    self.update_scale(self.image.scale * (1.02 ** step))
        self.render()

    def on_mouse_press(self, event):
        if not self.check_relevancy(event):
            return
        
        self.dragging = True
        pt0 = [event.x, event.y]
        pt = apply_homography(np.linalg.inv(self.M_global()), pt0)
        if self.homography_mode and not self.viewport_mode:
            if self.image.state != ImageState.LOCKED:
                self.image.push_anchor(pt)
                self.image.state = ImageState.MOVED
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
        
        self.last_mouse_position = (event.x, event.y)
        pt0 = [event.x, event.y]
        pt = apply_homography(np.linalg.inv(self.M_global()), pt0)
        
        if self.homography_mode and not self.viewport_mode:
            if self.image.state != ImageState.LOCKED:
                self.image.anchors[-1].move(*pt)
                self.render()
            return

        self.move_anchors()
        if self.viewport_mode:
            self.global_x_offset += pt0[0] - self.drag_start_x
            self.global_y_offset += pt0[1] - self.drag_start_y
            self.drag_start_x = pt0[0]
            self.drag_start_y = pt0[1]
        else:
            if self.image.state != ImageState.LOCKED:
                self.image.x_offset += pt[0] - self.drag_start_x
                self.image.y_offset += pt[1] - self.drag_start_y
                self.image.state = ImageState.MOVED
            self.drag_start_x = pt[0]
            self.drag_start_y = pt[1]
        self.render()

    def on_mouse_release(self, event):
        self.dragging = False
        if self.homography_mode and not self.viewport_mode:
            self.image.M_anchors = calc_homography(self.image.anchors) @ self.image.M_anchors
            for anchor in self.image.anchors:
                anchor.reset()
        self.render()
        return
        
    def move_anchors(self):
        self.image.M_anchors = calc_homography(self.image.anchors) @ self.image.M_anchors
        self.image.reset_anchors()

    def clear_messages(self):
        self.button_panel.help_text_box.config(state=tk.NORMAL)
        self.button_panel.help_text_box.delete('1.0', tk.END)
        self.button_panel.help_text_box.config(state=tk.DISABLED)
        
    def display_message(self, message):
        if self.help_mode:
            self.toggle_help_mode()
        self.button_panel.help_text_box.config(state=tk.NORMAL)
        self.button_panel.help_text_box.insert(tk.END, message + '\n')
        self.button_panel.help_text_box.config(state=tk.DISABLED)

    def exit(self, event=None):
        self.root.quit()
        self.root.destroy()

def configure_screen_size(root):
    global window_size, screen_size
    screen_size = [int(root.winfo_screenwidth()*SCREEN_FACTOR), int(root.winfo_screenheight()*SCREEN_FACTOR)]
    screen_size[0] = min(screen_size[0], int(screen_size[1]*1.9))
    window_size = [int(screen_size[0]*0.85), int(screen_size[1]*0.96)]

def main():
    root = tk.Tk()
    global window_size, screen_size
    configure_screen_size(root)
    root.title("TagIm Aligning App")
    root.geometry(f"{screen_size[0]}x{screen_size[1]}")
    if os.path.exists('resources/logo.jpg'):
        photo = ImageTk.PhotoImage(Image.open('resources/logo.jpg'))
        root.wm_iconphoto(False, photo)

    app = ImageAlignerApp(root)
    app.button_panel.load_csv("output/simulated_list2.csv")

    root.mainloop()

if __name__ == "__main__":
    main()
