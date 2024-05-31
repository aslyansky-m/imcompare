# credit goes to Maksym Aslianskyi, ChatGPT, GitHub Copilot and StackOverflow

import os
import cv2
import numpy as np
import types
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

window_size = (1050, 700)
DEBUG = True

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

def decompose_homography(H):
    tx = H[0, 2]
    ty = H[1, 2]
    H1 = H[:, 0]
    H2 = H[:, 1]
    scale_x = np.linalg.norm(H1)
    scale_y = np.linalg.norm(H2)
    scale = (scale_x + scale_y) / 2.0
    H1_normalized = H1 / scale_x
    H2_normalized = H2 / scale_y
    rotation_rad = np.arctan2(H1_normalized[1], H1_normalized[0])
    rotation_deg = np.degrees(rotation_rad)
    
    return scale, rotation_deg, tx, ty

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

def apply_homography(H, point):
    pt = np.array([point[0], point[1], 1])
    pt = H @ pt
    pt = pt[:2]/pt[2]
    return pt

class ImagePair:
    def __init__(self, img1_path, img2_path):
        self.img1_path = img1_path
        self.img2_path = img2_path
        self.valid = True
        self.scale = 1.0
        self.rotation = 0
        self.x_offset = 0
        self.y_offset = 0
        self.anchors = []
        self.scale_ratio = 1.0
        self.M_anchors = np.eye(3)
        self.M_original = np.eye(3)
        self.error_message = ''
        self.img1 = None
        self.img2 = None
        
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

        self.scale_ratio = min(window_size[0]/self.img1.shape[1], window_size[1]/self.img1.shape[0])
        self.M_original = np.diag([self.scale_ratio, self.scale_ratio, 1])
        
    def reset_anchors(self):        
        if self.img2 is not None:
            m = 30
            w = self.img2.shape[1]
            h = self.img2.shape[0]
            anchors_pos = [(m, m), (m, h - m), (w - m, m), (w - m, h - m)]
            M = calc_transform((self.img2.shape[1]*self.scale_ratio,self.img2.shape[0]*self.scale_ratio), self.scale, self.rotation, self.x_offset, self.y_offset)
            anchors_pos = [apply_homography(self.M_anchors @ M @ self.M_original, pos) for pos in anchors_pos]
            self.anchors = [Anchor(x, y, original=True) for x, y in anchors_pos]
        else:
            m = 100
            w = window_size[0]
            h = window_size[1]
            anchors_pos = [(m, m), (m, h - m), (w - m, m), (w - m, h - m)]
            self.anchors = [Anchor(x, y, original=True) for x, y in anchors_pos]
    
    def run_matching(self):
        H = sift_matching_with_homography(self.img2, self.img1)
        if H is None:
            return False
        
        self.scale = 1.0
        self.rotation = 0
        self.x_offset = 0
        self.y_offset = 0
        self.M_anchors = self.M_original @ H @ np.linalg.inv(self.M_original)
        self.reset_anchors()
        return True
 
    def render(self, app):
        if not self.valid:
            return np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)
        
        M = calc_transform([self.img2.shape[1]*self.scale_ratio, self.img2.shape[0]*self.scale_ratio], self.scale, self.rotation, self.x_offset, self.y_offset)
        H = calc_homography(self.anchors)
        M_global = app.M_global()
        
        im1 = cv2.warpPerspective(self.img1, M_global @ self.M_original, window_size)
        im2 = cv2.warpPerspective(self.img2, M_global @ H @ self.M_anchors @ M @ self.M_original, window_size)

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
        min_dist = -1
        closest_anchor = None
        for anchor in self.anchors:
            dist = (anchor.pos[0] - pt[0]) ** 2 + (anchor.pos[1] - pt[1]) ** 2
            if min_dist < 0 or dist < min_dist:
                min_dist = dist
                closest_anchor = anchor
        new_anchor = Anchor(pt[0], pt[1])
        self.anchors.remove(closest_anchor)
        self.anchors.append(new_anchor)
    
    def relative_transform(self):
        M = calc_transform([self.img2.shape[1]*self.scale_ratio, self.img2.shape[0]*self.scale_ratio], self.scale, self.rotation, self.x_offset, self.y_offset)
        H = calc_homography(self.anchors)
        T = H @ self.M_anchors @ M
        return T
    
    def __str__(self):
        T = self.relative_transform()
        return f"{self.img1_path}, {self.img2_path}, {T.flatten().tolist()}"
       
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
        for var_name, var_value in app_vars.items():
            if isinstance(var_value,np.ndarray):
                continue
            label = tk.Label(self.debug_frame, text=f"{var_name}: {var_value}", bg='grey')
            label.pack(side="top", anchor="w", padx=10, pady=5)
            self.label_widgets.append(label)

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
        self.alpha_slider = tk.Scale(self.frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Alpha Blending", command=self.app.update_alpha)
        self.alpha_slider.set(self.app.alpha)
        
        self.rotation_slider = tk.Scale(self.frame, from_=-180, to=180, resolution=1, orient=tk.HORIZONTAL, label="Rotation", command=self.app.update_rotation)
        self.rotation_slider.set(self.app.images.rotation)

        self.scale_slider = tk.Scale(self.frame, from_=-1, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Scale Factor", command=lambda x: self.app.update_scale(np.power(10,float(x))))
        self.scale_slider.set(np.log10(self.app.images.scale))

        self.homography_button = tk.Button(self.frame, text="Homography Mode: OFF", command=self.app.toggle_homography_mode)
        self.homography_reset_button = tk.Button(self.frame, text="Reset Homography", command=self.app.reset_homography)
        self.homography_calculate_button = tk.Button(self.frame, text="Automatic Homography", command=self.app.run_matching)
        self.viewport_button = tk.Button(self.frame, text="Viewport Mode: OFF", command=self.app.toggle_viewport_mode)
        self.contrast_button = tk.Button(self.frame, text="Contrast Mode: OFF", command=self.app.toggle_contrast_mode)
        self.switch_button = tk.Button(self.frame, text="Switch: OFF", command=self.app.toggle_switch)
        self.debug_button = tk.Button(self.frame, text="Debug Mode: OFF", command=self.app.toggle_debug_mode)
        self.help_button = tk.Button(self.frame, text="Help: OFF", command=self.app.toggle_help_mode)
        self.help_frame = tk.Frame(self.frame)
        self.help_text_box = tk.Text(self.help_frame, height=7, width=30, wrap="word")
        self.help_text_box.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.help_scrollbar = tk.Scrollbar(self.help_frame, command=self.help_text_box.yview)
        self.help_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.help_text_box['yscrollcommand'] = self.help_scrollbar.set
        self.upload_button = tk.Button(self.frame, text="Upload Images", command=self.app.upload_images)
        
        self.load_csv_button = tk.Button(self.frame, text="Load CSV", command=self.load_csv)
        self.save_results_button = tk.Button(self.frame, text="Save Results", command=self.save_results)
        self.next_image_button = tk.Button(self.frame, text="Next Image", command=self.next_image)
        self.image_list_frame = tk.Frame(self.frame)
        self.image_list_scrollbar = tk.Scrollbar(self.image_list_frame)
        self.image_list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_listbox = tk.Listbox(self.image_list_frame, yscrollcommand=self.image_list_scrollbar.set)
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_listbox.bind('<<ListboxSelect>>', self.select_image)
        self.image_list_scrollbar.config(command=self.image_listbox.yview)

        self.alpha_slider.grid(row=0, column=0, sticky='ew')
        self.rotation_slider.grid(row=1, column=0, sticky='ew')
        self.scale_slider.grid(row=2, column=0, sticky='ew')
        self.homography_button.grid(row=3, column=0, sticky='ew')
        self.homography_reset_button.grid(row=4, column=0, sticky='ew')
        self.homography_calculate_button.grid(row=5, column=0, sticky='ew')
        self.viewport_button.grid(row=6, column=0, sticky='ew')
        self.contrast_button.grid(row=7, column=0, sticky='ew')
        self.switch_button.grid(row=8, column=0, sticky='ew')
        self.debug_button.grid(row=9, column=0, sticky='ew')
        self.help_button.grid(row=10, column=0, sticky='ew')
        self.help_frame.grid(row=11, column=0, sticky='ew')
        self.upload_button.grid(row=12, column=0, sticky='ew')
        self.load_csv_button.grid(row=13, column=0, sticky='ew')
        self.next_image_button.grid(row=14, column=0, sticky='ew')
        self.save_results_button.grid(row=15, column=0, sticky='ew')
        self.image_list_frame.grid(row=16, column=0, sticky='ew')

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
                line_parsed = line.strip().split(',')[:2]  # Assuming 2 columns in CSV
                if len(line_parsed) != 2:
                    self.app.display_message(f"ERROR: {line} is not a valid image pair")
                    continue
                image_path1, image_path2 = line_parsed
                image_path1, image_path2 = image_path1.replace(' ',''), image_path2.replace(' ','')
                new_pair = ImagePair(image_path1, image_path2)
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
        if not self.app.images.valid and len(self.image_pairs) > 0:
            self.next_image()

    def save_results(self):
        self.app.clear_messages()
        output_folder = filedialog.askdirectory(title='Select output folder...')
        if not output_folder:
            self.app.display_message("ERROR: Please select an output folder")
            return
        output_file = f"{output_folder}/results.csv"
        self.app.display_message(f"Saved results to {output_file}")
        with open(output_file, 'a') as file:
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
            return
        self.current_index = (self.current_index + sign) % len(self.image_pairs)
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(self.current_index)
        self.image_listbox.see(self.current_index)
        self.app.images = self.image_pairs[self.current_index]
        self.app.render()
        
    def next_image(self):
        self.step_image(+1)
        
    def previous_image(self):
        self.step_image(-1)

    def select_image(self, event):
        self.current_index = self.image_listbox.curselection()[0]
        self.app.images = self.image_pairs[self.current_index]
        self.app.render()

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
    
    def render(self):
        rendered_image = self.images.render(self)
        img_pil = Image.fromarray(rendered_image)
        self.tk_image = ImageTk.PhotoImage(img_pil)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        if self.homography_mode:
            M_global = self.M_global()
            for anchor in self.images.anchors:
                anchor.plot(self.canvas, M_global)

        if self.debug_mode:
            self.debug_info.show_debug_info()

    # Event Handlers
    def update_alpha(self, val):
        self.alpha = float(val)
        self.button_panel.alpha_slider.set(self.alpha)
        self.render()
    
    def update_rotation(self, val):
        self.move_anchors()
        self.images.rotation = (int(val) + 180)%360 -180
        self.button_panel.rotation_slider.set(self.images.rotation)
        self.render()
        
    def update_scale(self, val):
        self.move_anchors()
        self.images.scale = np.clip(float(val),0.1,10)
        self.button_panel.scale_slider.set(np.log10(self.images.scale))
        self.render()

    def toggle_debug_mode(self):
        self.debug_mode = not self.debug_mode
        self.button_panel.debug_button.config(text="Debug Mode:  ON" if self.debug_mode else "Debug Mode: OFF")
        
    def toggle_help_mode(self):
        self.help_mode = not self.help_mode
        self.button_panel.help_button.config(text="Help:  ON" if self.help_mode else "Help: OFF")
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
        self.button_panel.homography_button.config(text="Homography Mode:  ON" if self.homography_mode else "Homography Mode: OFF")
        self.render()

    def toggle_contrast_mode(self):
        self.contrast_mode = not self.contrast_mode
        self.button_panel.contrast_button.config(text="Contrast Mode:  ON" if self.contrast_mode else "Contrast Mode: OFF")
        self.render()

    def toggle_switch(self):
        self.toggle = not self.toggle
        self.button_panel.switch_button.config(text="Switch:  ON" if self.toggle else "Switch: OFF")
        self.render()

    def reset_homography(self):
        self.images.reset_anchors()
        self.images.M_anchors = np.eye(3)
        self.render()
        
    def run_matching(self):
        ret = self.images.run_matching()
        if not ret:
            self.display_message('ERROR: Could not calculate homography')
        self.render()

    def toggle_viewport_mode(self):
        self.viewport_mode = not self.viewport_mode
        self.button_panel.viewport_button.config(text="Viewport Mode:  ON" if self.viewport_mode else "Viewport Mode: OFF")
    
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
        elif event.keysym == 'Control_L' or event.keysym == 'Control_R':
            self.toggle_homography_mode()
        elif event.char == 'o':
            self.reset_homography()
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
        if self.homography_mode and not self.viewport_mode:
            self.images.M_anchors = calc_homography(self.images.anchors) @ self.images.M_anchors
            for anchor in self.images.anchors:
                anchor.reset()
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
    root.title("TagIm Aligner")
    root.geometry(f"{int(window_size[0]*1.2)}x{window_size[1]}")
    photo = ImageTk.PhotoImage(Image.open('resources/logo.jpg'))
    root.wm_iconphoto(False, photo)

    image_pair = ImagePair("input/im1.jpeg", "input/im3.jpeg")
    
    app = ImageAlignerApp(root, image_pair if DEBUG else None)

    root.mainloop()

if __name__ == "__main__":
    main()

