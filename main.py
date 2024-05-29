# credit goes to Maksym Aslianskyi, ChatGPT, GitHub Copilot and StackOverflow

import cv2
import numpy as np
import types
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

window_dimensions = (1500, 900)
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

def calc_transform(shape, scale, rotation, x_offset, y_offset):
    rows, cols = shape[:2]
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
        self.image_paths = []

    def create_widgets(self):
        self.alpha_slider = tk.Scale(self.frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Alpha Blending", command=self.app.on_alpha_blending_change)
        self.alpha_slider.set(self.app.alpha_blending)
        self.alpha_slider.bind('<ButtonPress-1>', self.app.on_slider_click)
        self.alpha_slider.bind('<ButtonRelease-1>', self.app.on_slider_release)

        self.rotation_slider = tk.Scale(self.frame, from_=0, to=360, resolution=1, orient=tk.HORIZONTAL, label="Rotation", command=self.app.on_rotation_change)
        self.rotation_slider.set(self.app.rotation)
        self.rotation_slider.bind('<ButtonPress-1>', self.app.on_slider_click)
        self.rotation_slider.bind('<ButtonRelease-1>', self.app.on_slider_release)
        
        self.scale_slider = tk.Scale(self.frame, from_=0.1, to=10, resolution=0.01, orient=tk.HORIZONTAL, label="Scale Factor", command=self.app.update_scale_factor)
        self.scale_slider.set(self.app.scale_factor)
        self.scale_slider.bind('<ButtonPress-1>', self.app.on_slider_click)
        self.scale_slider.bind('<ButtonRelease-1>', self.app.on_slider_release)

        self.app.homography_button = tk.Button(self.frame, text="Homography Mode: OFF", command=self.app.toggle_homography_mode)
        self.app.homography_reset_button = tk.Button(self.frame, text="Reset Homography", command=self.app.reset_homography)
        self.app.viewport_button = tk.Button(self.frame, text="Viewport Mode: OFF", command=self.app.toggle_viewport_mode)
        self.app.contrast_button = tk.Button(self.frame, text="Contrast Mode: OFF", command=self.app.toggle_contrast_mode)
        self.app.switch_button = tk.Button(self.frame, text="Switch: OFF", command=self.app.toggle_switch)
        self.app.debug_button = tk.Button(self.frame, text="Debug Mode: OFF", command=self.app.toggle_debug_mode)
        self.app.help_button = tk.Button(self.frame, text="Help", command=self.app.toggle_help_mode)
        self.app.help_text_box = tk.Text(self.frame, height=11, width=50, wrap="word")
        self.app.upload_button = tk.Button(self.frame, text="Upload Images", command=self.app.upload_images)
        
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
        self.app.homography_button.grid(row=3, column=0, sticky='ew')
        self.app.homography_reset_button.grid(row=4, column=0, sticky='ew')
        self.app.viewport_button.grid(row=5, column=0, sticky='ew')
        self.app.contrast_button.grid(row=6, column=0, sticky='ew')
        self.app.switch_button.grid(row=7, column=0, sticky='ew')
        self.app.debug_button.grid(row=8, column=0, sticky='ew')
        self.app.help_button.grid(row=9, column=0, sticky='ew')
        self.app.help_text_box.grid(row=10, column=0, sticky='ew')
        self.app.upload_button.grid(row=11, column=0, sticky='ew')
        self.load_csv_button.grid(row=12, column=0, sticky='ew')
        self.next_image_button.grid(row=13, column=0, sticky='ew')
        self.save_results_button.grid(row=14, column=0, sticky='ew')
        self.image_list_frame.grid(row=15, column=0, sticky='ew')

        self.frame.grid_columnconfigure(0, weight=1)
        
    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            with open(file_path, 'r') as file:
                for line in file:
                    image_path1, image_path2 = line.strip().split(',')[:2]  # Assuming 2 columns in CSV
                    self.image_paths.append([image_path1.replace(' ',''), image_path2.replace(' ','')])
                    self.image_listbox.insert(tk.END, image_path2)
        self.current_index = -1

    def save_results(self):
        # Add saving logic here
        pass

    def next_image(self):
        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(self.current_index)
        self.image_listbox.see(self.current_index)
        self.app.__init__(self.root, images=self.image_paths[self.current_index])

    def select_image(self, event):
        self.current_index = self.image_listbox.curselection()[0]
        self.app.__init__(self.root, images=self.image_paths[self.current_index])

class ImageAlignerApp:
    def __init__(self, root, images = None):
        if not 'root' in dir(self):
            self.root = root
        
        self.alpha_blending = 0.75
        self.scale_factor = 1.0
        self.rotation = 0
        self.dragging = False
        self.x_offset = 0
        self.y_offset = 0
        self.global_scale_factor = 1.0
        self.global_x_offset = 0
        self.global_y_offset = 0
        self.cur_H = np.eye(3)
        self.M_global = np.eye(3)
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.slider_active = False
        self.viewport_mode = False
        self.toggle = False
        self.debug_mode = False
        self.help_mode = False
        self.contrast_mode = False
        self.homography_mode = False
        self.rotation_mode = False
        self.last_shape = None
        self.anchors = []
        self.img1 = None
        self.img2 = None
        
        if not 'debug_info' in dir(self):
            self.debug_info = DebugInfo(root, self)
            self.button_panel = ButtonPanel(root, self)
            self.canvas = tk.Canvas(self.root, width=window_dimensions[0], height=window_dimensions[1])
            self.canvas.pack()
            self.setup_bindings()
            
        self.load_images(images)
        self.update_image()

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
        image_paths = filedialog.askopenfilenames(title='Select images to align...')
        self.__init__(self.root, images=image_paths)

    def load_images(self, image_paths):
        if not isinstance(image_paths, list) or len(image_paths) != 2:
            print("Supplied images are not valid: {}".format(image_paths))
            if self.img1 is None or self.img2 is None:
                self.exit()
            else:
                return
            
        self.img1 = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
        self.img2 = cv2.imread(image_paths[1], cv2.IMREAD_UNCHANGED)
        
        scale_ratio = min(window_dimensions[0]/self.img1.shape[1], window_dimensions[1]/self.img1.shape[0])
        self.img1 = cv2.resize(self.img1, (0, 0), fx=scale_ratio, fy=scale_ratio)
        self.img2 = cv2.resize(self.img2, (0, 0), fx=scale_ratio, fy=scale_ratio)

    def reset_anchors(self):
        
        if self.img2 is not None:
            m = 30
            w = self.img2.shape[1]
            h = self.img2.shape[0]
            anchors_pos = [(m, m), (m, h - m), (w - m, m), (w - m, h - m)]
            M = calc_transform(self.img2.shape, self.scale_factor, self.rotation, self.x_offset, self.y_offset)
            anchors_pos = [apply_homography(M, pos) for pos in anchors_pos]
            self.anchors = [Anchor(x, y, original=True) for x, y in anchors_pos]
        else:
            m = 100
            w = window_dimensions[0]
            h = window_dimensions[1]
            anchors_pos = [(m, m), (m, h - m), (w - m, m), (w - m, h - m)]
            self.anchors = [Anchor(x, y, original=True) for x, y in anchors_pos]

    def update_image(self):
        if self.img1 is None or self.img2 is None:
            return
        
        im1 = self.img1.copy()
        im2 = self.img2.copy()

        M = calc_transform(im2.shape, self.scale_factor, self.rotation, self.x_offset, self.y_offset)
        H = calc_homography(self.anchors)
        im2 = cv2.warpPerspective(im2, H @ self.cur_H @ M, (im1.shape[1], im1.shape[0]))

        if self.toggle:
            im1, im2 = im2, im1

        if self.contrast_mode:
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
            blend_image = np.stack([im1, im2, im1], axis=-1)
        else:
            blend_image = cv2.addWeighted(im1, 1 - self.alpha_blending, im2, self.alpha_blending, 0)

        self.M_global = calc_transform(blend_image.shape, self.global_scale_factor, 0, self.global_x_offset, self.global_y_offset)
        blend_image = cv2.warpPerspective(blend_image, self.M_global, (blend_image.shape[1], blend_image.shape[0]))

        if self.viewport_mode:
            blend_image = (blend_image * 0.8).astype(np.uint8)

        img_rgb = cv2.cvtColor(blend_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.tk_image = ImageTk.PhotoImage(img_pil)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        if self.homography_mode:
            for anchor in self.anchors:
                anchor.plot(self.canvas, self.M_global)

        if self.debug_mode:
            self.debug_info.show_debug_info()

    # Event Handlers
    def on_alpha_blending_change(self, val):
        self.alpha_blending = float(val)
        self.update_image()

    def on_rotation_change(self, val):
        self.rotation = int(val)
        self.update_image()

    def on_slider_click(self, event):
        self.slider_active = True

    def on_slider_release(self, event):
        self.slider_active = False
        
    def update_alpha(self, val):
        self.alpha = float(val)
        self.button_panel.alpha_slider.set(self.alpha)
        self.update_image()
    
    def update_rotation(self, val):
        self.rotation = int(val)
        self.button_panel.rotation_slider.set(self.rotation)
        self.update_image()
        
    def update_scale_factor(self, val):
        self.scale_factor = float(val)
        self.button_panel.scale_slider.set(self.scale_factor)
        self.update_image()

    def toggle_debug_mode(self):
        self.debug_mode = not self.debug_mode
        self.update_image()
        self.button_panel.app.debug_button.config(text="Debug Mode:  ON" if self.debug_mode else "Debug Mode: OFF")
        
    def toggle_help_mode(self):
        self.help_mode = not self.help_mode
        self.update_image()
        self.button_panel.app.help_button.config(text="Help:  ON" if self.help_mode else "Help: OFF")
        descriptions = [('r', "Rotate by 90 degrees"),
                        ('+', "Increase global scale factor by 10%"),
                        ('-', "Decrease global scale factor by 10%"),
                        ('d', "Toggle debug mode"),
                        ('c', "Toggle contrast mode"),
                        ('t/right click', "Toggle switch"),
                        ('ctrl', "Toggle homography mode"),
                        ('o', "Reset homography"),
                        ('space', "Toggle viewport mode"),
                        ('mouse wheel', "Zoom in/out"),
                        ('middle click', "Toggle rotation mode")]
        self.help_text_box.delete('1.0', tk.END)
        if self.help_mode:
            for key, description in descriptions:
                self.help_text_box.insert(tk.END, f"{key}: {description}\n")
            self.help_text_box.config(state=tk.NORMAL)
    
    def toggle_homography_mode(self):
        self.homography_mode = not self.homography_mode
        if len(self.anchors) == 0:
            self.reset_anchors()
        self.update_image()
        self.button_panel.app.homography_button.config(text="Homography Mode:  ON" if self.homography_mode else "Homography Mode: OFF")

    def toggle_contrast_mode(self):
        self.contrast_mode = not self.contrast_mode
        self.update_image()
        self.button_panel.app.contrast_button.config(text="Contrast Mode:  ON" if self.contrast_mode else "Contrast Mode: OFF")

    def toggle_switch(self):
        self.toggle = not self.toggle
        self.update_image()
        self.button_panel.app.switch_button.config(text="Switch:  ON" if self.toggle else "Switch: OFF")

    def reset_homography(self):
        self.reset_anchors()
        self.cur_H = np.eye(3)

    def toggle_viewport_mode(self):
        self.viewport_mode = not self.viewport_mode
        self.button_panel.app.viewport_button.config(text="Viewport Mode:  ON" if self.viewport_mode else "Viewport Mode: OFF")
    
    def on_key_press(self, event):
        if event.char == 'r':
            self.update_rotation((self.rotation + 90) % 360)
        elif event.char == '-':
            self.global_scale_factor *= 0.9
        elif event.char == '=':
            self.global_scale_factor *= 1.1
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
        elif event.char == ' ':
            self.toggle_viewport_mode()
        self.update_image()

    def on_key_release(self, event):
        self.update_image()
        
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
            self.global_scale_factor = max(self.global_scale_factor * (1.01 ** step), 0.1)
        else:
            if self.rotation_mode:
                self.update_rotation((self.rotation - step)%360)
            elif not self.homography_mode:
                self.update_scale_factor(self.scale_factor * (1.01 ** step))
        self.update_image()

    def on_mouse_press(self, event):
        if not self.check_relevancy(event):
            return
        self.dragging = True
        pt0 = [event.x, event.y]
        pt = apply_homography(np.linalg.inv(self.M_global), pt0)
        if self.homography_mode and not self.viewport_mode:
            min_dist = -1
            closest_anchor = None
            for anchor in self.anchors:
                dist = (anchor.pos[0] - pt[0]) ** 2 + (anchor.pos[1] - pt[1]) ** 2
                if min_dist < 0 or dist < min_dist:
                    min_dist = dist
                    closest_anchor = anchor
            # create new anchor
            new_anchor = Anchor(pt[0], pt[1])
            self.anchors.remove(closest_anchor)
            self.anchors.append(new_anchor)
            self.update_image()
            return

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
        pt = apply_homography(np.linalg.inv(self.M_global), pt0)
        
        if self.homography_mode and not self.viewport_mode:
            self.anchors[-1].move(*pt)
            self.update_image()
            return

        if self.viewport_mode:
            self.global_x_offset += pt0[0] - self.drag_start_x
            self.global_y_offset += pt0[1] - self.drag_start_y
            self.drag_start_x = pt0[0]
            self.drag_start_y = pt0[1]
        else:
            self.x_offset += pt[0] - self.drag_start_x
            self.y_offset += pt[1] - self.drag_start_y
            self.drag_start_x = pt[0]
            self.drag_start_y = pt[1]
        self.update_image()

    def on_mouse_release(self, event):
        self.dragging = False
        if self.homography_mode and not self.viewport_mode:
            self.cur_H = calc_homography(self.anchors) @ self.cur_H
            for anchor in self.anchors:
                anchor.reset()
            self.update_image()
            return

    def exit(self, event=None):
        self.root.quit()
        self.root.destroy()

def main():
    root = tk.Tk()
    root.title("Manual Image Alignment Tool")
    root.geometry(f"{window_dimensions[0]+400}x{window_dimensions[1]}")

    app = ImageAlignerApp(root, images=["input/im1.jpeg", "input/im3.jpeg"])

    root.mainloop()

if __name__ == "__main__":
    main()

