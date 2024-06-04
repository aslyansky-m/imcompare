import cv2
import numpy as np
import types
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import time



from common import PyramidMap

class ImageManipulator:
    def __init__(self, master, image_paths):
        self.master = master
        self.image_paths = image_paths
        self.current_index = 0
        self.x_offset = 0
        self.y_offset = 0
        self.scale_factor = 1.0
        self.rotation = 0
        self.alpha_blending = 0.65
        self.alt_pressed = False
        self.debug_mode = False
        self.contrast_mode = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.dragging = False
        self.slider_active = False
        self.global_x_offset = 0
        self.global_y_offset = 0
        self.global_scale_factor = 1.0
        self.panel = None
        self.window_dimensions = (1500, 900)
        
        self.setup_ui()
        self.load_images()
        self.update_image()

    def setup_ui(self):
        self.master.title("Manual Image Alignment Tool")
        self.master.geometry(f"{self.window_dimensions[0]}x{self.window_dimensions[1]}")
        self.master.configure(background='grey')
        
        self.master.bind('<KeyPress>', self.on_key_press)
        self.master.bind('<KeyRelease>', self.on_key_release)
        self.master.bind('<MouseWheel>', self.on_mouse_wheel)
        self.master.bind('<ButtonPress-1>', self.on_mouse_press)
        self.master.bind("<Button-4>", self.on_mouse_wheel)
        self.master.bind("<Button-5>", self.on_mouse_wheel)
        self.master.bind('<B1-Motion>', self.on_mouse_drag)
        self.master.bind('<ButtonRelease-1>', self.on_mouse_release)
        self.master.bind("<Escape>", self.exit)

        self.panel = tk.Label(self.master)
        self.panel.pack(side="top", fill="both", expand="yes")

        slider_frame = tk.Frame(self.master, bg='grey')
        slider_frame.pack(side="left", fill="y")

        self.alpha_slider = tk.Scale(slider_frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Alpha Blending", command=self.on_alpha_blending_change, bg='grey')
        self.alpha_slider.set(self.alpha_blending)
        self.alpha_slider.bind('<ButtonPress-1>', self.on_slider_click)
        self.alpha_slider.bind('<ButtonRelease-1>', self.on_slider_release)
        self.alpha_slider.pack(side="top", padx=10, pady=5)

        self.rotation_slider = tk.Scale(slider_frame, from_=0, to=360, resolution=1, orient=tk.HORIZONTAL, label="Rotation", command=self.on_rotation_change, bg='grey')
        self.rotation_slider.set(self.rotation)
        self.rotation_slider.bind('<ButtonPress-1>', self.on_slider_click)
        self.rotation_slider.bind('<ButtonRelease-1>', self.on_slider_release)
        self.rotation_slider.pack(side="top", padx=10, pady=5)

    def load_images(self):

        # self.img1 = cv2.imread(self.image_paths, cv2.IMREAD_UNCHANGED)
        # self.img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB)

        self.img = PyramidMap(self.image_paths)

        self.scale_ratio = min(self.window_dimensions[0] / self.img.map_shape[1], self.window_dimensions[1] / self.img.map_shape[0])
        self.M_original = np.diag([self.scale_ratio, self.scale_ratio, 1])


    def update_image(self):

        def calc_transform(shape, scale, rotation, x_offset, y_offset):
            rows, cols = shape[:2]
            M1 = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, scale)
            M2 = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
            M1_3x3 = np.vstack([M1, [0, 0, 1]])
            M2_3x3 = np.vstack([M2, [0, 0, 1]])
            M_combined_3x3 = np.dot(M2_3x3, M1_3x3)
            return M_combined_3x3

        M = calc_transform(np.array(self.img.map_shape), self.scale_factor, self.rotation, self.x_offset/self.scale_ratio, self.y_offset/self.scale_ratio)

        # t1 = time.time()
        # img_cropped = cv2.warpPerspective(self.img1, M  @ self.M_original, self.window_dimensions)

        t1 = time.time()
        img_cropped = self.img.warp_map(self.M_original @ M, self.window_dimensions)

        t2 = time.time()

        print(t2-t1)

        img_pil = Image.fromarray(img_cropped)
        tk_image = ImageTk.PhotoImage(img_pil)
        self.panel.config(image=tk_image)
        self.panel.image = tk_image

    def on_key_press(self, event):
        if event.char == 'r':
            self.rotation = (self.rotation + 90) % 360
        elif event.char == '-':
            self.global_scale_factor *= 0.9
        elif event.char == '=':
            self.global_scale_factor *= 1.1
        elif event.char == 'd':
            self.debug_mode = not self.debug_mode
        elif event.char == 'c':
            self.contrast_mode = not self.contrast_mode
        elif event.keysym == 'Alt_L' or event.keysym == 'Alt_R':
            self.alt_pressed = True
        self.update_image()

    def on_key_release(self, event):
        if event.keysym == 'Alt_L' or event.keysym == 'Alt_R':
            self.alt_pressed = False
        self.update_image()

    def on_mouse_wheel(self, event):
        # if not self.slider_active:
        step_size = 120
        step = event.delta / step_size
        if event.num == 4:
            step = 3
        elif event.num == 5:
            step = -3
        self.scale_factor = max(self.scale_factor * (1.01 ** step), 0.1)
        self.update_image()

    def on_mouse_press(self, event):
        if not self.slider_active:
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            self.dragging = True

    def on_mouse_drag(self, event):
        if self.dragging and not self.slider_active:
            if self.alt_pressed:
                self.global_x_offset += event.x - self.drag_start_x
                self.global_y_offset += event.y - self.drag_start_y
            else:
                self.x_offset += (event.x - self.drag_start_x) / self.global_scale_factor
                self.y_offset += (event.y - self.drag_start_y) / self.global_scale_factor
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            self.update_image()

    def on_mouse_release(self, event):
        self.dragging = False

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

    def exit(self, message=None):
        if message:
            print(message)
        self.master.quit()
        self.master.destroy()

def main():
    root = tk.Tk()
    ImageManipulator(root, "input/hrscd/map.tif")
    root.mainloop()

if __name__ == "__main__":
    main()
