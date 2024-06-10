# credit goes to Maksym Aslianskyi, ChatGPT, GitHub Copilot and StackOverflow

import os
import cv2
import numpy as np
import types
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

screen_size = (1080, 700)
window_size = screen_size
SCREEN_FACTOR = 0.7
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
            color = 'green'
            canvas.create_line(pos0_t[0], pos0_t[1], pos_t[0], pos_t[1], fill=color, width=2)
            r0 = 3
            canvas.create_oval(pos0_t[0] - r0, pos0_t[1] - r0, pos0_t[0] + r0, pos0_t[1] + r0, fill='yellow')
            color = 'red' if self.moved else 'blue'
            canvas.create_oval(pos_t[0] - r, pos_t[1] - r, pos_t[0] + r, pos_t[1] + r, fill=color)
            

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

        self.scale_ratio = min(window_size[0] / self.img1.shape[1], window_size[1] / self.img1.shape[0])
        self.M_original = np.diag([self.scale_ratio, self.scale_ratio, 1])
        self.reset_anchors()



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


    def render(self, app):
        if not self.valid:
            return np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)

        M = calc_transform([self.img2.shape[1] * self.scale_ratio, self.img2.shape[0] * self.scale_ratio], self.scale, self.rotation, self.x_offset, self.y_offset)
        H = calc_homography(self.anchors)

        im1 = cv2.warpPerspective(self.img1, self.M_original, window_size)
        im2 = cv2.warpPerspective(self.img2, H @ self.M_anchors @ M @ self.M_original, window_size)

        blend_image = cv2.addWeighted(im1, 1 - app.alpha, im2, app.alpha, 0)

        return blend_image

    def push_anchor(self, pt):
        min_dist = 100
        closest_anchor = None
        for anchor in self.anchors:
            dist = np.sqrt((anchor.pos[0] - pt[0]) ** 2 + (anchor.pos[1] - pt[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_anchor = anchor
        new_anchor = Anchor(pt[0], pt[1])
        if closest_anchor:
            self.anchors.remove(closest_anchor)
        
        self.anchors.append(new_anchor)

    def relative_transform(self):
        M = calc_transform([self.img2.shape[1] * self.scale_ratio, self.img2.shape[0] * self.scale_ratio], self.scale, self.rotation, self.x_offset, self.y_offset)
        H = calc_homography(self.anchors)
        T = H @ self.M_anchors @ M
        return T

    def __str__(self):
        T = self.relative_transform()
        return f"{self.img1_path}, {self.img2_path}, {','.join([str(x) for x in T.flatten().tolist()])}"
  
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
        
        buttons = [
            self.alpha_slider,
        ]
        for i, b in enumerate(buttons):
            b.grid(row=i, column=0, sticky='ew')

        self.frame.grid_columnconfigure(0, weight=1)
    

class ImageAlignerApp:
    def __init__(self, root, image_pair=None):
        if not 'root' in dir(self):
            self.root = root
        
        if image_pair is None:
            image_pair = ImagePair("", "")

        self.images = image_pair
        self.alpha = 0.75
    
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0


        if not 'button_panel' in dir(self):
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
        self.root.bind("<Escape>", self.exit)
    
        
    
    def render(self, update_state=True):
        rendered_image = self.images.render(self)
        img_pil = Image.fromarray(rendered_image)
        self.tk_image = ImageTk.PhotoImage(img_pil)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        for anchor in self.images.anchors:
            anchor.plot(self.canvas, np.eye(3))


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
        #TODO
        pass

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
        pt = pt0 #apply_homography(np.linalg.inv(self.M_global()), pt0)

        self.images.push_anchor(pt)
        self.render()
            

    def on_mouse_drag(self, event):
        if not self.dragging:
            return
        
        pt0 = [event.x, event.y]
        pt = pt0 #apply_homography(np.linalg.inv(self.M_global()), pt0)
        self.images.anchors[-1].move(*pt)
        self.render()

    def on_mouse_release(self, event):
        self.dragging = False
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

    image_pair = ImagePair("input/im2.png", "input/im1.png")
    
    app = ImageAlignerApp(root, image_pair if DEBUG else None)

    root.mainloop()

if __name__ == "__main__":
    main()
