import cv2
import numpy as np
import types
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


window_dimensions = (1500, 900)

# Global variables for image manipulation
label_widgets = []
x_offset = 0
y_offset = 0
scale_factor = 1.0
rotation = 0
alpha_blending = 0.25
next_image = False
drag_start_x = 0
drag_start_y = 0
dragging = False
slider_active = False
viewport_mode = False
global_x_offset = 0
global_y_offset = 0
global_scale_factor = 1.0
debug_window = None
debug_frame = None
debug_mode = False
contrast_mode = False
homography_mode = False
toggle = False
anchors = []
cur_H = np.eye(3)
M_global = np.eye(3)
debug_button = None
homography_button = None
switch_button = None
viewport_button = None
contrast_button = None
homography_reset_button = None


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
        
    def plot(self, canvas):
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

def calc_homography(anchors):
    pts0 = np.array([anchor.pos0 for anchor in anchors], dtype=np.float32)
    pts1 = np.array([anchor.pos for anchor in anchors], dtype=np.float32)
    H, _ = cv2.findHomography(pts0, pts1)
    return H

def apply_homography(H, point):
    pt = np.array([point[0], point[1], 1])
    pt = H @ pt
    pt = pt[:2]/pt[2]
    return pt

def draw_anchors():
    for anchor in anchors:
        anchor.plot(canvas)
        
def reset_anchors():
    global anchors
    w = window_dimensions[0]
    h = window_dimensions[1]
    m = 100

    anchors_pos = [(m,m), (m, h-m), (w-m, m), (w-m, h-m)]
    anchors = []
    for pos in anchors_pos:
        anchors.append(Anchor(pos[0], pos[1], original=True))

def calc_transform(shape, scale, rotation, x_offset, y_offset):
    rows, cols = shape[:2]
    M1 = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, scale)
    M2 = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
    M1 = np.vstack([M1, [0, 0, 1]])
    M2 = np.vstack([M2, [0, 0, 1]])
    M = np.dot(M2,M1)
    return M

def toggle_debug_mode():
    global debug_mode, debug_button
    debug_mode = not debug_mode
    update_image()
    debug_button.config(text="Debug Mode:  ON" if debug_mode else "Debug Mode: OFF")
    
def toggle_homography_mode():
    global homography_mode, homography_button
    homography_mode = not homography_mode
    update_image()
    homography_button.config(text="Homography Mode:  ON" if homography_mode else "Homography Mode: OFF")


def toggle_contrast_mode():
    global contrast_mode, contrast_button
    contrast_mode = not contrast_mode
    update_image()
    contrast_button.config(text="Contrast Mode:  ON" if contrast_mode else "Contrast Mode: OFF")

def toggle_switch():
    global toggle, switch_button
    toggle = not toggle
    update_image()
    switch_button.config(text="Switch:  ON" if toggle else "Switch: OFF")
    
def reset_homography():
    global cur_H
    cur_H = np.eye(3)

def toggle_viewport_mode():
    global viewport_mode, viewport_button
    viewport_mode = not viewport_mode
    viewport_button.config(text="Viewport Mode:  ON" if viewport_mode else "Viewport Mode: OFF")
    

def on_key_press(event):
    global rotation, global_scale_factor, debug_mode, contrast_mode, homography_mode, viewport_mode, toggle, cur_H
    if event.char == 'r':
        rotation = (rotation + 90) % 360
    elif event.char == '-':
        global_scale_factor *= 0.9
    elif event.char == '=':
        global_scale_factor *= 1.1
    elif event.char == 'd':
        toggle_debug_mode()
    elif event.char == 'c':
        toggle_contrast_mode()
    elif event.char == 't':
        toggle_switch()
    elif event.char == 'h':
        toggle_homography_mode()
    elif event.char == 'o':
        reset_homography()
    elif event.keysym == 'Alt_L' or event.keysym == 'Alt_R':
        toggle_viewport_mode()
    update_image()

def on_key_release(event):
    # if event.keysym == 'Alt_L' or event.keysym == 'Alt_R':
    #     viewport_mode = False
    update_image()
    
def on_right_click(event):
    toggle_switch()

def on_mouse_wheel(event):
    global scale_factor, slider_active, global_scale_factor, homography_mode
    if not slider_active:
        step_size = 120
        step = event.delta / step_size
        if event.num == 4:
            step = 3
        elif event.num == 5:
            step = -3
        if viewport_mode:
            global_scale_factor = max(global_scale_factor*(1.01 ** step), 0.1)
        else: 
            if not homography_mode:
                scale_factor = max(scale_factor*(1.01 ** step), 0.1)
        update_image()

def on_mouse_press(event):
    global drag_start_x, drag_start_y, dragging, slider_active
    if slider_active:
        return
    
    pt0 = [event.x, event.y]
    pt = apply_homography(np.linalg.inv(M_global),pt0)
    if homography_mode and not viewport_mode:
        global anchors
        min_dist = -1
        closest_anchor = None
        for anchor in anchors:
            dist = (anchor.pos[0] - pt[0])**2 + (anchor.pos[1] - pt[1])**2
            if min_dist < 0 or dist < min_dist:
                min_dist = dist
                closest_anchor = anchor
        # create new anchor
        new_anchor = Anchor(pt[0], pt[1])
        anchors.remove(closest_anchor)
        anchors.append(new_anchor)
        update_image()
        return
        
    if viewport_mode:
        drag_start_x = pt0[0]
        drag_start_y = pt0[1]
    else:   
        drag_start_x = pt[0]
        drag_start_y = pt[1]
    dragging = True

def on_mouse_drag(event):
    global x_offset, y_offset, drag_start_x, drag_start_y, dragging, slider_active, viewport_mode, global_x_offset, global_y_offset, global_scale_factor, M_global

    if slider_active:
        return

    pt0 = [event.x, event.y]
    pt = apply_homography(np.linalg.inv(M_global),pt0)

    if homography_mode and not viewport_mode:
        global anchors
        anchors[-1].move(*pt)
        update_image()
        return
    
    if dragging:
        if viewport_mode:
            global_x_offset += pt0[0] - drag_start_x
            global_y_offset += pt0[1] - drag_start_y
            drag_start_x = pt0[0]
            drag_start_y = pt0[1]
        else:
            x_offset += pt[0] - drag_start_x
            y_offset += pt[1] - drag_start_y
            drag_start_x = pt[0]
            drag_start_y = pt[1]
        update_image()

def on_mouse_release(event):
    global dragging
    if homography_mode and not viewport_mode:
        global anchors, cur_H
        cur_H = calc_homography(anchors) @ cur_H
        for anchor in anchors:
            anchor.reset()  
        update_image()
        return
    dragging = False

def show_debug_info():
    global debug_window, debug_frame, label_widgets
    
    if debug_window is None:
        debug_window = tk.Toplevel()
        debug_window.title("Debug Information")
    
    if debug_frame is None:
        debug_frame = tk.Frame(debug_window, bg='grey')
        debug_frame.pack(side="left", fill="y")
        tk.Label(debug_frame, text="Debug Information", font=('Helvetica', 16, 'bold'), bg='grey').pack(side="top", pady=10)

    # Remove existing labels
    for label in label_widgets:
        label.destroy()
    label_widgets.clear()
    
    # Display all global variables
    globals_dict = globals()
    for var_name, var_value in globals_dict.items():
        if not var_name.startswith("__") and not callable(var_value) \
            and not isinstance(var_value, types.ModuleType) and not isinstance(var_value, np.ndarray):
            label = tk.Label(debug_frame, text=f"{var_name}: {var_value}", bg='grey')
            label.pack(side="top", anchor="w", padx=10, pady=5)
            label_widgets.append(label)

def update_image():
    global img1, img2, x_offset, y_offset, scale_factor, rotation, alpha_blending, viewport_mode
    global tk_image, anchors, homography_mode, toggle, M_global

    if img1 is None or img2 is None:
        return
    
    im1 = img1.copy()
    im2 = img2.copy()
    
    M = calc_transform(im2.shape, scale_factor, rotation, x_offset, y_offset)
    H = calc_homography(anchors)
    im2 = cv2.warpPerspective(im2, H @ cur_H @ M, (im2.shape[1], im2.shape[0]))
    

    if toggle:
        [im1, im2] = [im2, im1]
            
    if contrast_mode:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        blend_image = np.stack([im1, im2, im1], axis=-1)
    else:
        blend_image = cv2.addWeighted(im1, 1 - alpha_blending, im2, alpha_blending, 0)
        
    
    M_global = calc_transform(blend_image.shape, global_scale_factor, 0, global_x_offset, global_y_offset)
    blend_image = cv2.warpPerspective(blend_image, M_global, (blend_image.shape[1], blend_image.shape[0]))
    
    if viewport_mode:
        blend_image = (blend_image*0.8).astype(np.uint8)
    
    # Convert to ImageTk format and update canvas
    img_rgb = cv2.cvtColor(blend_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    tk_image = ImageTk.PhotoImage(img_pil)
    canvas.delete('all')
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)

    
    if homography_mode:
        draw_anchors()
    
    if debug_mode:
        show_debug_info()
    

def on_alpha_blending_change(val):
    global alpha_blending, slider_active
    alpha_blending = float(val)
    update_image()

def on_rotation_change(val):
    global rotation, slider_active
    rotation = int(val)
    update_image()

def on_slider_click(event):
    global slider_active
    slider_active = True

def on_slider_release(event):
    global slider_active
    slider_active = False

def main():
    global img1, img2, canvas, window_dimensions
    global x_offset, y_offset, scale_factor, rotation, anchors

    # Set up the TKinter window
    window = tk.Tk()
    window.title("Manual Image Alignment Tool")
    window.geometry(f"{window_dimensions[0]+200}x{window_dimensions[1]}")
    # window.configure(background='grey')
    
    def exit(event):
        window.quit()
        window.destroy()

    # Set up a key press to trigger a callback
    window.bind('<KeyPress>', on_key_press)
    window.bind('<KeyRelease>', on_key_release)
    window.bind('<MouseWheel>', on_mouse_wheel)
    window.bind("<Button-2>", on_right_click)
    window.bind("<Button-3>", on_right_click)
    window.bind("<Button-4>", on_mouse_wheel)
    window.bind("<Button-5>", on_mouse_wheel)
    window.bind('<ButtonPress-1>', on_mouse_press)
    window.bind('<B1-Motion>', on_mouse_drag)
    window.bind('<ButtonRelease-1>', on_mouse_release)
    window.bind("<Escape>", exit)

    # Ask the user to select images
    # image_paths = filedialog.askopenfilenames(title='Select images to align...')
    image_paths = ["input/im1.jpeg", "input/im2.jpeg"]

    if len(image_paths) < 2:
        exit("Error: Select at least two images")

    # Ask the user for a destination directory
    # output_folder = filedialog.askdirectory(title='Select output folder...')
    output_folder = 'output/'


    # Frame for sliders
    slider_frame = tk.Frame(window, bg='grey')
    slider_frame.pack(side="left", fill="y")

    # Alpha blending slider
    alpha_slider = tk.Scale(slider_frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Alpha Blending", command=on_alpha_blending_change, bg='grey')
    alpha_slider.set(alpha_blending)
    alpha_slider.bind('<ButtonPress-1>', on_slider_click)
    alpha_slider.bind('<ButtonRelease-1>', on_slider_release)

    # Rotation slider
    rotation_slider = tk.Scale(slider_frame, from_=0, to=360, resolution=1, orient=tk.HORIZONTAL, label="Rotation", command=on_rotation_change, bg='grey')
    rotation_slider.set(rotation)
    rotation_slider.bind('<ButtonPress-1>', on_slider_click)
    rotation_slider.bind('<ButtonRelease-1>', on_slider_release)

    global debug_button, homography_button, viewport_button, contrast_button, switch_button, homography_reset_button
    

    homography_button = tk.Button(slider_frame, text="Homography Mode: OFF", command=toggle_homography_mode, bg='grey')
    homography_reset_button = tk.Button(slider_frame, text="Reset Homography", command=reset_homography, bg='grey')
    viewport_button = tk.Button(slider_frame, text="Viewport Mode: OFF", command=toggle_viewport_mode, bg='grey')
    contrast_button = tk.Button(slider_frame, text="Contrast Mode: OFF", command=toggle_contrast_mode, bg='grey')
    switch_button = tk.Button(slider_frame, text="Switch: OFF", command=toggle_switch, bg='grey')
    debug_button = tk.Button(slider_frame, text="Debug Mode: OFF", command=toggle_debug_mode, bg='grey')
    
    alpha_slider.grid(row=0, column=0, sticky='ew')
    rotation_slider.grid(row=1, column=0, sticky='ew')
    homography_button.grid(row=2, column=0, sticky='ew')
    homography_reset_button.grid(row=3, column=0, sticky='ew')
    viewport_button.grid(row=4, column=0, sticky='ew')
    contrast_button.grid(row=5, column=0, sticky='ew')
    switch_button.grid(row=6, column=0, sticky='ew')
    debug_button.grid(row=7, column=0, sticky='ew')

    # Configure the column to have the same width
    slider_frame.grid_columnconfigure(0, weight=1)
        
    
    # Call the initial update_image
    img1 = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(image_paths[1], cv2.IMREAD_UNCHANGED)
    
    scale_ratio = min(window_dimensions[0]/img1.shape[1], window_dimensions[1]/img1.shape[0])
    img1 = cv2.resize(img1, (0,0), fx=scale_ratio, fy=scale_ratio)
    img2 = cv2.resize(img2, (0,0), fx=scale_ratio, fy=scale_ratio)
    
    reset_anchors()

    
    # canvas to display images
    canvas = tk.Canvas(window, width=window_dimensions[0], height=window_dimensions[1])
    img_pil = Image.fromarray(img1)
    tk_image = ImageTk.PhotoImage(img_pil)
    image_container = canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
    canvas.pack()

    x_offset, y_offset, scale_factor, rotation = 0, 0, 1.0, 0

    update_image()

    # Main loop
    window.mainloop()

if __name__ == "__main__":
    main()
