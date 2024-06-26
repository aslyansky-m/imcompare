import cv2
import numpy as np
import types
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


# Global variables for image manipulation
x_offset = 0
y_offset = 0
scale_factor = 1.0
rotation = 0
alpha_blending = 0.65
next_image = False
drag_start_x = 0
drag_start_y = 0
dragging = False
slider_active = False
alt_pressed = False
global_x_offset = 0
global_y_offset = 0
global_scale_factor = 1.0
debug_window = None
debug_frame = None
debug_mode = False
contrast_mode = False

def on_key_press(event):
    global next_image, rotation, alt_pressed, global_scale_factor, debug_mode, contrast_mode
    if event.char == 'r':
        rotation = (rotation + 90) % 360
    elif event.char == '-':
        global_scale_factor *= 0.9
    elif event.char == '=':
        global_scale_factor *= 1.1
    elif event.char == 'd':
        debug_mode = not debug_mode
    elif event.char == 'c':
        contrast_mode = not contrast_mode
    elif event.keysym == 'Alt_L' or event.keysym == 'Alt_R':
        alt_pressed = True
    update_image()

def on_key_release(event):
    global alt_pressed
    if event.keysym == 'Alt_L' or event.keysym == 'Alt_R':
        alt_pressed = False
    update_image()

def on_mouse_wheel(event):
    global scale_factor, slider_active
    if not slider_active:
        step_size = 120
        step = event.delta / step_size
        scale_factor = max(scale_factor*(1.01 ** step), 0.1)
        update_image()

def on_mouse_press(event):
    global drag_start_x, drag_start_y, dragging, slider_active
    if not slider_active:
        drag_start_x = event.x
        drag_start_y = event.y
        dragging = True

def on_mouse_drag(event):
    global x_offset, y_offset, drag_start_x, drag_start_y, dragging, slider_active, alt_pressed, global_x_offset, global_y_offset, global_scale_factor
    if dragging and not slider_active:
        if alt_pressed:
            global_x_offset += event.x - drag_start_x
            global_y_offset += event.y - drag_start_y
        else:
            x_offset += (event.x - drag_start_x)/global_scale_factor
            y_offset += (event.y - drag_start_y)/global_scale_factor
        drag_start_x = event.x
        drag_start_y = event.y
        update_image()

def on_mouse_release(event):
    global dragging
    dragging = False

def show_debug_info():
    global debug_window, debug_frame
    if debug_window is None:
        debug_window = tk.Toplevel()
        debug_window.title("Debug Information")
    
    if debug_frame is not None:
        debug_frame.destroy()

    debug_frame = tk.Frame(debug_window, bg='grey')
    debug_frame.pack(side="left", fill="y")

    tk.Label(debug_frame, text="Debug Information", font=('Helvetica', 16, 'bold'), bg='grey').pack(side="top", pady=10)

    # Display all global variables
    globals_dict = globals()
    for var_name, var_value in globals_dict.items():
        if not var_name.startswith("__") and not callable(var_value) \
            and not isinstance(var_value, types.ModuleType) and not isinstance(var_value, np.ndarray):
            tk.Label(debug_frame, text=f"{var_name}: {var_value}", bg='grey').pack(side="top", anchor="w", padx=10, pady=5)

def update_image():
    global img1, img2, x_offset, y_offset, scale_factor, rotation, alpha_blending, alt_pressed

    if img1 is None or img2 is None:
        return
    
    def calc_transform(shape, scale, rotation, x_offset, y_offset):
        rows, cols = shape[:2]
        M1 = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, scale)
        M2 = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
        M1_3x3 = np.vstack([M1, [0, 0, 1]])
        M2_3x3 = np.vstack([M2, [0, 0, 1]])
        M_combined_3x3 = np.dot(M2_3x3,M1_3x3)
        M_combined = M_combined_3x3[:2, :]
        return M_combined
    
    M = calc_transform(img2.shape, scale_factor, rotation, x_offset, y_offset)
    img2_cropped = cv2.warpAffine(img2, M, (img2.shape[1],img2.shape[0]))
    if contrast_mode:
        img_a = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img_b = cv2.cvtColor(img2_cropped, cv2.COLOR_BGR2GRAY)
        blend_image = np.stack([img_a, img_b, img_b], axis=-1)
    else:
        blend_image = cv2.addWeighted(img1, 1 - alpha_blending, img2_cropped, alpha_blending, 0)
    
    M_global = calc_transform(blend_image.shape, global_scale_factor, 0, global_x_offset, global_y_offset)
    blend_image = cv2.warpAffine(blend_image, M_global, (blend_image.shape[1], blend_image.shape[0]))
    
    if alt_pressed:
        blend_image = (blend_image*0.8).astype(np.uint8)
    
    # Convert to ImageTk format and update panel
    img_rgb = cv2.cvtColor(blend_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    tk_image = ImageTk.PhotoImage(img_pil)
    panel.config(image=tk_image)
    panel.image = tk_image
    
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
    global img1, img2, panel, window_dimensions
    global x_offset, y_offset, scale_factor, rotation, next_image

     # Set up the TKinter window
    window_dimensions = (1500, 900)
    window = tk.Tk()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    print(screen_width, screen_height)
    window.title("Manual Image Alignment Tool")
    window.geometry(f"{window_dimensions[0]}x{window_dimensions[1]}")
    window.configure(background='grey')
    
    def exit(event):
        window.quit()
        window.destroy()

    # Set up a key press to trigger a callback
    window.bind('<KeyPress>', on_key_press)
    window.bind('<KeyRelease>', on_key_release)
    window.bind('<MouseWheel>', on_mouse_wheel)
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

    # Panel to display images
    panel = tk.Label(window)
    panel.pack(side="top", fill="both", expand="yes")

    # Frame for sliders
    slider_frame = tk.Frame(window, bg='grey')
    slider_frame.pack(side="left", fill="y")

    # Alpha blending slider
    alpha_slider = tk.Scale(slider_frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Alpha Blending", command=on_alpha_blending_change, bg='grey')
    alpha_slider.set(alpha_blending)
    alpha_slider.bind('<ButtonPress-1>', on_slider_click)
    alpha_slider.bind('<ButtonRelease-1>', on_slider_release)
    alpha_slider.pack(side="top", padx=10, pady=5)

    # Rotation slider
    rotation_slider = tk.Scale(slider_frame, from_=0, to=360, resolution=1, orient=tk.HORIZONTAL, label="Rotation", command=on_rotation_change, bg='grey')
    rotation_slider.set(rotation)
    rotation_slider.bind('<ButtonPress-1>', on_slider_click)
    rotation_slider.bind('<ButtonRelease-1>', on_slider_release)
    rotation_slider.pack(side="top", padx=10, pady=5)

    # Call the initial update_image
    img1 = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(image_paths[1], cv2.IMREAD_UNCHANGED)
    
    scale_ratio = min(window_dimensions[0]/img1.shape[1], window_dimensions[1]/img1.shape[0])
    img1 = cv2.resize(img1, (0,0), fx=scale_ratio, fy=scale_ratio)
    img2 = cv2.resize(img2, (0,0), fx=scale_ratio, fy=scale_ratio)

    x_offset, y_offset, scale_factor, rotation = 0, 0, 1.0, 0

    update_image()

    # Main loop
    window.mainloop()

if __name__ == "__main__":
    main()
