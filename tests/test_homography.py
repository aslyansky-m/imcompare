import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import cv2

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
        if self.original:
            canvas.create_oval(self.pos[0] - r, self.pos[1] - r, self.pos[0] + r, self.pos[1] + r, fill='green')
        else:
            r0 = 3
            canvas.create_oval(self.pos0[0] - r0, self.pos0[1] - r0, self.pos0[0] + r0, self.pos0[1] + r0, fill='yellow')
            color = 'red' if self.moved else 'blue'
            canvas.create_oval(self.pos[0] - r, self.pos[1] - r, self.pos[0] + r, self.pos[1] + r, fill=color)

def calc_homography(anchors):
    pts0 = np.array([anchor.pos0 for anchor in anchors], dtype=np.float32)
    pts1 = np.array([anchor.pos for anchor in anchors], dtype=np.float32)
    H, _ = cv2.findHomography(pts0, pts1)
    return H

# Global variables for image manipulation

cur_H = np.eye(3)
ctrl_pressed = False  # Flag to track if Ctrl key is pressed
raw_image = cv2.cvtColor(cv2.imread('input/im1.jpeg'), cv2.COLOR_BGR2RGB)

w = raw_image.shape[1]
h = raw_image.shape[0]

anchors = []
anchors_pos = [(50, 50), (50, h-50), (w-50, 50), (w-50, h-50)]
for pos in anchors_pos:
    anchors.append(Anchor(pos[0], pos[1], original=True))

def draw_anchors():
    global anchors
    for anchor in anchors:
        anchor.plot(canvas)

def on_key_press(event):
    global ctrl_pressed
    if event.keysym == 'Control_L' or event.keysym == 'Control_R':
        ctrl_pressed = True

def on_key_release(event):
    global ctrl_pressed
    if event.keysym == 'Control_L' or event.keysym == 'Control_R':
        ctrl_pressed = False

def on_mouse_press(event):
    global anchors
    min_dist = -1
    closest_anchor = None
    for anchor in anchors:
        dist = (anchor.pos[0] - event.x)**2 + (anchor.pos[1] - event.y)**2
        if min_dist < 0 or dist < min_dist:
            min_dist = dist
            closest_anchor = anchor
    # create new anchor
    new_anchor = Anchor(event.x, event.y)
    anchors.remove(closest_anchor)
    anchors.append(new_anchor)

    update_image()

def on_mouse_drag(event):
    global anchors
    anchors[-1].move(event.x, event.y)
    update_image()

def on_mouse_release(event):
    global anchors, cur_H
    cur_H = calc_homography(anchors) @ cur_H
    for anchor in anchors:
        anchor.reset()  
    update_image()

def update_image():
    global pil_image, image

    H = calc_homography(anchors)

    img_warped = cv2.warpPerspective(raw_image, H @ cur_H, (w, h))
    img_pil = Image.fromarray(img_warped)
    image = ImageTk.PhotoImage(img_pil)
    canvas.delete('all')
    canvas.create_image(0, 0, anchor=tk.NW, image=image)
    draw_anchors()
    




# Set up the TKinter window
window = tk.Tk()
window.title("Anchor Points Example")

# Load the image and display it on the canvas
pil_image = Image.open("input/im1.jpeg")
image = ImageTk.PhotoImage(pil_image)
canvas = tk.Canvas(window, width=w, height=h)
image_container = canvas.create_image(0, 0, anchor=tk.NW, image=image)
canvas.pack()

# Bind events
window.bind('<KeyPress>', on_key_press)
window.bind('<KeyRelease>', on_key_release)
window.bind('<ButtonPress-1>', on_mouse_press)
window.bind('<B1-Motion>', on_mouse_drag)
window.bind('<ButtonRelease-1>', on_mouse_release)
update_image()

# Main loop
window.mainloop()
