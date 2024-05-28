import tkinter as tk
from PIL import Image, ImageTk

# Global variables for image manipulation
anchors = [(50, 50), (50, 250), (250, 50), (250, 250)]  # Initial anchor points
new_anchor = None  # Position of new anchor point
selected_anchor = None  # Index of the selected anchor point
ctrl_pressed = False  # Flag to track if Ctrl key is pressed

i = 0

def draw_anchors():

    for anchor in anchors:
        canvas.create_oval(anchor[0] - 5, anchor[1] - 5, anchor[0] + 5, anchor[1] + 5, fill='green')  # Green circle for existing anchors
    if new_anchor:
        canvas.create_oval(new_anchor[0] - 5, new_anchor[1] - 5, new_anchor[0] + 5, new_anchor[1] + 5, fill='blue')  # Blue circle for new anchor
    if selected_anchor is not None:
        canvas.create_oval(anchors[selected_anchor][0] - 5, anchors[selected_anchor][1] - 5, anchors[selected_anchor][0] + 5, anchors[selected_anchor][1] + 5, fill='red')  # Red circle for selected anchor

def on_key_press(event):
    global ctrl_pressed
    if event.keysym == 'Control_L' or event.keysym == 'Control_R':
        ctrl_pressed = True

def on_key_release(event):
    global ctrl_pressed
    if event.keysym == 'Control_L' or event.keysym == 'Control_R':
        ctrl_pressed = False

def on_mouse_press(event):
    global selected_anchor
    if selected_anchor is not None:  # If an anchor is selected, deselect it
        selected_anchor = None
    else:  # Check if the cursor is near any existing anchor
        for i, anchor in enumerate(anchors):
            if abs(anchor[0] - event.x) <= 5 and abs(anchor[1] - event.y) <= 5:
                selected_anchor = i
                break
    update_image()

def on_mouse_drag(event):
    global i
    i+=1
    print(i)
    move_new_anchor(event.x, event.y)

def on_mouse_release(event):
    global selected_anchor, new_anchor
    if selected_anchor is not None:  # If an anchor was selected, move it to cursor position
        anchors[selected_anchor] = (event.x, event.y)
        selected_anchor = None
        update_image()
    elif ctrl_pressed and new_anchor:  # If Ctrl is pressed and a new anchor exists, place it at cursor position
        new_anchor = None  # Clear the new anchor
        update_image()

def create_new_anchor(x, y):
    global new_anchor
    new_anchor = (x, y)  # Set the position of the new anchor
    update_image()

def move_new_anchor(x, y):
    global new_anchor
    # if new_anchor is not None:
    new_anchor = (x, y)  # Update the position of the new anchor
    update_image()

def update_image():
    canvas.delete('all')  # Clear the canvas
    draw_anchors()

# Set up the TKinter window
window = tk.Tk()
window.title("Anchor Points Example")

# Canvas to display circles
canvas = tk.Canvas(window, width=300, height=300, bg='white')
canvas.pack()

# Bind events
window.bind('<KeyPress>', on_key_press)
window.bind('<KeyRelease>', on_key_release)
window.bind('<ButtonPress-1>', on_mouse_press)
window.bind('<B1-Motion>', on_mouse_drag)
window.bind('<ButtonRelease-1>', on_mouse_release)

# Main loop
window.mainloop()
