import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

class ImageCompareApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Compare Tool")

        # Initialize variables
        self.image1 = None
        self.image2 = None
        self.alpha = 0.5
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.points = []
        self.homography_mode = False

        # Create GUI components
        self.create_widgets()

    
        file_path1 = "C://Users//maxima//Downloads//planes.jpeg"
        file_path2 = "C://Users//maxima//Downloads//planes2.jpeg"
        self.image1 = cv2.cvtColor(cv2.imread(file_path1), cv2.COLOR_BGR2RGB)
        self.image2 = cv2.cvtColor(cv2.imread(file_path2), cv2.COLOR_BGR2RGB)
        self.update_image()

    def create_widgets(self):
        # Canvas for displaying images
        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack()

        # Alpha slider
        self.alpha_slider = tk.Scale(self.root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Overlay Alpha", command=self.update_alpha)
        self.alpha_slider.set(self.alpha)
        self.alpha_slider.pack()

        # Load images button
        self.load_images_button = tk.Button(self.root, text="Load Images", command=self.load_images)
        self.load_images_button.pack()

        # Bind events
        self.canvas.bind("<MouseWheel>", self.zoom_image)
        self.canvas.bind("<B1-Motion>", self.pan_image)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Control-Button-1>", self.start_homography_mode)

    def load_images(self):
        # Load two images
        file_path1 = filedialog.askopenfilename(title="Select First Image")
        file_path2 = filedialog.askopenfilename(title="Select Second Image")

        if file_path1 and file_path2:
            self.image1 = cv2.cvtColor(cv2.imread(file_path1), cv2.COLOR_BGR2RGB)
            self.image2 = cv2.cvtColor(cv2.imread(file_path2), cv2.COLOR_BGR2RGB)
            self.update_image()

    def update_image(self):
        if self.image1 is None or self.image2 is None:
            return
        # Blend the two images using the alpha value
        blended_image = cv2.addWeighted(self.image1, self.alpha, self.image2, 1 - self.alpha, 0)

        # Apply zoom and pan transformations
        height, width, _ = blended_image.shape
        M = np.float32([[self.zoom, 0, self.offset_x], [0, self.zoom, self.offset_y]])
        transformed_image = cv2.warpAffine(blended_image, M, (width, height))

        # Convert to PIL Image and display on canvas
        img = Image.fromarray(transformed_image)
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def update_alpha(self, val):
        self.alpha = float(val)
        self.update_image()

    def zoom_image(self, event):
        self.zoom *= 1.1 if event.delta > 0 else 0.9
        self.update_image()

    def pan_image(self, event):
        self.offset_x += event.x - self.canvas.canvasx(0)
        self.offset_y += event.y - self.canvas.canvasy(0)
        self.update_image()

    def on_click(self, event):
        if self.homography_mode:
            self.points.append((event.x, event.y))
            if len(self.points) == 4:
                self.apply_homography()
                self.homography_mode = False
                self.points = []

    def start_homography_mode(self, event):
        self.homography_mode = True
        self.points = []

    def apply_homography(self):
        if len(self.points) != 4:
            return

        pts1 = np.float32(self.points)
        pts2 = np.float32([[0, 0], [self.image1.shape[1], 0], [self.image1.shape[1], self.image1.shape[0]], [0, self.image1.shape[0]]])
        H, _ = cv2.findHomography(pts1, pts2)
        self.image1 = cv2.warpPerspective(self.image1, H, (self.image1.shape[1], self.image1.shape[0]))
        self.update_image()

# Create the main window
root = tk.Tk()
app = ImageCompareApp(root)
root.mainloop()