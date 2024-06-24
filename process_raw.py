import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import time
import rasterio
from rasterio.windows import Window
from common import gps2enu, enu2gps, gps2pix, pix2gps
import subprocess
import shutil
import pandas as pd
from PIL import Image, ImageTk

def fix_name(filename):
    name = os.path.basename(filename).split('.')[0]
    num = [int(s) for s in name.split('_')]
    new_name = f"{num[0]:03d}_{num[1]:03d}"
    return filename.replace(name, new_name)

def process_tiff(path):
    im = Image.open(path)
    im = np.array(im)
    maximum = np.max(im)
    crop_max = np.where(im == maximum)[1].max()
    im = im[:,6:]
    if crop_max != 5:
        m = np.mean(im)
        s = np.std(im)
        p = 4
        th_low = max(0, m - p*s)
        th_high = min(pow(2,16)-1, m + p*s)
        im = np.clip(im, th_low, th_high)
    
    im -= np.min(im)
    im = im/np.max(im)
    im = (im*255).astype(np.uint8)
        
    return im

def get_image(path):
    if path.endswith('.tiff'):
        im = process_tiff(path)
    else:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return im
    

def track_frames(im0, im1):
    detection_params = dict(maxCorners=1000,qualityLevel=0.0001,minDistance=12,blockSize=11)
    tracking_params = dict(winSize=(8, 8),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.03),minEigThreshold=5e-5)
    bidirectional_thresh=2.0
    affine_params = dict(ransacReprojThreshold=2,maxIters=2000,confidence=0.999,refineIters=10)
    
    p0 = cv2.goodFeaturesToTrack(im0, mask=None, **detection_params)

    p1, st1, err1 = cv2.calcOpticalFlowPyrLK(im0, im1, p0, None, **tracking_params)
    if bidirectional_thresh > 0:
        p2, st2, err2 = cv2.calcOpticalFlowPyrLK(im1, im0, p1, None, **tracking_params)
        proj_err = np.linalg.norm(np.squeeze(p2 - p0), axis=1)
        st = np.squeeze(st1 * st2) * (proj_err < bidirectional_thresh).astype(np.uint8)
    else:
        st = np.squeeze(st1)
    p0 = np.squeeze(p0)[st == 1]
    p1 = np.squeeze(p1)[st == 1]
    
    if len(p0) > 10:
        H, mask = cv2.estimateAffinePartial2D(p0, p1, **affine_params)
        H = np.vstack([H, [0, 0, 1]])
        if H is None:
            H = np.eye(3)
    else:
        H = np.eye(3)
    return H

def select_keyframes(Hs):
    angles = []
    scales = []
    shifts = []
    for H in Hs:
        angle = np.arctan2(H[1,0], H[0,0])
        scale = np.sqrt(np.linalg.det(H[:2,:2]))
        shift = H[:2,2]
        angles.append(angle)
        scales.append(scale)
        shifts.append(shift)
    
    scales = np.array(scales)
    angles = np.unwrap(np.array(angles))
    shifts = np.array(shifts)
    
    first = 0
    kfs = [first]
    
    for i in range(first+1, len(angles)):
        to_append = False
        if i - kfs[-1] > 30:
            to_append = True
        if np.abs(np.log(scales[i]/scales[kfs[-1]])) > np.log(1.2): 
            to_append = True
        if np.linalg.norm((shifts[i] - shifts[kfs[-1]])) > 100:
            to_append = True
        if to_append:
            kfs.append(i)
            
    kfs = np.array(kfs)
    
    return kfs, scales, angles, shifts


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.configure_widgets()
        
        self.boundaries = None
        self.crop = None
        self.selected_frames = None
        self.saved_frames = None
        self.saved_map = None

    def configure_widgets(self):
        root = self.root
        self.root.title("TagIm Pre-Processing")
        # Ensure the app exits when the window is closed
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.bind("<Escape>", self.on_closing)
        
        # Folder selection for images
        self.image_folder_label = tk.Label(root, text="Select Image Folder:")
        self.image_folder_label.grid(row=0, column=0, padx=10, pady=5, sticky='e')
        self.image_folder_path = tk.StringVar()
        self.image_folder_entry = tk.Entry(root, textvariable=self.image_folder_path, width=40)
        self.image_folder_entry.grid(row=0, column=1, padx=10, pady=5, sticky='w')
        self.image_folder_button = tk.Button(root, text="Browse", command=self.browse_image_folder)
        self.image_folder_button.grid(row=0, column=2, padx=10, pady=5)

        # Folder selection for GeoTIFF
        self.geotif_label = tk.Label(root, text="Select GeoTIFF Image:")
        self.geotif_label.grid(row=1, column=0, padx=10, pady=5, sticky='e')
        self.geotif_path = tk.StringVar()
        self.geotif_entry = tk.Entry(root, textvariable=self.geotif_path, width=40)
        self.geotif_entry.grid(row=1, column=1, padx=10, pady=5, sticky='w')
        self.geotif_button = tk.Button(root, text="Browse", command=self.browse_geotif)
        self.geotif_button.grid(row=1, column=2, padx=10, pady=5)

        # Sliders for video processing parameters
        self.param1_label = tk.Label(root, text="CLAHE Clip Limit:")
        self.param1_label.grid(row=2, column=0, padx=10, pady=5, sticky='e')
        self.param1_slider = tk.Scale(root, from_=0.1, to=20, resolution=0.1, orient='horizontal', command=self.update_image)
        self.param1_slider.set(2)
        self.param1_slider.grid(row=2, column=1, padx=10, pady=5, columnspan=2, sticky='w')

        self.param2_label = tk.Label(root, text="Frame Step:")
        self.param2_label.grid(row=3, column=0, padx=10, pady=5, sticky='e')
        self.param2_slider = tk.Scale(root, from_=1, to=50, orient='horizontal')
        self.param2_slider.set(5)
        self.param2_slider.grid(row=3, column=1, padx=10, pady=5, columnspan=2, sticky='w')

        # Text boxes for center coordinates and radius
        self.center_coord_label = tk.Label(root, text="Center Coordinates (lat, long):")
        self.center_coord_label.grid(row=4, column=0, padx=10, pady=5, sticky='e')
        
        self.center_coord_lat = tk.Entry(root, width=20)
        self.center_coord_lat.grid(row=4, column=1, padx=5, pady=5, sticky='w')
        self.center_coord_long = tk.Entry(root, width=20)
        self.center_coord_long.grid(row=5, column=1, padx=5, pady=5, sticky='w')

        self.radius_label = tk.Label(root, text="Radius:")
        self.radius_label.grid(row=6, column=0, padx=10, pady=5, sticky='e')
        self.radius_entry = tk.Entry(root, width=10)
        self.radius_entry.grid(row=6, column=1, padx=10, pady=5, sticky='w')
        
        self.range_label = tk.Label(root, text="")
        self.range_label.grid(row=4, column=2, rowspan=3, padx=10, pady=5, sticky='w')

        # Output folder selection
        self.output_folder_label = tk.Label(root, text="Select Output Folder:")
        self.output_folder_label.grid(row=7, column=0, padx=10, pady=5, sticky='e')
        self.output_folder_path = tk.StringVar()
        self.output_folder_entry = tk.Entry(root, textvariable=self.output_folder_path, width=40)
        self.output_folder_entry.grid(row=7, column=1, padx=10, pady=5, sticky='w')
        self.output_folder_button = tk.Button(root, text="Browse", command=self.browse_output_folder)
        self.output_folder_button.grid(row=7, column=2, padx=10, pady=5)

        # Frame to contain all buttons
        button_frame = tk.Frame(root)
        button_frame.grid(row=8, column=0, columnspan=5, pady=20, padx=10)
        
        # Buttons
        self.run_button = tk.Button(button_frame, text="Run Processing", command=self.run_processing)
        self.run_button.grid(row=0, column=0, padx=5, sticky='ew')
        
        self.save_button = tk.Button(button_frame, text="Save Frames", command=self.save_frames)
        self.save_button.grid(row=0, column=1, padx=5, sticky='ew')
        
        self.display_button = tk.Button(button_frame, text="Display Map", command=self.display_map)
        self.display_button.grid(row=0, column=2, padx=5, sticky='ew')
        
        self.map_button = tk.Button(button_frame, text="Save Map", command=self.save_map)
        self.map_button.grid(row=0, column=3, padx=5, sticky='ew')
        
        self.list_button = tk.Button(button_frame, text="Generate List", command=self.generate_list)
        self.list_button.grid(row=0, column=4, padx=5, sticky='ew')

        # Progress bars
        self.current_step_label = tk.Label(root, text="Current Step:")
        self.current_step_label.grid(row=9, column=0, padx=10, pady=5, sticky='e')
        self.current_progress = ttk.Progressbar(root, orient='horizontal', length=300, mode='determinate')
        self.current_progress.grid(row=9, column=1, padx=10, pady=5, columnspan=2, sticky='w')

        self.plot_frame = tk.Frame(root)
        self.plot_frame.grid(row=10, column=0, columnspan=3, pady=10)
        
    def on_closing(self,_=None):
        self.root.destroy()

    def browse_image_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            if any(file.endswith(('.png', '.jpg', '.jpeg', '.tiff')) for file in os.listdir(folder)):
                self.image_folder_path.set(folder)
            else:
                messagebox.showerror("Error", "Selected folder does not contain valid image files.")
                return
        else:
            messagebox.showerror("Error", "Selected folder does not exist.")
            return
        self.update_image(self.param1_slider.get())

    def browse_geotif(self):
        file = filedialog.askopenfilename(filetypes=[("GeoTIFF files", "*.tif")])
        if file:
            self.geotif_path.set(file)
            self.show_geotif_ranges(file)
    
    def browse_output_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder_path.set(folder)

    def run_processing(self):
        # Validation checks
        if not os.path.isdir(self.image_folder_path.get()):
            messagebox.showerror("Error", "Invalid image folder selected.")
            return

        image_folder = self.image_folder_path.get()
        if not image_folder:
            return
        images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.tiff'))]
        if not images:
            return
        
        if images[0].endswith('.tiff'):
            ind = np.argsort([fix_name(f) for f in images])
            images = [images[i] for i in ind]
        else:
            images = sorted(images)

        frame_step = self.param2_slider.get()
        clip_limit = self.param1_slider.get()
        
        selected_images = images[::frame_step]
        clahe = cv2.createCLAHE(clipLimit=clip_limit)
        
        ratio = 0.9
        total_steps = len(selected_images)
        
        Hs = [np.eye(3)]
        prev_image = clahe.apply(get_image(selected_images[0]))
        for i in range(1,len(selected_images)):
            im0 = prev_image
            file = selected_images[i]
            try:
                im1 = clahe.apply(get_image(file))
            except:
                continue
            self.current_step_label.config(text=f"Processing {os.path.basename(file)}...")
            self.current_progress['value'] = ratio * (i / total_steps) * 100
            self.root.update_idletasks()
            
            H = track_frames(im1, im0)
            Hs.append(H @ Hs[-1])
            prev_image = im1.copy()

        self.current_step_label.config(text=f"Selecting Keyframes...")
        self.current_progress['value'] = ratio * 100
        self.root.update_idletasks()
            
        kfs, scales, angles, shifts = select_keyframes(Hs)
        self.display_keyframes(kfs, scales, angles, shifts)
        self.selected_frames = [selected_images[i] for i in kfs]

        self.current_step_label.config(text="Processing completed.")
        self.current_progress['value'] = 100
        self.root.update_idletasks()
    
    def update_image(self, value):
        # Update image based on the slider's current value
        value = float(value)
        image_folder = self.image_folder_path.get()
        
        if not image_folder:
            return
        images = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.tiff'))]
        
        if not images:
            return
        
        filename = os.path.join(image_folder, images[0])
        image = get_image(filename)
        new_image = cv2.createCLAHE(clipLimit=value).apply(image)
        
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(image, cmap='gray')
        ax[1].imshow(new_image, cmap='gray')
        ax[0].set_title("Before CLAHE")
        ax[1].set_title("After CLAHE")
        ax[0].axis('off')   
        ax[1].axis('off')

        for widget in self.plot_frame.winfo_children():
            widget.destroy()  # Clear previous plot if any

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        
    def display_keyframes(self, kfs, scales, angles, shifts):
            
        fig, ax = plt.subplots(4,1, figsize=(6,6))
        ax[0].plot(np.log2(scales),label='Scale')
        ax[1].plot(angles,label='Angle')
        ax[2].plot(shifts[:,0],label='Shift X')
        ax[3].plot(shifts[:,1],label='Shift Y')
        ax[0].scatter(kfs, np.log2(scales)[kfs], color='red')
        ax[1].scatter(kfs, angles[kfs], color='red')
        ax[2].scatter(kfs, shifts[:,0][kfs], color='red')
        ax[3].scatter(kfs, shifts[:,1][kfs], color='red')
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        ax[3].legend()
        ax[0].set_title(f'Selected {len(kfs)} Keyframes')

        for widget in self.plot_frame.winfo_children():
            widget.destroy() 

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
    
    def save_frames(self):
        if self.selected_frames is None:
            messagebox.showerror("Error", "No keyframes selected.")
            return
        
        if not os.path.isdir(self.output_folder_path.get()):
            messagebox.showerror("Error", "Invalid output folder selected.")
            return
        
        os.makedirs(self.output_folder_path.get() + '/images/', exist_ok=True)
        
        clip_limit = self.param1_slider.get()
        clahe = cv2.createCLAHE(clipLimit=clip_limit)

        saved_frames = []
        for i, image_file in enumerate(self.selected_frames):
            self.current_step_label.config(text=f"Saving {image_file}...")
            self.current_progress['value'] = (i / len(self.selected_frames)) * 100
            self.root.update_idletasks()
            
            output_file = os.path.join(self.output_folder_path.get() + '/images/', f'im{i:03d}_{os.path.basename(image_file)}')
            frame = clahe.apply(get_image(image_file))
            cv2.imwrite(output_file, frame)
            
            saved_frames.append(image_file)

        self.saved_frames = saved_frames
        self.current_step_label.config(text=f"Frames Saved.")
        self.current_progress['value'] = 100
        self.root.update_idletasks()
        
    def show_geotif_ranges(self, geotif_file):
        # Simulated range for center coordinates and radius (example values)
        try:
            map_object = rasterio.open(geotif_file)
        except:
            messagebox.showerror("Error", f"Invalid GeoTIFF file selected: {geotif_file}")
            return
        
        gps0 = pix2gps(map_object, [0,map_object.height])
        gps1 = pix2gps(map_object, [map_object.width, 0])
        ranges = gps2enu(map_object, pix2gps(map_object, [map_object.width,0]))[:2]
        max_range = max(ranges)
        self.boundaries = [gps0, gps1, max_range]

        self.range_label.config(text=f"Latitude range:  ({gps0[0]:.6f},{gps1[0]:.6f}) \nLongitude range: ({gps0[1]:.6f},{gps1[1]:.6f}) \nRadius range: (0,{max_range:.1f})")
        
        self.center_coord_lat.delete(0, tk.END)
        self.center_coord_long.delete(0, tk.END)
        self.radius_entry.delete(0, tk.END)
        self.center_coord_lat.insert(tk.INSERT, f"{((gps0[0] + gps1[0])/2):.6f}")
        self.center_coord_long.insert(tk.INSERT, f"{((gps0[1] + gps1[1])/2):.6f}")
        self.radius_entry.insert(tk.INSERT, "500.0")
    
    def display_map(self):
        
        if self.boundaries is None:
            self.show_geotif_ranges(self.geotif_path.get())
        
        for widget in self.plot_frame.winfo_children():
            widget.destroy()  # Clear previous plot if any
        
        geotif_file = self.geotif_path.get()
        map_object = rasterio.open(geotif_file)
        
        def calc_bounds(gps, shift):
            enu = gps2enu(map_object, gps)
            enu[:2] += shift*np.array([1, -1])
            gps = enu2gps(map_object, enu)
            pix = gps2pix(map_object, gps)
            return pix, gps
        
        try:
            lat = float(self.center_coord_lat.get())
            lon = float(self.center_coord_long.get())
            radius = float(self.radius_entry.get())
        except:
            messagebox.showerror("Error", "Invalid values for coordinates or radius.")
            return
        
        gps = [lat, lon]
        [gps0, gps1, max_range] = self.boundaries
        if lat < gps0[0] or lat > gps1[0]:
            messagebox.showerror("Error", "Invalid latitude.")
            return
        if lon < gps0[1] or lon > gps1[1]:
            messagebox.showerror("Error", "Invalid longitude.")
            return
        if radius <= 0 or radius > max_range:
            messagebox.showerror("Error", "Invalid radius.")
            return
        
        [x_min, y_min], crop0 = calc_bounds(gps, -radius)
        [x_max, y_max], crop1 = calc_bounds(gps, +radius)
        
        self.crop = [crop0, crop1]
        
        width = map_object.width
        height = map_object.height
        
        x_min = max(0, min(x_min, width))
        x_max = max(0, min(x_max, width))
        y_min = max(0, min(y_min, height))
        y_max = max(0, min(y_max, height))
        
        self.output_size = [x_max - x_min, y_max - y_min]

        window = Window(x_min, y_min, self.output_size[0], self.output_size[1])

        # Read the selected region
        selected_region = map_object.read(window=window)

        # Plotting the selected region
        fig, ax = plt.subplots()
        ax.imshow(selected_region.transpose(1, 2, 0)) 
        ax.set_title("Selected Map Region")
        ax.axis('off')
        
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
    
    def save_map(self):
        if not os.path.isdir(self.output_folder_path.get()):
            messagebox.showerror("Error", "Invalid output folder selected.")
            return
        
        if self.crop is None:
            self.display_map()
            return
        
        output_folder = self.output_folder_path.get()
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        input_tif = self.geotif_path.get()
        
        map_fld = output_folder + '/map'
        if os.path.exists(map_fld):
            shutil.rmtree(map_fld)
        os.makedirs(map_fld)

        resize_factors = [1, 2, 4, 8]

        projwin = f'-projwin {self.crop[0][1]} {self.crop[0][0]} {self.crop[1][1]} {self.crop[1][0]} '
        filename = None
    
        if filename is None:
            filename = os.path.basename(input_tif).split('.')[0] + '_tiled'
    
        output_size = self.output_size
        exe = f'gdal_translate -co TILED=YES {projwin} {input_tif} {map_fld}/cropped.tif' 
        
        total_time = [0.2] + [1/r**2 for r in resize_factors]
        total_time = np.array(total_time)
        total_time += 0.1
        progress = np.cumsum(total_time)/sum(total_time)
        
        self.current_step_label.config(text=f"Cropping Map...")
        self.current_progress['value'] = 0
        self.root.update_idletasks()

        subprocess.Popen(exe, shell=True).wait()
        
        self.current_progress['value'] = progress[0]*100
        self.root.update_idletasks()

        for i, factor in enumerate(resize_factors):
            exe = 'gdalwarp -co TILED=YES '
            
            new_output_size = [int(output_size[0]/factor), int(output_size[0]/factor)]
            
            exe += f'-ts {new_output_size[0]} {new_output_size[1]} '
            
            tif_src = f'{map_fld}/cropped.tif'

            tif_dst = f'{map_fld}/{filename}_{factor:02d}x.tif'
            
            exe += tif_src + ' ' + tif_dst
            
            self.current_step_label.config(text=f"Resizing Scale {factor}...")
            self.current_progress['value'] = progress[i+1]*100
            self.root.update_idletasks()

            subprocess.Popen(exe, shell=True).wait()
            time.sleep(0.1)
        
        os.remove(f'{map_fld}/cropped.tif')
        self.saved_map = f'{map_fld}/{filename}_{resize_factors[0]:02d}x.tif'
        
        self.current_step_label.config(text=f"Maps Saved.")
        self.current_progress['value'] = 100
        self.root.update_idletasks()
    
    def generate_list(self):
        if self.saved_frames is None:
            messagebox.showerror("Error", "No frames saved.")
            return
        
        if self.saved_map is None:
            messagebox.showerror("Error", "No map saved.")
            return
        
        # save pandas dataframe to csv with fields images,map_path
        saved_frames = [os.path.abspath(f) for f in self.saved_frames]
        saved_map = os.path.abspath(self.saved_map)
        df = pd.DataFrame({'images': saved_frames, 'map_path': [saved_map]*len(saved_frames)})
        output_file = os.path.abspath(f'{self.output_folder_path.get()}/output_list.csv')
        df.to_csv(output_file, index=False)
        
        messagebox.showinfo("Success", f"List generated successfully:\n {output_file}")

    def test_inputs(self):
        self.image_folder_path.set("output/simulated4")
        self.geotif_path.set("output/hrscd.tif")
        self.center_coord_lat.insert(0, "49.05602")
        self.center_coord_long.insert(0, "-0.53157")
        self.radius_entry.insert(0, "400")
        self.output_folder_path.set("output/test")
        self.show_geotif_ranges("output/hrscd.tif")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    if os.path.exists('resources/logo.jpg'):
        photo = ImageTk.PhotoImage(Image.open('resources/logo.jpg'))
        root.wm_iconphoto(False, photo)
    app.test_inputs()
    root.mainloop()
