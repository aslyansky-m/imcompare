# credit goes to Maksym Aslianskyi, ChatGPT, GitHub Copilot and StackOverflow

# data is here: https://ufile.io/8hcsrlo1

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import font
from PIL import Image, ImageTk
import pandas as pd
import time
from pathlib import Path

from image import *
from common import *
from process_raw import ImageProcessorApp
from panorama import sift_matching_with_homography, create_cache, stitch_images

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

        for label in self.label_widgets:
            label.destroy()
        self.label_widgets.clear()
    
        app_vars = vars(self.app)
        image_vars = vars(self.app.image)
        for var_name, var_value in {**app_vars, **image_vars}.items():
            if isinstance(var_value,np.ndarray) or var_value == self.app.image:
                continue
            try:
                label = tk.Label(self.debug_frame, text=f"{var_name}: {var_value}", bg='grey')
                label.pack(side="top", anchor="w", padx=10, pady=5)
                self.label_widgets.append(label)
            except:
                self.debug_window = None
                self.debug_frame = None
                self.app.toggle_debug_mode()
                return

class InfoPanel:
    def __init__(self, root, app):
        self.root = root
        self.app = app
        self.frame = tk.Frame(self.root, bg='grey')
        self.frame.pack(side="bottom", fill="y")
        self.create_widgets()

    def create_widgets(self):
        self.position_box = tk.Text(self.frame, height=1, width=40,bg='white')
        self.position_box.pack(side="left", padx=2, pady=2)
        self.scale_box = tk.Text(self.frame, height=1, width=30,bg='white')
        self.scale_box.pack(side="left", padx=2, pady=2)
        self.fps_box = tk.Text(self.frame, height=1, width=30,bg='white')
        self.fps_box.pack(side="left", padx=2, pady=2)
        
        self.update_position([0,0])
        self.update_scale(1.0)
        self.update_fps(0) 

    def update_position(self, position):
        self.position_box.config(state=tk.NORMAL)
        self.position_box.delete('1.0', tk.END)
        self.position_box.insert(tk.END, f"Coordinate: [{position[0]:.6f}°, {position[1]:.6f}°]")
        self.position_box.config(state=tk.DISABLED)
        
    def update_scale(self, scale):
        self.scale_box.config(state=tk.NORMAL)
        self.scale_box.delete('1.0', tk.END)
        self.scale_box.insert(tk.END, f"Scale: [1:{1/scale:.2f}]")
        self.scale_box.config(state=tk.DISABLED)
    
    def update_fps(self, fps):
        self.fps_box.config(state=tk.NORMAL)
        self.fps_box.delete('1.0', tk.END)
        self.fps_box.insert(tk.END, f"FPS: {fps:.2f}")
        self.fps_box.config(state=tk.DISABLED)
        
class ButtonPanel:
    def __init__(self, root, app):
        self.root = root
        self.app = app
        self.frame = tk.Frame(self.root, bg='white')
        self.frame.pack(side="left", fill="y")
        self.create_widgets()
        self.current_index = 0
        self.images = []
        self.output_folder = None

    def create_widgets(self):
        self.alpha_slider = tk.Scale(self.frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Alpha Blending", command=self.app.update_alpha,bg='white')
        self.alpha_slider.set(self.app.alpha)
        
        self.rotation_slider = tk.Scale(self.frame, from_=-180, to=180, resolution=0.01, orient=tk.HORIZONTAL, label="Rotation", command=self.app.update_rotation,bg='white')
        self.rotation_slider.set(self.app.image.rotation)

        self.scale_slider = tk.Scale(self.frame, from_=-1, to=1, resolution=0.001, orient=tk.HORIZONTAL, label="Scale Factor", command=lambda x: self.app.update_scale(np.power(10,float(x))),bg='white')
        self.scale_slider.set(np.log10(self.app.image.scale))

        self.homography_button = tk.Button(self.frame, text="Homography Mode: OFF", command=self.app.toggle_homography_mode,bg='white')
        self.homography_reset_button = tk.Button(self.frame, text="Reset Homography", command=self.app.reset_homography,bg='white')
        self.panorama_button = tk.Button(self.frame, text="Create Panorama", command=self.app.create_panorama,bg='white')
        self.homography_calculate_button = tk.Button(self.frame, text="Calculate Homography", command=self.app.run_matching,bg='white')
        self.automatic_matching_button = tk.Button(self.frame, text="Automatic Matching: OFF", command=self.app.toggle_automatic_matching,bg='white')
        self.viewport_button = tk.Button(self.frame, text="Viewport Mode: OFF", command=self.app.toggle_viewport_mode,bg='white')
        self.contrast_button = tk.Button(self.frame, text="Contrast Mode: OFF", command=self.app.toggle_contrast_mode,bg='white')
        self.switch_button = tk.Button(self.frame, text="Switch Images: OFF", command=self.app.toggle_images,bg='white')
        self.grid_button = tk.Button(self.frame, text="Show Grid: OFF", command=self.app.toggle_grid,bg='white')
        self.borders_button = tk.Button(self.frame, text="Show Borders: OFF", command=self.app.toggle_borders,bg='white')
        #self.debug_button = tk.Button(self.frame, text="Debug Mode: OFF", command=self.app.toggle_debug_mode,bg='white')
        self.help_button = tk.Button(self.frame, text="Help: OFF", command=self.app.toggle_help_mode,bg='white')
        self.help_frame = tk.Frame(self.frame)
        self.help_text_box = tk.Text(self.help_frame, height=7, width=30, wrap="word",bg='white')
        self.help_text_box.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.help_scrollbar = tk.Scrollbar(self.help_frame, command=self.help_text_box.yview)
        self.help_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.help_text_box['yscrollcommand'] = self.help_scrollbar.set
        
        self.raw_button = tk.Button(self.frame, text="Raw Processing", command=self.run_raw_processing, bg='white')
        self.upload_button = tk.Button(self.frame, text="Upload Image", command=self.app.upload_image,bg='white')
        self.load_csv_button = tk.Button(self.frame, text="Load CSV", command=self.load_csv,bg='white')
        self.save_image_button = tk.Button(self.frame, text="Save Image", command=self.on_save_image,bg='white')
        self.save_results_button = tk.Button(self.frame, text="Save Results", command=self.save_results,bg='white')
        self.image_list_frame = tk.Frame(self.frame,bg='white')
        self.image_list_scrollbar = tk.Scrollbar(self.image_list_frame)
        self.image_list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_listbox = tk.Listbox(self.image_list_frame, yscrollcommand=self.image_list_scrollbar.set,bg='white')
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_selection)
        self.image_list_scrollbar.config(command=self.image_listbox.yview)

        buttons = [
            self.alpha_slider,
            self.rotation_slider,
            self.scale_slider,
            self.homography_button,
            self.homography_reset_button,
            self.panorama_button,
            self.homography_calculate_button,
            self.automatic_matching_button,
            self.viewport_button,
            self.contrast_button,
            self.switch_button,
            self.grid_button,
            self.borders_button,
            #self.debug_button,
            self.help_button,
            self.help_frame,
            
            self.raw_button,
            self.upload_button,
            self.load_csv_button,
            self.save_image_button,
            self.save_results_button,
            self.image_list_frame
        ]
        for i, b in enumerate(buttons):
            b.grid(row=i, column=0, sticky='ew')

        self.frame.grid_columnconfigure(0, weight=1)
        
    def load_csv(self, file_path=None):
        self.app.clear_messages()
        if not file_path:
            file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not os.path.exists(file_path):
            self.app.display_message(f"ERROR: {file_path} does not exist")
            return
        file_dir = os.path.dirname(file_path)
        df = pd.read_csv(file_path)
        df["images"] = df["images"].apply(lambda x: x if os.path.isabs(x) else os.path.abspath(os.path.join(file_dir, x)))
    
        if "map_path" in df:
            df["map_path"] = df["map_path"].apply(lambda x: x if os.path.isabs(x) else os.path.abspath(os.path.join(file_dir, x)))
            count = df["map_path"].nunique()
            map_path = df["map_path"][0]
            if count > 1:
                self.app.display_message(f"WARNING: Multiple maps found in CSV, slecting: {map_path}")
                df = df[df["map_path"] == map_path]
            if not os.path.exists(map_path):
                self.app.display_message(f"ERROR: {map_path} does not exist")
                return
            self.app.map = PyramidMap(map_path)
            self.app.last_map = None
            self.app.display_message(f"Loaded map from: \n{map_path}")
        elif not self.app.map:
            self.app.display_message("ERROR: No map loaded")
            return
        new_objects = []
        for index, row in df.iterrows():
            image_path = row["images"]
            H = None
            if "homography" in row:
                parsed_H = row["homography"].replace("[","").replace("]","").replace("\n","").split()
                parsed_H = [float(x) for x in parsed_H if x]
                H = np.array(parsed_H).reshape(3,3)
                if np.linalg.norm(H - np.eye(3)) < 1e-6:
                    H = None
            new_object = ImageObject(image_path, H, window_size=self.app.window_size)
            if not new_object.state == ImageState.NOT_VALID:
                new_objects.append(new_object)
            else:
                self.app.display_message(f"ERROR: {new_object.error_message}")
        if len(new_objects) > 0:
            self.image_listbox.delete(0,tk.END)
            self.app.display_message(f"Loaded {len(new_objects)} image(s) from: \n{file_path}")
            for new_object in new_objects:
                normalized_path = os.path.normpath(new_object.image_path)
                parts = normalized_path.split(os.sep)
                if len(parts) > 3:
                    parts = parts[-3:]
                result = os.path.join(*parts)
                self.image_listbox.insert(tk.END, result)
            self.images = new_objects
        else:
            self.app.display_message(f"ERROR: No valid image pairs found in CSV: \n{file_path}")
        self.app.image = None
        self.app.panorama_cache = None
        self.select_image(0)
        self.app.sync_sliders()

    def save_results(self):
        self.app.clear_messages()
        if True or not self.output_folder:
            self.output_folder = filedialog.askdirectory(title='Select output folder...')
        if not self.output_folder:
            self.app.display_message("ERROR: Please select an output folder")
            return
        output_file = f"{self.output_folder}/results.csv"
        images_to_save = [self.app.image]
        if len(self.images) > 0:
            images_to_save = self.images
            
        data_to_write = []

        for i, image_object in enumerate(images_to_save):
            data_to_write.append(dict(images=image_object.image_path, homography=image_object.relative_transform().flatten(),map_path=self.app.map.map_file))
        
        pd.DataFrame(data_to_write).to_csv(output_file, index=False)
        self.app.display_message(f"Saved results to {output_file}")
        
    def on_save_image(self, event=None):
        self.app.clear_messages()
        if True or not self.output_folder:
            self.output_folder = filedialog.askdirectory(title='Select output folder...')
        if not self.output_folder:
            self.app.display_message("ERROR: Please select an output folder")
            return
        
        selected_image = self.app.image
        output_file = f"{self.output_folder}/{Path(selected_image.image_path).stem}.tif"
        H_tot = np.linalg.inv(selected_image.relative_transform())
        query_aligned, bbox_gps = align_image(selected_image.get_image(), self.app.map, H_tot, target_size = max(selected_image.get_image().shape))
        dump_geotif(query_aligned, bbox_gps, output_file)
        
        self.app.display_message(f"Saved results to {output_file}")
        
    def on_delete(self, event):
        
        if self.app.homography_mode:
            return
        
        if len(self.images) == 0:
            return
        
        self.app.clear_messages()
        self.app.display_message(f"Deleted image pair: {self.images[self.current_index].image_path.split('/')[-1]}")
        del self.images[self.current_index]
        
        if len(self.images) == 0:
            return
        self.image_listbox.delete(self.current_index)
        self.select_image(self.current_index)
            
    def select_image(self, new_index):
        if len(self.images) == 0:
            self.app.clear_messages()
            self.app.display_message("ERROR: No image pairs loaded")
            return
        prev = self.app.image
        self.current_index = new_index % len(self.images)
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(self.current_index)
        self.image_listbox.see(self.current_index)
        self.app.image = self.images[self.current_index]
        cur = self.app.image
        if self.app.automatic_matching and \
            prev is not None and (cur.state == ImageState.NOT_LOADED or cur.state == ImageState.LOADED or cur.state == ImageState.MATCHED) and \
            not (prev.state == ImageState.MOVED and cur.state == ImageState.MOVED): 
 
            H_rel = sift_matching_with_homography(cur.get_image(), prev.get_image())
            if H_rel is None:
                cur.scale, cur.rotation, cur.x_offset, cur.y_offset, cur.M_anchors = prev.scale, prev.rotation, prev.x_offset, prev.y_offset, prev.M_anchors
            else:
                prev.derived_images.append((cur, H_rel))
                H_prev = prev.M_anchors @ calc_transform([prev.image.shape[1] * prev.scale_ratio, prev.image.shape[0] * prev.scale_ratio], prev.scale, prev.rotation, prev.x_offset, prev.y_offset) @ prev.M_original
                cur_H =  H_prev @ H_rel @ np.linalg.inv(cur.M_original)
                cur.initialize_from_homography(cur_H)
            cur.reset_anchors()
            cur.state = ImageState.MATCHED
        self.app.render()
        self.app.sync_sliders()
        
    def next_image(self):
        self.select_image(self.current_index+1)
        
    def previous_image(self):
        self.select_image(self.current_index-1)

    def on_selection(self, event):
        if len(self.image_listbox.curselection())==0:
            return
        selected_index = self.image_listbox.curselection()[0]
        self.select_image(selected_index)
        
    def update_listbox(self):
        for i, image in enumerate(self.images):
            self.image_listbox.itemconfig(i, bg=ImageState.to_color(image.state), fg='black')
        cur_image = self.app.image
        if cur_image is None:
            return
        for image, _ in cur_image.derived_images:
            if image is None or image.state == ImageState.MOVED:
                continue
            index = self.images.index(image)
            self.image_listbox.itemconfig(index, bg=ImageState.to_color(image.state), fg='navy')
    
    def run_raw_processing(self):
        try:
            root = tk.Toplevel()
            app = ImageProcessorApp(root)
        except Exception as e:
            print(e)


class ImageAlignerApp:
    def __init__(self, root, image_object=None, map_object=None):
        self.root = root
        
        if image_object is None:
            image_object = ImageObject("",window_size=(800,600))
        
        self.window_size = None

        self.image = image_object
        self.map = map_object
        self.alpha = 0.75
        self.last_map = None
        self.last_state = None
        self.zoom_center = None
        self.last_global_scale = 1.0
        
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
        self.edge_mode = False
        self.homography_mode = False
        self.rotation_mode = False
        self.draw_grid = False
        self.automatic_matching = False
        self.show_borders = False
        
        self.panorama_cache = None
        self.num_panoramas = 0

        self.configure_window()
        self.debug_info = DebugInfo(root, self)
        self.info_panel = InfoPanel(root, self)
        self.button_panel = ButtonPanel(root, self)
        self.canvas = tk.Canvas(self.root, width=self.window_size[0], height=self.window_size[1])
        self.canvas.pack(fill='both', expand=True)
        self.setup_bindings()
        
        if self.image.state == ImageState.NOT_VALID:
            self.display_message('ERROR: ' + self.image.error_message)
        self.render()
        
    def configure_window(self):
        SCREEN_FACTOR = 0.9
        screen_size = [int(self.root.winfo_screenwidth()*SCREEN_FACTOR), int(self.root.winfo_screenheight()*SCREEN_FACTOR)]
        screen_size[0] = min(screen_size[0], int(screen_size[1]*1.9))
        self.window_size = [int(screen_size[0]*0.85), int(screen_size[1]*0.96)]
        self.root.title("TagIm Aligning App")
        self.root.geometry(f"{screen_size[0]}x{screen_size[1]}")
        if os.path.exists('resources/logo.jpg'):
            photo = ImageTk.PhotoImage(Image.open('resources/logo.jpg'))
            self.root.wm_iconphoto(False, photo)

    def setup_bindings(self):
        self.root.bind('<KeyPress>', self.on_key_press)
        self.root.bind('<KeyRelease>', self.on_key_release)
        self.root.bind('<MouseWheel>', self.on_mouse_wheel)
        self.root.bind("<Button-2>", self.on_middle_click)
        self.root.bind("<Button-3>", self.toggle_images)
        self.root.bind("<Button-4>", self.on_mouse_wheel)
        self.root.bind("<Button-5>", self.on_mouse_wheel)
        self.root.bind('<ButtonPress-1>', self.on_mouse_press)
        self.root.bind('<B1-Motion>', self.on_mouse_drag)
        self.root.bind('<ButtonRelease-1>', self.on_mouse_release)
        self.root.bind('<Control-z>', self.undo)
        self.root.bind('<Control-Z>', self.redo)
        self.root.bind('<Control-s>', self.button_panel.on_save_image)
        self.root.bind('<Control-l>', self.toggle_lock_image)
        self.root.bind('<Delete>', self.button_panel.on_delete)
        self.root.bind("<Escape>", self.exit)
        self.root.bind("<Motion>", self.on_mouse_position)
        self.root.bind("<Configure>", self.on_root_resize)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def on_root_resize(self, event):
        self.canvas.config(width=event.width, height=event.height)
    
    def on_canvas_resize(self, event):
        self.window_size = [event.width, event.height]
        self.last_map = None
        self.render(update_state=False)
    
    def upload_image(self):
        self.clear_messages()
        image_path = filedialog.askopenfilename(title='Select image to align...')
        self.image = ImageObject(image_path,window_size=self.window_size) 
        if self.image.state == ImageState.NOT_VALID:
            self.display_message('ERROR: ' + self.image.error_message)
            return
        self.display_message('Image uploaded successfully')
        self.render()
        
    def M_global(self):
        center = np.array(self.window_size)/2
        
        if self.zoom_center is not None:
            prev_matrix = translation_matrix([self.global_x_offset, self.global_y_offset]) @ translation_matrix(center)@scale_matrix(self.last_global_scale)@translation_matrix(-center)
            desired_change = translation_matrix(self.zoom_center)@scale_matrix(self.global_scale/self.last_global_scale)@translation_matrix(-self.zoom_center)
            desired_matrix = desired_change@prev_matrix
            cur_scale = translation_matrix(center)@scale_matrix(self.global_scale)@translation_matrix(-center)
            T = desired_matrix @ np.linalg.inv(cur_scale)
            self.global_x_offset = T[0, 2]
            self.global_y_offset = T[1, 2]
            self.zoom_center = None
            
        self.last_global_scale = self.global_scale
            
        M = translation_matrix([self.global_x_offset, self.global_y_offset]) @ translation_matrix(center)@scale_matrix(self.global_scale)@translation_matrix(-center)
        return M
    
    def blend_images(self):
        if self.map is None:
            return np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)
        
        M_global = self.M_global()
        im2 = self.image.render(M_global, window_size=self.window_size)
        
        if im2 is None:
            return np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)
        
        M_global = self.M_global()
        if self.last_map is None or not (M_global==self.last_state).all():
            self.last_map = self.map.warp_map(M_global, self.window_size)
            if self.last_map is None:
                self.last_map = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)
            self.last_state = M_global
        
        im1 = self.last_map.copy()
        
        if im1 is None:
            im1 = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)

        if self.contrast_mode or self.edge_mode:
            if self.contrast_mode:
                im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
                im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
            else:
                im1 = edge_detection(im1,blur=3)
                im2 = edge_detection(im2,blur=9)
            if self.toggle:
                im1, im2 = im2, im1
            blend_image = np.stack([im1, im2, im1], axis=-1)
        else:
            alpha = self.alpha
            if self.toggle:
                alpha = 1 - alpha
            blend_image = cv2.addWeighted(im1, 1 - alpha, im2, alpha, 0)
            if self.show_borders:
                mask = (im2[:,:,0] == 0).astype(np.uint8)
                im1_new = cv2.multiply(im1,alpha)
                blend_image = cv2.add(blend_image, cv2.bitwise_and(im1_new, im1_new, mask=mask))
            
        return blend_image
        
    
    def render(self, update_state=True):
        t = time.time()
        
        if not self.dragging and update_state:
            self.image.save_state()
            
        rendered_image = self.blend_images()
        
        if self.draw_grid:
            rendered_image = draw_grid(rendered_image, self.M_global())

        if self.viewport_mode:
            rendered_image = cv2.rectangle(rendered_image, (0, 0), (self.window_size[0], self.window_size[1]), (0, 0, 255), 10)

        img_pil = Image.fromarray(rendered_image)
        self.tk_image = ImageTk.PhotoImage(img_pil)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        if self.homography_mode:
            M_global = self.M_global()
            for anchor in self.image.anchors:
                anchor.plot(self.canvas, M_global)
                
        fps = 1/(time.time()-t)
        self.info_panel.update_fps(fps)
        self.info_panel.update_scale(self.global_scale)
        
        self.button_panel.update_listbox()

        if self.debug_mode:
            self.debug_info.show_debug_info()
            
    def create_panorama(self):
        #create panorama
        if self.image.is_panorama:
            self.clear_messages()
            self.display_message("ERROR: Cannot create a panorama from another one")
            return
        selected_index = self.button_panel.current_index
        good_indices = []
        good_images = []
        good_files = []
        dst = None
        i = 0
        for num, image in enumerate(self.button_panel.images):
            if image.is_panorama:
                continue
            if num == selected_index:
                dst = i
            good_indices.append(i)
            i += 1
            good_images.append(image.get_image())
            good_files.append(image.image_path)
        if self.panorama_cache is None:
            self.panorama_cache = create_cache(good_images)
            self.panorama_cache["files"] = good_files
        else:
            if self.panorama_cache["files"] != good_files:
                # TODO: if some images is deleted you can re-use cache
                self.panorama_cache = create_cache(good_images)
                self.panorama_cache["files"] = good_files
        panorama_image, final_transforms = stitch_images(good_images, self.panorama_cache, dst)
        num_good = len([x for x in final_transforms if x is not None])
        if num_good < 2:
            self.clear_messages()
            self.display_message("ERROR: Failed to create panorama")
            return
        #calculate it's homography based on selected image
        H_rel = np.linalg.inv(final_transforms[dst])
        selected = self.image
        selected.state = ImageState.MATCHED
        M_selected = calc_transform([selected.image.shape[1] * selected.scale_ratio, selected.image.shape[0] * selected.scale_ratio], selected.scale, selected.rotation, selected.x_offset, selected.y_offset)
        cur_H = selected.M_anchors @ M_selected @ selected.M_original @ H_rel 
        self.num_panoramas += 1
        new_pair = ImageObject.create_panorama(panorama_image, cur_H, name=f"panorama_{self.num_panoramas}",window_size=self.window_size)
        #add to the list
        self.button_panel.images.append(new_pair)
        self.button_panel.image_listbox.delete(0,tk.END)
        for new_object in self.button_panel.images:
            normalized_path = os.path.normpath(new_object.image_path)
            parts = normalized_path.split(os.sep)
            if len(parts) > 3:
                parts = parts[-3:]
            result = os.path.join(*parts)
            self.button_panel.image_listbox.insert(tk.END, result)
        self.button_panel.select_image(len(self.button_panel.images)-1)
        #update relative transforms
        if self.automatic_matching:
            self.toggle_automatic_matching()
        prev = new_pair 
        H_prev = prev.M_anchors @ calc_transform([prev.image.shape[1] * prev.scale_ratio, prev.image.shape[0] * prev.scale_ratio], prev.scale, prev.rotation, prev.x_offset, prev.y_offset) @ prev.M_original
        for cur_index, ind in enumerate(good_indices):
            cur = self.button_panel.images[ind]
            if final_transforms[cur_index] is None or cur.is_panorama:
                continue
            H_rel = final_transforms[cur_index]
            prev.derived_images.append([cur, H_rel])
            if cur.state == ImageState.MOVED or cur.state == ImageState.LOCKED:
                continue
            cur_H = H_prev @ H_rel @ np.linalg.inv(cur.M_original)
            cur.initialize_from_homography(cur_H)
            cur.reset_anchors()
            cur.state = ImageState.MATCHED
        #success
        self.button_panel.update_listbox()
        self.clear_messages()
        self.display_message(f"Successfully created panorama from {num_good} images")
        

    # Event Handlers
    def update_alpha(self, val):
        self.alpha = float(val)
        self.button_panel.alpha_slider.set(self.alpha)
        self.render()
    
    def update_rotation(self, val):
        self.move_anchors()
        val = float(val)
        self.image.rotation = ((val + 180)%360 -180) if val != 180 else 180
        self.button_panel.rotation_slider.set(self.image.rotation)
        self.render()
        
    def update_scale(self, val):
        self.move_anchors()
        self.image.scale = np.clip(float(val),0.1,10)
        self.button_panel.scale_slider.set(np.log10(self.image.scale))
        self.render()
    
    def sync_sliders(self):
        self.button_panel.rotation_slider.set(self.image.rotation)
        self.button_panel.scale_slider.set(np.log10(self.image.scale))
        
    def toggle_help_mode(self):
        self.help_mode = not self.help_mode
        self.button_panel.help_button.config(text="Help:  ON" if self.help_mode else "Help: OFF", bg=('grey' if self.help_mode else 'white'))
        descriptions = [
            ('r', "Rotate by 90 degrees"),
            ('+', "Zoom in 10%"),
            ('-', "Zoom out 10%"),
            ('d', "Debug mode"),
            ('c', "Contrast mode"),
            ('e', "Edge detection mode"),
            ('space', "Change field of view"),
            ('h', "Homography mode"),
            ('o', "Reset homography"),
            ('p', "Create panorama"),
            ('s', "Print coordinates"),
            ('q', "Reset settings"),
            ('m', "Automatic matching"),
            ('g', "Toggle grid visibility"),
            ('b', "Toggle borders visibility"),
            ('a', "Run image matching"),
            ('->', "Next image"),
            ('<-', "Previous image"),
            ('Control-z', "Undo last action"),
            ('Control-Shift-z', "Redo last undone action"),
            ('Control-s', "Save the current image"),
            ('Control-l', "Toggle image lock"),
            ('Delete', "Delete the current selection"),
            ('Mouse Wheel', "Zoom in/out"),
            ('Middle Click', "Toggle rotation mode"),
            ('Right Click', "Toggle images"),
            ('Escape', "Exit the application")
        ]
        self.clear_messages()
        if self.help_mode:
            self.button_panel.help_text_box.config(state=tk.NORMAL)
            for key, description in descriptions:
                self.button_panel.help_text_box.insert(tk.END, f"{key}: {description}\n")
            self.button_panel.help_text_box.config(state=tk.DISABLED)
    
    def toggle_homography_mode(self):
        self.homography_mode = not self.homography_mode
        if len(self.image.anchors) == 0:
            self.image.reset_anchors()
        self.button_panel.homography_button.config(text="Homography Mode:  ON" if self.homography_mode else "Homography Mode: OFF", bg=('grey' if self.homography_mode else 'white'))
        self.render()

    def toggle_contrast_mode(self):
        self.contrast_mode = not self.contrast_mode
        if self.contrast_mode and self.edge_mode:
            self.toggle_edge_mode()
        self.button_panel.contrast_button.config(text="Contrast Mode:  ON" if self.contrast_mode else "Contrast Mode: OFF", bg=('grey' if self.contrast_mode else 'white'))
        self.render()
    
    def toggle_edge_mode(self):
        self.edge_mode = not self.edge_mode
        if self.edge_mode and self.contrast_mode:
            self.toggle_contrast_mode()
        self.render()

    def toggle_images(self, _ = None):
        self.toggle = not self.toggle
        self.button_panel.switch_button.config(text="Switch Images:  ON" if self.toggle else "Switch Images: OFF", bg=('grey' if self.toggle else 'white'))
        self.render()

    def toggle_grid(self):
        self.draw_grid = not self.draw_grid
        self.button_panel.grid_button.config(text="Show Grid:  ON" if self.draw_grid else "Show Grid: OFF", bg=('grey' if self.draw_grid else 'white'))
        self.render()

    def reset_homography(self):
        self.image.M_anchors = np.eye(3)
        self.image.reset_anchors()
        self.render()

    def reset(self):
        self.update_scale(1.0)
        self.update_rotation(0)
        self.image.x_offset = 0
        self.image.y_offset = 0
        self.image.M_anchors = np.eye(3)
        self.image.reset_anchors()
        self.image.state = ImageState.LOADED
        self.render()
        
    def run_matching(self):
        M_global = self.M_global()
        im1 = self.map.warp_map(M_global, self.window_size)
        im2 = self.image.get_image()
        H = sift_matching_with_homography(im2, im1)
        if H is None:
            self.display_message('ERROR: Could not calculate homography')
            return
        
        self.image.initialize_from_homography( np.linalg.inv(M_global) @ H @ np.linalg.inv(self.image.M_original))
        self.image.state = ImageState.MOVED
        self.sync_sliders()
        self.render() 
    
    def toggle_lock_image(self, _ = None):
        if self.image.state == ImageState.LOCKED:
            self.image.state = ImageState.MOVED
        else:
            self.image.state = ImageState.LOCKED
        self.button_panel.update_listbox()

    def toggle_viewport_mode(self, _ = None):
        self.viewport_mode = not self.viewport_mode
        self.button_panel.viewport_button.config(text="Viewport Mode:  ON" if self.viewport_mode else "Viewport Mode: OFF", bg=('grey' if self.viewport_mode else 'white'))
        self.root.config(cursor=('fleur' if self.viewport_mode else 'arrow'))
        self.render()
    
    def toggle_automatic_matching(self):
        self.automatic_matching = not self.automatic_matching
        self.button_panel.automatic_matching_button.config(text="Automatic Matching:  ON" if self.automatic_matching else "Automatic Matching: OFF", bg=('grey' if self.automatic_matching else 'white'))
        
    def toggle_debug_mode(self):
        self.debug_mode = not self.debug_mode
        self.button_panel.debug_button.config(text="Debug Mode:  ON" if self.debug_mode else "Debug Mode: OFF", bg=('grey' if self.debug_mode else 'white'))
        self.render()
        
    def toggle_borders(self):
        self.show_borders = not self.show_borders
        self.button_panel.borders_button.config(text="Show Borders:  ON" if self.show_borders else "Show Borders: OFF", bg=('grey' if self.show_borders else 'white'))
        self.render()
    
    def undo(self, event):
        self.image.undo()
        self.render(False)

    def redo(self, event):
        self.image.redo()
        self.render(False)
        
    def on_mouse_position(self, event):
        relevant = self.check_relevancy(event)
        self.root.config(cursor=('fleur' if self.viewport_mode and relevant else 'arrow'))
        if not relevant:
            return

        pos = [event.x, event.y]
        if self.map is None:
            self.info_panel.update_position(pos)
            return
        pos = apply_homography(np.linalg.inv(self.M_global()), pos)
        pos = self.map.pix2gps(pos)
        self.info_panel.update_position(pos)
    
    def print_coords(self):
        text = self.info_panel.position_box.get("1.0",tk.END)
        self.clear_messages()
        self.display_message(text)
    
    def on_key_press(self, event):
        if event.char == 'r':
            self.update_rotation(self.image.rotation + 90)
        elif event.char == '-':
            self.global_scale *= 0.9
        elif event.char == '=':
            self.global_scale *= 1.1
        elif event.char == 'd':
            self.toggle_debug_mode()
        elif event.char == 'c':
            self.toggle_contrast_mode()
        elif event.char == 'e':
            self.toggle_edge_mode()
        elif event.char == ' ':
            self.toggle_viewport_mode()
        elif event.char == 'h':
            self.toggle_homography_mode()
        elif event.char == 'o':
            self.reset_homography()
        elif event.char == 'p':
            self.create_panorama()
        elif event.char == 's':
            self.print_coords()
        elif event.char == 'q':
            self.reset()
        elif event.char == 'm':
            self.toggle_automatic_matching()
        elif event.char == 'g':
            self.toggle_grid()
        elif event.char == 'b':
            self.toggle_borders()
        elif event.char == 'a':
            self.run_matching()
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
            self.zoom_center = np.array([event.x, event.y])
        else:
            if self.image.state != ImageState.LOCKED:
                if self.rotation_mode:
                    self.update_rotation(self.image.rotation - step)
                elif not self.homography_mode:
                    self.update_scale(self.image.scale * (1.02 ** step))
        self.render()

    def on_mouse_press(self, event):
        if not self.check_relevancy(event):
            return
        
        self.dragging = True
        pt0 = [event.x, event.y]
        pt = apply_homography(np.linalg.inv(self.M_global()), pt0)
        if self.homography_mode and not self.viewport_mode:
            if self.image.state != ImageState.LOCKED:
                self.image.push_anchor(pt)
                self.image.state = ImageState.MOVED
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
        
        self.last_mouse_position = (event.x, event.y)
        pt0 = [event.x, event.y]
        pt = apply_homography(np.linalg.inv(self.M_global()), pt0)
        
        if self.homography_mode and not self.viewport_mode:
            if self.image.state != ImageState.LOCKED:
                self.image.anchors[-1].move(*pt)
                self.render()
            return

        self.move_anchors()
        if self.viewport_mode:
            self.global_x_offset += pt0[0] - self.drag_start_x
            self.global_y_offset += pt0[1] - self.drag_start_y
            self.drag_start_x = pt0[0]
            self.drag_start_y = pt0[1]
        else:
            if self.image.state != ImageState.LOCKED:
                self.image.x_offset += pt[0] - self.drag_start_x
                self.image.y_offset += pt[1] - self.drag_start_y
                self.image.state = ImageState.MOVED
            self.drag_start_x = pt[0]
            self.drag_start_y = pt[1]
        self.render()

    def on_mouse_release(self, event):
        self.dragging = False
        if self.homography_mode and not self.viewport_mode:
            self.image.M_anchors = calc_homography(self.image.anchors) @ self.image.M_anchors
            for anchor in self.image.anchors:
                anchor.reset()
        self.render()
        return
        
    def move_anchors(self):
        self.image.M_anchors = calc_homography(self.image.anchors) @ self.image.M_anchors
        self.image.reset_anchors()

    def clear_messages(self):
        self.button_panel.help_text_box.config(state=tk.NORMAL)
        self.button_panel.help_text_box.delete('1.0', tk.END)
        self.button_panel.help_text_box.config(state=tk.DISABLED)
        
    def display_message(self, message):
        if self.help_mode:
            self.toggle_help_mode()
        self.button_panel.help_text_box.config(state=tk.NORMAL)
        self.button_panel.help_text_box.insert(tk.END, message + '\n')
        self.button_panel.help_text_box.config(state=tk.DISABLED)

    def exit(self, event=None):
        self.root.quit()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = ImageAlignerApp(root)
    app.button_panel.load_csv("output/simulated_list2.csv")

    root.mainloop()

if __name__ == "__main__":
    main()
