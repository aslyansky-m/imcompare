import os
import numpy as np
import tkinter as tk
import pandas as pd
from tkinter import filedialog, Menu
from pathlib import Path
from process_raw import ImageProcessorApp
from image import *
from common import *
import time
import sys

def resource_path(relative_path):
    """ Get the absolute path to the resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def filename_to_title(filename):
    normalized_path = os.path.normpath(filename)
    parts = normalized_path.split(os.sep)
    if len(parts) > 3:
        parts = parts[-3:]
    result = os.path.join(*parts)
    
    parts = os.path.basename(filename).split('_')
    if len(parts) == 3:
        date = parts[0]
        sensor = parts[2].split('.')[0]
        result = date[:4] + '-' + date[4:6] + '-' + date[6:] + ' ' + sensor
    return result

class MenuBar:
    def __init__(self, root, app):
        self.root = root
        self.app = app
        self.create_menu()
        
    def create_menu(self):
        menubar = Menu(self.root)

        # File menu
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Upload Image", command=self.app.upload_image)
        filemenu.add_command(label="Load CSV", command=self.app.load_csv)
        filemenu.add_separator()
        filemenu.add_command(label="Save Image", command=self.app.button_panel.on_save_image)
        filemenu.add_command(label="Save Results", command=self.app.button_panel.save_results)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        # Edit menu
        editmenu = Menu(menubar, tearoff=0)
        editmenu.add_command(label="Rotate Right", command=lambda: self.app.update_rotation(self.app.image.rotation + 90))
        editmenu.add_command(label="Reset Homography", command=self.app.reset_homography)
        editmenu.add_command(label="Reset Setting", command=self.app.reset)
        editmenu.add_command(label="Undo", command=self.app.undo)
        editmenu.add_command(label="Redo", command=self.app.redo)    
        menubar.add_cascade(label="Edit", menu=editmenu)

        # View menu
        viewmenu = Menu(menubar, tearoff=0)
        viewmenu.add_command(label="Viewport Mode", command=self.app.toggle_viewport_mode)
        viewmenu.add_command(label="Switch Images", command=self.app.toggle_images)
        viewmenu.add_command(label="Contrast Mode", command=self.app.toggle_contrast_mode)
        viewmenu.add_command(label="Edge Detection Mode", command=self.app.toggle_edge_mode)
        viewmenu.add_command(label="Show Grid", command=self.app.toggle_grid)
        viewmenu.add_command(label="Show Borders", command=self.app.toggle_borders)
        viewmenu.add_command(label="Zoom In", command=lambda: self.app.update_global_scale(self.app.global_scale*1.5))
        viewmenu.add_command(label="Zoom Out", command=lambda: self.app.update_global_scale(self.app.global_scale/1.5))
        menubar.add_cascade(label="View", menu=viewmenu)

        # Tools menu
        toolsmenu = Menu(menubar, tearoff=0)
        toolsmenu.add_command(label="Calculate Homography", command=self.app.run_matching)
        toolsmenu.add_command(label="Create Panorama", command=self.app.create_panorama)
        toolsmenu.add_command(label="Toggle Automatic Matching", command=self.app.toggle_automatic_matching)
        toolsmenu.add_command(label="Homography Mode", command=self.app.toggle_homography_mode)
        toolsmenu.add_command(label="Raw Processing", command=self.app.button_panel.run_raw_processing)
        menubar.add_cascade(label="Tools", menu=toolsmenu)

        # Help menu
        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Help Index", command=self.app.toggle_help_mode)
        menubar.add_cascade(label="Help", menu=helpmenu)

        self.root.config(menu=menubar)
        
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
        self.heading_box = tk.Text(self.frame, height=1, width=30,bg='white')
        self.heading_box.pack(side="left", padx=2, pady=2)
        self.fps_box = tk.Text(self.frame, height=1, width=30,bg='white')
        self.fps_box.pack(side="left", padx=2, pady=2)
        self.enhance_box = tk.Text(self.frame, height=1, width=30,bg='white')
        self.enhance_box.pack(side="left", padx=2, pady=2)
        
        self.update_position([0,0])
        self.update_scale(1.0)
        self.update_heading(0)
        self.update_fps(0) 
        self.update_enhance(0)

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
        
    def update_heading(self, heading):
        self.heading_box.config(state=tk.NORMAL)
        self.heading_box.delete('1.0', tk.END)
        self.heading_box.insert(tk.END, f"Heading: {heading:.2f}°")
        self.heading_box.config(state=tk.DISABLED)
    
    def update_fps(self, fps):
        self.fps_box.config(state=tk.NORMAL)
        self.fps_box.delete('1.0', tk.END)
        self.fps_box.insert(tk.END, f"FPS: {fps:.2f}")
        self.fps_box.config(state=tk.DISABLED)
    
    def update_enhance(self, enhance):
        self.enhance_box.config(state=tk.NORMAL)
        self.enhance_box.delete('1.0', tk.END)
        self.enhance_box.insert(tk.END, f"Enhance level: {enhance}")
        self.enhance_box.config(state=tk.DISABLED)
        
        
class ButtonPanel:
    def __init__(self, root, app):
        self.root = root
        self.app = app
        self.frame = tk.Frame(self.root, bg='white')
        self.frame.pack(side="left", fill="y")
        self.create_widgets()
        self.current_index = 0
        self.current_starred_index = 0
        self.images = []
        self.starred_images = []
        self.lru_cache = []
        self.selection_time = 0
        self.output_folder = None
        
    def toggle_panel(self):
        if self.panel.winfo_ismapped():
            self.panel.grid_remove()
            self.panel_button.config(text="Show Controls")
        else:
            self.panel.grid()
            self.panel_button.config(text="Hide Controls")
    
    def create_panel(self):
        self.panel = tk.Frame(self.frame,bg='white')
        
        self.homography_reset_button = tk.Button(self.panel, text="Reset Homography (h)", command=self.app.reset_homography,bg='white')
        self.reset_button = tk.Button(self.panel, text="Reset Edit (q)", command=self.app.reset,bg='white')
        self.panorama_button = tk.Button(self.panel, text="Create Panorama (p)", command=self.app.create_panorama,bg='white')
        self.homography_calculate_button = tk.Button(self.panel, text="Calculate Homography (a)", command=self.app.run_matching,bg='white')
        self.homography_mode_button = tk.Button(self.panel, text="Homography Mode: OFF", command=self.app.toggle_homography_mode,bg='white')
        self.automatic_matching_button = tk.Button(self.panel, text="Automatic Matching: OFF", command=self.app.toggle_automatic_matching,bg='white')
        self.viewport_button = tk.Button(self.panel, text="Viewport Mode: OFF", command=self.app.toggle_viewport_mode,bg='white')
        self.contrast_button = tk.Button(self.panel, text="Contrast Mode: OFF", command=self.app.toggle_contrast_mode,bg='white')
        self.switch_button = tk.Button(self.panel, text="Switch Images: OFF", command=self.app.toggle_images,bg='white')
        self.grid_button = tk.Button(self.panel, text="Show Grid: OFF", command=self.app.toggle_grid,bg='white')
        self.borders_button = tk.Button(self.panel, text="Show Borders: OFF", command=self.app.toggle_borders,bg='white')
        
        panel_buttons = [
            self.homography_reset_button,
            self.reset_button,
            self.panorama_button,
            self.homography_calculate_button,
            self.homography_mode_button,
            self.automatic_matching_button,
            self.viewport_button,
            self.contrast_button,
            self.switch_button,
            self.grid_button,
            self.borders_button
        ]
        for i, b in enumerate(panel_buttons):
            b.grid(row=i, column=0, sticky='ew')
        return self.panel
        

    def create_widgets(self):
        # sliders
        self.alpha_slider = tk.Scale(self.frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Alpha Blending", command=self.app.update_alpha,bg='white')
        self.alpha_slider.set(self.app.alpha)
        self.rotation_slider = tk.Scale(self.frame, from_=-180, to=180, resolution=0.01, orient=tk.HORIZONTAL, label="Rotation", command=self.app.update_rotation,bg='white')
        self.rotation_slider.set(self.app.image.rotation)
        self.scale_slider = tk.Scale(self.frame, from_=-1, to=1, resolution=0.001, orient=tk.HORIZONTAL, label="Scale Factor", command=lambda x: self.app.update_scale(np.power(10,float(x))),bg='white')
        self.scale_slider.set(np.log10(self.app.image.scale))
        
        # sub panel
        self.panel = self.create_panel()
        self.panel_button = tk.Button(self.frame, text="Show Controls", command=self.toggle_panel,bg='gray81')
        self.help_button = tk.Button(self.frame, text="Help: OFF", command=self.app.toggle_help_mode,bg='white')
        
        self.help_frame = tk.Frame(self.frame)
        self.help_text_box = tk.Text(self.help_frame, height=7, width=30, wrap="word",bg='white')
        self.help_text_box.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.help_scrollbar = tk.Scrollbar(self.help_frame, command=self.help_text_box.yview)
        self.help_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.help_text_box['yscrollcommand'] = self.help_scrollbar.set
        
        self.raw_button = tk.Button(self.frame, text="Raw Processing", command=self.run_raw_processing, bg='white')
        self.upload_button = tk.Button(self.frame, text="Upload Image", command=self.app.upload_image,bg='white')
        self.load_csv_button = tk.Button(self.frame, text="Load CSV", command=self.app.load_csv,bg='white')
        self.save_image_button = tk.Button(self.frame, text="Save Image", command=self.on_save_image,bg='white')
        self.save_results_button = tk.Button(self.frame, text="Save Results", command=self.save_results,bg='white')
        
        self.image_list_frame = tk.Frame(self.frame,bg='white')
        self.image_list_scrollbar = tk.Scrollbar(self.image_list_frame)
        self.image_list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_listbox = tk.Listbox(self.image_list_frame, exportselection=False, yscrollcommand=self.image_list_scrollbar.set,bg='white')
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_selection)
        self.image_list_scrollbar.config(command=self.image_listbox.yview)
        
        self.starred_list_frame = tk.Frame(self.frame,bg='white')
        self.starred_list_scrollbar = tk.Scrollbar(self.starred_list_frame)
        self.starred_list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.starred_listbox = tk.Listbox(self.starred_list_frame, exportselection=False, yscrollcommand=self.starred_list_scrollbar.set,bg='white')
        self.starred_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.starred_listbox.bind('<<ListboxSelect>>', self.on_selection_starred)
        self.starred_list_scrollbar.config(command=self.starred_listbox.yview)

        buttons = [
            self.alpha_slider,
            self.rotation_slider,
            self.scale_slider,
            self.panel_button,
            self.panel, 
            self.help_button,
            self.help_frame,
            self.raw_button,
            #self.upload_button,
            self.load_csv_button,
            self.save_image_button,
            self.save_results_button,
            tk.Label(self.frame, text="Images:", bg='white'),
            self.image_list_frame,
            tk.Label(self.frame, text="Starred Images:", bg='white'),
            self.starred_list_frame
        ]
        for i, b in enumerate(buttons):
            b.grid(row=i, column=0, sticky='ew')

        self.frame.grid_columnconfigure(0, weight=1)
        self.panel.grid_columnconfigure(0, weight=1)
        self.help_frame.grid_columnconfigure(0, weight=1)
        self.panel.grid_remove()
    
    def show_help(self, is_active):
        descriptions = [
            ('space', "Change field of view"),
            ('+', "Increse enhance level"),
            ('-', "Decrease enhance level"),
            ('>', "Increse heading"),
            ('<', "Decrease heading"),
            ('/', "Zero heading"),
            ('r', "Rotate by 90 degrees"),
            ('f', "Set/unset reference image"),
            ('d', "Debug mode"),
            ('c', "Contrast mode"),
            ('e', "Edge detection mode"),
            ('h', "Homography mode"),
            ('o', "Reset homography"),
            ('p', "Create panorama"),
            ('s', "Add image to starred list"),
            ('i', "Print coordinates"),
            ('k', "Save screenshot"),
            ('q', "Reset settings"),
            ('a', "Toggle automatic matching"),
            ('g', "Toggle grid visibility"),
            ('b', "Toggle borders visibility"),
            ('m', "Run image matching"),
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
        if is_active:
            self.help_text_box.config(state=tk.NORMAL)
            for key, description in descriptions:
                self.help_text_box.insert(tk.END, f"{key}: {description}\n")
            self.help_text_box.config(state=tk.DISABLED)
        
    def add_new_images(self, new_objects):
        self.image_listbox.delete(0,tk.END)
        for new_object in new_objects:
            result = filename_to_title(new_object.image_path)
            self.image_listbox.insert(tk.END, result)
        self.images = new_objects
        self.select_image(0)
        self.app.sync_sliders()
        
    def sync_images(self):
        prev = self.app.image
        for cur in self.images:
            if cur == prev:
                continue
            cur.scale, cur.rotation, cur.x_offset, cur.y_offset = prev.scale, prev.rotation, prev.x_offset, prev.y_offset
            cur.M_anchors = prev.M_anchors 
            cur.state = ImageState.SYNCED

    def save_results(self):
        self.app.clear_messages()
        output_file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not output_file:
            self.app.display_message("ERROR: Please select an output folder")
            return
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
        if not self.output_folder:
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
        
    def save_screen_shot(self):
        self.app.clear_messages()
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if not file_path:
            return
        M_global = self.app.M_global()
        im = self.app.image.render(M_global, window_size=self.app.window_size)
        if im is None:
            return 
        image_name = filename_to_title(self.app.image.image_path)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        pos = [self.app.window_size[0]//2, self.app.window_size[0]//2]
        if self.app.map is not None:
            pos = apply_homography(np.linalg.inv(M_global), pos)
            pos = self.app.map.pix2gps(pos)
        
        heading = self.app.global_rotation
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        margin = 10
        
        text1 = image_name
        text2 = f"GPS: {pos[0]:.6f}, {pos[1]:.6f}"
        window_size = im.shape
        (text_width1, text_height1), _ = cv2.getTextSize(text1, font, font_scale, thickness)
        (text_width2, text_height2), _ = cv2.getTextSize(text2, font, font_scale, thickness)
        
        x = window_size[1] - max(text_width1, text_width2) - margin
        y1 = window_size[0] - margin - text_height1 - text_height2 - 10
        y2 = window_size[0] - margin - text_height2
        box_color = (255, 255, 255) 
        cv2.rectangle(im,(x, y1 - text_height1 - margin),(window_size[1], window_size[0]),(255, 255, 255),cv2.FILLED)

        # Put the text in black color
        text_color = (0, 0, 0)  # Black color
        im = cv2.putText(im, text1, (x, y1), font, font_scale, text_color, thickness, cv2.LINE_AA)
        im = cv2.putText(im, text2, (x, y2), font, font_scale, text_color, thickness, cv2.LINE_AA)

        # # Draw the north arrow on a circle
        # arrow_thickness = 6
        # circle_radius = 30
        # circle_center = (window_size[1] - margin - circle_radius - 10, window_size[0] - margin - 2*circle_radius - 80)
        # cv2.circle(im, circle_center, int(circle_radius*1.2), (200, 200, 200), -1)  # Filled gray circle
        # angle = math.radians(heading)
        # end_x = int(circle_center[0] + circle_radius * math.sin(angle))
        # end_y = int(circle_center[1] - circle_radius * math.cos(angle))
        # cv2.arrowedLine(im, circle_center, (end_x, end_y), (255, 0, 0), arrow_thickness, tipLength=0.3)

        compass_img = cv2.imread(resource_path('resources/compass.png'), cv2.IMREAD_UNCHANGED)
        compass_size = 200
        compass_size = [compass_size, int(compass_size*compass_img.shape[0]/compass_img.shape[1])]
        compass_img = cv2.resize(compass_img, compass_size, interpolation=cv2.INTER_AREA)
        (h, w) = compass_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -heading, 1.0)  
        rotated_compass = cv2.warpAffine(compass_img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        compass_pos = (window_size[1] - compass_size[0], window_size[0] - 2*margin - compass_size[1] - 80)
        im[compass_pos[1]:compass_pos[1]+compass_size[1], compass_pos[0]:compass_pos[0]+compass_size[0], :] = rotated_compass 
        cv2.imwrite(file_path, im)
        self.app.display_message(f"Screen shot saved to: {file_path}")
        
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
        
    def select_image_impl(self, new_index, from_starred=False):
        if len(self.images) == 0:
            self.app.clear_messages()
            self.app.display_message("ERROR: No image pairs loaded")
            return
        prev = self.app.image
        self.current_index = new_index % len(self.images)
        
        self.app.image = self.images[self.current_index]
        if self.app.image in self.lru_cache:
            self.lru_cache.remove(self.app.image)
        self.lru_cache.insert(0, self.app.image)
        if len(self.lru_cache) > 30:
            evicted = self.lru_cache.pop()
            del evicted.image
            evicted.image = None
            evicted.state = ImageState.EVICTED
            
        cur = self.app.image
        if self.app.automatic_matching:
            self.app.match_images(cur, prev)
            
        self.app.render()
        self.app.sync_sliders()
        
             
    def select_image(self, new_index):
        # hack to fix 2 listbox issue
        if time.time() - self.selection_time < 0.1:
            return
        self.select_image_impl(new_index)
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(self.current_index)
        self.image_listbox.see(self.current_index)
        
        
    def select_starred_image(self, new_index):
        if len(self.starred_images) == 0:
            return
        new_index = new_index % len(self.starred_images)
        self.current_starred_index = new_index
        self.starred_listbox.selection_set(new_index)
        self.starred_listbox.see(new_index)
        selected_image = self.starred_images[new_index]
        image_index = self.images.index(selected_image)
        self.select_image_impl(image_index,True)
        self.image_listbox.unbind('<<ListboxSelect>>')
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(self.current_index)
        self.image_listbox.see(self.current_index)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_selection)
        self.starred_listbox.selection_clear(0, tk.END)
        self.starred_listbox.selection_set(self.current_starred_index)
        self.starred_listbox.see(self.current_starred_index)
        self.selection_time = time.time()


    def on_selection(self, event):
        selected_index = self.image_listbox.curselection()
        if len(selected_index)==0:
            return
        selected_index = selected_index[0]
        self.select_image(selected_index)
        
    def on_selection_starred(self, event):
        selected_index = self.starred_listbox.curselection()
        if len(selected_index)==0:
            return
        selected_index = selected_index[0]
        self.select_starred_image(selected_index)
        
    def add_starred_image(self, image):
        if image in self.starred_images:
            self.starred_images.remove(image)
        else:
            self.starred_images.append(image)
        self.update_starred_listbox()
        
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
        
        for image in self.starred_images:
            index = self.images.index(image)
            self.image_listbox.itemconfig(index, bg=ImageState.to_color(image.state), fg='DarkGoldenrod4')
        
        image = self.app.reference_image
        if image is not None:
            index = self.images.index(image)
            self.image_listbox.itemconfig(index, bg='red', fg='black')
    
    def update_starred_listbox(self):
        self.starred_listbox.delete(0,tk.END)
        for image in self.starred_images:
            result = filename_to_title(image.image_path)
            self.starred_listbox.insert(tk.END, result)
            
        image = self.app.reference_image
        if image is not None:
            index = self.starred_images.index(image)
            self.starred_listbox.itemconfig(index, bg='red', fg='black')
    
    def run_raw_processing(self):
        try:
            root = tk.Toplevel()
            app = ImageProcessorApp(root)
        except Exception as e:
            print(e)

