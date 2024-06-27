import os
import numpy as np
import tkinter as tk
import pandas as pd
from tkinter import filedialog, Menu
from pathlib import Path
from process_raw import ImageProcessorApp
from image import *
from common import *

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
        self.panel_button = tk.Button(self.frame, text="Show Controls", command=self.toggle_panel,bg='white')
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
        self.image_listbox = tk.Listbox(self.image_list_frame, yscrollcommand=self.image_list_scrollbar.set,bg='white')
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_selection)
        self.image_list_scrollbar.config(command=self.image_listbox.yview)
        
        self.selected_list_frame = tk.Frame(self.frame,bg='white')
        self.selected_list_scrollbar = tk.Scrollbar(self.selected_list_frame)
        self.selected_list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.selected_listbox = tk.Listbox(self.selected_list_frame, yscrollcommand=self.selected_list_scrollbar.set,bg='white')
        self.selected_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.selected_listbox.bind('<<ListboxSelect>>', self.on_selection)
        self.selected_list_scrollbar.config(command=self.image_listbox.yview)

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
            tk.Label(self.frame, text="Selected Images:", bg='white'),
            self.selected_list_frame
        ]
        for i, b in enumerate(buttons):
            b.grid(row=i, column=0, sticky='ew')

        self.frame.grid_columnconfigure(0, weight=1)
        self.panel.grid_columnconfigure(0, weight=1)
        self.help_frame.grid_columnconfigure(0, weight=1)
        self.panel.grid_remove()
    
    def show_help(self, is_active):
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
        if is_active:
            self.help_text_box.config(state=tk.NORMAL)
            for key, description in descriptions:
                self.help_text_box.insert(tk.END, f"{key}: {description}\n")
            self.help_text_box.config(state=tk.DISABLED)
        
        
    def add_new_images(self, new_objects):
        self.image_listbox.delete(0,tk.END)
        for new_object in new_objects:
            normalized_path = os.path.normpath(new_object.image_path)
            parts = normalized_path.split(os.sep)
            if len(parts) > 3:
                parts = parts[-3:]
            result = os.path.join(*parts)
            self.image_listbox.insert(tk.END, result)
        self.images = new_objects
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
        if self.app.automatic_matching:
            self.app.match_images(cur, prev)
            
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

