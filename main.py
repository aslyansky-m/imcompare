# credit goes to Maksym Aslianskyi, ChatGPT, GitHub Copilot and StackOverflow

# data is here: https://ufile.io/8hcsrlo1

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import pandas as pd
import time


from image import *
from common import *
from panels import *
from panorama import sift_matching_with_homography, create_cache, stitch_images


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
        self.menu_bar = MenuBar(root, self)
        self.canvas = tk.Canvas(root, width=self.window_size[0], height=self.window_size[1])
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
    
    def load_csv(self, file_path=None):
        self.clear_messages()
        if not file_path:
            file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not os.path.exists(file_path):
            self.display_message(f"ERROR: {file_path} does not exist")
            return
        file_dir = os.path.dirname(file_path)
        df = pd.read_csv(file_path)
        df["images"] = df["images"].apply(lambda x: x if os.path.isabs(x) else os.path.abspath(os.path.join(file_dir, x)))
    
        if "map_path" in df:
            df["map_path"] = df["map_path"].apply(lambda x: x if os.path.isabs(x) else os.path.abspath(os.path.join(file_dir, x)))
            count = df["map_path"].nunique()
            map_path = df["map_path"][0]
            if count > 1:
                self.display_message(f"WARNING: Multiple maps found in CSV, slecting: {map_path}")
                df = df[df["map_path"] == map_path]
            if not os.path.exists(map_path):
                self.display_message(f"ERROR: {map_path} does not exist")
                return
            self.map = PyramidMap(map_path)
            self.last_map = None
            self.display_message(f"Loaded map from: \n{map_path}")
        elif not self.map:
            self.display_message("ERROR: No map loaded")
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
            new_object = ImageObject(image_path, H, window_size=self.window_size)
            if not new_object.state == ImageState.NOT_VALID:
                new_objects.append(new_object)
            else:
                self.display_message(f"ERROR: {new_object.error_message}")
        
        if len(new_objects) > 0:
            self.image = None
            self.panorama_cache = None  
            self.button_panel.add_new_images(new_objects)
            self.display_message(f"Loaded {len(new_objects)} image(s) from: \n{file_path}")
        else:
            self.display_message(f"ERROR: No valid image pairs found in CSV: \n{file_path}")
        
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

    def match_images(self, cur, prev):
        if prev is None:
            return
        if not (cur.state == ImageState.NOT_LOADED or cur.state == ImageState.LOADED or cur.state == ImageState.MATCHED):
            return
        if (prev.state == ImageState.MATCHED and cur.state == ImageState.MATCHED): 
            return
 
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
        
    def update_global_scale(self, val):
        self.global_scale = np.clip(float(val),0.1,10)
        self.render()
    
    def sync_sliders(self):
        self.button_panel.rotation_slider.set(self.image.rotation)
        self.button_panel.scale_slider.set(np.log10(self.image.scale))
        
    def toggle_help_mode(self):
        self.help_mode = not self.help_mode
        self.button_panel.help_button.config(text="Help:  ON" if self.help_mode else "Help: OFF", bg=('grey' if self.help_mode else 'white'))
        self.clear_messages()
        self.button_panel.show_help(self.help_mode)
        
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
        elif event.char == 'i':
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
        elif event.char == 's':
            self.button_panel.add_starred_image(self.image)
        elif event.keysym == 'Right':
            self.button_panel.select_image(self.button_panel.current_index + 1)
        elif event.keysym == 'Left' or event.keysym == 'Up':
            self.button_panel.select_image(self.button_panel.current_index - 1)
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
    app.load_csv("output/simulated_list2.csv")

    root.mainloop()

if __name__ == "__main__":
    main()
