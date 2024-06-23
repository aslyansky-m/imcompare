import os
import cv2
import numpy as np
from enum import Enum


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
        
    def plot(self, canvas, M_global):
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


def calc_transform(shape, scale, rotation, x_offset, y_offset):
    w, h = shape[:2]
    M1 = cv2.getRotationMatrix2D((w/2, h/2), rotation, scale)
    M2 = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
    M1 = np.vstack([M1, [0, 0, 1]])
    M2 = np.vstack([M2, [0, 0, 1]])
    M = np.dot(M2,M1)
    return M

def calc_homography(anchors):
    if len(anchors) < 4:
        return np.eye(3)
    pts0 = np.array([anchor.pos0 for anchor in anchors], dtype=np.float32)
    pts1 = np.array([anchor.pos for anchor in anchors], dtype=np.float32)
    H, _ = cv2.findHomography(pts0, pts1)
    if H is None:
        return np.eye(3)
    return H

def decompose_homography(H, image_size, scale_ratio):
    h, w = image_size[:2]
    corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    M, _ = cv2.estimateAffinePartial2D(corners.reshape(-1, 2), transformed_corners)

    scale = np.sqrt(np.linalg.det(M[:2,:2]))
    rotation = -np.rad2deg(np.arctan2(M[1, 0], M[0, 0]))

    M2 = calc_transform((w*scale_ratio,h*scale_ratio),scale,rotation,0,0)
    M1 = np.vstack([M, [0, 0, 1]])
    M3 = M1@np.linalg.inv(M2)
    translation = M3[:2, 2]

    H1 = calc_transform((w*scale_ratio,h*scale_ratio), scale, rotation, translation[0], translation[1])
    H2 = H@np.linalg.inv(H1)
    
    return translation, rotation, scale, H2

def apply_homography(H, point):
    pt = np.array([point[0], point[1], 1])
    pt = H @ pt
    pt = pt[:2]/pt[2]
    return pt

def draw_grid(image, H, grid_spacing=100, color=(192, 192, 192), thickness=1):
    height, width = image.shape[:2]
    
    new_grid_spacing_x = H[0, 0] * grid_spacing
    new_grid_spacing_y = H[1, 1] * grid_spacing
    
    max_diff = 5.0
    factor = np.ceil(np.log(grid_spacing/new_grid_spacing_x)/np.log(max_diff))
    new_grid_spacing_x *= max_diff**factor
    new_grid_spacing_y *= max_diff**factor
    new_start_x = H[0, 2]
    new_start_y = H[1, 2]
    new_start_x -= np.round(new_start_x / new_grid_spacing_x)*new_grid_spacing_x
    new_start_y -= np.round(new_start_y / new_grid_spacing_y)*new_grid_spacing_y

    x = new_start_x
    while x < width:
        cv2.line(image, (int(x), 0), (int(x), height), color, thickness)
        x += new_grid_spacing_x
    
    y = new_start_y
    while y < height:
        cv2.line(image, (0, int(y)), (width, int(y)), color, thickness)
        y += new_grid_spacing_y
    
    return image

def edge_detection(image, blur=5, low_threshold=80, high_threshold=150):
    blur = max(1, blur)
    blur = blur + 1 if blur % 2 == 0 else blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur, blur), 1.4)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    edges = (cv2.boxFilter(edges, -1, (3, 3)) > 0).astype(np.uint8) * 255
    output = (np.clip(edges + gray*0.6,0,255)).astype(np.uint8)
    return output

# enum for state
class ImageState(Enum):
    NOT_VALID = 0
    NOT_LOADED = 1
    INITIALIZED = 2
    LOADED = 3
    MOVED = 4
    MATCHED = 5
    LOCKED = 6
    PANORAMA = 7
    
    @staticmethod
    def to_color(state):
        colors = {
            ImageState.NOT_VALID: 'red',
            ImageState.NOT_LOADED: 'white',
            ImageState.INITIALIZED: 'Slategray1',
            ImageState.LOADED: 'ivory',
            ImageState.MOVED: 'SeaGreen2',
            ImageState.MATCHED: 'RosyBrown1',
            ImageState.LOCKED: 'saddle brown',
            ImageState.PANORAMA: 'violet'
        }
        return colors[state]
    
class ImageObject:
    def __init__(self, image_path, M_anchors = None, window_size=(800, 600)):
        self.image_path = image_path
        self.state = ImageState.NOT_VALID
        self.scale = 1.0
        self.rotation = 0
        self.x_offset = 0
        self.y_offset = 0
        self.anchors = []
        self.scale_ratio = 1.0
        self.M_anchors = M_anchors
        self.M_original = np.eye(3)
        self.error_message = ''
        self.image = None
        self.window_size = window_size

        self.state_stack = []
        self.current_state_index = -1
        
        self.derived_images = []
        self.is_panorama = False
        
        if not os.path.exists(image_path):
            self.error_message = f"Could not find image: {image_path}"
            return
        
        if M_anchors is None:
            self.state = ImageState.NOT_LOADED
        else:
            self.state = ImageState.INITIALIZED

        
    def get_image(self):
        if self.state == ImageState.NOT_VALID:
            return None
        if self.state == ImageState.NOT_LOADED or self.image is None:
            try:
                self.image = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)
                self.scale_ratio = min(self.window_size[0] / self.image.shape[1], self.window_size[1] / self.image.shape[0])
                self.M_original = np.diag([self.scale_ratio, self.scale_ratio, 1])
                if self.M_anchors is not None:
                    H = self.M_anchors @ np.linalg.inv(self.M_original)
                    self.initialize_from_homography(H)
                    self.state = ImageState.MOVED
                else:
                    self.M_anchors = np.eye(3)
                    self.state = ImageState.LOADED
                    
                self.save_state()
            except Exception as e:
                print(e)
                return None
            
        return self.image

    def save_state(self):
        if self.image is None:
            return
        current_state = (self.scale, self.rotation, self.x_offset, self.y_offset, [(a.pos, a.original) for a in self.anchors], self.M_anchors, self.state)
        if len(self.state_stack) > 0:
            previous_state = self.state_stack[self.current_state_index]
            identical = (np.linalg.norm(np.array(previous_state[:4]) - np.array(current_state[:4]) ) < 1e-3) and (np.linalg.norm(previous_state[5]-current_state[5]) < 1e-3)
            if identical:
                return
        self.state_stack = self.state_stack[:self.current_state_index + 1]
        self.state_stack.append(current_state)
        self.current_state_index += 1

    def undo(self):
        if self.current_state_index > 0:
            self.current_state_index -= 1
            self.load_state()

    def redo(self):
        if self.current_state_index < len(self.state_stack) - 1:
            self.current_state_index += 1
            self.load_state()
    
    def is_identity(self):
        if self.M_anchors is not None and np.linalg.norm(self.M_anchors-np.eye(3)) > 1e-6:
            return False
        return self.x_offset == 0 and self.y_offset == 0 and self.scale == 1.0 and self.rotation == 0

    def load_state(self):
        if 0 <= self.current_state_index < len(self.state_stack):
            state = self.state_stack[self.current_state_index]
            self.scale, self.rotation, self.x_offset, self.y_offset, anchor_states, self.M_anchors, self.state = state
            self.anchors = [Anchor(pos[0], pos[1], original) for pos, original in anchor_states]

    def reset_anchors(self):
        image = self.get_image()
        if image is not None:
            m = 30
            w = image.shape[1]
            h = image.shape[0]
            anchors_pos = [(m, m), (m, h - m), (w - m, m), (w - m, h - m)]
            M = calc_transform((image.shape[1] * self.scale_ratio, image.shape[0] * self.scale_ratio), self.scale, self.rotation, self.x_offset, self.y_offset)
            anchors_pos = [apply_homography(self.M_anchors @ M @ self.M_original, pos) for pos in anchors_pos]
            self.anchors = [Anchor(x, y, original=True) for x, y in anchors_pos]
        else:
            m = 100
            w = self.window_size[0]
            h = self.window_size[1]
            anchors_pos = [(m, m), (m, h - m), (w - m, m), (w - m, h - m)]
            self.anchors = [Anchor(x, y, original=True) for x, y in anchors_pos]
    
    def initialize_from_homography(self, H):
        translation, rotation, scale, H_residual = decompose_homography(H, self.image.shape, self.scale_ratio)

        self.scale = scale
        self.rotation = rotation
        self.x_offset = translation[0]
        self.y_offset = translation[1]
        self.M_anchors = H_residual
        self.reset_anchors()
        return 

    def render(self, M_global, window_size):
        if self.state == ImageState.NOT_VALID:
            return None
        image = self.get_image()
        if image is None:
            return None
        M = calc_transform([image.shape[1] * self.scale_ratio, image.shape[0] * self.scale_ratio], self.scale, self.rotation, self.x_offset, self.y_offset)
        H = calc_homography(self.anchors)
        
        result = cv2.warpPerspective(image, M_global @ H @ self.M_anchors @ M @ self.M_original, window_size)
        self.update_derived_images()
        return result
    
    def update_derived_images(self):
        H_prev = self.M_anchors @ calc_transform([self.image.shape[1] * self.scale_ratio, self.image.shape[0] * self.scale_ratio], self.scale, self.rotation, self.x_offset, self.y_offset) @ self.M_original
        for cur, H_rel in self.derived_images:
            if cur is None:
                continue
            if not (cur.state == ImageState.MATCHED or cur.state == ImageState.LOADED):
                continue
            cur_H = H_prev @ H_rel @ np.linalg.inv(cur.M_original)
            cur.initialize_from_homography(cur_H)
            cur.reset_anchors()
            cur.state = ImageState.MATCHED

    def push_anchor(self, pt):
        min_dist = -1
        closest_anchor = None
        for anchor in self.anchors:
            dist = np.linalg.norm(np.array(anchor.pos) - np.array(pt))
            if min_dist < 0 or dist < min_dist:
                min_dist = dist
                closest_anchor = anchor
        new_anchor = Anchor(pt[0], pt[1])
        self.anchors.remove(closest_anchor)
        self.anchors.append(new_anchor)
        self.save_state()

    def relative_transform(self):
        if self.image is None:
            return np.eye(3)
        M = calc_transform([self.image.shape[1] * self.scale_ratio, self.image.shape[0] * self.scale_ratio], self.scale, self.rotation, self.x_offset, self.y_offset)
        H = calc_homography(self.anchors)
        T = H @ self.M_anchors @ M @ self.M_original
        return T

    def __str__(self):
        T = self.relative_transform()
        return f"{self.image_path}, {','.join([str(x) for x in T.flatten().tolist()])}"
    
    @staticmethod
    def create_panorama(image, H = None, name="panorama", window_size=(800, 600)):
        o = ImageObject(name,window_size=window_size)
        o.is_panorama = True
        o.state = ImageState.PANORAMA
        o.image = image
        o.scale_ratio = min(window_size[0] / o.image.shape[1], window_size[1] / o.image.shape[0])
        o.M_original = np.diag([o.scale_ratio, o.scale_ratio, 1])
        if H is not None:
            cur_H = H @ np.linalg.inv(o.M_original)
            o.initialize_from_homography(cur_H)
        else:
            o.M_anchors = np.eye(3)
        return o
       
