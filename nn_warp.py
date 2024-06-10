from time import time
import numpy as np
import cv2

class SmoothWarp:
    def __init__(self, im_size, k=5):
        self.imx = im_size[1]
        self.imy = im_size[0]
        self.k = k
        
    def warp(self, image, src, dst):
        Nx = 100
        Ny = 100
        dnx = (Nx-3)
        dx = (self.imx-1)
        dny = (Nx-3)
        dy = (self.imy-1)

        x_ = np.linspace(-dx/dnx, dx + dx/dnx, Nx)
        y_ = np.linspace(-dy/dny, dy + dy/dny, Ny)
        
        x_grid, y_grid = np.meshgrid(x_, y_)
        
        pt_in = np.stack([x_grid.ravel(), y_grid.ravel(), np.ones_like(x_grid.ravel())], axis=-1)
        
        pts_out = np.zeros((pt_in.shape[0], 2))

        for i in range(pt_in.shape[0]):
            weights = self._compute_weights(pt_in[i, :2], src)
            H = self._weighted_homography(src, dst, weights)
            transformed_pt = (H @ pt_in[i].T).T
            pts_out[i] = transformed_pt[:2] / transformed_pt[2]

        pts_out = pts_out.reshape(Ny, Nx, 2).astype(np.float32)
        
        xs = np.linspace(1, Nx-2, self.imx, dtype=np.float32)
        ys = np.linspace(1, Ny-2, self.imy, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        
        coords_out = cv2.remap(pts_out, xx, yy, interpolation=cv2.INTER_LINEAR)
        coords_out = coords_out.reshape(self.imy, self.imx, 2)
        
        image_out = cv2.remap(image, coords_out, None, cv2.INTER_CUBIC)
        
        return image_out
    
    def _compute_weights(self, point, src):
        # Compute weights based on distance
        distances = np.linalg.norm(src - point, axis=1)
        sigma = np.mean(distances) / 2
        weights = np.exp(-distances**2 / (2 * sigma**2))
        return weights
    
    def _weighted_homography(self, src, dst, weights):
        A = []
        for i in range(len(src)):
            x, y = src[i][0], src[i][1]
            xp, yp = dst[i][0], dst[i][1]
            w = weights[i]
            A.append([-x*w, -y*w, -w, 0, 0, 0, xp*x*w, xp*y*w, xp*w])
            A.append([0, 0, 0, -x*w, -y*w, -w, yp*x*w, yp*y*w, yp*w])
        
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        
        return H
