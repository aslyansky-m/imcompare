from time import time
import numpy as np
import cv2

class SmoothWarp:
    def __init__(self, im_size, nx=5, ny = 5, w_smooth=0.1, w_rot=0.1):
        self.imx = im_size[1]
        self.imy = im_size[0]
        self.nx = nx
        self.ny = ny
        self.w_smooth = w_smooth
        self.w_rot = w_rot
        self.transforms = np.zeros((ny, nx, 4))
        self.transforms[:, :, 0] = 1
        
    def warp(self, image):
        Nx = 100
        Ny = 100
        dnx = (Nx-3)
        dx = (self.imx-1)
        dny = (Nx-3)
        dy = (self.imy-1)

        x_ = np.linspace(-dx/dnx, dx + dx/dnx, Nx)
        y_ = np.linspace(-dy/dny, dy + dy/dny, Ny)
        
        x_grid, y_grid = np.meshgrid(x_, y_)
        
        wx = x_grid / self.imx * self.nx
        wy = y_grid / self.imy * self.ny

        ix = np.clip(np.floor(wx).astype(int), 0, self.nx-2)
        iy = np.clip(np.floor(wy).astype(int), 0, self.ny-2)

        wx -= ix
        wy -= iy

        w00 = (1 - wx) * (1 - wy)
        w01 = wx * (1 - wy)
        w10 = (1 - wx) * wy
        w11 = wx * wy

        t00 = self.transforms[iy, ix]
        t01 = self.transforms[iy, ix+1]
        t10 = self.transforms[iy+1, ix]
        t11 = self.transforms[iy+1, ix+1]

        coefs = (w00[..., None] * t00 + w01[..., None] * t01 +
                 w10[..., None] * t10 + w11[..., None] * t11)

        pt_in = np.stack([x_grid, y_grid, np.ones_like(x_grid)], axis=-1)
        
        transform = np.zeros((Ny, Nx, 2, 3))
        transform[..., 0, 0] = coefs[..., 0]
        transform[..., 0, 1] = coefs[..., 1]
        transform[..., 0, 2] = coefs[..., 2]
        transform[..., 1, 0] = -coefs[..., 1]
        transform[..., 1, 1] = coefs[..., 0]
        transform[..., 1, 2] = coefs[..., 3]
        
        pts_out = np.einsum('...ij,...j->...i', transform, pt_in)
        
        xs = np.linspace(1, Nx-2, self.imx, dtype = np.float32)
        ys = np.linspace(1, Ny-2, self.imy, dtype = np.float32)
        xx, yy = np.meshgrid(xs, ys)
        coords_out = cv2.remap(pts_out, xx, yy, interpolation=cv2.INTER_LINEAR)
        image_out = cv2.remap(image, coords_out.astype(np.float32), None, cv2.INTER_CUBIC)
        return image_out
    
    def calculate_indices_weights(self, src):
        x = src[:, 0]
        y = src[:, 1]

        wx = x / self.imx * self.nx
        wy = y / self.imy * self.ny

        ix = np.clip(np.floor(wx).astype(int), 0, self.nx-2)
        iy = np.clip(np.floor(wy).astype(int), 0, self.ny-2)

        wx -= ix
        wy -= iy

        w00 = (1 - wx) * (1 - wy)
        w01 = wx * (1 - wy)
        w10 = (1 - wx) * wy
        w11 = wx * wy

        return ix, iy, w00, w01, w10, w11

    def solve(self, src, dst):
        N = src.shape[0]
        Nh = self.nx * self.ny * 4
        H = np.zeros((Nh, Nh))
        v = np.zeros(Nh)
        for n in range(N):
            ix, iy, w00, w01, w10, w11 = self.calculate_indices_weights(src[n,:].reshape(1,2))
            w_block = np.array([w00, w01, w10, w11])
            ind0 = self.nx * iy + ix
            ind1 = (4*np.array([ind0, ind0+1, ind0+self.nx, ind0+self.nx+1]) + np.arange(4)).flatten()
            x, y = src[n,0], src[n,1]
            J_block1 = np.stack([np.array([x, y, 1, 0]), np.array([y, -x, 0, 1])], axis=-1)
            J_block = np.kron(w_block, J_block1)
            r_block = dst[n, :] 
            H_block = J_block @ J_block.T
            v_block = J_block @ r_block.T
            H[np.ix_(ind1, ind1)] += H_block
            v[ind1] += v_block

        H /= N
        v /= N

        J_reg = np.kron(np.array([1,-1]), np.eye(4))
        H_reg = J_reg.T @ J_reg
        for i in range(self.nx):
            for j in range(self.ny):
                ind0 = self.nx * j + i
                if i > 0:
                    ind1 = (4*np.array([ind0, ind0-1]).reshape(-1,1) + np.arange(4).reshape(1,-1)).flatten()
                    H[np.ix_(ind1, ind1)] += self.w_smooth * H_reg
                if j > 0:
                    ind1 = (4*np.array([ind0, ind0-self.nx]).reshape(-1,1) + np.arange(4).reshape(1,-1)).flatten()
                    H[np.ix_(ind1, ind1)] += self.w_smooth * H_reg
                
                ind1 = 4*ind0 + np.arange(2)
                J_rot = np.eye(2)
                r_rot = np.array([1, 0])
                H_rot = J_rot @ J_rot.T
                v_rot = J_rot @ r_rot.T
                H[np.ix_(ind1, ind1)] += self.w_rot * H_rot
                v[ind1] += self.w_rot * v_rot
    
        res = np.linalg.solve(H, v)
        self.transforms = res.reshape(self.ny, self.nx, 4)