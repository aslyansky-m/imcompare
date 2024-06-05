import cv2
import numpy as np
import math
import pymap3d as pm
from skimage import transform
from tifffile import TiffFile
import rasterio
from scipy.ndimage import gaussian_filter1d
from glob import glob


def calc_bbox2(map_corners):
	bx0 = np.min(map_corners[:, 0])
	bx1 = np.max(map_corners[:, 0])
	by0 = np.min(map_corners[:, 1])
	by1 = np.max(map_corners[:, 1])
	return bx0, bx1, by0, by1


def estimate_transform(src, dst, ttype=3):
	if ttype == 0:
		min_x = np.min(dst[:, 0])
		max_x = np.max(dst[:, 0])
		min_y = np.min(dst[:, 1])
		max_y = np.max(dst[:, 1])
		dst = np.array([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])
		ttype = 3

	tform_type = ['euclidean', 'similarity', 'affine', 'projective'][ttype]
	tform = transform.estimate_transform(tform_type, src, dst)
	return tform.params


def warp_map(map_image, tform, output_shape):
	[H, W] = map_image.shape[:2]
	(w, h) = output_shape
	corners = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
	new_corners = apply_tform(np.linalg.inv(tform), corners)
	min_x = np.floor(np.min(new_corners[:, 0])).astype(int)
	max_x = np.ceil(np.max(new_corners[:, 0])).astype(int)
	min_y = np.floor(np.min(new_corners[:, 1])).astype(int)
	max_y = np.ceil(np.max(new_corners[:, 1])).astype(int)
	min_x = max(0, min_x)
	max_x = min(W, max_x)
	min_y = max(0, min_y)
	max_y = min(H, max_y)
	topleft = np.hstack([min_x, min_y])
	crop = map_image[min_y:max_y, min_x:max_x, :]
	T = np.eye(3)
	T[:2, 2] += topleft
	tform_fixed = np.matmul(tform, T)
	im_out = cv2.warpPerspective(crop, tform_fixed, output_shape)
	return im_out


def euler2mat(theta):
	R_x = np.array([[1, 0, 0],
					[0, math.cos(theta[0]), -math.sin(theta[0])],
					[0, math.sin(theta[0]), math.cos(theta[0])]
					])

	R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
					[0, 1, 0],
					[-math.sin(theta[1]), 0, math.cos(theta[1])]
					])

	R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
					[math.sin(theta[2]), math.cos(theta[2]), 0],
					[0, 0, 1]
					])

	R = np.dot(R_z, np.dot(R_y, R_x))

	return R


def apply_tform(H, corners):
	res = H @ np.hstack([corners, np.ones((corners.shape[0], 1))]).T
	res = (res[:2] / res[2]).T
	return res

def gps2enu(map_object, x, y=0):
	reference_point = (map_object.bounds.bottom, map_object.bounds.left, 0)
	return np.array(pm.geodetic2enu(*x, y, *reference_point))


def enu2gps(map_object, x):
	reference_point = (map_object.bounds.bottom, map_object.bounds.left, 0)
	return np.array(pm.enu2geodetic(*x, *reference_point))


def gps2pix(map_object, gps):
	return np.array(map_object.index(gps[1], gps[0])[::-1])


def pix2gps(map_object, xy):
	return np.array(map_object.xy(xy[1], xy[0])[::-1])


def project_cam2world(points, K, Ki, angle, position, altitude):
	angle_fixed = angle.copy()
	angle_fixed[2] = math.pi - angle_fixed[2]
	H = euler2mat(angle_fixed) @ Ki
	coord = apply_tform(H, points) * altitude + position[:2]
	coord = np.hstack([coord, np.zeros((coord.shape[0], 1))])
	return coord

def rotation_matrix(theta):
	R = np.array([[math.cos(theta), -math.sin(theta), 0],
					[math.sin(theta), math.cos(theta), 0],
					[0, 0, 1]
					])
	return R

def translation_matrix(t):
	T = np.eye(3)
	T[:2, 2] = t
	return T

def scale_matrix(s):
	S = np.diag([s,s,1])
	return S


def calc_map_aspect_ratio(map_object):
	map_aspect_ratio = gps2pix(map_object, enu2gps(map_object, (10000, -10000, 0)))
	map_aspect_ratio = map_aspect_ratio[0] / map_aspect_ratio[1]
	return map_aspect_ratio


def imdiff(A,B):
	C = np.dstack((B,A,B))
	return C


def normalize_image(im_in):
	if im_in.dtype==np.uint8:
		im = im_in.astype(float)/255.0
	a = np.std(im,axis=(0,1))
	b = np.mean(im,axis=(0,1))
	im = (im-b)/(5*a) + 0.5
	im = np.clip(im,0,1)
	if im_in.dtype==np.uint8:
		im = (im*255.0).astype(np.uint8)
	return im


def calc_camera_matrix(f,frame_shape):
	w = frame_shape[0]
	h = frame_shape[1]
	cx = (w - 1) / 2
	cy = (h - 1) / 2
	K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
	return K


def get_crop(page, i0, j0, h, w):
	"""Extract a crop from a TIFF image file directory (IFD).
	Parameters
	----------
	page : TiffPage
		TIFF image file directory (IFD) from which the crop must be extracted.
	i0, j0: int
		Coordinates of the top left corner of the desired crop.
	h: int
		Desired crop height.
	w: int
		Desired crop width.
	Returns
	-------
	out : ndarray of shape (imagedepth, h, w, sampleperpixel)
		Extracted crop.
	"""

	if not page.is_tiled:
		raise ValueError("Input page must be tiled.")

	im_width = page.imagewidth
	im_height = page.imagelength

	if h < 1 or w < 1:
		raise ValueError("h and w must be strictly positive.")

	if i0 < 0 or j0 < 0 or i0 + h >= im_height or j0 + w >= im_width:
		raise ValueError("Requested crop area is out of image bounds.")

	tile_width, tile_height = page.tilewidth, page.tilelength
	i1, j1 = i0 + h, j0 + w

	tile_i0, tile_j0 = i0 // tile_height, j0 // tile_width
	tile_i1, tile_j1 = np.ceil([i1 / tile_height, j1 / tile_width]).astype(int)

	tile_per_line = int(np.ceil(im_width / tile_width))

	out = np.empty((page.imagedepth,
					(tile_i1 - tile_i0) * tile_height,
					(tile_j1 - tile_j0) * tile_width,
					page.samplesperpixel), dtype=page.dtype)

	fh = page.parent.filehandle

	jpegtables = page.tags.get('JPEGTables', None)
	if jpegtables is not None:
		jpegtables = jpegtables.value

	for i in range(tile_i0, tile_i1):
		for j in range(tile_j0, tile_j1):
			index = int(i * tile_per_line + j)

			offset = page.dataoffsets[index]
			bytecount = page.databytecounts[index]

			fh.seek(offset)
			data = fh.read(bytecount)

			tile, indices, shape = page.decode(data, index)

			im_i = (i - tile_i0) * tile_height
			im_j = (j - tile_j0) * tile_width
			out[:, im_i: im_i + tile_height, im_j: im_j + tile_width, :] = tile

	im_i0 = i0 - tile_i0 * tile_height
	im_j0 = j0 - tile_j0 * tile_width

	return np.squeeze(out[:, im_i0: im_i0 + h, im_j0: im_j0 + w, :])


def warp_map_tiled(map_page, tform, output_shape):
	[H, W] = [map_page.imagelength,map_page.imagewidth]
	(w, h) = output_shape
	corners = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
	new_corners = apply_tform(np.linalg.inv(tform), corners)
	min_x = np.floor(np.min(new_corners[:, 0])).astype(int)
	max_x = np.ceil(np.max(new_corners[:, 0])).astype(int)
	min_y = np.floor(np.min(new_corners[:, 1])).astype(int)
	max_y = np.ceil(np.max(new_corners[:, 1])).astype(int)
	min_x = min(W-1, max(0, min_x))
	max_x = min(W-1, max(0, max_x))
	min_y = min(H-1, max(0, min_y))
	max_y = min(H-1, max(0, max_y))
	topleft = np.hstack([min_x, min_y])
	T = np.eye(3)
	T[:2, 2] += topleft
	tform_fixed = np.matmul(tform, T)
 
	if min_x == max_x or min_y == max_y:
		return None
	
	crop = get_crop(map_page, min_y, min_x, max_y-min_y, max_x-min_x)
	
	im_out = cv2.warpPerspective(crop, tform_fixed, output_shape)
	
	return im_out


def calc_crop(map_shape, map_corners, template_size, target_size, random=True, perm_index=-1, fixed_margin=1000):
	scale_ratio = target_size/template_size
	# calculate possible crop
	mx = map_shape[1]
	my = map_shape[0]
	bx0, bx1, by0, by1 = calc_bbox2(map_corners)
	template_size_map = max(bx1 - bx0, by1 - by0)
	target_size_map = template_size_map * scale_ratio
	sx = sy = target_size_map
	cx0 = max(0, bx1 - sx)
	cx1 = min(bx0, mx - sx)
	cy0 = max(0, by1 - sy)
	cy1 = min(by0, my - sy)
	cx1 = max(cx0, cx1)
	cy1 = max(cy0, cy1)
 
	margin = 10
	center = np.array([(cx0 + cx1) / 2, (cy0 + cy1) / 2])
	# choose translation
	if random:
		start_x = np.random.randint(min(cx0+margin,center[0]),max(cx1-margin,center[0]+1))
		start_y = np.random.randint(min(cy0+margin,center[1]),max(cy1-margin,center[1]+1))
		start = np.array([start_x, start_y])
	elif perm_index == -1:
		start = center
	else:
		possible_perms = [np.array([center[0] - fixed_margin, center[1] - fixed_margin]),
                          np.array([center[0] + fixed_margin, center[1] - fixed_margin]),
                          np.array([center[0] - fixed_margin, center[1] + fixed_margin]),
                          np.array([center[0] + fixed_margin, center[1] + fixed_margin]),
                          np.array([center[0], center[1] - fixed_margin]),
                          np.array([center[0], center[1] + fixed_margin]),
                          np.array([center[0] - fixed_margin, center[1]]),
                          np.array([center[0] + fixed_margin, center[1]]),
                          center]
		start = possible_perms[perm_index % len(possible_perms)]
		

	# calculate scale
	ds_ratio = template_size / template_size_map
	H = np.diag([ds_ratio, ds_ratio, 1])

	# combine transforms
	T = translation_matrix(-start)
	H_tot = np.matmul(H, T)

	#perform check
	new_corners = apply_tform(H_tot, map_corners).astype(int)
	bx0, bx1, by0, by1 = calc_bbox2(new_corners)

	fx0 = (bx0 + bx1) / 2 - template_size / 2
	fy0 = (by0 + by1) / 2 - template_size / 2
	fx1 = fx0 + template_size
	fy1 = fy0 + template_size

	if fx0 < 0 or fx1 >= target_size or fy0 < 0 or fy1 >= target_size:
		if not random:
			return calc_crop(map_shape, map_corners, template_size, target_size, random=False)
		else:
			return None
	return H_tot


def add_noise(im_in, noise_std):
	im_shape = im_in.shape

	gaussian_std = np.random.rand()*noise_std*3
	row_std = np.random.rand()*noise_std

	gaussian_noise = np.random.randn(*im_shape)*gaussian_std
	gaussian_noise = np.random.randn(*im_shape)*gaussian_std + 3*cv2.GaussianBlur(gaussian_noise,(61,61),cv2.BORDER_DEFAULT)
	row_noise = np.random.randn(im_shape[0])*row_std
	row_noise = np.random.randn(im_shape[0])*row_std + 3*gaussian_filter1d(row_noise,201)
	row_noise = np.repeat(np.expand_dims(row_noise,1),im_shape[1],1)

	im_out = np.clip(im_in + gaussian_noise + row_noise,0,1)

	return im_out

class PyramidMap:
    def __init__(self, map_file) -> None:
        effective_altitude = 500
        self.map_file = map_file
        self.map_files = sorted(glob(map_file.split('.')[-2] + '*'))
        self.map_pages = [TiffFile(map_file).pages[0] for map_file in self.map_files]
        self.map_object = rasterio.open(self.map_file)
        self.map_boundaries = gps2enu(self.map_object, pix2gps(self.map_object, [self.map_object.width,0]))[:2]
        self.shape = [self.map_pages[0].imagelength,self.map_pages[0].imagewidth]
        self.dataset_length = 2*int(np.prod(self.map_boundaries)/effective_altitude**2)
        self.map_initialized = True
        self.scales = [None]*len(self.map_pages)
        for n in range(len(self.map_pages)):
            self.scales[n] = self.map_pages[0].imagelength / self.map_pages[n].imagelength
    
    def warp_map(self, tform, output_shape):
        [H, W] = self.shape
        (w, h) = output_shape
        corners = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
        new_corners = apply_tform(np.linalg.inv(tform), corners)
        min_x = np.floor(np.min(new_corners[:, 0])).astype(int)
        max_x = np.ceil(np.max(new_corners[:, 0])).astype(int)
        min_x = min(W-1, max(0, min_x))
        max_x = min(W-1, max(0, max_x))
        
        target_scale = (max_x-min_x)/output_shape[0]
        
        cur_map = 0
        
        for i in range(1,len(self.scales)):
            if (self.scales[i] > target_scale):
                break
            cur_map = i

        cur_scale = self.scales[cur_map]
        H_scale = scale_matrix(cur_scale)
        H_fixed =  tform @ H_scale

        im_out = warp_map_tiled(self.map_pages[cur_map], H_fixed, output_shape)
        
        return im_out
    
    def pix2gps(self, xy):
        return np.array(self.map_object.xy(xy[1], xy[0])[::-1])

    def gps2pix(self, gps):
        return np.array(self.map_object.index(gps[1], gps[0])[::-1])