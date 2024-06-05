import numpy as np
import os
from osgeo import gdal, osr
from common import *



image = cv2.imread('output/simulated/image_00.jpg')
map_object = PyramidMap('output/hrscd/map.tif')
H_tot = "[ 2.22442709e+00 -1.28667621e+00  4.88697573e+02  1.28357689e+00  2.21928138e+00  3.71062741e+00  4.04685986e-06 -3.32916608e-06  1.00000005e+00]"
H_tot = H_tot.replace('[','').replace(']','').split()
H_tot = [float(i) for i in H_tot]
H_tot = np.array(H_tot).reshape(3,3)

H_tot = np.linalg.inv(H_tot)
query_aligned, bbox_gps = align_image(image, map_object, H_tot, target_size = 504)
dump_geotif(query_aligned, bbox_gps, 'crop.tif')

