#!/usr/bin/env python3

import subprocess
import rasterio
import time
import os
import numpy as np
import pymap3d as pm
import shutil

def enu2gps(map_object, x):
	reference_point = (map_object.bounds.bottom, map_object.bounds.left, 0)
	return np.array(pm.enu2geodetic(*x, *reference_point))

def gps2enu(map_object, x, y=0):
	reference_point = (map_object.bounds.bottom, map_object.bounds.left, 0)
	return np.array(pm.geodetic2enu(*x, y, *reference_point))

def gps2pix(map_object, gps):
	return np.array(map_object.index(gps[1], gps[0])[::-1])

def pix2gps(map_object, xy):
	return np.array(map_object.xy(xy[1], xy[0])[::-1])

def calc_map_aspect_ratio(map_object):
    eps = 10000
    shift_enu = gps2pix(map_object, enu2gps(map_object, (eps, eps, 0))) - gps2pix(map_object, enu2gps(map_object, (0, 0, 0)))
    map_aspect_ratio = np.abs(shift_enu[:2]).astype(float)
    map_aspect_ratio /= np.max(map_aspect_ratio)
    return map_aspect_ratio

def transform_geotiff(input_tif, output_folder, filename=None, grayscale=False, resize_factors=None, crop_box=None, fix_aspect_ratio=True):
    # Open the input GeoTIFF file
    input_ds = rasterio.open(input_tif)
    
    if fix_aspect_ratio:
        map_aspect_ratio = calc_map_aspect_ratio(input_ds)
    else:
        map_aspect_ratio = [1, 1]
    # print(map_aspect_ratio)
    # return

    if input_ds is None:
        print("Error: Could not open input GeoTIFF.")
        return
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    
    if crop_box is not None:
        projwin = f'-projwin {crop_box[0]} {crop_box[3]} {crop_box[2]} {crop_box[1]} '
    else:
        projwin = ''
    
    if filename is None:
        filename = os.path.basename(input_tif).split('.')[0] + '_tiled'
    

    if grayscale:
        exe = f'gdal_calc.py -R {input_tif} --R_band=1 -G {input_tif} --G_band=2 -B {input_tif} --B_band=3 --outfile={output_folder}/{filename}.tif --calc="R*0.2989+G*0.5870+B*0.1140" '# --co TILED=YES'
        if projwin:
            exe += f' -{projwin}'
    else:
        output_size = [int(input_ds.width/map_aspect_ratio[0]), int(input_ds.height/map_aspect_ratio[1])]
        exe = f'gdalwarp -co TILED=YES -ts {output_size[0]} {output_size[1]} {projwin} {input_tif} {output_folder}/{filename}.tif' 

    subprocess.Popen(exe, shell=True).wait()

    for factor in resize_factors:
        exe = 'gdalwarp -co TILED=YES '# -ot Byte '
        
        output_size = [int(input_ds.width/factor/map_aspect_ratio[0]), int(input_ds.height/factor/map_aspect_ratio[1])]
        
        exe += f'-ts {output_size[0]} {output_size[1]} '
        
        tif_src = f'{output_folder}/{filename}.tif'

        tif_dst = f'{output_folder}/{filename}_{factor:02d}x.tif'
        
        exe += tif_src + ' ' + tif_dst

        subprocess.Popen(exe, shell=True).wait()
        
if __name__ == "__main__":
    input_tiff = 'input/hrscd.tif'
    output_folder = 'input/hrscd/'
    filename = 'map'
    grayscale_conversion = False
    resize_factors = [2, 4, 8, 16]
    crop_region = None #= (start_lon, end_lon, start_lat, end_lat)  # Define your crop box here
    fix_aspect_ratio = False

    transform_geotiff(input_tiff, output_folder, filename, grayscale=grayscale_conversion, resize_factors=resize_factors, crop_box=crop_region, fix_aspect_ratio=fix_aspect_ratio)

