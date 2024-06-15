import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.enums import ColorInterp
from pyproj import CRS

import os
import pickle as pkl

def getGeoTransform(extent, nx, ny):
    resx = (extent[2] - extent[0]) / ny
    resy = (extent[3] - extent[1]) / nx
    return [extent[1], resy, 0, extent[0] , 0, resx]

from osgeo import gdal, osr
def dump_geotif_old(query_aligned, bbox_gps, output_file):
    
    extent = bbox_gps.flatten()
    driver = gdal.GetDriverByName('GTiff')
    
    (im_h, im_w, _) = query_aligned.shape
    data_type = gdal.GDT_Byte
    
    #options = ['COMPRESS=JPEG', 'JPEG_QUALITY=80', 'TILED=YES']
    grid_data = driver.Create('grid_data', im_w, im_h, 4, data_type, options=["ALPHA=YES"])
    mask = ((np.sum(query_aligned, axis=2) > 0)*255).astype(np.uint8)
    
    colors = [
        gdal.GCI_RedBand,
        gdal.GCI_GreenBand,
        gdal.GCI_BlueBand,
    ]
    for n in range(3):
        grid_data.GetRasterBand(n+1).WriteArray(query_aligned[:,:,n])
        grid_data.GetRasterBand(n+1).SetRasterColorInterpretation(colors[n])
    grid_data.GetRasterBand(4).WriteArray(mask)
    grid_data.GetRasterBand(4).SetRasterColorInterpretation(gdal.GCI_AlphaBand)

    srs = osr.SpatialReference()
    srs.ImportFromProj4('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
    
    grid_data.SetProjection(srs.ExportToWkt())
    grid_data.SetGeoTransform(getGeoTransform(extent, im_w, im_h))
    
    driver.CreateCopy(output_file, grid_data, 0)  
    
    driver = None
    grid_data = None
    os.remove('grid_data')
    return

def dump_geotif(query_aligned, bbox_gps, output_file):
    extent = bbox_gps.flatten()
    (im_h, im_w, _) = query_aligned.shape
    data_type = rasterio.uint8
    transform = from_bounds(extent[1], extent[2], extent[3], extent[0], width=im_w, height=im_h)
    crs = CRS.from_proj4('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs').to_wkt()

    mask = ((np.sum(query_aligned, axis=2) > 0) * 255).astype(np.uint8)
    
    with rasterio.open(
        output_file, 'w',
        driver='GTiff',
        height=im_h,
        width=im_w,
        count=4,
        dtype=data_type,
        crs=crs,
        transform=transform,
        photometric='RGB'
    ) as dst:
        dst.write(query_aligned[:, :, 0].squeeze(), 1)
        dst.write(query_aligned[:, :, 1].squeeze(), 2)
        dst.write(query_aligned[:, :, 2].squeeze(), 3)
        dst.write(mask.squeeze(), 4)
        
        dst.colorinterp = [
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha
        ]


with open('debug.pkl', 'rb') as f:
    query_aligned, bbox_gps = pkl.load(f)

for i in range(10):
    output_file = f'debug/new_{i:02d}.tif'
    old_file = f'debug/old_{i:02d}.tif'
    # if not os.path.exists(output_file):
    dump_geotif(query_aligned, bbox_gps, output_file)
    dump_geotif_old(query_aligned, bbox_gps, old_file)
    break