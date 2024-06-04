import numpy as np
import os
from osgeo import gdal, osr

def align_image(query_image, H_loftr, H_tile, target_size = 504):
    
    H_tot = np.linalg.inv(H_loftr) @ H_tile
    
    (h,w) = query_image.shape[:2]
    corners = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).astype(np.float32)
    map_corners = apply_tform(np.linalg.inv(H_tot), corners)

    bx0, bx1, by0, by1 = calc_bbox2(map_corners)

    bbox_pix = np.array([[bx0,by0], [bx1,  by1]])

    ds_ratio = target_size / abs(bx1 - bx0)
    S = np.diag([ds_ratio, ds_ratio, 1])

    output_size = [target_size,abs(int(target_size*(by1 - by0)/(bx1 - bx0)))]

    start = np.array([bx0, by0])
    T = translation_matrix(-start)
    H_crop = np.matmul(S, T)

    crop = map_object.warp_map(H_crop, output_size)

    warped_corners = apply_tform(H_crop, map_corners).astype(int)
    bx0, bx1, by0, by1 = calc_bbox2(warped_corners)
    loc_guess = np.array([(bx0 + bx1) / 2, (by0 + by1) / 2]) - np.array(output_size) / 2

    # calculate second image warp
    new_corners2 = warped_corners - np.array(loc_guess)[np.newaxis]
    H_align = estimate_transform(corners, new_corners2, 3)
    query_aligned = cv2.warpPerspective(query_image, H_align, output_size)
    
    bbox_gps = []
    for corner in bbox_pix:
        gps_corner = pix2gps(map_object.map_object,corner)
        bbox_gps.append(gps_corner)
    bbox_gps = np.array(bbox_gps)
    
    return query_aligned, bbox_gps

def getGeoTransform(extent, nx, ny):
    resx = (extent[2] - extent[0]) / ny
    resy = (extent[3] - extent[1]) / nx
    return [extent[1], resy, 0, extent[0] , 0, resx]

def dump_geotif(query_aligned, bbox_gps, output_file):
    
    extent = bbox_gps.flatten()
    driver = gdal.GetDriverByName('GTiff')
    
    (im_h, im_w, im_c) = query_aligned.shape
    data_type = gdal.GDT_Float32 
    
    #options = ['COMPRESS=JPEG', 'JPEG_QUALITY=80', 'TILED=YES']
    grid_data = driver.Create('grid_data', im_w, im_h, im_c, data_type)#, options)
    
    for n in range(im_c):
        grid_data.GetRasterBand(n+1).WriteArray(query_aligned[:,:,n])

    srs = osr.SpatialReference()
    srs.ImportFromProj4('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
    
    grid_data.SetProjection(srs.ExportToWkt())
    grid_data.SetGeoTransform(getGeoTransform(extent, im_w, im_h))
    
    driver.CreateCopy(output_file, grid_data, 0)  
    
    driver = None
    grid_data = None
    os.remove('grid_data')
    
query_aligned, bbox_gps = align_image(query_resized, H_loftr, dataset[ind_gt]['H'])
dump_geotif(query_aligned, bbox_gps, 'crop.tif')
