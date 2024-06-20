import cv2
import numpy as np
import os
from random import uniform
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from common import rotation_matrix, scale_matrix, translation_matrix


# Load the image
image = cv2.imread('output/hrscd/map.tif')

# Define the size of the output images
output_size = (640, 512)

# Define the number of images to generate
num_images = 30

# Create the output directory if it doesn't exist
output_fld = 'output/simulated2/'
os.makedirs(output_fld, exist_ok=True)

height, width = image.shape[:2] 
image_center = np.array([width/2, height/2]) 
center = np.array(output_size)/2

t = np.linspace(0, 1, num_images)

scales = 1.3*interp1d([0,0.2,0.3,0.4, 1], [0.3,0.4,0.9,1.0,1.0],kind='linear')(t)
shift_x = -image_center[0]*0.5+10*interp1d([0,0.2,0.5, 1], [-50, 50, 20, -20],kind='cubic')(t)
shift_y = -image_center[1]*0.5+10*interp1d([0,0.2,0.5, 1], [-50, -50, 20, 20],kind='cubic')(t)
rotations = interp1d([0,0.2,0.5, 1], [-np.pi/6, 0, 0, np.pi/6],kind='cubic')(t)

# fig, ax = plt.subplots(4, 1, figsize=(6, 6))
# ax[0].plot(t, scales)   
# ax[1].plot(t, shift_x)
# ax[2].plot(t, shift_y)
# ax[3].plot(t, rotations)
# plt.show()

# plt.figure()
# plt.imshow(image)
# plt.show()

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Generate the images
imfile_path = 'output/simulated_list2.csv'
imfiles = []
for i in range(num_images):

    M = translation_matrix(center)@scale_matrix(scales[i])@rotation_matrix(rotations[i])@translation_matrix(-center)@translation_matrix([shift_x[i], shift_y[i]])

    # Warp the image
    warped_image = cv2.warpPerspective(image, M, output_size)
    
    warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    warped_image = cv2.GaussianBlur(warped_image, (5, 5), 0)
    warped_image = (warped_image/255.0) + np.random.normal(0, 0.02, warped_image.shape) + cv2.GaussianBlur(np.random.normal(0, 0.1, warped_image.shape), (5, 5), 0)
    warped_image = (np.clip(warped_image, 0, 1)*255).astype(np.uint8)
    warped_image = clahe.apply(warped_image)

    # Save the image
    # cv2.imshow('output', warped_image)
    # cv2.waitKey(100)
    imfile = f'{output_fld}image_{i:02}.jpg'
    cv2.imwrite(imfile, warped_image)
    
    imfiles.append(imfile)
    
with open(imfile_path, 'w') as f:
    f.write('\n'.join(imfiles))