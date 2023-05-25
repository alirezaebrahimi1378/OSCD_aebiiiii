import rasterio as rio
import os
import numpy as np
from tqdm import tqdm
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt
os.chdir('..')
root = os.getcwd()
date1_path = os.path.join(root, 'data/chips/date1')
date2_path = os.path.join(root, 'data/chips/date2')
output_path = os.path.join(root, 'data/merged/images')
images = os.listdir(date1_path)
os.chdir(output_path)
for i in tqdm(range(len(images))):
    image_path_date1 = os.path.join(date1_path, images[i])
    image_path_date2 = os.path.join(date2_path, images[i])
    with rio.open(image_path_date1) as src1:
        bands1 = src1.read()  # in paranthes : (1 , number of bands + 1 )
        bands1 = bands1.transpose(1, 2, 0)
        profile = src1.profile

    # Open the second image and read the specified number of bands
    with rio.open(image_path_date2) as src2:
        bands2 = src2.read()  # in paranthes : (1 , number of bands + 1 )
        bands2 = bands2.transpose(1 , 2, 0)
        bands1 = match_histograms(bands1, bands2, channel_axis=-1)
    # img1 = bands1
    # img2 = bands2
    # img1 = img1[: , : , [2 , 1 , 0]]
    # img2 = img2[: , : , [2 , 1 , 0]]
    # plt.subplot(1 , 2 , 1)
    # plt.imshow(img1.astype(np.uint8))
    # plt.subplot(1 , 2 , 2)
    # plt.imshow(img2.astype(np.uint8))
    # plt.show()
    # break
    bands1 = bands1.transpose(2 , 0 , 1)
    bands2 = bands2.transpose(2 , 0 , 1)
    # Stack the bands into a new array
    new_bands = np.concatenate((bands1, bands2), axis=0)

    # Update the profile to reflect the new number of bands
    profile.update(count=len(new_bands))

    # Write the new image to disk
    with rio.open(images[i], 'w', **profile) as dst:
        dst.write(new_bands)
