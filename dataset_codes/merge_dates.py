import rasterio as rio
import os
import numpy as np
from tqdm import tqdm

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
        bands1 = src1.read(range(1, 11))  # in paranthes : (1 , number of bands + 1 )
        profile = src1.profile

    # Open the second image and read the specified number of bands
    with rio.open(image_path_date2) as src2:
        bands2 = src2.read(range(1, 11))  # in paranthes : (1 , number of bands + 1 )

    # Stack the bands into a new array
    new_bands = np.concatenate((bands1, bands2), axis=0)

    # Update the profile to reflect the new number of bands
    profile.update(count=len(new_bands))

    # Write the new image to disk
    with rio.open(images[i], 'w', **profile) as dst:
        dst.write(new_bands)
