"""
this code gets images that are in different band , stack them and transform them in a way that matches input images given by oscd dataset,
then makes 128 by 128 chips from images . it can delete images with size smaller than 128 or resize them to 128 and same them .
"""


import os
import numpy as np
import rasterio as rio
import cv2
import re
from rasterio.windows import Window

class prepare_data_earthengine:
    def __init__(self, root_path , remove_small_chips):
        self.root_path = root_path                                                       # path to root path where all codes and datasets are
        self.date1_model_input_path = os.path.join(root_path, 'data/model/date1')        # path to directory where stacked images for date 1 get saved
        self.date2_model_input_path = os.path.join(root_path, 'data/model/date2')        # path to directory where stacked images for date 2 get saved
        self.mask_model_input_path = os.path.join(root_path, 'data/model/mask')          # path to directory where masks get saved
        self.image_path = os.path.join(root_path, 'data/OSCD/images')                    # path to directory that has input images in split bands format
        self.mask_path = os.path.join(root_path, 'data/OSCD/masks')                      # path to directory that has masks given from oscd dataset
        self.output_date1_path = os.path.join(root_path, 'data/chips/date1')             # path to directory where small chips of date1 will be saved
        self.output_date2_path = os.path.join(root_path, 'data/chips/date2')             # path to directory where small chips of date2 will be saved
        self.output_mask_path = os.path.join(root_path, 'data/chips/mask')               # path to directory where small chips of mask will be saved
        self.city_names = os.listdir(self.image_path)
        self.remove_small_chips = remove_small_chips
    ######################################################################################################################
    def get_index(self, path):  # returns index of image or band number
        path = path.replace("/", " ")
        path = path.replace("_", " ")
        path = path.replace(".", " ")
        index = path.split()[-2]
        return index

    ######################################################################################################################
    def get_format(self, path):  # returns format of file such as jp2 , tif ,etc
        path = path.replace("/", " ")
        path = path.replace("_", " ")
        path = path.replace(".", " ")
        format = path.split()[-1]
        return format

    ######################################################################################################################
    # this function gets the input image and the source image and reproject the input image to source one ,it will loop through all bands
    def transform(self, city_path, date_path):
        source_image_path = os.path.join(city_path, 'b02.tif')
        os.chdir(date_path)
        for input_image_path in os.listdir(date_path):
            image_main = rio.open(source_image_path)
            image_incorrect = rio.open(input_image_path)

            img1 = image_main.read()
            img2 = image_incorrect.read()
            img2 = img2[0, :, :]
            img1 = img1.transpose(1, 2, 0) # changing shape format of image from (C , H , W) to (H , W , C)

            x, y = img1.shape[0], img1.shape[1]
            shape = image_main.shape
            crs = image_main.crs
            transform = image_main.transform

            img2_new = cv2.resize(img2, (y, x), interpolation=cv2.INTER_NEAREST).astype(int)
            new_dataset = rio.open(input_image_path, 'w', driver='GTiff',
                                   height=shape[0], width=shape[1],
                                   count=1, dtype=str('uint16'),
                                   crs=crs,
                                   transform=transform)

            new_dataset.write(np.array([img2_new]))
            new_dataset.close()

    ######################################################################################################################
    def stack_bands(self, date_path, index, date):  # this function merges different bands to create a tif image
        bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
        os.chdir(date_path)
        image_paths = sorted(os.listdir(), key=lambda x: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', x)]) # sorts files in order given in bands list , doesnt use bands list
        arrs = []
        for file in image_paths:
            if self.get_index(file) not in bands: # remove any file that doesn't have names given in bands list
                os.remove(file)
            else:
                with rio.open(file) as src:
                    arr = src.read(1)  # read the first band
                    arrs.append(np.expand_dims(arr, axis=0))  # add a new dimension for stacking

                # stack the arrays from all input files into a single multi-band array
                mosaic = np.vstack(arrs)

                # update the metadata of the output TIFF
                out_meta = src.meta.copy()
                out_meta.update({"driver": "GTiff",
                                 "height": mosaic.shape[1],
                                 "width": mosaic.shape[2],
                                 "count": mosaic.shape[0]})
                if date == 'date1':
                    os.chdir(self.date1_model_input_path)
                else:
                    os.chdir(self.date2_model_input_path)
                # save the output TIFF
                with rio.open(os.path.join(f'sent2_{index}.tif'), "w", **out_meta) as dest:
                    dest.write(mosaic)
                os.chdir(date_path)

    ######################################################################################################################
    # again transforming masks downloaded to shape of tif images given by OSCD dataset
    def transform_mask(self, city_name, city_path, index):
        image_path = os.path.join(self.image_path, f'{city_name}/b02.tif')
        os.chdir(os.path.join(city_path, 'cm'))
        mask_path = f'{city_name}-cm.tif'
        image_main = rio.open(image_path)
        image_incorrect = rio.open(mask_path)

        img = image_main.read()
        mask = image_incorrect.read()
        mask = mask[0, :, :]
        img = img.transpose(1, 2, 0) # changing position of band numbers in mask shape

        x, y = img.shape[0], img.shape[1]
        shape = image_main.shape
        crs = image_main.crs
        transform = image_main.transform

        mask_new = cv2.resize(mask, (y, x), interpolation=cv2.INTER_NEAREST).astype(int)
        os.chdir(self.mask_model_input_path)
        new_dataset = rio.open(f'CM_{index}.tif', 'w', driver='GTiff',
                               height=shape[0], width=shape[1],
                               count=1, dtype=str('uint16'),
                               crs=crs,
                               transform=transform)

        new_dataset.write(np.array([mask_new]))
        new_dataset.close()

    ######################################################################################################################
    # this function gets the input_path for directory that have the images and output directory that saves images , then makes 128 by 128 chips from input images
    def make_chips(self, input_path, output_path, num_band, remove_small_chips=True): # num_bands is needed to say wheter its label or image , remove_small_image mean if a chip size is less than 64 it wouldn't be saved
        chip_size = 128 # determines chip_size
        meta = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': None,
            'width': chip_size,
            'height': chip_size,
            'count': num_band,
            'crs': 'EPSG:4326',
            'transform': rio.transform.from_bounds(0, 0, chip_size, chip_size, chip_size, chip_size) # by this transform all chips have same spatial coordinates , can change it to have original coordinates
        }
        for filename in os.listdir(input_path):
            if filename.endswith(".tif"):
                # Open the TIFF image using rasterio
                with rio.open(os.path.join(input_path, filename)) as src:
                    # Get the width and height of the image
                    width = src.width
                    height = src.height

                    # Calculate the number of chips in each dimension
                    chips_x = (width + chip_size - 1) // chip_size
                    chips_y = (height + chip_size - 1) // chip_size

                    # Loop through each chip and extract it
                    for i in range(chips_x):
                        for j in range(chips_y):
                            # Calculate the window for this chip
                            window = Window(i * chip_size, j * chip_size, chip_size, chip_size)

                            # Read the data for this chip
                            chip = src.read(window=window)
                            if remove_small_chips:
                                if chip.shape[1] < 64 or chip.shape[2] < 64:
                                    continue
                            else:
                                # Save the chip to the output directory
                                chip_filename = os.path.splitext(filename)[0] + '_{}_{}.tif'.format(i, j)
                                with rio.open(os.path.join(output_path, chip_filename), 'w', **meta) as dst:
                                    dst.write(chip)

    ######################################################################################################################
    # this function has all processes that will be applied to images in date1 and date2
    def image_process(self, date1_path, date2_path, city_path, index):
        self.transform(city_path, date1_path)
        self.transform(city_path, date2_path)
        self.stack_bands(date1_path, index, 'date1')
        self.stack_bands(date2_path, index, 'date2')
        self.make_chips(self.date1_model_input_path, self.output_date1_path, num_band=10, remove_small_chips=self.remove_small_chips)
        self.make_chips(self.date2_model_input_path, self.output_date2_path, num_band=10, remove_small_chips=self.remove_small_chips)

    ######################################################################################################################
    def prepare(self , city_filter = None):
        for i, city_name in enumerate(self.city_names):
            if city_filter == None :
                city_path_image = os.path.join(self.image_path, city_name)
                city_path_mask = os.path.join(self.mask_path, city_name)
                date1_path = os.path.join(city_path_image, 'date1')
                date2_path = os.path.join(city_path_image, 'date2')
                self.image_process(date1_path, date2_path, city_path_image, index=i)
                self.transform_mask(city_name, city_path_mask, index=i)
                self.make_chips(self.mask_model_input_path, self.output_mask_path, num_band=1,remove_small_chips=self.remove_small_chips)
            else:
                if city_name == city_filter :
                    city_path_image = os.path.join(self.image_path, city_name)
                    city_path_mask = os.path.join(self.mask_path, city_name)
                    date1_path = os.path.join(city_path_image, 'date1')
                    date2_path = os.path.join(city_path_image, 'date2')
                    self.image_process(date1_path, date2_path, city_path_image, index=i)
                    self.transform_mask(city_name, city_path_mask, index=i)
                    self.make_chips(self.mask_model_input_path, self.output_mask_path, num_band=1, remove_small_chips=self.remove_small_chips)


os.chdir('..')
root_path = os.getcwd()
p1 = prepare_data_earthengine(root_path , remove_small_chips=False)
p1.prepare()
