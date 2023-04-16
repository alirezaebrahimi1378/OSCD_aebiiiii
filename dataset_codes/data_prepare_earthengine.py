import os
import numpy as np
import rasterio as rio
import cv2
import re


class prepare_data_earthengine:
    def __init__(self, root_path):
        self.root_path = root_path
        self.date1_model_input_path = os.path.join(root_path, 'data/model/date1')
        self.date2_model_input_path = os.path.join(root_path, 'data/model/date2')
        self.mask_model_input_path = os.path.join(root_path, 'data/model/mask')
        self.image_path = os.path.join(root_path, 'data/OSCD/images')
        self.mask_path = os.path.join(root_path, 'data/OSCD/masks')
        self.city_names = os.listdir(self.image_path)

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
            img1 = img1.transpose(1, 2, 0)

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
        image_paths = sorted(os.listdir(), key=lambda x: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', x)])
        arrs = []
        for file in image_paths:
            if self.get_index(file) not in bands:
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
    def transform_mask(self, city_name , city_path , index):
        image_path = os.path.join(self.image_path ,f'{city_name}/b02.tif')
        os.chdir(os.path.join(city_path, 'cm'))
        mask_path = f'{city_name}-cm.tif'
        image_main = rio.open(image_path)
        image_incorrect = rio.open(mask_path)

        img = image_main.read()
        mask = image_incorrect.read()
        mask = mask[0, :, :]
        img = img.transpose(1, 2, 0)

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
    def image_process(self, date1_path, date2_path, city_path, index):
        self.transform(city_path, date1_path)
        self.transform(city_path, date2_path)
        self.stack_bands(date1_path, index, 'date1')
        self.stack_bands(date2_path, index, 'date2')

    ######################################################################################################################
    def prepare(self):
        for i, city_name in enumerate(self.city_names):
            if city_name == 'mumbai':
                city_path_image = os.path.join(self.image_path, city_name)
                city_path_mask = os.path.join(self.mask_path, city_name)
                date1_path = os.path.join(city_path_image, 'date1')
                date2_path = os.path.join(city_path_image, 'date2')
                self.image_process(date1_path, date2_path, city_path_image, index=i)
                self.transform_mask(city_name , city_path_mask , index = i)


os.chdir('..')
root_path = os.getcwd()
p1 = prepare_data_earthengine(root_path)
p1.prepare()
