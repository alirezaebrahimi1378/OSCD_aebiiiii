import os
import glob
import shutil
import geopandas as gpd
import zipfile39
import rasterio as rio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling

class prepare_data_copernicus:
    def __init__(self, root_path):
        self.root_path = root_path
        self.image_path = os.path.join(root_path, 'data/OSCD/images')
        self.city_names = os.listdir(self.image_path)

    def extract_data(self, date_path):  # extracts zip files downloaded from copernicus api
        file_path = os.listdir(date_path)
        for item in file_path:
            if self.get_format(item) == 'zip':
                file_path = os.path.join(date_path, item)
                with zipfile39.ZipFile(file_path, 'r') as zip:
                    zip.extractall(date_path)

    # first removes TCI and PVI image from  files then renames the remaining JP2 images and moving them to date_path directory and deleting all junk files and folders
    def clean_folder(self,date_path):
        bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
        junk_indexs = ['B01','B09','B10','PVI','TCI']
        file_list = glob.glob(os.path.join(date_path, '**/*.jp2'), recursive=True)
        src_path = os.path.dirname(file_list[3])
        os.chdir(src_path)
        for i, name in enumerate(file_list):
            index = self.get_index(name)
            if index in junk_indexs:
                os.remove(name)
            else:
                ind = bands.index(index)
                dir_path = os.path.dirname(name)
                os.rename(name, dir_path + f'/{bands[ind]}.jp2')

        for filename in os.listdir(src_path):
            # Create the full file paths
            src_data = os.path.join(src_path, filename)
            dst_path = os.path.join(date_path, filename)
            # Move the file
            shutil.move(src_data, dst_path)

        for file_name in os.listdir(date_path):
            if self.get_format(file_name) != 'jp2':
                if os.path.isfile(file_name):
                    print(file_name)
                    os.remove(file_name)
                elif os.path.isdir(file_name):
                    shutil.rmtree(file_name)

    def get_index(self, path):  # returns index of image or band number
        path = path.replace("/", " ")
        path = path.replace("_", " ")
        path = path.replace(".", " ")
        index = path.split()[-2]
        return index

    def get_format(self, path):  # returns format of file such as jp2 , tif ,etc
        path = path.replace("/", " ")
        path = path.replace("_", " ")
        path = path.replace(".", " ")
        format = path.split()[-1]
        return format

    def crop_transform(self, date_path, json_path):
        os.chdir(date_path)
        images = os.listdir(date_path)
        geojson = gpd.read_file(json_path)

        # Extract geometry of GeoJSON file
        geometry = geojson.geometry.values[0]

        for jp2_image in images:
            index = self.get_index(jp2_image)
            with rio.open(jp2_image) as src:
                transform, width, height = calculate_default_transform(src.crs, geojson.crs, src.width, src.height,
                                                                       *src.bounds)
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': geojson.crs,
                    'transform': transform,
                    'width': width,
                    'height': height
                })
                with rio.open(f'sent2_{index}.tif', 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rio.band(src, i),
                            destination=rio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=geojson.crs,
                            resampling=Resampling.nearest
                        )

            # Clip image using GeoJSON file extent
            with rio.open(f'sent2_{index}.tif') as src:
                out_image, out_transform = mask(src, [geometry], crop=True)

                # Update metadata with new shape and transform
                out_meta = src.meta.copy()
                out_meta.update({'height': out_image.shape[1],
                                 'width': out_image.shape[2],
                                 'transform': out_transform})

                # Write clipped image to disk
                with rio.open(f'sent2_{index}.tif', 'w', **out_meta) as dst:
                    dst.write(out_image)


    def prepare(self):
        for i in range(len(self.city_names)):
            if self.city_names[i] == 'mumbai':
                city_path = os.path.join(self.image_path, self.city_names[i])
                json_path = os.path.join(city_path, f'{self.city_names[i]}.geojson')
                date1_path = os.path.join(city_path, 'date1')
                date2_path = os.path.join(city_path, 'date2')
                self.extract_data(date1_path)
                self.extract_data(date2_path)
                self.clean_folder(date1_path)
                self.clean_folder(date2_path)
                self.crop_transform(date1_path, json_path)


os.chdir('..')
root_path = os.getcwd()
p1 = prepare_data_copernicus(root_path)
p1.prepare()
