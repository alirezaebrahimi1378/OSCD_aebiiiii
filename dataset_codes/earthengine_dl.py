import os
import rasterio as rio
import ee
import requests
from tqdm import tqdm
import sys
from datetime import datetime, timedelta

service_account = 'geo-test@geotest-317218.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, 'geotest-privkey.json')
ee.Initialize(credentials)


def get_next_day(date):
    date = datetime.strptime(date, '%Y%m%d')
    next_day = date + timedelta(days=1)
    return next_day.strftime('%Y%m%d')


def get_dates(text):
    with open(text) as date_file:
        lines = date_file.readlines()
        line1 = lines[0]
        line2 = lines[1]
        date1 = line1[-9:]
        date2 = line2[-9:]
        return str(int(date1)), str(int(date2))


def get_index(names):
    idx = names.replace("_", " ")
    idx = idx.replace(".", " ")
    idx = idx.replace("/", " ")
    idx = idx.split()
    return idx


def make_dirs(city_path):
    date1_path = os.path.join(city_path, 'date1')
    date2_path = os.path.join(city_path, 'date2')
    if not os.path.isdir(date1_path):
        os.mkdir(date1_path)
    if not os.path.isdir(date2_path):
        os.mkdir(date2_path)
    return date1_path, date2_path


def get_lalbel_address(names):
    idx = names.replace("_", " ")
    idx = idx.replace(".", " ")
    idx = idx.replace("/", " ")
    idx = idx.split()
    index = idx[-2]
    root = idx[:-4]
    return index, root


def dl_progress(response):
    total = int(response.headers.get('content-length', 0))
    fname = '/home/alireza/change_detection/data/file.txt'
    with open(fname, 'wb') as file, tqdm(
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def reformat_date(date):
    year = date[:4]
    month = date[4:6]
    day = date[6:8]
    formatted_date = f"{year}-{month}-{day}"
    return formatted_date

######################################DOWNLOAD_CODE###########################################
class Processor:
    def __init__(self, start1, end1, start2, end2, north, south, east, west, date1_path, date2_path, pixles):
        self.start1 = start1
        self.end1 = end1
        self.start2 = start2
        self.end2 = end2
        self.north = north
        self.south = south
        self.east = east
        self.west = west
        self.pixels = pixles
        self.polygon = []
        self.date1_path = date1_path
        self.date2_path = date2_path

    ######################################################################
    def geojson(self):

        coords_list = [[self.west, self.south],
                       [self.west, self.north],
                       [self.east, self.north],
                       [self.east, self.south],
                       [self.west, self.south]
                       ]
        geoJSON = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            coords_list
                        ]
                    }
                }
            ]
        }
        coords = geoJSON['features'][0]['geometry']['coordinates']
        self.polygon.append(ee.Geometry.Polygon(coords))

    ######################################################################
    def get_images_sentinel1(self, coordinates , band):
        image1 = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                  .filterDate(self.start1, self.end1)
                  .select(band)
                  .filterBounds(coordinates)
                  .mean()
                  .clip(coordinates))

        image2 = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                  .filterDate(self.start2, self.end2)
                  .select(band)
                  .filterBounds(coordinates)
                  .mean()
                  .clip(coordinates))
        return image1, image2

    ######################################################################
    def turn_image_to_raster(self, image, title, coordinate, folder):
        # download image from google earth engine
        url = image.getDownloadURL(
            params={'name': title, 'scale': 10, 'region': coordinate,
                    'crs': 'EPSG:4326', 'filePerBand': False, 'format': 'GEO_TIFF'})

        with open(folder + title + '.tif', "wb") as f:
            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')
            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    print("\r[%s%s]" % ('=' * done, ' ' * (50 - done)),
                          f' {dl / 10 ** 6 :0.3f}/{total_length / 10 ** 6 :0.3f} mb', end='')
                    sys.stdout.flush()

    ##################################################################################################
    def main(self):
        self.geojson()
        s2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
        for polygon in self.polygon:
            for band in s2_bands:
                image_date1, image_date2 = self.get_images_sentinel1(coordinates=polygon , band = band)
                print(f'\n************ downloading image one band {band} ************')
                self.turn_image_to_raster(image=image_date1, title=f'sent_date1_{band}', coordinate=polygon,
                                        folder=self.date1_path + '/')
                print(f'\n************ downloading image two band {band} ************')
                self.turn_image_to_raster(image=image_date2, title=f'sent_date2_{band}', coordinate=polygon,
                                        folder=self.date2_path + '/')


os.chdir('../data/OSCD/images')
dataset_path = os.getcwd()
cityname = os.listdir(dataset_path)
for i in range(len(cityname)):
    city_path = os.path.join(dataset_path, cityname[i])
    json_path = os.path.join(city_path, f'{cityname[i]}.geojson')
    date1, date2 = get_dates(os.path.join(city_path , 'dates.txt'))
    date1_path, date2_path = make_dirs(city_path)
    sent_path = os.path.join(city_path, 'b02.tif')
    sent2 = rio.open(sent_path)
    coords = sent2.bounds
    north = coords[3]
    south = coords[1]
    East = coords[2]
    West = coords[0]
    try : 
        print()
        print('#'*40 , f'downloading images of city {cityname[i]}' , '#'*40)
        p1 = Processor(reformat_date(date1), reformat_date(get_next_day(date1)), reformat_date(date2), reformat_date(get_next_day(date2)), north=north, south=south,
                        east=East, west=West, date1_path=date1_path, date2_path=date2_path, pixles=1)
        p1.main()
    except: 
        print()
        print(f'couldnt download images for city {cityname[i]}')
        continue