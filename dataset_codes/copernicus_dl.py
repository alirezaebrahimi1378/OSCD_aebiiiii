"""
this code will download data from copernicus using one geojson file and dates read from date.txt

"""
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import os
from datetime import datetime , timedelta

os.chdir('../data/OSCD/images')
dataset_path = os.getcwd()
api = SentinelAPI('mantis98', '0123456789')

def get_next_day(date_int):
    date_str = str(date_int)
    date = datetime.strptime(date_str, '%Y%m%d')
    next_day = date + timedelta(days=1)
    return next_day.strftime('%Y%m%d')

def download_cop(dl_path,json_path , date ):
    os.chdir(dl_path)
    footprint = geojson_to_wkt(read_geojson(json_path))
    products = api.query(footprint,
                         date=(date , get_next_day(int(date))),
                         platformname='Sentinel-2',
                         orbitdirection='DESCENDING',
                         cloudcoverpercentage=(0, 20))
    api.download_all(products , output_crs='EPSG:4326')

def get_dates(text):
    with open(text) as date_file:
        lines = date_file.readlines()
        line1 = lines[0]
        line2 = lines[1]
        date1 = line1[-9:]
        date2 = line2[-9:]
        return date1 , date2

def make_dirs(city_path):
    date1_path = os.path.join(city_path, 'date1')
    date2_path = os.path.join(city_path, 'date2')
    if not os.path.isdir(date1_path) :
        os.mkdir(date1_path)
    if not os.path.isdir(date2_path) :
        os.mkdir(date2_path)
    return date1_path , date2_path


for root, cityname, files in os.walk(dataset_path, topdown=True):
    for i in range(len(cityname)):
        city_path = os.path.join(dataset_path , cityname[i])
        os.chdir(city_path)
        json_path = os.path.join(city_path , f'{cityname[i]}.geojson')
        date1 , date2 = get_dates('dates.txt')
        date1_path , date2_path = make_dirs(city_path)
        print(city_path)
        if cityname[i] != 'mumbai' :
            continue
        else :
            print('downloading date1')
            download_cop(date1_path , json_path , date1[:9])
            print('downloading date2')
            download_cop(date2_path , json_path , date2[:9])
