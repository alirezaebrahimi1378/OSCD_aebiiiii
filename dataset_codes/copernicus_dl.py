"""
this code will download data from copernicus using one geojson file and dates read from date.txt

"""
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import geopandas as gpd
import os

os.chdir('../data/OSCD/images')
dataset_path = os.getcwd()
api = SentinelAPI('mantis98', '0123456789')

def download_cop(dl_path,json_path , date ):
    os.chdir(dl_path)
    footprint = geojson_to_wkt(read_geojson(json_path))
    query_params = {
        'date': (date, str(int(date) + 1)),
        'platformname': 'Sentinel-2',
        'orbitdirection': 'DESCENDING',
        'cloudcoverpercentage': (0, 20)
    }
    # Query the API and download products
    products = api.query(footprint, **query_params)
    api.download_all(products)
    # Convert products to a geodataframe
    gdf = api.to_geodataframe(products)
    # Load extent as a geodataframe and convert it to the same CRS as the products
    extent_gdf = gpd.read_file(json_path)
    extent_gdf = extent_gdf.to_crs(gdf.crs)
    # Crop products by extent
    cropped_gdf = gpd.overlay(gdf, extent_gdf, how='intersection')
    # Download the cropped products
    for index, row in cropped_gdf.iterrows():
        api.download(row['uuid'])

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
        download_cop(date1_path , json_path , date1)
        download_cop(date2_path , json_path , date2)
        break
