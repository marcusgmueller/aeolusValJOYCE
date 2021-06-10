import os
from multiprocessing import Process, Queue
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import seaborn as sns
os.putenv('CODA_DEFINITION', '/home/mmueller/hiwi/aeolus/')
import coda
from numpy import vstack, zeros
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import pyproj
import geopandas as gpd
from shapely.ops import nearest_points
from shapely.geometry import LineString
import sys, os
from radarlidar_analysis.RadarLidarWindSpeed import RadarLidarWindSpeed
from datetime import datetime, timedelta
import tarfile
import math


def calculate_nearest(row, destination, val, col='geometry'):
    # 1 - create unary union    
    dest_unary = destination['geometry'].unary_union
    # 2 - find closest point
    nearest_geom = nearest_points(row[col], dest_unary)
    # 3 - Find the corresponding geom
    match_geom = destination.loc[destination.geometry 
                == nearest_geom[1]]
    # 4 - get the corresponding value
    match_value = match_geom[val].to_numpy()[0]
    return match_value
def create_gdf(df, x='lat', y='lon'):
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[y], df[x]), crs={'init':'EPSG:4326'})
def readToGDF(product,target):
    if target == "rayleigh":
        latitude = coda.fetch(product, 'rayleigh_geolocation', -1, 'windresult_geolocation/latitude_cog')
        longitude = coda.fetch(product, 'rayleigh_geolocation', -1, 'windresult_geolocation/longitude_cog')
        altitude = coda.fetch(product, 'rayleigh_geolocation', -1, 'windresult_geolocation/altitude_vcog')
        Velocity = coda.fetch(product, 'rayleigh_hloswind', -1, 'windresult/rayleigh_wind_velocity')
        Validity = coda.fetch(product, 'rayleigh_hloswind', -1, 'windresult/validity_flag')
        resultId = coda.fetch(product, 'rayleigh_profile', -1, 'l2b_wind_profiles/wind_result_id_number')
        time = coda.fetch(product, 'rayleigh_profile', -1, 'Start_of_Obs_DateTime')
        orbit = coda.fetch(product, 'rayleigh_geolocation', -1, 'windresult_geolocation/altitude_vcog')
        azimuth = coda.fetch(product, 'rayleigh_geolocation', -1, 'windresult_geolocation/los_azimuth')
    elif target == 'mie':
        latitude = coda.fetch(product, 'mie_geolocation', -1, 'windresult_geolocation/latitude_cog')
        longitude = coda.fetch(product, 'mie_geolocation', -1, 'windresult_geolocation/longitude_cog')
        altitude = coda.fetch(product, 'mie_geolocation', -1, 'windresult_geolocation/altitude_vcog')
        Velocity = coda.fetch(product, 'mie_hloswind', -1, 'windresult/mie_wind_velocity')
        Validity = coda.fetch(product, 'mie_hloswind', -1, 'windresult/validity_flag')
        resultId = coda.fetch(product, 'mie_profile', -1, 'l2b_wind_profiles/wind_result_id_number')
        time = coda.fetch(product, 'mie_profile', -1, 'Start_of_Obs_DateTime')
        orbit = coda.fetch(product, 'mie_geolocation', -1, 'windresult_geolocation/altitude_vcog')
        azimuth = coda.fetch(product, 'mie_geolocation', -1, 'windresult_geolocation/los_azimuth')
    Velocity = Velocity*0.01
    resultId = vstack(resultId)
    windVelocity = zeros(resultId.shape)
    windVelocity[resultId != 0] = Velocity[resultId[resultId != 0] - 1]
    windValidity = zeros(resultId.shape)
    windValidity[resultId != 0] = Validity[resultId[resultId != 0] - 1]
    lats = zeros(resultId.shape)
    lats[resultId != 0] = latitude[resultId[resultId != 0] - 1]
    lons = zeros(resultId.shape)
    lons[resultId != 0] = longitude[resultId[resultId != 0] - 1]
    alt = zeros(resultId.shape)
    alt[resultId != 0] = altitude[resultId[resultId != 0] - 1]
    azimuth_hlos = zeros(resultId.shape)
    azimuth_hlos[resultId != 0] = azimuth[resultId[resultId != 0] - 1]
    df = pd.DataFrame([],  columns =['column', 'alt', 'lat', 'lon', 'speed','azimuth'])
    for i in range(windVelocity.shape[0]):
        for j in range(24):
            newDF = pd.DataFrame.from_dict({
                'column': [i],
                'time': time[i],
                'alt': [alt[i,j]],
                'lat': [lats[i,j]],
                'lon': [lons[i,j]],
                'speed': [windVelocity[i,j]],
                'azimuth': [azimuth_hlos[i,j]],
                'validity':   [windValidity[i,j]]      
            })
            df = df.append(newDF)
    gdf = create_gdf(df)
    #gdf = gdf[gdf.lat != 0.0]
    gdf = gdf[gdf.validity != 0.0]
    return gdf
def joyceNN(gdf):
    lon = 6.41
    lat = 50.90
    joyceDf = pd.DataFrame.from_dict({
        'lat': [lat], 
        'lon': [lon],
    })
    joyceGdf = create_gdf(joyceDf)
    joyceGdf['nearest_geom'] = joyceGdf.apply(calculate_nearest, destination=gdf, val='geometry', axis=1)
    joyceGdf['column'] = joyceGdf.apply(calculate_nearest, destination=gdf, val='column', axis=1)
    column = joyceGdf.values[0,4]
    gdf = gdf.loc[gdf['column'] == column]
    gdf = gdf[gdf.alt >= 0]
    return(gdf)
def getMeasurementTime(rayleighGdf):
    date = datetime(2000, 1, 1)
    aeolusTime = rayleighGdf.time.to_list()[0]
    delta = timedelta(seconds=aeolusTime)
    measurementDatetime = (date+delta).replace(hour=5, minute=30, second=0, microsecond=0)
    return(measurementDatetime)
def getObservationData(measurementDatetime, rayleighGdf, aolusHlosAngle):
    end = measurementDatetime.replace(hour=23, minute=59, second=0, microsecond=0)
    begin = measurementDatetime.replace(hour=0, minute=0, second=0, microsecond=0)
    analysis = RadarLidarWindSpeed(begin, end)
    analysis.importDataset()
    analysis.calculateSpeedFusion()
    analysis.calculateDirectionFusion()
    analysis.dataframe.reset_index(level=0, inplace=True)
    analysis.dataframe.reset_index(level=0, inplace=True)
    resultAnalysis = analysis.dataframe.loc[analysis.dataframe.time == measurementDatetime]
    alt_observation = resultAnalysis.height.to_list()
    speed_observation = resultAnalysis.speedFusion.to_list()
    direction = resultAnalysis.directionFusion.to_list()
    speed_joyce_hlos = []
    for i in range(len(direction)):
        difference = aolusHlosAngle-direction[i]
        rad = math.radians(difference)
        speed = speed_observation[i]*math.cos(rad)
        speed_joyce_hlos.append(speed) 
    df = pd.DataFrame(data={
        'speed': speed_joyce_hlos,
        'alt': alt_observation
    })
    return(df)
def readFile(path, list):
    for filename in list:
        filename = filename[:-4]
        print(filename)
        tf = tarfile.open(path+filename+".TGZ", "r:gz")
        tf.extractall(path)
        tf.close()
        sys.path.append(path)
        #try:
            #get Data
        product = coda.open(path+filename+".DBL")
        rayleighGdf = readToGDF(product,'rayleigh')
        mieGdf = readToGDF(product,'mie')
        os.remove(path+filename+".DBL")
        os.remove(path+filename+".HDR")
        rayleighGdf = joyceNN(rayleighGdf)
        mieGdf = joyceNN(mieGdf)
        aeolus_hlos_angle = rayleighGdf.azimuth.mean()       
        measurementDatetime =  getMeasurementTime(rayleighGdf)
        observationDf = getObservationData(measurementDatetime,rayleighGdf,aeolus_hlos_angle )
        #Plot
        fig = plt.figure(figsize=(20,10))
        plt.title("AEOLUS: Wind-Speed "+measurementDatetime.strftime("%Y-%m-%d"))
        ax = plt.axes()
        sns.scatterplot(x = 'speed', y = 'alt', data = rayleighGdf,ax=ax, label="Aeolus Rayleigh")
        sns.scatterplot(x = 'speed', y = 'alt', data = mieGdf,ax=ax, label="Aeolus Mie")
        sns.scatterplot(x = 'speed', y = 'alt', data = observationDf,ax=ax, label="JOYCE")
        ax.set_xlabel("horizontal windspeed [m/s]")
        ax.set_ylabel("height AGL [m]")
        ax.legend()
        filename=path+'plots/'+measurementDatetime.strftime("%Y-%m-%d")+'.png'
        plt.savefig(filename,dpi=150)
        plt.show()
        plt.close()
        # except Exception as e:
        #     print("- error -")
        #     print(e)
def runParallel(path):
    # get all files
    path = '/work/marcus_mueller/aeolus/3082/'
    fileList =  os.listdir(path)
    tasks = []
    tasks.append(fileList[::4])
    tasks.append(fileList[1::4])
    tasks.append(fileList[2::4])
    tasks.append(fileList[3::4])
    #start multiprocessing
    queue = Queue()
    processes = [Process(target=readFile, args=([path, list])) for list in tasks]
    for p in processes:
        p.start()

    for p in processes:
        p.join()
def runSinle(path, filename):    
    fileList =  [filename]
    readFile(path, fileList)

if __name__ == "__main__":
    runSinle('/work/marcus_mueller/aeolus/3082/', 'AE_OPER_ALD_U_N_2B_20191008T054411_20191008T071459_0002.TGZ')
    #runParallel('/work/marcus_mueller/aeolus/3082/')







    

    







