imagePath = "/work/marcus_mueller/aeolus/3082/plots/"

from plot import *
import os
from multiprocessing import Process, Queue
from sklearn.neighbors import NearestNeighbors
import pandas as pd
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
import xarray as xr

from radarlidar_analysis.RadarLidarWindSpeed import RadarLidarWindSpeed
from datetime import datetime, timedelta
import tarfile
import math

import seaborn as sns
sns.set_theme()

orbit = '3082'
path = '/work/marcus_mueller/aeolus/'

fileList =  os.listdir(path+orbit+"/")
tasks = []
tasks.append(fileList[::4])
tasks.append(fileList[1::4])
tasks.append(fileList[2::4])
tasks.append(fileList[3::4])

def analysis(path,list):
    rayleighResultList = []    
    mieResultList = []
    for filename in list:
        dt = getMeasurementTime(filename)

        filename = filename[:-4]
        print(filename)
        tf = tarfile.open(path+orbit+'/'+filename+".TGZ", "r:gz")
        tf.extractall(path+orbit+'/')
        tf.close()
        sys.path.append(path+orbit+'/')
        try:
            #get Data
            measurementDatetime =  getMeasurementTime(filename)
            product = coda.open(path+orbit+'/'+filename+".DBL")
            rayleighGdf = readToGDF(product,'rayleigh', measurementDatetime)
            mieGdf = readToGDF(product,'mie', measurementDatetime)

            rayleighGdf = joyceNN(rayleighGdf)
            mieGdf = joyceNN(mieGdf)

            rayleighGdf = rayleighGdf.loc[rayleighGdf.validity == 1.0]
            rayleighGdf = rayleighGdf.loc[rayleighGdf.speed < 50]
            rayleighGdf = rayleighGdf.loc[rayleighGdf.speed > -50]
            #rayleighGdf = rayleighGdf.loc[rayleighGdf.error < 7.0]
            mieGdf = mieGdf.loc[mieGdf.validity == 1.0]
            mieGdf = mieGdf.loc[mieGdf.speed < 50]
            mieGdf = mieGdf.loc[mieGdf.speed > -50]
            #mieGdf = mieGdf.loc[mieGdf.error < 5.0]



            os.remove(path+orbit+'/'+filename+".DBL")
            os.remove(path+orbit+'/'+filename+".HDR")
            rayleighResultList.append(joyceNN(rayleighGdf))
            mieResultList.append(joyceNN(mieGdf))
        except Exception as e:
            print("- error aeolus-")
            print(e)
            #exit()


        try:
            #get JOYCE data
            measurementDatetime = getMeasurementTime(filename)
            #measurementDatetime = x = datetime(2000, 1, 10)  #dringend noch aendern
            aolusHlosAngle = rayleighGdf.azimuth.mean() 
            joyceDf = getObservationData(measurementDatetime, aolusHlosAngle)
            iconDf = getICONdata(measurementDatetime,aolusHlosAngle)


            fig = plt.figure(figsize=(20,10))
            plt.title("Wind-Speed "+dt.strftime("%Y-%m-%d"))
            ax = plt.axes()
            plt.scatter(rayleighGdf['speed'],rayleighGdf['alt'], label="Aeolus - Rayleigh")
            plt.scatter(mieGdf['speed'],mieGdf['alt'], label="Aeolus - Mie")
            plt.scatter(joyceDf['speed'],joyceDf['alt'], label="JOYCE - Radar/Lidar")
            plt.scatter(iconDf['speed'],iconDf['alt'], label="ICON")
            plt.ylim([0,20000])
            plt.xlim([-50,50])
            ax.set_xlabel("windspeed aeolus HLOS [m/s]")
            ax.set_ylabel("height AGL [m]")
            ax.legend()
            plt.savefig(imagePath+dt.strftime("%Y-%m-%d")+'.png',dpi=150)
            plt.show()
        except Exception as e:
            print("- error joyce -")
            print(e)
            #exit()

       
#analysis(path,[fileList[0]])

queue = Queue()
processes = [Process(target=analysis, args=(path, list)) for list in tasks]
for p in processes:
    p.start()
for p in processes:
    p.join()



























































# path = 

# def calculate_nearest(row, destination, val, col='geometry'):
#     # 1 - create unary union    
#     dest_unary = destination['geometry'].unary_union
#     # 2 - find closest point
#     nearest_geom = nearest_points(row[col], dest_unary)
#     # 3 - Find the corresponding geom
#     match_geom = destination.loc[destination.geometry 
#                 == nearest_geom[1]]
#     # 4 - get the corresponding value
#     match_value = match_geom[val].to_numpy()[0]
#     return match_value
# def create_gdf(df, x='lat', y='lon'):
#     return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[y], df[x]), crs={'init':'EPSG:4326'})
# def readFile(list,band, orbit):
#     for filename in list:

#         filename = filename[:-4]
#         print(filename)
#         tf = tarfile.open(path+filename+".TGZ", "r:gz")
#         tf.extractall(path+orbit)
#         tf.close()
#         sys.path.append(path+orbit)

#         try:
#             product = coda.open(path+orbit+filename+".DBL")
#             if band == 'mie'
#             latitude = coda.fetch(product, 'mie_geolocation', -1, 'windresult_geolocation/latitude_cog')
#             longitude = coda.fetch(product, 'mie_geolocation', -1, 'windresult_geolocation/longitude_cog')
#             altitude = coda.fetch(product, 'mie_geolocation', -1, 'windresult_geolocation/altitude_vcog')
#             velocity = coda.fetch(product, 'mie_hloswind', -1, 'windresult/mie_wind_velocity')
#             validity = coda.fetch(product, 'mie_hloswind', -1, 'windresult/validity_flag')
#             velocity = velocity*0.01
#             result_id = coda.fetch(product, 'mie_profile', -1, 'l2b_wind_profiles/wind_result_id_number')
#             time = coda.fetch(product, 'mie_profile', -1, 'Start_of_Obs_DateTime')
#             orbit = coda.fetch(product, 'mie_geolocation', -1, 'windresult_geolocation/altitude_vcog')
            

#             result_id = vstack(result_id)

#             wind_velocity = zeros(result_id.shape)
#             wind_velocity[result_id != 0] = mie_wind_velocity[result_id[result_id != 0] - 1]

#             wind_validity = zeros(result_id.shape)
#             wind_validity[result_id != 0] = mie_wind_validity[result_id[result_id != 0] - 1]

#             lats = zeros(result_id.shape)
#             lats[result_id != 0] = latitude[result_id[result_id != 0] - 1]

#             lons = zeros(result_id.shape)
#             lons[result_id != 0] = longitude[result_id[result_id != 0] - 1]

#             alt = zeros(result_id.shape)
#             alt[result_id != 0] = altitude[result_id[result_id != 0] - 1]

#             # azimuth_hlos = zeros(result_id.shape)
#             # azimuth_hlos[result_id != 0] = azimuth[result_id[result_id != 0] - 1]

#             #Rayleigh Azimuth
#             azimuth = coda.fetch(product, 'rayleigh_geolocation', -1, 'windresult_geolocation/los_azimuth')
#             result_id_ray = coda.fetch(product, 'rayleigh_profile', -1, 'l2b_wind_profiles/wind_result_id_number')
#             result_id_ray = vstack(result_id_ray)
#             azimuth_hlos = zeros(result_id_ray.shape)
#             azimuth_hlos[result_id_ray != 0] = azimuth[result_id_ray[result_id_ray != 0] - 1]

#             df = pd.DataFrame([],  columns =['column', 'alt', 'lat', 'lon', 'speed','azimuth'])
#             for i in range(wind_velocity.shape[0]):
#                 for j in range(24):
#                     newDF = pd.DataFrame.from_dict({
#                         'column': [i],
#                         'time': time[i],
#                         'alt': [alt[i,j]],
#                         'lat': [lats[i,j]],
#                         'lon': [lons[i,j]],
#                         'speed': [wind_velocity[i,j]],   
#                         'validity':   [wind_validity[i,j]]    
#                     })
#                     df = df.append(newDF)

#             #joyce
#             lon = 6.41
#             lat = 50.90
#             joyceDf = pd.DataFrame.from_dict({
#                 'lat': [lat], 
#                 'lon': [lon],
#             })
#             joyceDf.head()





#             measurements_gdf = create_gdf(df)
#             joyceGdf = create_gdf(joyceDf)






#             # Get the nearest geometry
#             joyceGdf['nearest_geom'] = joyceGdf.apply(calculate_nearest, destination=measurements_gdf, val='geometry', axis=1)
#             # Get the nearest Bike station name
#             joyceGdf['column'] = joyceGdf.apply(calculate_nearest, destination=measurements_gdf, val='column', axis=1)



#             column = joyceGdf.values[0,4]


#             resultGDF = df.loc[df['column'] == column]
#             resultGDF = resultGDF[resultGDF.alt >= 0]


#             aolus_hlos_angle = azimuth_hlos.flatten().mean()

#             #aolus_hlos_angle = math.radians(aolus_hlos_angle)
#             print(aolus_hlos_angle)



#             #get observation data
#             x = datetime(2000, 1, 1)
#             aeolusTime = resultGDF.time.to_list()[0]
#             delta = timedelta(seconds=aeolusTime)

#             measurementDatetime = (x+delta).replace(hour=5, minute=30, second=0, microsecond=0)


#             end = (x+delta).replace(hour=23, minute=59, second=0, microsecond=0)
#             begin = (x+delta).replace(hour=0, minute=0, second=0, microsecond=0)



#             analysis = RadarLidarWindSpeed(begin, end)
#             analysis.importDataset()
#             analysis.calculateSpeedFusion()
#             analysis.calculateDirectionFusion()



#             analysis.dataframe.reset_index(level=0, inplace=True)
#             analysis.dataframe.reset_index(level=0, inplace=True)
#             resultAnalysis = analysis.dataframe.loc[analysis.dataframe.time == measurementDatetime]
#             alt_observation = resultAnalysis.height.to_list()
#             speed_observation = resultAnalysis.speedFusion.to_list()
#             direction = resultAnalysis.directionFusion.to_list()



#             speed_joyce_hlos = []
#             for i in range(len(direction)):
#                 difference = aolus_hlos_angle-direction[i]
#                 rad = math.radians(difference)
#                 speed = speed_observation[i]*math.cos(rad)
#                 speed_joyce_hlos.append(speed)  


#             #resultGDF['speed'] = resultGDF['speed'].abs()

#             # Get names of indexes for which column Age has value 30
#             #indexNames = resultGDF[ resultGDF['lat'] == 0.0 ].index
#             # Delete these row indexes from dataFrame
#             #resultGDF.drop(indexNames , inplace=True)
#             resultGDF = resultGDF[resultGDF.lat != 0.0]
#             resultGDF = resultGDF[resultGDF.validity != 0.0]

            

#         except Exception as e:
#             print("error")
#             print(e)

#         #remove file
#         os.remove(path+filename+".DBL")
#         os.remove(path+filename+".HDR")





# # get all files

# fileList =  os.listdir(path)
# tasks = []
# tasks.append(fileList[::4])
# tasks.append(fileList[1::4])
# tasks.append(fileList[2::4])
# tasks.append(fileList[3::4])




# # #start multiprocessing

# #queue = Queue()
# #processes = [Process(target=readFile, args=([list])) for list in tasks]
# #for p in processes:
# #   p.start()#

# #for p in processes:
# #   p.join()
# readFile(['AE_OPER_ALD_U_N_2B_20210209T054235_20210209T071323_0001.TGZ'])