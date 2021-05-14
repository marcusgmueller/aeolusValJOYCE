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


from .radarlidar_analysis.RadarLidarWindSpeed import RadarLidarWindSpeed
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
def readFile(list):
    for filename in list:
        filename = filename[:-4]
        print(filename)
        tf = tarfile.open(path+filename+".TGZ", "r:gz")
        tf.extractall("/work/marcus_mueller/aeolus/3082/")
        tf.close()
        sys.path.append(path)

        #e/mmueller/hiwi/aeolusE_OPER_ALD_U_N_2B_20190630T171935_20190630T194247_0003.DBL"



        try:
            product = coda.open(path+filename+".DBL")
            latitude = coda.fetch(product, 'rayleigh_geolocation', -1, 'windresult_geolocation/latitude_cog')

            longitude = coda.fetch(product, 'rayleigh_geolocation', -1, 'windresult_geolocation/longitude_cog')

            altitude = coda.fetch(product, 'rayleigh_geolocation', -1, 'windresult_geolocation/altitude_vcog')




            mie_wind_velocity = coda.fetch(product, 'rayleigh_hloswind', -1, 'windresult/rayleigh_wind_velocity')

            mie_wind_velocity = mie_wind_velocity*0.01
            #mie_wind_result_wind_velocity

            result_id = coda.fetch(product, 'rayleigh_profile', -1, 'l2b_wind_profiles/wind_result_id_number')
            time = coda.fetch(product, 'rayleigh_profile', -1, 'Start_of_Obs_DateTime')
            orbit = coda.fetch(product, 'rayleigh_geolocation', -1, 'windresult_geolocation/altitude_vcog')
            azimuth = coda.fetch(product, 'rayleigh_geolocation', -1, 'windresult_geolocation/los_azimuth')

            result_id = vstack(result_id)

            wind_velocity = zeros(result_id.shape)
            wind_velocity[result_id != 0] = mie_wind_velocity[result_id[result_id != 0] - 1]

            lats = zeros(result_id.shape)
            lats[result_id != 0] = latitude[result_id[result_id != 0] - 1]

            lons = zeros(result_id.shape)
            lons[result_id != 0] = longitude[result_id[result_id != 0] - 1]

            alt = zeros(result_id.shape)
            alt[result_id != 0] = altitude[result_id[result_id != 0] - 1]

            azimuth_hlos = zeros(result_id.shape)
            azimuth_hlos[result_id != 0] = azimuth[result_id[result_id != 0] - 1]



            os.remove(path+filename+".DBL")
            os.remove(path+filename+".HDR")
            df = pd.DataFrame([],  columns =['column', 'alt', 'lat', 'lon', 'speed','azimuth'])
            for i in range(wind_velocity.shape[0]):
                for j in range(24):
                    newDF = pd.DataFrame.from_dict({
                        'column': [i],
                        'time': time[i],
                        'alt': [alt[i,j]],
                        'lat': [lats[i,j]],
                        'lon': [lons[i,j]],
                        'speed': [wind_velocity[i,j]],   
                        'azimuth': azimuth_hlos[i,j]      
                    })
                    df = df.append(newDF)

            #joyce
            lon = 6.41
            lat = 50.90
            joyceDf = pd.DataFrame.from_dict({
                'lat': [lat], 
                'lon': [lon],
            })
            joyceDf.head()





            measurements_gdf = create_gdf(df)
            joyceGdf = create_gdf(joyceDf)






            # Get the nearest geometry
            joyceGdf['nearest_geom'] = joyceGdf.apply(calculate_nearest, destination=measurements_gdf, val='geometry', axis=1)
            # Get the nearest Bike station name
            joyceGdf['column'] = joyceGdf.apply(calculate_nearest, destination=measurements_gdf, val='column', axis=1)



            column = joyceGdf.values[0,4]


            resultGDF = df.loc[df['column'] == column]
            resultGDF = resultGDF[resultGDF.alt >= 0]


            aolus_hlos_angle = resultGDF.azimuth.mean()

            #aolus_hlos_angle = math.radians(aolus_hlos_angle)
            print(aolus_hlos_angle)



            #get observation data
            x = datetime(2000, 1, 1)
            aeolusTime = resultGDF.time.to_list()[0]
            delta = timedelta(seconds=aeolusTime)

            measurementDatetime = (x+delta).replace(hour=5, minute=30, second=0, microsecond=0)


            end = (x+delta).replace(hour=23, minute=59, second=0, microsecond=0)
            begin = (x+delta).replace(hour=0, minute=0, second=0, microsecond=0)



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
                difference = aolus_hlos_angle-direction[i]
                rad = math.radians(difference)
                speed = np.abs(speed_observation[i])*math.cos(rad)
                speed_joyce_hlos.append(speed)  


            #resultGDF['speed'] = resultGDF['speed'].abs()

            # Get names of indexes for which column Age has value 30
            #indexNames = resultGDF[ resultGDF['lat'] == 0.0 ].index
            # Delete these row indexes from dataFrame
            #resultGDF.drop(indexNames , inplace=True)
            resultGDF = resultGDF[resultGDF.lat != 0.0]

            fig = plt.figure(figsize=(20,10))
            plt.title("AEOLUS: Rayleigh Wind-Speed "+(x+delta).strftime("%Y-%m-%d"))
            ax = plt.axes()

            resultGDF.plot(x="speed", y="alt", ax=ax, label="Aeolus")
            plt.plot(speed_joyce_hlos, alt_observation, label="JOYCE")
            ax.set_xlabel("horizontal windspeed [m/s]")
            ax.set_ylabel("height AGL [m]")
            ax.legend()
            plt.savefig('/work/marcus_mueller/aeolus/3082/plots/'+(x+delta).strftime("%Y-%m-%d")+'.png',dpi=150)
            plt.show()
        except:
            print("error")




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
processes = [Process(target=readFile, args=([list])) for list in tasks]
for p in processes:
    p.start()

for p in processes:
    p.join()
