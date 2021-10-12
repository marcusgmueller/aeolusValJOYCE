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
from datetime import datetime, time, timedelta
import tarfile
import math
from sklearn.neighbors import KDTree
import xarray as xr

def create_gdf(df, x='lat', y='lon'):
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[y], df[x]), crs={'init':'EPSG:4326'})
def readToGDF(product,target, measurementDatetime):
    if target == "rayleigh":
        latitude = coda.fetch(product, 'rayleigh_geolocation', -1, 'windresult_geolocation/latitude_cog')
        longitude = coda.fetch(product, 'rayleigh_geolocation', -1, 'windresult_geolocation/longitude_cog')
        altitude = coda.fetch(product, 'rayleigh_geolocation', -1, 'windresult_geolocation/altitude_vcog')
        Velocity = coda.fetch(product, 'rayleigh_hloswind', -1, 'windresult/rayleigh_wind_velocity')
        error = coda.fetch(product, 'rayleigh_wind_prod_conf_data', -1, 'rayleigh_wind_qc/hlos_error_estimate')
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
        error = coda.fetch(product, 'mie_wind_prod_conf_data', -1, 'Mie_Wind_QC/hlos_error_estimate')
        Validity = coda.fetch(product, 'mie_hloswind', -1, 'windresult/validity_flag')
        resultId = coda.fetch(product, 'mie_profile', -1, 'l2b_wind_profiles/wind_result_id_number')
        time = coda.fetch(product, 'mie_profile', -1, 'Start_of_Obs_DateTime')
        orbit = coda.fetch(product, 'mie_geolocation', -1, 'windresult_geolocation/altitude_vcog')
        azimuth = coda.fetch(product, 'mie_geolocation', -1, 'windresult_geolocation/los_azimuth')
    Velocity = Velocity*0.01
    error = error*0.01
    df = pd.DataFrame(data={
        'measurementDatetime': measurementDatetime,
        'alt': altitude,
        'lat': latitude,
        'lon': longitude,
        'speed': Velocity,
        'azimuth': azimuth,
        'validity':   Validity,
        'error': error
    })
    gdf = create_gdf(df)
    #gdf = gdf[gdf.lat != 0.0]
    #print(gdf.validity)
    #gdf = gdf[gdf.validity == 1.0]
    return gdf
def joyceNN(gdf):
    points = np.transpose(np.array([gdf.lat.to_list(), gdf.lon.to_list()]))
    tree = KDTree(points)
    joyce = np.array([[50.90, 6.41]])
    nearest_ind = tree.query_radius(joyce, r=0.5)#3082
    #nearest_ind = tree.query_radius(joyce, r=50)#3058
    gdf = gdf.iloc[nearest_ind[0].tolist()]
    return(gdf)
def getMeasurementTime(filename):   
    day = filename[25:27]
    month = filename[23:25]
    year = filename[19:23]
    date = datetime(int(year), int(month), int(day))
    measurementDatetime = (date).replace(hour=5, minute=30, second=0, microsecond=0)#3082
    #measurementDatetime = (date).replace(hour=17, minute=20, second=0, microsecond=0)#3058
    return(measurementDatetime)
def getObservationData(measurementDatetime, aolusHlosAngle):
    end = measurementDatetime.replace(hour=23, minute=59, second=0, microsecond=0)
    begin = measurementDatetime.replace(hour=0, minute=0, second=0, microsecond=0)
    analysis = RadarLidarWindSpeed(begin, end)
    analysis.importDataset()
    analysis.calculateSpeedFusion()
    analysis.calculateDirectionFusion()
    analysis.dataframe.reset_index(level=0, inplace=True)
    analysis.dataframe.reset_index(level=0, inplace=True)
    time_begin = measurementDatetime.strftime("%H")+":00"
    time_end = measurementDatetime.strftime("%H")+":00"
    analysis.dataframe = analysis.dataframe.set_index('time')
    resultAnalysis = analysis.dataframe.between_time(time_begin, time_end) # dringend noch aendern/automatisieren
    #resultAnalysis = analysis.dataframe.loc[analysis.dataframe.time == measurementDatetime]
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
def getICONdata(dt, aolusHlosAngle):
    filename = "meteogram.iglo.h."+dt.strftime("%Y%m%d")+"00.nc"
    path = "/data/mod/icon_op/iglo/site/"+dt.strftime("%Y/%m/")+filename
    ds = xr.open_dataset(path)
    nStation = ds.station_name.values.tolist().index(b'Juelich')
    nU = ds.var_name.values.tolist().index(b'U')
    nV = ds.var_name.values.tolist().index(b'V')
    height = ds.sel(nstations=nStation, nvars = [nU],nsfcvars=[],time=4)['heights'].values.flatten().tolist()
    u = ds.sel(nstations=nStation, nvars = [nU],nsfcvars=[],time=4)['values'].values.flatten().tolist()
    v = ds.sel(nstations=nStation, nvars = [nV],nsfcvars=[],time=4)['values'].values.flatten().tolist()
    speed_icon = []
    direction = []
    for i in range(len(u)):
        speed_icon.append(math.sqrt(u[i]*u[i]+v[i]*v[i]))
        if v[i]== 0.0:
            direction.append(math.pi/2)
        else:
            direction.append(math.atan(u[i]/v[i]))
    speed_icon_hlos = []
    for i in range(len(direction)):
        difference = aolusHlosAngle-math.degrees(direction[i])
        rad = math.radians(difference)
        speed = speed_icon[i]*math.cos(rad)
        speed_icon_hlos.append(speed) 
    df = pd.DataFrame(data={
        'speed': speed_icon_hlos,
        'alt': height
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
        try:
            #get Data
            measurementDatetime =  getMeasurementTime(filename)
            print(path+filename+".DBL")
            product = coda.open(path+filename+".DBL")
            rayleighGdf = readToGDF(product,'rayleigh',measurementDatetime)
            mieGdf = readToGDF(product,'mie',measurementDatetime)
            rayleighGdf = rayleighGdf.loc[rayleighGdf.validity == 1.0]
            #rayleighGdf = rayleighGdf.loc[rayleighGdf.speed < 50]
            #rayleighGdf = rayleighGdf.loc[rayleighGdf.speed > -50]
            #rayleighGdf = rayleighGdf.loc[rayleighGdf.error < 7.0]
            mieGdf = mieGdf.loc[mieGdf.validity == 1.0]
            #mieGdf = mieGdf.loc[mieGdf.speed < 50]
            #mieGdf = mieGdf.loc[mieGdf.speed > -50]
            #mieGdf = mieGdf.loc[mieGdf.error < 5.0]
            os.remove(path+filename+".DBL")
            os.remove(path+filename+".HDR")
            rayleighGdf = joyceNN(rayleighGdf)
            mieGdf = joyceNN(mieGdf)
            aeolus_hlos_angle = rayleighGdf.azimuth.mean()       
            
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
            #plt.xlim([-50, 50])
            filename=path+'plots/'+measurementDatetime.strftime("%Y-%m-%d")+'.png'
            plt.savefig(filename,dpi=150)
            plt.show()
            plt.close()
        except Exception as e:
            print("- error -")
            print(e)
def runParallel(path):
    # get all files
    path = '/work/marcus_mueller/aeolus/3058/'
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
    #runSinle('/work/marcus_mueller/aeolus/3058/', 'AE_OPER_ALD_U_N_2B_20210502T171744_20210502T194056_0001.TGZ')
    runParallel('/work/marcus_mueller/aeolus/3058/')







    

    







