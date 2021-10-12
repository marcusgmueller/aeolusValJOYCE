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
from sklearn.neighbors import KDTree
from plot import *
from radarlidar_analysis.RadarLidarWindSpeed import RadarLidarWindSpeed
import pickle

def analysis(path,list):
    rayleighResultList = []    
    mieResultList = []
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
            product = coda.open(path+filename+".DBL")
            rayleighGdf = readToGDF(product,'rayleigh', measurementDatetime)
            mieGdf = readToGDF(product,'mie', measurementDatetime)
            rayleighGdf = rayleighGdf.loc[rayleighGdf.validity == 1.0]
            rayleighGdf = rayleighGdf.loc[rayleighGdf.speed < 50]
            rayleighGdf = rayleighGdf.loc[rayleighGdf.speed > -50]
            #rayleighGdf = rayleighGdf.loc[rayleighGdf.error < 7.0]
            mieGdf = mieGdf.loc[mieGdf.validity == 1.0]
            mieGdf = mieGdf.loc[mieGdf.speed < 50]
            mieGdf = mieGdf.loc[mieGdf.speed > -50]
            #mieGdf = mieGdf.loc[mieGdf.error < 5.0]
            os.remove(path+filename+".DBL")
            os.remove(path+filename+".HDR")
            rayleighResultList.append(joyceNN(rayleighGdf))
            mieResultList.append(joyceNN(mieGdf))
        except Exception as e:
            print("- error -")
            #print(e)
    return {
        'rayleighList': rayleighResultList,
        'mieList': mieResultList
    }

# get all files
path = '/work/marcus_mueller/aeolus/3082/'
# fileList = []
# for filename in os.listdir(path):
#     day = filename[25:27]
#     month = filename[23:25]
#     year = filename[19:23]
#     if year == "2020":
#         fileList.append(filename)
# print(len(fileList))


# result = analysis(path,fileList)
# rayleighGdf = pd.concat(result['rayleighList'])
# mieGdf = pd.concat(result['mieList'])









#start
result = pd.read_pickle("joycedf.pkl")

radarCoverage = result.pivot(index="height", columns="day", values="radar Coverage")#.to_numpy()
lidarCoverage = result.pivot(index="height", columns="day", values="lidar Coverage")#.to_numpy()
totalCoverage = result.pivot(index="height", columns="day", values="total Coverage")#.to_numpy()
print(radarCoverage.shape)
X,Y = np.meshgrid(radarCoverage.shape)
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 10), sharex=True, sharey=False)
#fig.suptitle("September 2020", fontsize=16)


#days_extende = np.arange(self.dateBegin, self.dateEnd+timedelta(days=1), timedelta(days=1)).astype(datetime)

axes[0].set_title("coverage by radar")
#axes[0].axis([0, 24, 0, maxHeight])
im = axes[0].pcolor(X,Y,radarCoverage,cmap='viridis',vmin=0,vmax=100)
axes[0].set_ylabel("height AGL [m]")

axes[1].set_title("coverage by lidar")
#axes[0].axis([0, 24, 0, maxHeight])
im = axes[1].pcolor(X,Y,lidarCoverage,cmap='viridis',vmin=0,vmax=100)
axes[1].set_ylabel("height AGL [m]")

axes[2].set_title("total coverage by height")
#axes[0].axis([0, 24, 0, maxHeight])
im = axes[2].pcolor(X,Y,totalCoverage,cmap='viridis',vmin=0,vmax=100)
axes[2].set_ylabel("height AGL [m]")


axes[2].set_xlabel("date UTC")


cb_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
cbar = fig.colorbar(im, cax=cb_ax)
cbar.set_label('daily coverage by measurements [%]')
axes[3].legend()
axes[3].tick_params(axis='x', rotation=45)
#end








# joyceDf.plot.scatter(x='day',
#                       y='height',
#                       )




# plt.figure(figsize=(20,10))
# plt.subplot(311) 
# plt.title('Aeolus Wind Speed Availability Orbit 3082', size=20)
# ax = sns.scatterplot(x="measurementDatetime", y="alt", hue="speed", data=mieGdf, label="Mie Wind Speed")
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.subplot(312) 
# sns.scatterplot(x="measurementDatetime", y="alt", hue="speed", data=rayleighGdf, label="Rayleigh Wind Speed")
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.subplot(313) 
# sns.heatmap(x="day", y="height", hue="radar Coverage", data=joyceDf, label="JOYCE")
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


plt.savefig("availability.png", dpi=150)