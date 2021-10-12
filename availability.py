import os
from multiprocessing import Process, Queue
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import seaborn as sns
from numpy import array, vstack, zeros
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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


begin = datetime(2020, 1, 1)
end = datetime(2020, 12, 31)
analysis = RadarLidarWindSpeed(begin, end)
analysis.importDataset()
joyceDf = analysis.getCoverageHeightTimeSeries()

joyceDf.to_pickle("joycedf.pkl")


