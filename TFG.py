import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import folium
import mpimg
from matplotlib.animation import FuncAnimation
# from pymaps import Map, PyMap, Icon
from geopy.geocoders import Nominatim
from mpl_toolkits.basemap import Basemap
import animatplot
import matplotlib.animation as an
import matplotlib.animation as animation
from shapely import geometry
from math import radians, cos, sin, asin, sqrt
import pyproj as proj
import cartopy
import cartopy.crs as ccrs
from pyproj import Transformer
from cartopy.geodesic import Geodesic
from shapely import geometry
from shapely.geometry import Point
from shapely.ops import transform
import shapely.geometry as sgeom

tree = ET.parse('airsidee.xml')
root = tree.getroot()
for child in root:
    print(child.tag, child.attrib)
SR = []
lat = []
long = []

# extraemos del xml
for serviceroads in root.iter('sr'):
    SR.append(serviceroads.attrib)
    for sr in serviceroads.iter('entry'):
        SR.append(sr.attrib)
        for latitude in sr.iter('latitude'):
            lat.append(latitude.attrib)
        for longitude in sr.iter('longitude'):
            long.append(longitude.attrib)
            for decimal_degrees in latitude.iter('decimal_degrees'):
                # print(decimal_degrees.text)
                SR.append(float(decimal_degrees.text))
            for decimal_degrees in longitude.iter('decimal_degrees'):
                #  print(decimal_degrees.text)
                SR.append(float(decimal_degrees.text))
print(SR)
TW = []
lat1 = []
long1 = []
for taxiways in root.iter('TWY'):
    TW.append(taxiways.attrib)
    for TWY in taxiways.iter('holding_point'):
        TW.append(TWY.attrib)
        for latitude in TWY.iter('latitude'):
            lat1.append(latitude.attrib)
        for longitude in TWY.iter('longitude'):
            long1.append(longitude.attrib)
            for decimal_degrees in latitude.iter('decimal_degrees'):
                # print(decimal_degrees.text)
                TW.append(float(decimal_degrees.text))
            for decimal_degrees in longitude.iter('decimal_degrees'):
                #  print(decimal_degrees.text)
                TW.append(float(decimal_degrees.text))
for taxiways in root.iter('TWY'):
    TW.append(taxiways.attrib)
    for TWY in taxiways.iter('crossing_point'):
        TW.append(TWY.attrib)
        for latitude in TWY.iter('latitude'):
            lat1.append(latitude.attrib)
        for longitude in TWY.iter('longitude'):
            long1.append(longitude.attrib)
            for decimal_degrees in latitude.iter('decimal_degrees'):
                # print(decimal_degrees.text)
                TW.append(float(decimal_degrees.text))
            for decimal_degrees in longitude.iter('decimal_degrees'):
                #  print(decimal_degrees.text)
                TW.append(float(decimal_degrees.text))
print(TW)
corner2 = []
lat3 = []
long3 = []
for taxiways in root.iter('TWY'):
    corner2.append(taxiways.attrib)
    for TWY in taxiways.iter('corner'):
        corner2.append(TWY.attrib)
        for corner in TWY.iter('entry'):
            corner2.append(corner.attrib)
            for latitude in corner.iter('latitude'):
                lat1.append(latitude.attrib)
            for longitude in corner.iter('longitude'):
                long1.append(longitude.attrib)
                for decimal_degrees in latitude.iter('decimal_degrees'):
                    # print(decimal_degrees.text)
                    corner2.append(float(decimal_degrees.text))
                for decimal_degrees in longitude.iter('decimal_degrees'):
                    #  print(decimal_degrees.text)
                    corner2.append(float(decimal_degrees.text))
print(corner2)
RWY_exit = []
lat2 = []
long2 = []

for runways in root.iter('rwy_exit'):
    RWY_exit.append(runways.attrib)
    for rwy_exit in runways.iter('entry'):
        RWY_exit.append(rwy_exit.attrib)
        for latitude in rwy_exit.iter('latitude'):
            lat2.append(latitude.attrib)
        for longitude in rwy_exit.iter('longitude'):
            long2.append(longitude.attrib)
            for decimal_degrees in latitude.iter('decimal_degrees'):
                # print(decimal_degrees.text)
                RWY_exit.append(float(decimal_degrees.text))
            for decimal_degrees in longitude.iter('decimal_degrees'):
                #  print(decimal_degrees.text)
                RWY_exit.append(float(decimal_degrees.text))
print(RWY_exit)
Fueltrajectory_x = []
Fueltrajectory_y = []
initial_point = {'id': 'TB'}
subinitial = {'id': '1'}
final_point = {'id': 'THR29'}
subfinal = {'id': '3'}

i = 0
j = 1
encontrado = False
# initial trajectory (de momento solo irá por via de servicio)
while (SR[i] != 0 and not encontrado):
    while SR[i] != initial_point:
        if SR[i] != initial_point:
            i = i + 1
    if SR[i] == initial_point:
        encontrado = True
        j = i + 2
while (type(SR[i]) == dict and type(SR[j]) == float):
    Fueltrajectory_x.append(SR[j])
    Fueltrajectory_y.append(SR[j + 1])
    j = j + 3

print(Fueltrajectory_x)
print(Fueltrajectory_y)
i = 0

t = []
vx = 0
Fueltrajectory_x_final = []

distance = []
# convertim en distancia
i = 0
while (i + 1) < len(Fueltrajectory_x):
    rad = math.pi / 180
    dlat = Fueltrajectory_x[i + 1] - Fueltrajectory_x[i]
    dlon = Fueltrajectory_y[i + 1] - Fueltrajectory_y[i]
    R = 6372.795477598
    a = (math.sin(rad * dlat / 2)) ** 2 + math.cos(rad * Fueltrajectory_x[i]) * math.cos(
        rad * Fueltrajectory_x[i + 1]) * (math.sin(rad * dlon / 2)) ** 2
    d = 2 * R * math.asin(math.sqrt(a))
    distance.append(d)
    # time.append((d/ServiceRoadsVelocity)*3600)
    i = i + 1

# print (distance)
Fueltrajectory_y_final = []
i = 0
j = 1
val = 0
# Punts de trajectoria
Fueltrajectory_x_final = []
ServiceRoadsVelocity = 30;

val = 0
val2 = 0
while (i < len(Fueltrajectory_y)):
    if (j < len(Fueltrajectory_y)):
        if (distance[i] >= 0.08):
            val = (Fueltrajectory_y[j] - Fueltrajectory_y[i]) / 4
            Fueltrajectory_y_final.append(Fueltrajectory_y[i])
            Fueltrajectory_y_final.append(Fueltrajectory_y[i] + val)
            Fueltrajectory_y_final.append(Fueltrajectory_y[i] + 2 * val)
            Fueltrajectory_y_final.append(Fueltrajectory_y[i] + 3 * val)
            val2 = (Fueltrajectory_x[j] - Fueltrajectory_x[i]) / 4
            Fueltrajectory_x_final.append(Fueltrajectory_x[i])
            Fueltrajectory_x_final.append(Fueltrajectory_x[i] + val2)
            Fueltrajectory_x_final.append(Fueltrajectory_x[i] + 2 * val2)
            Fueltrajectory_x_final.append(Fueltrajectory_x[i] + 3 * val2)
        if (distance[i] >= 0.05 and distance[i] < 0.08):
            val = (Fueltrajectory_y[j] - Fueltrajectory_y[i]) / 2
            Fueltrajectory_y_final.append(Fueltrajectory_y[i])
            Fueltrajectory_y_final.append(Fueltrajectory_y[i] + val)
            val2 = (Fueltrajectory_x[j] - Fueltrajectory_x[i]) / 2
            Fueltrajectory_x_final.append(Fueltrajectory_x[i])
            Fueltrajectory_x_final.append(Fueltrajectory_x[i] + val2)
        if (distance[i] < 0.05):
            Fueltrajectory_y_final.append(Fueltrajectory_y[i])
            Fueltrajectory_x_final.append(Fueltrajectory_x[i])
    else:
        Fueltrajectory_y_final.append(Fueltrajectory_y[i])
        Fueltrajectory_x_final.append(Fueltrajectory_x[i])
    i = i + 1
    val = 0
    val2 = 0
    j = j + 1

i = 0
while (i + 1) < len(Fueltrajectory_x_final):
    if Fueltrajectory_x_final[i] == Fueltrajectory_x_final[i + 1] and Fueltrajectory_y_final[i] == \
            Fueltrajectory_y_final[i + 1]:
        Fueltrajectory_x_final.pop(i)
        Fueltrajectory_y_final.pop(i)
    i = i + 1
# print (Fueltrajectory_y_final)
# print (Fueltrajectory_x_final)
# Time as a function of distance
time = []
time.append(1.0)
i = 0

distance2 = []
while (i + 1) < len(Fueltrajectory_x_final):
    rad = math.pi / 180
    dlat = Fueltrajectory_x_final[i + 1] - Fueltrajectory_x_final[i]
    dlon = Fueltrajectory_y_final[i + 1] - Fueltrajectory_y_final[i]
    R = 6372.795477598
    a = (math.sin(rad * dlat / 2)) ** 2 + math.cos(rad * Fueltrajectory_x_final[i]) * math.cos(
        rad * Fueltrajectory_x_final[i + 1]) * (math.sin(rad * dlon / 2)) ** 2
    d = 2 * R * math.asin(math.sqrt(a))
    distance2.append(d)
    time.append(time[i]+(d / ServiceRoadsVelocity) * 3600)
    i = i + 1

# print(distance2)
# print(time)

# trajectory2
Fueltrajectory_x2 = []
Fueltrajectory_y2 = []
initial_point = {'id': 'THR29'}
subinitial = {'id': '1'}
final_point = {'id': 'T'}
subfinal = {'id': 'TB2'}

i = 0
j = 1
encontrado = False
# initial trajectory (THR29-TB2)
# primer tramo (THR29)
while (RWY_exit[i] != 0 and not encontrado):
    while RWY_exit[i] != initial_point:
        if RWY_exit[i] != initial_point:
            i = i + 1
    if RWY_exit[i] == initial_point:
        encontrado = True
        j = i + 2
while (type(RWY_exit[i]) == dict and type(RWY_exit[j]) == float):
    Fueltrajectory_x2.append(RWY_exit[j])
    Fueltrajectory_y2.append(RWY_exit[j + 1])
    j = j + 3

# segundo tramo (THR29-C1-2)
initial_point = {'id': 'C1'}
subinitial = {'id': '1'}
i = 1
j = 2
encontrado = False
while (corner2[i] != 0 and not encontrado):
    while corner2[i] != initial_point:
        if corner2[i] != initial_point:
            i = i + 1
    if corner2[i] == initial_point:
        encontrado = True
        j = i + 2
while (type(corner2[i]) == dict and type(corner2[j]) == float):
    Fueltrajectory_x2.append(corner2[j])
    Fueltrajectory_y2.append(corner2[j + 1])
    j = j + 3
print(Fueltrajectory_x2)
print(Fueltrajectory_y2)
# tercer tramo (C1-2 - TB3)
initial_point = {'id': 'T'}
subinitial = {'id': 'TB1'}
final_point = {'id': 'T'}
subfinal = {'id': 'TB3'}
i = 0
j = 2
k = 0
l = 1
encontrado = False
while (TW[i] != 0 and not encontrado):
    while TW[i] != initial_point:
        if TW[i] != initial_point:
            i = i + 1
    if TW[i] == initial_point:
        if TW[l] == subinitial:
            encontrado = True
            j = i + 2
            k = i + 1
        else:
            l = l + 3
j = l + 1
while (type(TW[i]) == dict and type(TW[j]) == float and TW[k] != subfinal):
    Fueltrajectory_x2.append(TW[j])
    Fueltrajectory_y2.append(TW[j + 1])
    j = j + 3
    k = k + 3

print(Fueltrajectory_x2)
print(Fueltrajectory_y2)

i = 0

t = []
vx = 0
Fueltrajectory_x_final2 = []
Fueltrajectory_y_final2 = []

distance22 = []
# convertim en distancia (km)
i = 0
while (i + 1) < len(Fueltrajectory_x2):
    rad = math.pi / 180
    dlat = Fueltrajectory_x2[i + 1] - Fueltrajectory_x2[i]
    dlon = Fueltrajectory_y2[i + 1] - Fueltrajectory_y2[i]
    R = 6372.795477598
    a = (math.sin(rad * dlat / 2)) ** 2 + math.cos(rad * Fueltrajectory_x2[i]) * math.cos(
        rad * Fueltrajectory_x2[i + 1]) * (math.sin(rad * dlon / 2)) ** 2
    d = 2 * R * math.asin(math.sqrt(a))
    distance22.append(d)
    # time.append((d/ServiceRoadsVelocity)*3600)
    i = i + 1

print(distance22)
i = 0
j = 1
val = 0
# Punts de trajectoria
val2 = 0
while (i < len(Fueltrajectory_y2)):
    if (j < len(Fueltrajectory_y2)):
        if (distance22[i] >= 0.08):
            val = (Fueltrajectory_y2[j] - Fueltrajectory_y2[i]) / 4
            Fueltrajectory_y_final2.append(Fueltrajectory_y2[i])
            Fueltrajectory_y_final2.append(Fueltrajectory_y2[i] + val)
            Fueltrajectory_y_final2.append(Fueltrajectory_y2[i] + 2 * val)
            Fueltrajectory_y_final2.append(Fueltrajectory_y2[i] + 3 * val)
            val2 = (Fueltrajectory_x2[j] - Fueltrajectory_x2[i]) / 4
            Fueltrajectory_x_final2.append(Fueltrajectory_x2[i])
            Fueltrajectory_x_final2.append(Fueltrajectory_x2[i] + val2)
            Fueltrajectory_x_final2.append(Fueltrajectory_x2[i] + 2 * val2)
            Fueltrajectory_x_final2.append(Fueltrajectory_x2[i] + 3 * val2)
        if (distance22[i] >= 0.05 and distance22[i] < 0.08):
            val = (Fueltrajectory_y2[j] - Fueltrajectory_y2[i]) / 2
            Fueltrajectory_y_final2.append(Fueltrajectory_y2[i])
            Fueltrajectory_y_final2.append(Fueltrajectory_y2[i] + val)
            val2 = (Fueltrajectory_x2[j] - Fueltrajectory_x2[i]) / 2
            Fueltrajectory_x_final2.append(Fueltrajectory_x2[i])
            Fueltrajectory_x_final2.append(Fueltrajectory_x2[i] + val2)
        if (distance22[i] < 0.05):
            Fueltrajectory_y_final2.append(Fueltrajectory_y2[i])
            Fueltrajectory_x_final2.append(Fueltrajectory_x2[i])
    else:
        Fueltrajectory_y_final2.append(Fueltrajectory_y2[i])
        Fueltrajectory_x_final2.append(Fueltrajectory_x2[i])
    i = i + 1
    val = 0
    val2 = 0
    j = j + 1

# print (Fueltrajectory_y_final)
# print (Fueltrajectory_x_final)
# Time as a function of distance
i = 0
while (i + 1) < len(Fueltrajectory_x_final2):
    if Fueltrajectory_x_final2[i] == Fueltrajectory_x_final2[i + 1] and Fueltrajectory_y_final2[i] == \
            Fueltrajectory_y_final2[i + 1]:
        Fueltrajectory_x_final2.pop(i)
        Fueltrajectory_y_final2.pop(i)
    i = i + 1
time2 = []
time2.append(1.0)
TWYvelocity = 75
i = 0
distance3 = []
while (i + 1) < len(Fueltrajectory_x_final2):
    rad = math.pi / 180
    dlat = Fueltrajectory_x_final2[i + 1] - Fueltrajectory_x_final2[i]
    dlon = Fueltrajectory_y_final2[i + 1] - Fueltrajectory_y_final2[i]
    R = 6372.795477598
    a = (math.sin(rad * dlat / 2)) ** 2 + math.cos(rad * Fueltrajectory_x_final2[i]) * math.cos(
        rad * Fueltrajectory_x_final2[i + 1]) * (math.sin(rad * dlon / 2)) ** 2
    d = 2 * R * math.asin(math.sqrt(a))
    distance3.append(d)
    time2.append(time2[i]+(d / TWYvelocity) * 3600)
    i = i + 1

timegraph=[]
i=1
j=1
timegraph.append(1)
b=0
c=0
while i<len(time) or j<len(time2):
    if i<len(time) and j<len(time2):
        b=time[i]
        c=time2[j]
        if b<c:
            timegraph.append(b)
            i=i+1
        if c<b:
            timegraph.append(c)
            j=j+1
    if i >= len(time) and j < len(time2):
        timegraph.append(time2[j])
        j = j + 1
    if j >= len(time2) and i < len(time):
        timegraph.append(time[i])
        i = i + 1
timegraphf=[]
timegraphf.append(timegraph[0])
i=0
j=0
while (i+1)<len(timegraph):
    timegraphf.append(timegraph[i+1]-timegraph[i])
    i=i+1

trackx1=[]
tracky1=[]
trackx2=[]
tracky2=[]
i=0
k=0
j=0
l=0
while k<len(timegraph) or l<len(timegraph):
    if k<len(timegraph):
        if i<len(time):
            if timegraph[k]==time[i]:
                trackx1.append(Fueltrajectory_x_final[i])
                tracky1.append(Fueltrajectory_y_final[i])
                k=k+1
                i = i + 1
            else:
                trackx1.append(Fueltrajectory_x_final[i-1])
                tracky1.append(Fueltrajectory_y_final[i-1])
                k=k+1
    if l<len(timegraph):
        if j<len(time2):
            if timegraph[l] == time2[j]:
                trackx2.append(Fueltrajectory_x_final2[j])
                tracky2.append(Fueltrajectory_y_final2[j])
                l=l+1
                j = j + 1
            else:
                trackx2.append(Fueltrajectory_x_final2[j-1])
                tracky2.append(Fueltrajectory_y_final2[j-1])
                l=l+1
    if i>=len(time) and k<len(timegraph):
        trackx1.append(Fueltrajectory_x_final[i-1])
        tracky1.append(Fueltrajectory_y_final[i-1])
        k=k+1
    if j>=len(time2) and l<len(timegraph):
        trackx2.append(Fueltrajectory_x_final2[j-1])
        tracky2.append(Fueltrajectory_y_final2[j-1])
        l=l+1

# Conflict detection
#Detectaremos el conflicto cuando se encuentre a cierta distancia uno del otro.
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r
i=0
for i in range(len(trackx1)):
    center_point = [{'lat': tracky1[i], 'lng': trackx1[i]}]
    test_point = [{'lat': tracky2[i], 'lng': trackx2[i]}]

    lat1 = center_point[0]['lat']
    lon1 = center_point[0]['lng']
    lat2 = test_point[0]['lat']
    lon2 = test_point[0]['lng']

    radius = 0.1 # in kilometer

    a = haversine(lon1, lat1, lon2, lat2)

    print('Distance (km) : ', a)
    if a <= radius:
        print('Inside the area')
        print('Vehicle 1: ', tracky1[i],trackx1[i])
        print('Vehicle 2: ', tracky2[i],trackx2[i])
    else:
        print('Outside the area')
#x1, y1 = proj.transform(proj.Proj(init='epsg:4326'), proj.Proj(init='epsg:27700'), tracky1[0], trackx1[0])

geoms=[]
#i=0
#for i in range(len(time2)):
i=0
for i in range(len(timegraph)):
    BBox = (-6.0512, -6.0184,
            43.557939, 43.5689)
    ruh_m = plt.imread('prueba3.png')
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_xlim(BBox[0], BBox[1])
    ax.set_ylim(BBox[2], BBox[3])
    ax.imshow(ruh_m, zorder=0, extent=BBox, aspect='auto', interpolation='nearest')
    ax.axis('off')
    fig.tight_layout()
    plt.scatter(tracky1[i],trackx1[i], s=200)
    gd=Geodesic()

    cp=gd.circle(tracky1[i],trackx1[i], radius=1000)
    geoms.append(sgeom.Polygon(cp))
    ax.add_geometries(geoms, crs=src_crs, edgecolor='r', alpha=0.5)
    # Get the polygon with lat lon coordinates
    #circle_poly = transform(aeqd_to_wgs84, buffer)
    #circle=plt.Circle(l, 0.1, color='g')

    #fig = plt.gcf()
    #ax = fig.gca()

    #ax.add_patch(circle)
    #ax.add_patch(circle1)
    plt.pause(timegraphf[i])


    #plt.pause(time[i])
    i=i+1
#ax.scatter(-6.043681,43.564080)
    plt.show()
