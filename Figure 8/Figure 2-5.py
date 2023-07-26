#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 21:58:59 2023

@author: fuzhenghang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:43:04 2023

@author: fuzhenghang
"""

# In[0]

import vertical_diff as V

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import cartopy.mpl.ticker as cticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr  
import numpy as np
import cmaps
from cartopy.io.shapereader import Reader
from scipy.stats import pearsonr


# In[1]
from matplotlib.colors import ListedColormap
cmap=plt.get_cmap(cmaps.GMT_red2green_r)
newcolors=cmap(np.linspace(0, 1, 256))
newcmap1 = ListedColormap(newcolors[20:235:10])

mpl.rcParams["mathtext.fontset"] = 'cm'  # 数学文字字体
mpl.rcParams["font.size"] = 8
mpl.rcParams["axes.linewidth"] = 0.9
plt.rcParams['ytick.direction'] = 'out'


d1 = xr.open_dataset('/Users/fuzhenghang/Documents/ERA5/temp_geo_5monthly9_1979_2022.nc')
lon = d1.variables['longitude'][:]
lat = d1.variables['latitude'][:]

proj = ccrs.PlateCarree()  #中国为左
fig = plt.figure(figsize=(12,7),dpi=1000)
ax=[]
x1 = [0,0,0.33,0.33,0,0.33]
yy = [0.72,0.43,0.72,0.43,0,0]
dx = [0.33,0.33,0.66,0.66,0.8,0.66]
dy = [0.25,0.25,0.25,0.25,0.9,0.375]
loc = [[0.335, 0.735, 0.0085, 0.22],[0.335, 0.445, 0.0085, 0.22],[0.903, 0.735,0.0085, 0.22],[0.903, 0.448, 0.0085, 0.22],[0.335, 0.028, 0.0085, 0.32],[0.903, 0.028, 0.0085, 0.32]]

for i in range(5):
    if i != 4:
        ax.append(fig.add_axes([x1[i],yy[i],dx[i],dy[i]],projection = proj))
    else: 
        ax.append(fig.add_axes([x1[i],yy[i],dx[i],dy[i]]))
label = ['A','B']
sc = ['mm/month','day/year','gpm','Pa/s','K','$10^{-11}/s$']
label2 = ['C','D']
#cofi = cmaps.pengsa
#cofi = plt.cm.bwr
cofi = cmaps.pengsa
wc = '#555555'
ic = '#3A3A3A'
bc = 'navy'
rc = 'darkred'
fs = 12

tik = [[-60,-30,0,30,60],[-0.04,-0.02,0,0.02,0.04]]
for i in range(5):
    if i <= 1:
        leftlon, rightlon, lowerlat, upperlat = (60,150,5,45)
        shp_path1 = r'/Users/fuzhenghang/Documents/python/Pakistan/tibetan/tibetan.shp'
        reader = Reader(shp_path1)
        tpfeat=cfeature.ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), edgecolor='grey', facecolor='none',zorder=2)
        ax[i].add_feature(tpfeat, linewidth=0.4)
    elif i in [2,3]:
        leftlon, rightlon, lowerlat, upperlat = (30,160,20,60)
    elif i == 5:
        leftlon, rightlon, lowerlat, upperlat = (30,160,0,60)
    if i != 4:
        
        lon_formatter = cticker.LongitudeFormatter()
        lat_formatter = cticker.LatitudeFormatter()
        ax[i].set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
        ax[i].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k',zorder=2)
        ax[i].add_feature(cfeature.OCEAN.with_scale('50m'),zorder=1,color='w',lw=0)
        
        gl=ax[i].gridlines(draw_labels=True, linewidth=0.35, color='k', alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator(np.arange(0,181,20))
        gl.ylocator = mticker.FixedLocator(np.arange(10,80,10))
        gl.top_labels    = False    
        gl.right_labels  = False
        gl.ypadding=50
        gl.xpadding=50
    
   
 


    if i == 4:
        ax[i].set_yscale('symlog')
        ax[i].set_yticks([1000, 500,300, 200, 100])
        ax[i].set_yticklabels(['1000','500','300','200','100'])
        ax[i].invert_yaxis()
        ax[i].set_ylabel('hPa',fontsize=9)
        c = ax[i].contourf(V.x[i-4],V.level ,V.data[i-3],levels=[-2,-1.6,-1.2,-0.8,-0.4,0.4,0.8,1.2,1.6,2], extend = 'both',zorder=1, cmap=cofi)
        
        ax[i].xaxis.set_major_formatter(cticker.LongitudeFormatter())  
        
        #CS=ax[i].contour(V.lon,V.level,V.theta1,[310+5*j for j in range(15)],linewidths=0.92,alpha=0.4,colors=[ic],zorder=1)
        #ax[i].clabel(CS,[310+5*j for j in range(15)], inline=1, fontsize=12)
        ax[i].set_xlim(40,160)
        
        #cq=ax[i].quiver(V.lon[::4],V.level,V.wind[2][:,::4],V.wind[3][:,::4]*(-100),color=wc,scale=180,width=0.0024,zorder=10)
        #cq=ax[i].quiver(V.lon[::4],V.level,V.winds[2][:,::4],V.winds[3][:,::4]*(-100),color=wc,scale=120,width=0.0048,linewidth=0.3)
        #ax[i].quiverkey(cq, X=0.87, Y = 1.046, U = 10,angle = 0,label='10m/s',labelpos='E', color = wc,labelcolor = 'k')
        ax[i].text(142.3,1000,sc[i],fontsize=8)
        

         
    