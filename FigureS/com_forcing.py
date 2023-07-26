#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 18:17:35 2023

@author: fuzhenghang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 16:47:09 2023

@author: fuzhenghang
"""

# In[0]
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import cartopy.mpl.ticker as cticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr  
import numpy as np
import cmaps
from cartopy.util import add_cyclic_point
from cartopy.io.shapereader import Reader
from scipy.stats import pearsonr
import pandas as pd
import scipy
from scipy.stats.mstats import ttest_ind
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

mpl.rcParams["font.family"] = 'cm' 
mpl.rcParams["mathtext.fontset"] = 'cm'  # 数学文字字体
mpl.rcParams["font.size"] = 7
mpl.rcParams["axes.linewidth"] = 0.5
plt.rcParams['ytick.direction'] = 'out'


d1 = nc.Dataset(r'/Users/fuzhenghang/Documents/大四下/毕业论文/LBM/frc.prer.nc')
lon = d1['lon'][:]
lat = d1['lat'][:]
t1 = d1['t'][0,]*86400
l = d1['lev']
aa=t1[:,21,26]
aaa=t1[8,:,:]
d2 = nc.Dataset(r'/Users/fuzhenghang/Documents/大四下/毕业论文/LBM/frc.pretr.nc')
t2 = d2['t'][0,]*86400
aat=t2[:,23,28]
aaat=t2[8,:,:]
d3 = xr.open_dataset(r'/Users/fuzhenghang/Documents/大四下/毕业论文/LBM/era40.clim.t42.nc')
#print(d3.variables.keys())
time = d3['time'][:]
lon3 = d3['longitude'][:]
lat3 = d3['latitude'][:]
#print(d3['lev'][:])

va = [aa,aa,aat]
tt = [t1,t1,t2]
fig = plt.figure(figsize=(8,4),dpi=900)
proj = ccrs.PlateCarree(central_longitude=180)  #中国为左
ax=[]
x1 = [0.1,0.52,0.1,0.52,0.1,0.52]
yy = [1.01,1.01,0.54,0.54,0.07,0.07]
dy = 0.4
for i in range(3):
    ax.append(fig.add_axes([x1[2*i],yy[2*i],0.4,dy],projection = proj))
    ax.append(fig.add_axes([x1[2*i+1],yy[2*i+1],0.12,dy]))
for i in range(1,6,2):
    ax[i].plot(va[i//2],l,'r-',lw = 1.5,zorder=0)
    ax[i].set_ylim(0,1)
    ax[i].set_xlim(0,8)
    ax[i].yaxis.tick_right()
    ax[i].yaxis.set_label_position("right") 
    ax[i].set_xticks([0,4,8])
    ax[i].set_ylabel('$\sigma$',labelpad=1,fontsize=10)
ax[5].set_xlabel('Rate  (K/day)',labelpad=1)

for i in range(0,6,2):
    leftlon, rightlon, lowerlat, upperlat = (60,140,15,55)
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax[i].set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax[i].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.3,color='k',zorder=2)

    gl=ax[i].gridlines(draw_labels=True, linewidth=0.3, color='k', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator(np.arange(-180,181,20))
    gl.ylocator = mticker.FixedLocator(np.arange(-80,80,10))
    gl.ypadding=15
    gl.xpadding=15
    gl.top_labels    = False    
    gl.right_labels  = False
    if i in [0,2]:
        gl.bottom_labels    = False 
    
    shp_path1 = r'/Users/fuzhenghang/Documents/python/tibetan/tibetan.shp'
    reader = Reader(shp_path1)
    tpfeat=cfeature.ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), edgecolor='grey', facecolor='none',zorder=2)
    ax[i].add_feature(tpfeat, linewidth=0.4)
    cb1=ax[i].contourf(lon,lat,tt[i//2][8], levels=np.arange(1,9.01,0.8),cmap=cmaps.MPL_Reds ,transform=ccrs.PlateCarree(),extend='max',zorder=0)
position2=fig.add_axes([0.15, 0.01, 0.3, 0.03])#位置[左,下,长度,宽度]
cbar1=plt.colorbar(cb1,cax=position2,orientation='horizontal',ticks=[1,3,5,7,9],
                     aspect=20,shrink=0.2,pad=0.06)#方向 
cbar1.ax.tick_params(length=1.8,width=0.4,pad=1.5)
ax[0].text(-119,56.5,'Exp 1',fontsize=10)
ax[2].text(-119,56.5,'Exp 2',fontsize=10)
ax[4].text(-119,56.5,'Exp 3',fontsize=10)
ax[0].text(-125,56.5,'a',fontsize=12,fontweight='bold')
ax[2].text(-125,56.5,'b',fontsize=12,fontweight='bold')
ax[4].text(-125,56.5,'c',fontsize=12,fontweight='bold')

u850 = d3['u'][5,10,:,:]
#u850 = u850.groupby('time.year').mean(dim='time')
#u850c = np.mean(u850,axis=0)
CS=ax[2].contour(lon3,lat3,u850,[10,15,20],linewidths=1.5,alpha=0.75,colors=['lightgreen','g','darkgreen'],zorder=1)
#ax[2].clabel(CS, inline=1, fontsize=8)

u850 = d3['u'][6:8,10,:,:]
#u850 = u850.groupby('time.year').mean(dim='time')
u850c = np.mean(u850,axis=0)
CS=ax[0].contour(lon3,lat3,u850c,[10,15,20],linewidths=1.5,alpha=0.75,colors=['lightgreen','g','darkgreen'],zorder=1)
#ax[0].clabel(CS, inline=1, fontsize=8)
CS=ax[4].contour(lon3,lat3,u850c,[10,15,20],linewidths=1.5,alpha=0.75,colors=['lightgreen','g','darkgreen'],zorder=1)
#ax[4].clabel(CS,inline=1, fontsize=8)
