#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 16:46:32 2023

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
cmap=plt.get_cmap(cmaps.CBR_wet)
newcolors=cmap(np.linspace(0, 1, 12))
newcmap = ListedColormap(newcolors[0:9])
cmap=plt.get_cmap(cmaps.MPL_BrBG)
newcolors=cmap(np.linspace(0, 16, 256))
newcmap1 = ListedColormap(newcolors[2:14])
cmap=plt.get_cmap(cmaps.ewdifft)
newcolors=cmap(np.linspace(0, 1, 16))
newcmap2 = ListedColormap(newcolors[2:14])
cmap=plt.get_cmap(cmaps.BlueDarkRed18)
newcolors=cmap(np.linspace(0, 1, 18))
newcmap3 = ListedColormap(newcolors[2:16])
mpl.rcParams["font.family"] = 'cm' 
mpl.rcParams["mathtext.fontset"] = 'cm'  # 数学文字字体
mpl.rcParams["font.size"] = 6.5
mpl.rcParams["axes.linewidth"] = 0.5
plt.rcParams['ytick.direction'] = 'out'

d1 = nc.Dataset(r'/Users/fuzhenghang/Documents/大四下/毕业论文/LBM/frc.pretr.nc')
t = d1['t'][0,]*86400
aa=t[8,:,:]
print(aa)

d2 = nc.Dataset(r'/Users/fuzhenghang/Documents/大四下/毕业论文/LBM/pretr.nc')
time = d2['time'][:]
lon = d2['lon'][:]
lat = d2['lat'][:]
lev = d2['lev'][:]
lev_2 = d2['lev_2'][:]
psi = d2['psi'][:]
chi = d2['chi'][:]
u = d2['u'][:,:]
v = d2['v'][:,:]
w = d2['w'][:]
t1 = d2['t'][:]
z = d2['z'][:]
p = d2['p'][:,0,]
print(u.shape)
print(lev)
#850-3;200-10

# In[1]
proj = ccrs.PlateCarree(central_longitude=180)  #中国为左
fig = plt.figure(figsize=(8,6),dpi=1000)
ax=[]
x1 = [0,0,0,0,0.4]
yy = [0.99,0.745,0.50,0.50]
dx = 0.45
dy = 0.21

levels = [np.arange(1,9.01,0.8),np.arange(-24,24.01,4),np.arange(-12,12.0101,2),np.arange(-0.000048,0.0000480101,0.000008)]

for i in range(2):
    ax.append(fig.add_axes([x1[i],yy[i],dx,dy],projection = proj))
ti = ['(a) 200-hPa H200 & UV200','(b) H500 & UV850','(c) T1000','+ 20 day','+ 25 day','+ 30 day']
for i in range(2):
    leftlon, rightlon, lowerlat, upperlat = (20,160,0,60)
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax[i].set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax[i].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.3,color='k',zorder=2)
    shp_path1 = r'/Users/fuzhenghang/Documents/python/tibetan/tibetan.shp'
    reader = Reader(shp_path1)
    tpfeat=cfeature.ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), edgecolor='grey', facecolor='none',zorder=2)
    ax[i].add_feature(tpfeat, linewidth=0.4)
    gl=ax[i].gridlines(draw_labels=True, linewidth=0.3, color='k', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator(np.arange(-180,181,30))
    gl.ylocator = mticker.FixedLocator(np.arange(-80,80,20))
    gl.ypadding=15
    gl.xpadding=15
    gl.top_labels    = False    
    gl.right_labels  = False
    gl.bottom_labels  = True
    if  i <=0:
        gl.bottom_labels  = False
    
    ax[i].text(-160,62,ti[i])
    #position2=fig.add_axes(loc[i])#位置[左,下,长度,宽度]
    #cbar1=plt.colorbar(cb1,cax=position2,orientation='vertical',ticks=ti[i],
                     #aspect=20,shrink=0.2,pad=0.06)#方向 
    #cbar1.ax.tick_params(length=1.8,width=0.4,pad=1.5)
    if i ==0 :
        cb1=ax[i].contourf(lon,lat,z[-1,10], levels=levels[1],cmap=newcmap2 ,transform=ccrs.PlateCarree(),extend='both',zorder=0)
        position2=fig.add_axes([0.42, 1, 0.01, 0.185])#位置[左,下,长度,宽度]
        cbar1=plt.colorbar(cb1,cax=position2,orientation='vertical',ticks=[-24,-12,0,12,24],
                         aspect=20,shrink=0.2,pad=0.06)#方向 
        cq = ax[i].quiver(lon[::],lat[::],u[-1,10,::,::],v[-1,10,::,::],color='k',
                              transform=ccrs.PlateCarree(),scale=40,width=0.0031,edgecolor='w',linewidth=0.15)   
        ax[i].quiverkey(cq, X=0.87, Y = 1.046, U=3 ,angle = 0,label='3 m/s',labelpos='E', color = 'k',labelcolor = 'k')
    if i ==1 :
        cb1=ax[i].contourf(lon,lat,z[-1,6], levels=levels[2],cmap=newcmap2 ,transform=ccrs.PlateCarree(),extend='both',zorder=0)
        position2=fig.add_axes([0.42, 0.76, 0.01, 0.185])#位置[左,下,长度,宽度]
        cbar1=plt.colorbar(cb1,cax=position2,orientation='vertical',ticks=[-12,-6,0,6,12],
                         aspect=20,shrink=0.2,pad=0.06)#方向 
        cq = ax[i].quiver(lon[::],lat[::],u[-1,3,::,::],v[-1,3,::,::],color='k',
                              transform=ccrs.PlateCarree(),scale=60,width=0.0031,edgecolor='w',linewidth=0.15)   
        ax[i].quiverkey(cq, X=0.87, Y = 1.046, U=5 ,angle = 0,label='5 m/s',labelpos='E', color = 'k',labelcolor = 'k')
    
    if i==2:
        cb1=ax[i].contourf(lon,lat,w[-1,5], levels=levels[3],cmap=newcmap2 ,transform=ccrs.PlateCarree(),extend='both',zorder=0)
print(p.shape)