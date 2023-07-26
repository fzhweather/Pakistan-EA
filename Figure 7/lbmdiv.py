#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 20:05:13 2023

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
from windspharm.standard import VectorWind
from windspharm.tools import prep_data, recover_data, order_latdim
cmap=plt.get_cmap(cmaps.ewdifft)
newcolors=cmap(np.linspace(0, 1, 16))
newcmap2 = ListedColormap(newcolors[2:14])
mpl.rcParams["font.family"] = 'cm' 
mpl.rcParams["mathtext.fontset"] = 'cm'  # 数学文字字体
mpl.rcParams["font.size"] = 9
mpl.rcParams["axes.linewidth"] = 0.8
plt.rcParams['ytick.direction'] = 'out'


d3 = xr.open_dataset(r'/Users/fuzhenghang/Documents/大四下/毕业论文/LBM/era40.clim.t42.nc')
lon3 = d3['longitude'][:]
lat3 = d3['latitude'][:]
u3 = d3['u'][6:8,10,:]
v3 = d3['v'][6:8,10,:]
windcu = np.zeros((1,64,128))
windcv = np.zeros((1,64,128))
windcu[0] = np.mean(u3,axis=0)
windcv[0] = np.mean(v3,axis=0)

d = nc.Dataset(r'/Users/fuzhenghang/Documents/大四下/毕业论文/LBM/pre78r.nc')
u = d['u'][-1,10,:]
v = d['v'][-1,10,:]
z = d['z'][-1,10]
z = np.roll(z,64)
windu = np.zeros((1,64,128))
windv = np.zeros((1,64,128))
windu[0] = u
windv[0] = v

d = nc.Dataset(r'/Users/fuzhenghang/Documents/大四下/毕业论文/LBM/pre6r.nc')
u6 = d['u'][-1,10,:]
v6 = d['v'][-1,10,:]
z6 = d['z'][-1,10]
z6 = np.roll(z6,64)
windu6 = np.zeros((1,64,128))
windv6 = np.zeros((1,64,128))
windu6[0] = u6
windv6[0] = v6
d2 = nc.Dataset(r'/Users/fuzhenghang/Documents/大四下/毕业论文/LBM/pretr.nc')
time = d2['time'][:]
lons = d2['lon'][:]
lats = d2['lat'][:]
zt = d2['z'][-1,10]
zt = np.roll(zt,64)
ut = d2['u'][-1,10,:]
vt = d2['v'][-1,10,:]
windut = np.zeros((1,64,128))
windvt = np.zeros((1,64,128))
windut[0] = ut
windvt[0] = vt

windcu = np.zeros((1,64,128))
windcv = np.zeros((1,64,128))

zz = [z,z6,zt]

out = np.zeros((3,64,128))
d2u = np.zeros((3,64,128))
d2v = np.zeros((3,64,128))
uu = [windu,windu6,windut]
vv = [windv,windv6,windvt]

uwnd, uwnd_info = prep_data(windcu, 'tyx')
vwnd, vwnd_info = prep_data(windcv, 'tyx')

lats, uwnd, vwnd = order_latdim(lats, uwnd, vwnd)
w = VectorWind(uwnd, vwnd)

divu, divv = w.irrotationalcomponent()
div = w.divergence()
avrt = w.absolutevorticity()
avrt_zonal, avrt_meridional = w.gradient(avrt)
rbws2 = -(divu*avrt_zonal+divv*avrt_meridional)
rbws1 = -avrt*div
RWSc = rbws1+rbws2

for i in range(3):
    uwnd, uwnd_info = prep_data(uu[i]+windcu, 'tyx')
    vwnd, vwnd_info = prep_data(vv[i]+windcv, 'tyx')

    lats, uwnd, vwnd = order_latdim(lats, uwnd, vwnd)
    w = VectorWind(uwnd, vwnd)
    
    divu, divv = w.irrotationalcomponent()
    d2u[i] = divu[:,:,0]
    d2v[i] = divv[:,:,0]
    div = w.divergence()
    avrt = w.absolutevorticity()
    avrt_zonal, avrt_meridional = w.gradient(avrt)
    rbws2 = -(divu*avrt_zonal+divv*avrt_meridional)
    rbws1 = -avrt*div
    RWS = rbws1+rbws2
    out[i] = 1e11*(RWS[:,:,0]-RWSc[:,:,0])
for i in range(3):
    for j in range(64):
        for k in range(128):
            if (d2u[i,j,k]**2 + d2v[i,j,k]**2)<=0.04:
                d2u[i,j,k]=np.nan
                d2v[i,j,k]=np.nan
proj = ccrs.PlateCarree(central_longitude=180)  #中国为左
fig = plt.figure(figsize=(8,8),dpi=1000)
ax=[]
x1 = [0,0,0,]
yy = [0.9,0.66,0.42]
dx = 0.6
dy = 0.21

levels = [np.arange(1,9.01,0.8),np.arange(-18,18+0.00001,3.0000),np.arange(-2.1,2.101,0.3)]

for i in range(3):
    ax.append(fig.add_axes([x1[i],yy[i],dx,dy],projection = proj))
ti = ['(a) Exp 1','(b) Exp 2','(c) Exp 3']
for i in range(3):
    leftlon, rightlon, lowerlat, upperlat = (1,180,0,60)
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax[i].set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax[i].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.25,color='k',zorder=2)
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
    if  i <=1:
        gl.bottom_labels  = False
    
    ax[i].text(-175,62,ti[i])
    out0, cycle_lon =add_cyclic_point(out[i,:,:], coord=lons)
    cb1=ax[i].contourf(cycle_lon,lats,out0, levels=levels[1],cmap=newcmap2 ,transform=ccrs.PlateCarree(),extend='both',zorder=0)
    #position2=fig.add_axes(loc[i])#位置[左,下,长度,宽度]
    #cbar1=plt.colorbar(cb1,cax=position2,orientation='vertical',ticks=ti[i],
                     #aspect=20,shrink=0.2,pad=0.06)#方向 
    #cbar1.ax.tick_params(length=1.8,width=0.4,pad=1.5)
    if i == 0:
        cq0 = ax[i].quiver(lons[::1],lats[::1],d2u[i,::1,::1],d2v[i,::1,::1],color='k',
                          transform=ccrs.PlateCarree(),scale=12,width=0.0025,alpha=0.7,edgecolor='w',linewidth=0.15)   
    cq = ax[i].quiver(lons[::1],lats[::1],d2u[i,::1,::1],d2v[i,::1,::1],color='k',
                      transform=ccrs.PlateCarree(),scale=12,width=0.0025,alpha=0.7,edgecolor='w',linewidth=0.15)   
    #zz1, cycle_lon =add_cyclic_point(zz[i][:,:], coord=lons)
    
    out0, cycle_lon =add_cyclic_point(zz[i], coord=lons)
    
    ax[i].contour(cycle_lon,lats,out0,[-25,-20,-15,-10,10,15,20,25],linewidths=1,alpha=0.75,colors=['k'],zorder=1)

ax[0].quiverkey(cq, X=0.87, Y = 1.046, U=1 ,angle = 0,label='1 m/s',alpha=0.7,labelpos='E', color = 'k',labelcolor = 'k')
position2=fig.add_axes([0.615,0.57,0.025,0.4])#位置[左,下,长度,宽度]
cbar1=plt.colorbar(cb1,cax=position2,orientation='vertical',ticks=[-15,-10,-5,0,5,10,15],
                 aspect=20,shrink=0.2,pad=0.06)#方向 
cbar1.ax.tick_params(length=1.8,width=0.4,pad=1.5)


u850 = d3['u'][5,10,:,:]

#u850 = u850.groupby('time.year').mean(dim='time')
#u850c = np.mean(u850,axis=0)
ut01, cycle_lon =add_cyclic_point(u850[:,:], coord=lon3)
CS=ax[1].contour(cycle_lon,lat3,ut01,[15,20],linewidths=0.75,alpha=0.75,colors=['magenta'],zorder=1)
#ax[2].clabel(CS, inline=1, fontsize=8)

u850 = d3['u'][6:8,10,:,:]
#u850 = u850.groupby('time.year').mean(dim='time')
u850c = np.mean(u850,axis=0)
#u850c = np.mean(u850,axis=0)
ut0, cycle_lon =add_cyclic_point(u850c[:,:], coord=lon3)
CS=ax[0].contour(cycle_lon,lat3,ut0,[15,20],linewidths=0.75,alpha=0.75,colors=['magenta'],zorder=1)
#ax[0].clabel(CS, inline=1, fontsize=8)
CS=ax[2].contour(cycle_lon,lat3,ut0,[15,20],linewidths=0.75,alpha=0.75,colors=['magenta'],zorder=1)
#ax[4].clabel(CS,inline=1, fontsize=8)
    
    
    