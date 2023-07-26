#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 09:49:57 2023

@author: fuzhenghang
"""

#Comparation of monthly MCA

from xmca.array import MCA  # use with np.ndarray
from xmca.xarray import xMCA  # use with xr.DataArray
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
from matplotlib.colors import ListedColormap
cmap=plt.get_cmap(cmaps.MPL_BrBG_r)
newcolors=cmap(np.linspace(0, 8, 256))
newcmap1 = ListedColormap(newcolors[4:29])
cmap=plt.get_cmap(cmaps.BlueDarkRed18)
newcolors=cmap(np.linspace(0, 1, 18))
newcmap3 = ListedColormap(newcolors[1:17])

mpl.rcParams["font.family"] = 'cm' 
mpl.rcParams["mathtext.fontset"] = 'cm'  # 数学文字字体
mpl.rcParams["font.size"] = 7
mpl.rcParams["axes.linewidth"] = 0.6
plt.rcParams['ytick.direction'] = 'out'


d1 = xr.open_dataset(r'/Users/fuzhenghang/Documents/ERA5/OLR_5month9_1979_2022.nc',use_cftime=True)
lon = d1.variables['longitude'][:]
lat = d1.variables['latitude'][:]
#print(d1.variables.keys())
time = d1['time'][:]
pre6 = d1['ttr'][(time.dt.month>=6)&(time.dt.month<=6)][:,0,:,:]/(-86400)
pre6 = pre6.groupby('time.year').mean(dim='time')
scipy.signal.detrend(pre6, axis=0, type='linear', bp=0, overwrite_data=True)
#print(pre.shape)
pre7 = d1['ttr'][(time.dt.month>=7)&(time.dt.month<=7)][:,0,:,:]/(-86400)
pre7 = pre7.groupby('time.year').mean(dim='time')
scipy.signal.detrend(pre7, axis=0, type='linear', bp=0, overwrite_data=True)

pre8 = d1['ttr'][(time.dt.month>=8)&(time.dt.month<=8)][:,0,:,:]/(-86400)
pre8 = pre8.groupby('time.year').mean(dim='time')
scipy.signal.detrend(pre8, axis=0, type='linear', bp=0, overwrite_data=True)

a1 = 55
a2 = 81
b1 = 60
b2 = 101
pre6 = pre6[:,a1:a2,b1:b2]
pre7 = pre7[:,a1:a2,b1:b2]
pre8 = pre8[:,a1:a2,b1:b2]

lats = d1['latitude'][a1:a2]
lons = d1['longitude'][b1:b2]
times6 = d1['time'][(time.dt.month==6)]
times7 = d1['time'][(time.dt.month==7)]
times8 = d1['time'][(time.dt.month==8)]
pre6 = xr.DataArray(pre6,coords=[times6, lats, lons], dims=['time', 'lat','lon'])
pre7 = xr.DataArray(pre7,coords=[times7, lats, lons], dims=['time', 'lat','lon'])
pre8 = xr.DataArray(pre8,coords=[times8, lats, lons], dims=['time', 'lat','lon'])


"""
hw = np.load("/Users/fuzhenghang/Documents/大四上/热浪/中间数据/R_hw_days_1979-2022_monthly.npy")

#处理成距平
clihw = np.zeros((3,181,360))
for i in range(129):
    for j in range(181):
        for k in range(360):
            clihw[i%3,j,k]+=hw[i,j,k]

clihw = clihw/43

for i in range(129):
    hw[i,:,:] = hw[i,:,:]-clihw[i%3,:,:]
np.save("/Users/fuzhenghang/Documents/大四上/热浪/中间数据/R_hw_days_1979-2022_monthly_anomaly.npy",hw)
"""
hw6 = np.zeros((44,26,81))
hw7 = np.zeros((44,26,81))
hw8 = np.zeros((44,26,81))
hw = np.load("/Users/fuzhenghang/Documents/大四上/热浪/中间数据/R_hw_days_1979-2022_monthly_anomaly.npy")
lats1 = d1['latitude'][45:71]
lons1 = d1['longitude'][60:141]
for i in range(44):
    hw6[i]=hw[i*3:i*3+1,45:71,60:141]
    hw7[i]=hw[i*3+1:i*3+2,45:71,60:141]
    hw8[i]=hw[i*3+2:i*3+3,45:71,60:141]
scipy.signal.detrend(hw6, axis=0, type='linear', bp=0, overwrite_data=True)
scipy.signal.detrend(hw7, axis=0, type='linear', bp=0, overwrite_data=True)
scipy.signal.detrend(hw8, axis=0, type='linear', bp=0, overwrite_data=True)
hw6 = xr.DataArray(hw6,coords=[times6, lats1, lons1], dims=['time', 'lat','lon'])
hw7 = xr.DataArray(hw7,coords=[times7, lats1, lons1], dims=['time', 'lat','lon'])
hw8 = xr.DataArray(hw8,coords=[times8, lats1, lons1], dims=['time', 'lat','lon'])

cm = [newcmap1,newcmap3,newcmap1,newcmap3,newcmap1,newcmap3]
mca6 = xMCA(pre6, hw6)                  # MCA of field A and B
mca7 = xMCA(pre7, hw7)
mca8 = xMCA(pre8, hw8)
mca6.solve(complexify=False)            # True for complex MCA
mca7.solve(complexify=False)
mca8.solve(complexify=False)
eigenvalues = mca6.singular_values()     # singular vales
eigenvalues = mca7.singular_values()
eigenvalues = mca8.singular_values()
pcs6 = mca6.pcs()                         # expansion coefficient (PCs)
eofs6 = mca6.eofs()                       # spatial patterns (EOFs)
expvar6 = mca6.explained_variance()
pcs7 = mca7.pcs()                         # expansion coefficient (PCs)
eofs7 = mca7.eofs()                       # spatial patterns (EOFs)
expvar7 = mca7.explained_variance()
pcs8 = mca8.pcs()                         # expansion coefficient (PCs)
eofs8 = mca8.eofs()                       # spatial patterns (EOFs)
expvar8 = mca8.explained_variance()
#print(eofs.get('left'))
mode=0
pattern6 = [np.array(eofs6.get('left')[:,:,mode]),np.array(eofs6.get('right')[:,:,mode])]
pattern7 = [np.array(eofs7.get('left')[:,:,mode]),np.array(eofs7.get('right')[:,:,mode])]
pattern8 = [np.array(eofs8.get('left')[:,:,mode]),np.array(eofs8.get('right')[:,:,mode])]
#print(expvar6[0:2])
#print(expvar7[0:2])
#print(expvar8[0:2])
lo = [lons,lons1,lons,lons1,lons,lons1]
la = [lats,lats1,lats,lats1,lats,lats1]
proj = ccrs.PlateCarree()  #中国为左
fig = plt.figure(figsize=(6,8),dpi=600)
ax=[]
x1 = [0,0.37,0,0.37,0,0.37]
yy = [1,1,0.83,0.83,0.66,0.66]
dx = [0.3,0.6,0.3,0.6,0.3,0.6]
dy = 0.19
loc = [[0.025, 0.656, 0.24, 0.012],[0.543, 0.656, 0.24, 0.012]]

for i in range(6):
    ax.append(fig.add_axes([x1[i],yy[i],dx[i],dy],projection = proj))
tit = ['June OLR','June Heatwave','July OLR','July Heatwave','August OLR','August Heatwave']
tit2 = ['a','b','c','d','e','f']
ti = [np.arange(-0.06,0.0600001,0.03)]
for i in range(6):
    if i in [0,2,4]:
        leftlon, rightlon, lowerlat, upperlat = (60,100,10,35)
        mf = np.arange(60,181,10)
    else:
        leftlon, rightlon, lowerlat, upperlat = (60,140,20,45)
        mf = np.arange(60,181,20)
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax[i].set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax[i].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.3,color='k')
    if i in [1,3,5]:
        ax[i].add_feature(cfeature.OCEAN.with_scale('50m'),zorder=1,color='w',lw=0)
    shp_path1 = r'/Users/fuzhenghang/Documents/python/tibetan/tibetan.shp'
    reader = Reader(shp_path1)
    tpfeat=cfeature.ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), edgecolor='grey', facecolor='none')
    ax[i].add_feature(tpfeat, linestyle='--',linewidth=0.8)
    gl=ax[i].gridlines(draw_labels=True, linewidth=0.3, color='k', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator(mf)
    gl.ylocator = mticker.FixedLocator(np.arange(20,80,10))
    gl.ypadding=15
    gl.xpadding=15
    gl.top_labels    = False    
    gl.right_labels  = False
    gl.bottom_labels  = False
    if i ==4 or i==5:
        gl.bottom_labels  = True
    levels = np.arange(-0.064,0.064001,0.008)
    if i <=1:
        cb=ax[i].contourf(lo[i],la[i],-1*pattern6[i], levels=levels,cmap=cm[i] ,transform=ccrs.PlateCarree(),extend='both',zorder=0)
    elif i<=3:
        cb=ax[i].contourf(lo[i],la[i],-1*pattern7[i-2], levels=levels,cmap=cm[i] ,transform=ccrs.PlateCarree(),extend='both',zorder=0)
    elif i <=5:
        cb=ax[i].contourf(lo[i],la[i],-1*pattern8[i-4], levels=levels,cmap=cm[i] ,transform=ccrs.PlateCarree(),extend='both',zorder=0)
    if i == 2:   
        position1=fig.add_axes(loc[0])#位置[左,下,长度,宽度]
        cbar=plt.colorbar(cb,cax=position1,orientation='horizontal',ticks=ti[0],
                         aspect=20,shrink=0.2,pad=0.06)#方向 
        cbar.ax.tick_params(length=1.8,width=0.4,pad=1.5)
    if i == 5:   
        position1=fig.add_axes(loc[1])#位置[左,下,长度,宽度]
        cbar=plt.colorbar(cb,cax=position1,orientation='horizontal',ticks=ti[0],
                         aspect=20,shrink=0.2,pad=0.06)#方向 
        cbar.ax.tick_params(length=1.8,width=0.4,pad=1.5)
    if i in [0,2,4]:
        ax[i].text(61,36,tit[i],fontsize=7)
        ax[i].text(56,36,tit2[i],fontweight='bold',fontsize=9)
    else:
        ax[i].text(61,46,tit[i],fontsize=7)
        ax[i].text(56,46,tit2[i],fontweight='bold',fontsize=9)
ax[1].text(113,46,'SCF = 23.6%, R = 0.83**',fontsize=7)
ax[3].text(113,46,'SCF = 22.3%, R = 0.89**',fontsize=7)
ax[5].text(113,46,'SCF = 24.7%, R = 0.77**',fontsize=7)
      
y1 = np.array(pcs6.get('right')[:,mode])
y2 = np.array(pcs7.get('right')[:,mode])
y3 = np.array(pcs8.get('right')[:,mode])
y4 = np.array(pcs6.get('left')[:,mode])
y5 = np.array(pcs7.get('left')[:,mode])
y6 = np.array(pcs8.get('left')[:,mode])
const1,p1 = pearsonr(y1, y4)
const2,p2 = pearsonr(y2, y5)
const3,p3 = pearsonr(y3, y6)
print(const1,const2,const3,p1,p2,p3)
