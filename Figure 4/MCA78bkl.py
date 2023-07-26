#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 12:34:27 2023

@author: fuzhenghang
"""

# In[0]
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
from netCDF4 import Dataset
import cmaps
from cartopy.util import add_cyclic_point
from cartopy.io.shapereader import Reader
from scipy.stats import pearsonr
import pandas as pd
import scipy
from matplotlib.colors import ListedColormap
cmap=plt.get_cmap(cmaps.BlueDarkRed18)
newcolors=cmap(np.linspace(0, 1, 18))
newcmap3 = ListedColormap(newcolors[1:17])

mpl.rcParams["font.family"] = 'cm' 
mpl.rcParams["mathtext.fontset"] = 'cm'  # 数学文字字体
mpl.rcParams["font.size"] = 6
mpl.rcParams["axes.linewidth"] = 0.4
plt.rcParams['ytick.direction'] = 'out'


d1 =  Dataset(r'/Users/fuzhenghang/Documents/ERA5/ncep_monthly_olr_1deg.nc')
print(d1.variables.keys())
lon = d1.variables['lon'][:]
lat = d1.variables['lat'][:]

print(lon)
print(lat)
time = d1['T'][:]
#print(time)
pre = d1['olr'][:]
#print(pre.shape)
mon = [6,7,8,9,10,11,12]
yea = [1974,1974,1974,1974,1974,1974,1974]
for j in range(48):
    for i in range(12):
        mon.append(i+1)
        yea.append(1975+j)
olr = np.zeros((41,181,360))
for i in range(41):
    for m in range(12):
        if mon[55+i*12+m] in [7,8]:
            olr[i]+=pre[55+i*12+m]
olr /= 2
scipy.signal.detrend(olr, axis=0, type='linear', bp=0, overwrite_data=True)
#print(pre.shape)

olr = olr[:,110:126,60:81]

lats = d1['lat'][110:126]
lons = d1['lon'][60:81]
times = np.array([1979+i for i in range(41)])
pre = xr.DataArray(olr,coords=[times, lats, lons], dims=['time', 'lat','lon'])

# In[1]
d2 =xr.open_dataset(r'/Users/fuzhenghang/Documents/大四上/热浪/Complete_TMAX_Daily_LatLong1_1970-2010.nc',use_cftime=True)
hw = np.load("/Users/fuzhenghang/Documents/大四上/热浪/中间数据/R_Best_hw_days_1979-2019_monthly.npy")
lats1 = d2['latitude'][110:136]
lons1 = d2['longitude'][240:321]
for i in range(41):
    hw[i]=sum(hw[i*3+1:i*3+3])/2
scipy.signal.detrend(hw[0:41], axis=0, type='linear', bp=0, overwrite_data=True)
hw = hw[0:41,90+20:90+46,60+180:141+180]
hw = xr.DataArray(hw,coords=[times, lats1, lons1], dims=['time', 'lat','lon'])
"""
for y in range(44):
    for i in range(10,26):
        for j in range(20):
            hw[y,i,j]=0

"""
cm = [cmaps.MPL_BrBG_r,newcmap3]
mca = xMCA(pre, hw)                  # MCA of field A and B
mca.solve(complexify=False)            # True for complex MCA
 
eigenvalues = mca.singular_values()     # singular vales
pcs = mca.pcs()                         # expansion coefficient (PCs)
eofs = mca.eofs()                       # spatial patterns (EOFs)
expvar = mca.explained_variance()
#print(eofs.get('left'))
mode=0
pattern = [np.array(eofs.get('left')[:,:,mode]),np.array(eofs.get('right')[:,:,mode])]
#print(list(-1*np.array(pcs.get('right')[:,mode])))
#print(list(np.array(eofs.get('right')[:,:,0])))
print(expvar[0:2])
#print(eigenvalues[0:3])
# In[2]
lo = [lons,lons1]
la = [lats,lats1]
proj = ccrs.PlateCarree()  #中国为左
fig = plt.figure(figsize=(12,6),dpi=1000)
ax=[]
x1 = [0,0,0]
yy = [1,0.77,0.55]
dx = 0.3
dy = 0.19
loc = [[0.307, 1.02, 0.007, 0.15],[0.305, 1.02, 0.007, 0.15]]

for i in range(1):
    ax.append(fig.add_axes([x1[i],yy[i],dx,dy],projection = proj))
i=1
ax.append(fig.add_axes([x1[i],yy[i],dx,dy]))

tit = ['A   JA OLR (NOAA) & Heatwave (Berkeley)','B ']
ti = [np.arange(-0.06,0.0600001,0.03)]
for i in range(1):
    leftlon, rightlon, lowerlat, upperlat = (60,140,20,45)
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax[i].set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax[i].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.3,color='k',zorder=2)
    if i== 0:
        ax[i].add_feature(cfeature.OCEAN.with_scale('50m'),zorder=1,color='w',lw=0)
    shp_path1 = r'/Users/fuzhenghang/Documents/python/tibetan/tibetan.shp'
    reader = Reader(shp_path1)
    tpfeat=cfeature.ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), edgecolor='grey', facecolor='none',zorder=2)
    ax[i].add_feature(tpfeat, linewidth=0.4)
    gl=ax[i].gridlines(draw_labels=True, linewidth=0.3, color='k', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator(np.arange(60,181,20))
    gl.ylocator = mticker.FixedLocator(np.arange(20,80,10))
    gl.ypadding=15
    gl.xpadding=15
    gl.top_labels    = False    
    gl.right_labels  = False
    gl.bottom_labels  = True
    if i ==0:
        gl.bottom_labels  = True
    levels = np.arange(-0.064,0.064001,0.008)
    CS=ax[i].contour(lo[i],la[i],-1*pattern[i],[-0.08,-0.06,-0.04,-0.02,0,0.02,0.04,0.06,0.08], cmap=cm[i] ,linewidths = 1.2,transform=ccrs.PlateCarree(),extend='both',zorder=1)
    ax[i].clabel(CS, inline=1, fontsize=5)
    cb1=ax[i].contourf(lo[i+1],la[i+1],-1*pattern[i+1], levels=levels,cmap=cm[i+1] ,transform=ccrs.PlateCarree(),extend='both',zorder=0)
    position2=fig.add_axes(loc[i+1])#位置[左,下,长度,宽度]
    cbar1=plt.colorbar(cb1,cax=position2,orientation='vertical',ticks=ti[0],
                     aspect=20,shrink=0.2,pad=0.06)#方向 
    cbar1.ax.tick_params(length=1.8,width=0.4,pad=1.5)
    ax[i].text(57,46,tit[i],fontweight='bold',fontsize=7)
x = [1979+i for i in range(41)]
y1 = np.array(pcs.get('left')[:,mode])
y2 = np.array(pcs.get('right')[:,mode])

m1 = np.mean(y1)
m2 = np.mean(y2)

sd1 = np.std(y1)
sd2 = np.std(y2)

y1 = (y1-m1)/sd1
y2 = (y2-m2)/sd2

const1,p1 = pearsonr(y1, y2)
print(const1,p1)
print(list(y1))
print(list(y2))
lab = ['SCF = 43.5%, R = 0.86**']
ax[1].text(1977.5,3.25,tit[1],fontweight='bold',fontsize=7)
ax[1].text(2009,3.2,lab[mode],fontsize=6)
ax[1].yaxis.tick_right()
ax[1].yaxis.set_label_position("right") 
ax[1].plot(x,-1*y1,'-o',color='royalblue',markersize=2.5,lw = 1.2,label='JA OLR')
ax[1].plot(x,-1*y2,'-*',color='tomato',markersize=4,lw = 1.2,label='JA Heatwave')
ax[1].set_xlim(1979,2022)
ax[1].set_ylim(-3,3)
ax[1].set_ylabel('Normalized Value',labelpad=1)
ax[1].tick_params(length=2,width=0.4,pad=1.5)
ax[1].axhline(y=0,  linestyle='-',linewidth = 0.5,color='black',alpha=1,zorder=0)
y3 = np.zeros((46))-0.6
y4 = y3+1.2
x1 = [1978+i for i in range(46)]
ax[1].fill_between(x1, y3, y4, color='gray',alpha = 0.2,linewidths=0)
#ax[1].axhline(y=-1,  linestyle='--',linewidth = 0.35,color='black',alpha=1,zorder=0)
ax[1].legend(frameon=False,loc='lower right',ncol=2)
ax[1].text(2011.5,2.5,'OLR Corr.: 0.93**',fontweight='bold',c='royalblue')
ax[1].text(1996.5,2.5,'Heatwave Corr.: 0.94**',fontweight='bold',c='tomato')
#for i in range(44):
    #if abs(y1[i])>0.8 and abs(y2[i])>0.8:
        #ax[1].axvline(x=1979+i,  linestyle='-',linewidth = 2,color='black',alpha=0.1,zorder=0)
lon = np.empty(4)
lat = np.empty(4)
lon[0],lat[0] = 60, 20  # lower left (ll)
lon[1],lat[1] = 80, 20  # lower right (lr)
lon[2],lat[2] = 80, 35  # upper right (ur)
lon[3],lat[3] = 60, 35  # upper left (ul)
x, y =  lon, lat
xy = list(zip(x,y))
poly = plt.Polygon(xy,edgecolor='k',linestyle='--',fc="none", lw=1.25, alpha=0.8,transform=ccrs.PlateCarree(),zorder=7)
ax[0].add_patch(poly)
