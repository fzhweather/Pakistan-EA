#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 15:38:28 2023

@author: fuzhenghang
"""

# In[0]
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

mpl.rcParams["font.family"] = 'cm' 
mpl.rcParams["mathtext.fontset"] = 'cm'  # 数学文字字体
mpl.rcParams["font.size"] = 7
mpl.rcParams["axes.linewidth"] = 0.6
plt.rcParams['ytick.direction'] = 'out'


pb,pa=scipy.signal.butter(3,2/11,'highpass')
d2 = xr.open_dataset("/Users/fuzhenghang/Documents/ERA5/amip/ts.ensmean.amip.1880-2014_1deg.nc")
#print(d2.variables.keys())
lons = d2['lon']
lats = d2['lat']
#print(lons)
time1 = d2['time'][:]   
tas = d2['TS'][(time1.dt.month>=7)&(time1.dt.month<=8)&(time1.dt.year>=1979)&(time1.dt.year<=2020)]
tas = tas.groupby('time.year').mean(dim='time')

pre = np.zeros((114))
hw = np.zeros((36))
d1 = xr.open_dataset(r'/Users/fuzhenghang/Documents/ERA5/amip/olr.ensmean.amip.1880-2014_1deg.nc',use_cftime=True)
#print(d1.variables.keys())
time = d1['time'][:]
#tp = d1['PRECC'][(time.dt.month>=7)&(time.dt.month<=8)&(time.dt.year>=1901)&(time.dt.year<=2020)][:,:,:]
tp = d1['FLNT'][(time.dt.month>=7)&(time.dt.month<=8)&(time.dt.year>=1901)&(time.dt.year<=2020)][:,:,:]
tp = -tp.groupby('time.year').mean(dim='time')

la = 114
lo = 67
lonw=12
law=10

for i in range(law):
    for j in range(lonw):
        for k in range(114):
            pre[k]+=tp[k,i+la,j+lo]
pre/=120
pre1 = pre[78:]       
for i in range(8):
    for j in range(18):
        for k in range(36):   
            hw[k]+=tas[k,i+116,j+105]
hw/=144             
   
pre = scipy.signal.filtfilt(pb,pa,pre1, axis=0)
hw = scipy.signal.filtfilt(pb,pa,hw, axis=0)
tas = scipy.signal.filtfilt(pb,pa,tas, axis=0)
tp = scipy.signal.filtfilt(pb,pa,tp, axis=0)

# In[1]
corr = np.zeros((181,360))
corrp = np.zeros((181,360))
for i in range(181):
    for j in range(360):
        corr[i,j],corrp[i,j] = pearsonr(pre, tas[:,i,j])#days_map
    
#print(list(pre))
#print(list(hw))
# In[2]

proj = ccrs.PlateCarree()  #中国为左
leftlon, rightlon, lowerlat, upperlat = (60,140,15,45)
lon_formatter = cticker.LongitudeFormatter()
lat_formatter = cticker.LatitudeFormatter()
fig = plt.figure(figsize=(10,4),dpi=600)

ax=[]
x1 = [0,0]
yy = [1,0.6]
dx = [0.45,0.45]
dy = [0.6,0.4]
loc = [[0.46, 1.096, 0.013, 0.4]]
for i in range(2):
    if i == 0:
        ax.append(fig.add_axes([x1[i],yy[i],dx[i],dy[i]],projection = proj))
    else:
        ax.append(fig.add_axes([x1[i],yy[i],dx[i],dy[i]]))

i = 0
ax[i].set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
ax[i].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.6,color='k')
ax[i].add_feature(cfeature.OCEAN.with_scale('50m'),zorder=10,color='w',lw=0)
shp_path1 = r'/Users/fuzhenghang/Documents/python/tibetan/tibetan.shp'
reader = Reader(shp_path1)
tpfeat=cfeature.ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), edgecolor='grey', facecolor='none')
ax[i].add_feature(tpfeat, linewidth=0.4)
gl=ax[i].gridlines(draw_labels=True, linewidth=0.2, color='k', alpha=0.5, linestyle='--')
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,20))
gl.ylocator = mticker.FixedLocator(np.arange(0,80,10))
gl.ypadding=15
gl.xpadding=15
gl.top_labels    = False    
gl.right_labels  = False
gl.bottom_labels  = True
corr1, cycle_lon =add_cyclic_point(corr, coord=lons)
corrp1, cycle_lon =add_cyclic_point(corrp, coord=lons)
levels = np.arange(-0.7,0.7001,0.1)
cb=ax[i].contourf(cycle_lon,lats,corr1, levels=levels,cmap=cmaps.hotcold_18lev ,transform=ccrs.PlateCarree(),extend='both',zorder=0)
#print(days_map)
position1=fig.add_axes(loc[i])#位置[左,下,长度,宽度]
cbar=plt.colorbar(cb,cax=position1,orientation='vertical',ticks=np.arange(-0.6,0.6001,0.3),
                 aspect=20,shrink=0.2,pad=0.06)#方向 
cbar.set_label('Corr.',labelpad=1)
cbar.ax.tick_params(length=1.8,width=0.4,pad=1.5)

for l1 in range(105,135):
    for l2 in range(60,140):
        if corrp1[l1,l2]<0.05:
            ax[i].text(l2,l1-90,'.',fontsize=8)
la = 171-la
a1=lo
b1=lo+lonw-1
c1=90-la-law+1
d1=90-la

lon = np.empty(4)
lat = np.empty(4)
lon[0],lat[0] = a1, c1  # lower left (ll)
lon[1],lat[1] = b1, c1  # lower right (lr)
lon[2],lat[2] = b1, d1  # upper right (ur)
lon[3],lat[3] = a1, d1  # upper left (ul)
x, y =  lon, lat
xy1 = list(zip(x,y))
poly = plt.Polygon(xy1,edgecolor='red',linestyle='--',fc="none", lw=2, alpha=1,transform=ccrs.PlateCarree(),zorder=7)
ax[i].add_patch(poly)


lon = np.empty(4)
lat = np.empty(4)
lon[0],lat[0] = 105, 27  # lower left (ll)
lon[1],lat[1] = 123, 27  # lower right (lr)
lon[2],lat[2] = 123, 33  # upper right (ur)
lon[3],lat[3] = 105, 33  # upper left (ul)
x, y =  lon, lat
xy2 = list(zip(x,y))
poly = plt.Polygon(xy2,edgecolor='blue',linestyle='--',fc="none", lw=2, alpha=1,transform=ccrs.PlateCarree(),zorder=7)
ax[i].add_patch(poly)


tit = ['(a) JA NWSA Precip. & EA Temperature','(b) ']
ax[0].text(60,46.2,tit[0],fontsize=8,fontweight='bold')
ax[0].text(125,46.2,'AMIP mean',fontsize=8,fontweight='bold')
pres = np.std(pre)
hws = np.std(hw)

prem = np.mean(pre)
hwm = np.mean(hw)

pren = (pre-prem)/pres
hwn = (hw-hwm)/hws

print(hwn[-1])
print(pren[-1])
x = [1979+i for i in range(36)]
ax[1].yaxis.tick_right()
ax[1].yaxis.set_label_position("right")
ax[1].bar(x,pren,color='lightskyblue',lw=2,label='JA NWSA Precipitation')
ax[1].plot(x,hwn,'-o',color='tomato',lw=1.75,ms=3,label='JA YR Temperature')

ax[1].set_xlim(1978,2014)
ax[1].set_ylim(-3,3)
ax[1].set_ylabel('Normalized Value',labelpad=1)
ax[1].set_yticks([-3,-2,-1,0,1,2,3])
ax[1].text(1979,3.15,tit[1],fontsize=8,fontweight='bold')

ax[1].tick_params(length=2,width=0.4,pad=1.5)
ax[1].axhline(y=0,  linestyle='-',linewidth = 0.35,color='black',alpha=1,zorder=0)
ax[1].axhline(y=1,  linestyle='dashed',linewidth = 0.35,color='black',alpha=1,zorder=0)
ax[1].axhline(y=-1,  linestyle='dashed',linewidth = 0.35,color='black',alpha=1,zorder=0)
ax[1].legend(frameon=False,loc='lower left',ncol=3)

const1,p1 = pearsonr(pre, hw)

ax[1].text(1989,2.2,'r$_1$ = %.2f**'%const1,color='tomato',fontsize=9,fontweight='bold')
print(const1,p1)

#ax[1].text(2011.5,2.5,'Year 2022: 3.99→',fontsize=7.5,fontweight='bold',c='r')
