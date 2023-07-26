#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 10:58:54 2023

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
mpl.rcParams["font.size"] = 6
mpl.rcParams["axes.linewidth"] = 0.4
plt.rcParams['ytick.direction'] = 'out'
p = np.array([-0.4458979259977168, -0.3754201005607775, 0.22287066773855074, 0.34533658088718244, 1.1704240131977526, 0.986223339516427, 0.5701541181616033, -0.15149413521872804, -2.269020627183887, 1.2315705623446471, -0.05102135318383557, 0.36060136943032983, -1.2133492354338462, 0.6331102735206877, -1.4353942079486035, 2.5285921065108594, 0.29224660607772346, 0.16086850932042582, -0.5049127940486131, -0.4034543278170713, -0.7504493434598971, -0.4274946761581417, 0.2922067016589964, -1.8842874557549079, 0.8641457341229953, -1.0589782114000585, -1.0819997683730271, 0.9148659631829771, -0.03718458954388571, 0.1297318677187449, -1.3188473518411354, 1.7210289190888621, 0.8488284985215876, 0.056817439941308424, 0.6968063802635217, -1.2407427034043168, -0.5867595174387706, 0.06082986930696566, 0.04635906212011417, -0.19968995009210647, -0.2563642156099938, 0.1399648019923955, -1.0135454092003504, 2.4327245150450127])
h = np.array([0.09448453696533735, 0.050085415364903775, 0.6036041671880024, 0.009687046654238819, 0.9680626476063686, 0.6150070138161418, 0.3157553912775852, 0.4261713522246078, -2.654004943504871, 0.7269828377339608, 0.08966388834110811, 0.6561355469594375, -0.6403013939492661, 0.07122620485071847, -1.0891394606929887, 1.5434056507957736, -0.14450364301356203, -0.17230219332992722, 0.5173525660479271, -0.509873289731116, -0.6212133374800697, -0.33267909753767483, 0.18228654761364846, -1.0999432891641654, 0.4347044513436416, -0.721067530742988, -0.6955407347360891, 1.0716997872513254, -0.3858285079557736, -0.5437636800827149, -0.9388260137671296, 0.7275696155433872, 0.6091148958260554, -0.9341018693542557, 1.4291508503327417, -1.5386670866090693, -0.6141319311588014, 0.6254425742751526, 0.12557246464856228, -0.039274414778532205, -0.15253452836551815, -0.8446041057981198, -1.1085559970511905, 3.8876915961431977])

high = []
low = []
th=0.6
for i in range(44):
    if p[i]>th and h[i]>th:
        high.append(i)
        print(i+1979)
    elif p[i]<-th and h[i]<-th:
        low.append(i)
        print(low,i+1979)

d1 = xr.open_dataset('/Users/fuzhenghang/Documents/ERA5/temp_geo_5monthly9_1979_2022.nc')
lon1 = d1.variables['longitude'][:]
lat1 = d1.variables['latitude'][:]


d3 = xr.open_dataset(r'/Users/fuzhenghang/Documents/ERA5/cru/cru_1901-20221deg.nc',use_cftime=True)
time = d3['time'][:]
lon = d3.variables['lon'][:]
lat = d3.variables['lat'][:]
pre = d3['pre'][:,:,:][(time.dt.month>=7)&(time.dt.month<=8)&(time.dt.year>=1979)&(time.dt.year<=2022)]
pre = pre.groupby('time.year').mean(dim='time')
print(d3.variables.keys)
print(pre)
for i in range(181):
    for j in range(360):
        if not (np.isnan(pre[0,i,j])):
            scipy.signal.detrend(pre[:,i,j], axis=0, type='linear', bp=0, overwrite_data=True)
#scipy.signal.detrend(pre, axis=0, type='linear', bp=0, overwrite_data=True)
hw = np.load("/Users/fuzhenghang/Documents/python/Pakistan/npydata/R_hw_days_1979-2022_monthly_anomaly.npy")
for i in range(44):
    hw[i]=sum(hw[i*3+1:i*3+3])/2
scipy.signal.detrend(hw[0:44], axis=0, type='linear', bp=0, overwrite_data=True)



hwh = np.zeros((len(high),181,360))
hwl = np.zeros((len(low),181,360))
preh = np.zeros((len(high),181,360))
prel = np.zeros((len(low),181,360))
r=0

for i in high:
    hwh[r] = hw[i]
    preh[r] = pre[i]
    r+=1
r=0
for i in low:
    hwl[r] = hw[i]
    prel[r] = pre[i]
    r+=1


_,phw = ttest_ind(hwh,hwl,equal_var=False)
_,ppre = ttest_ind(preh,prel,equal_var=False)


hwh = np.mean(hwh,axis=0)
preh = np.mean(preh,axis=0)

hwl = np.mean(hwl,axis=0)
prel = np.mean(prel,axis=0)
# In[1]
print(hwl.shape)
proj = ccrs.PlateCarree()  #中国为左
fig = plt.figure(figsize=(12,6),dpi=1000)
ax=[]
x1 = [0,0]
yy = [1,0.745]
dx = 0.35
dy = 0.22
loc = [[0.303, 1.021, 0.007, 0.18],[0.303, 0.767, 0.007, 0.18]]
cm = [cmaps.MPL_BrBG,newcmap3,newcmap2,newcmap2]
for i in range(2):
    ax.append(fig.add_axes([x1[i],yy[i],dx,dy],projection = proj))
data = [preh-prel,hwh-hwl]
datap=[ppre,phw]

"""
i=1
ax.append(fig.add_axes([x1[i],yy[i],dx,dy]))
"""
tit = ['(a) Diff JA Precip','(b) Diff JA Heatwave']
ti = [np.arange(-120,120.00001,60),np.arange(-6,6.00001,3),np.arange(-6,6.00001,3),np.arange(-2,2.00001,1)]
for i in range(2):
    if i<=1:
        leftlon, rightlon, lowerlat, upperlat = (60,160,5,50)
    else:
        leftlon, rightlon, lowerlat, upperlat = (30,160,20,60)
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax[i].set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax[i].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.3,color='k',zorder=2)
    if i== 1:
        ax[i].add_feature(cfeature.OCEAN.with_scale('50m'),zorder=1,color='w',lw=0)
    shp_path1 = r'/Users/fuzhenghang/Documents/python/tibetan/tibetan.shp'
    reader = Reader(shp_path1)
    tpfeat=cfeature.ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), edgecolor='grey', facecolor='none',zorder=2)
    ax[i].add_feature(tpfeat, linewidth=0.4)
    gl=ax[i].gridlines(draw_labels=True, linewidth=0.3, color='k', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator(np.arange(0,181,20))
    gl.ylocator = mticker.FixedLocator(np.arange(10,80,10))
    gl.ypadding=15
    gl.xpadding=15
    gl.top_labels    = False    
    gl.right_labels  = False
    gl.bottom_labels  = True
    if  i ==0:
        gl.bottom_labels  = False
  
    levels = [np.arange(-120,120.01,20),np.arange(-7,7.01,1),np.arange(-6,6.01,1),np.arange(-3,3.01,0.5)]
    if i == 0:
        cb1=ax[i].contourf(lon,lat,data[i], levels=levels[i],cmap=cm[i] ,transform=ccrs.PlateCarree(),extend='both',zorder=0)
    else:
        cb1=ax[i].contourf(lon1,lat1,data[i], levels=levels[i],cmap=cm[i] ,transform=ccrs.PlateCarree(),extend='both',zorder=0)
    position2=fig.add_axes(loc[i])#位置[左,下,长度,宽度]
    cbar1=plt.colorbar(cb1,cax=position2,orientation='vertical',ticks=ti[i],
                     aspect=20,shrink=0.2,pad=0.06)#方向 
    cbar1.ax.tick_params(length=1.8,width=0.4,pad=1.5)
    
    if i ==1:
        ax[i].text(60,51,tit[i],fontweight='bold',fontsize=6)
        for ii in range(40,86):
            for jj in range(60,160):
                if datap[i][ii,jj]<=0.05:
                    ax[i].text(jj,90-ii,'.')
    else:
        ax[i].text(60,51,tit[i],fontweight='bold',fontsize=6)
        for ii in range(95,140):
            for jj in range(60,160):
                if datap[i][ii,jj]<=0.05:
                    ax[i].text(jj,ii-90,'.')
               
        






