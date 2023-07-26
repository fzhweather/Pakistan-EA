#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 21:01:01 2023

@author: fuzhenghang
"""

# In[0]
import xarray as xr
import numpy as np
from scipy.stats.mstats import ttest_ind
import matplotlib as mpl
import datetime as dt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
from scipy import optimize
from cartopy.io.shapereader import Reader
import cmaps
import matplotlib.ticker as mticker
from windspharm.standard import VectorWind
from windspharm.tools import prep_data, recover_data, order_latdim
import scipy
from matplotlib.colors import ListedColormap
def f_1(x, A, B):
 return A * x + B
def ww(t,P):
    theta=(t)*((1000/P)**0.286)
    return theta
cmap=plt.get_cmap(cmaps.ewdifft)
newcolors=cmap(np.linspace(0, 1, 16))
newcmap2 = ListedColormap(newcolors[2:14])
mpl.rcParams["mathtext.fontset"] = 'cm'  # 数学文字字体
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.linewidth"] = 1.2

pb,pa=scipy.signal.butter(3,2/11,'highpass')
d2 = xr.open_dataset("/Users/fuzhenghang/Documents/ERA5/amip/geo.ensmean.amip.1880-2014_1deg.nc")
#print(d2.variables.keys())
lons = d2['lon']
lats = d2['lat']
#print(lons)
time1 = d2['time'][:]   
tas = d2['Z3'][(time1.dt.month>=6)&(time1.dt.month<=6)&(time1.dt.year>=1979)&(time1.dt.year<=2020)][:,0,:,:]
tas = tas.groupby('time.year').mean(dim='time')
print(tas.shape)
pre = np.zeros((36))
hw = np.zeros((36))
d1 = xr.open_dataset(r'/Users/fuzhenghang/Documents/ERA5/amip/olr.ensmean.amip.1880-2014_1deg.nc',use_cftime=True)
#print(d1.variables.keys())
time = d1['time'][:]
#tp = d1['PRECC'][(time.dt.month>=7)&(time.dt.month<=8)&(time.dt.year>=1901)&(time.dt.year<=2020)][:,:,:]
tp = d1['FLNT'][(time.dt.month>=6)&(time.dt.month<=6)&(time.dt.year>=1979)&(time.dt.year<=2020)][:,:,:]
tp = -tp.groupby('time.year').mean(dim='time')
#print(tp[0])

la = 114
lo = 67
lonw=12
law=10

for i in range(law):
    for j in range(lonw):
        for k in range(36):
            pre[k]+=tp[k,i+la,j+lo]
pre /= 120    
             
tas = scipy.signal.filtfilt(pb,pa,tas, axis=0) 
pre = scipy.signal.filtfilt(pb,pa,pre, axis=0)

reg = np.zeros((181,360))
R = np.zeros((181,360))
for i in range(181):
    for j in range(360):
        reg[i,j], B1 = optimize.curve_fit(f_1, pre, tas[:,i,j])[0]
        
        
        y = tas[:,i,j]
        mean = np.mean(y)  # 1.y mean
        ss_tot = np.sum((y - mean) ** 2)  # 2.total sum of squares
        ss_res = np.sum((y - f_1(pre, reg[i,j],B1)) ** 2)  # 3.residual sum of squares
        R[i,j] = 34* ((ss_tot-ss_res) / ss_res) 
        

np.save("/Users/fuzhenghang/Documents/大四上/热浪/中间数据/june_amip_Reg_cesm-z200-36yr.npy",reg)
np.save("/Users/fuzhenghang/Documents/大四上/热浪/中间数据/june_amip_R_cesm-z200-36yr.npy",R)

reg = np.load("/Users/fuzhenghang/Documents/大四上/热浪/中间数据/june_amip_Reg_cesm-z200-36yr.npy")
R = np.load("/Users/fuzhenghang/Documents/大四上/热浪/中间数据/june_amip_R_cesm-z200-36yr.npy")
print(reg)
proj = ccrs.PlateCarree()  #中国为左
leftlon, rightlon, lowerlat, upperlat =(0,300,5,65)
lon_formatter = cticker.LongitudeFormatter()
lat_formatter = cticker.LatitudeFormatter()
fig = plt.figure(figsize=(10,6),dpi=600)

levels = [np.arange(-1.8,1.8001,0.2)]
cm = [newcmap2,cmaps.MPL_BrBG,cmaps.BlueDarkRed18]#cmaps.GMT_no_green
ax=[]
x1 = [0]
yy = [1]
dx = 0.8
dy = 0.4
loc = [[0.65, 1.02, 0.015, 0.35]]
for i in range(1):
    ax.append(fig.add_axes([x1[i],yy[i],dx,dy],projection = proj))

ti = ['CESM2-CAM6: H200']
lab=[np.arange(-0.018,0.018001,0.006)]
#la = ['dagpm/(W)']
for i in range(1):
    ax[i].set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax[i].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.6,color='k')
    
    shp_path1 = r'/Users/fuzhenghang/Documents/python/tibetan/tibetan.shp'
    reader = Reader(shp_path1)
    tpfeat=cfeature.ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), edgecolor='grey', facecolor='none')
    ax[i].add_feature(tpfeat, linewidth=0.6)
    gl=ax[i].gridlines(draw_labels=True, linewidth=0.5, color='k', alpha=0.6, linestyle='--')
    gl.xlocator = mticker.FixedLocator(np.arange(-180,181,30))
    gl.ylocator = mticker.FixedLocator(np.arange(0,80,20))
    gl.ypadding=15
    gl.xpadding=15
    gl.top_labels    = False    
    gl.right_labels  = False
    if i in [0,1]:
        gl.bottom_labels  = False
    if i in [1]:
        gl.left_labels  = False
    
    cb=ax[i].contourf(lons,lats,reg, levels=levels[i],cmap=cm[i] ,transform=ccrs.PlateCarree(),extend='both',zorder=0)
    position1=fig.add_axes(loc[i])#位置[左,下,长度,宽度]
    cbar=plt.colorbar(cb,cax=position1,orientation='vertical',ticks=lab[i],
                      aspect=20,shrink=0.2,pad=0.06)#方向 
    #cbar.set_label(la[i],labelpad=1)
    cbar.ax.tick_params(length=1.8,width=0.4,pad=1.5)
    #print(days_map)
    ax[i].text(40,66.6,ti[i],fontweight='bold',fontsize=12)
    #print(corrp)
    for l1 in range(5,65,2):
        for l2 in range(40,160,2):
            if R[90+l1,l2]>3.926:
                ax[i].text(l2,l1,'.',fontsize=12)

