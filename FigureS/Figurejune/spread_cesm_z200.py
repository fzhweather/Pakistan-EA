#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 10:45:56 2023

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

preall = np.zeros((1140))
hwall = np.zeros((1140))
tpall = np.zeros((1140,181,360))
tasall = np.zeros((1140,181,360))
pb,pa=scipy.signal.butter(3,2/11,'highpass')
d3 = xr.open_dataset("/Users/fuzhenghang/Documents/ERA5/amip/geo.ensmean.amip.1880-2014_1deg.nc")
#print(d3.variables.keys())
time1 = d3['time'][:]   
tmean = d3['Z3'][(time1.dt.month>=6)&(time1.dt.month<=6)&(time1.dt.year>=1901)&(time1.dt.year<=2020)][:,0,:,:]
#print(tmean.shape)
d1 = xr.open_dataset(r'/Users/fuzhenghang/Documents/ERA5/amip/olr.ensmean.amip.1880-2014_1deg.nc',use_cftime=True)
#print(d1.variables.keys())
time = d1['time'][:]
#tp = d1['PRECC'][(time.dt.month>=7)&(time.dt.month<=8)&(time.dt.year>=1901)&(time.dt.year<=2020)][:,:,:]
tpmean = d1['FLNT'][(time.dt.month>=6)&(time.dt.month<=6)&(time.dt.year>=1901)&(time.dt.year<=2020)][:,:,:]

for mm in range(10):
    d2 = xr.open_dataset("/Users/fuzhenghang/Documents/ERA5/amip/z200.ens_deg.nc")
    #print(d2.variables.keys())
    lons = d2['lon']
    lats = d2['lat']
    #print(lons)
    time1 = d2['time'][:]   
    tas = d2['Z3'][(time1.dt.month>=6)&(time1.dt.month<=6)&(time1.dt.year>=1901)&(time1.dt.year<=2020)][:,mm,:,:]
    print(tas.shape)
    tas = tas - tmean
    tas = tas.groupby('time.year').mean(dim='time')
    #print(tas[10,10,10])
    pre = np.zeros((114))
    hw = np.zeros((114))
    d1 = xr.open_dataset(r'/Users/fuzhenghang/Documents/ERA5/amip/olr.ens_1deg.nc',use_cftime=True)
    #print(d1.variables.keys())
    time = d1['time'][:]
    #tp = d1['PRECC'][(time.dt.month>=7)&(time.dt.month<=8)&(time.dt.year>=1901)&(time.dt.year<=2020)][:,:,:]
    tp = d1['FLNT'][(time.dt.month>=6)&(time.dt.month<=6)&(time.dt.year>=1901)&(time.dt.year<=2020)][:,:,:][:,mm,:,:]
    tp = tp - tpmean
    tp = -tp.groupby('time.year').mean(dim='time')
    #print(tp[10,10,10])
    la = 114
    lo = 67
    lonw=12
    law=10

    for i in range(law):
        for j in range(lonw):
            for k in range(114):
                pre[k]+=tp[k,i+la,j+lo]
    pre/=120
    for i in range(8):
        for j in range(18):
            for k in range(114):   
                hw[k]+=tas[k,i+116,j+105]
    hw/=144           
  
    preall[mm*114:(mm+1)*114] = scipy.signal.filtfilt(pb,pa,pre, axis=0)
    hwall[mm*114:(mm+1)*114] = scipy.signal.filtfilt(pb,pa,hw, axis=0)
    tasall[mm*114:(mm+1)*114,:,:] = scipy.signal.filtfilt(pb,pa,tas, axis=0)
    tpall[mm*114:(mm+1)*114,:,:] = scipy.signal.filtfilt(pb,pa,tp, axis=0)
    """
    preall[mm*114:(mm+1)*114] = pre
    hwall[mm*114:(mm+1)*114] = hw
    tasall[mm*114:(mm+1)*114,:,:] = tas
    tpall[mm*114:(mm+1)*114,:,:] = tp
"""

# In[1]
def f_1(x, A, B):
 return A * x + B
from scipy import optimize
import cmaps
from matplotlib.colors import ListedColormap
cmap=plt.get_cmap(cmaps.ewdifft)
newcolors=cmap(np.linspace(0, 1, 16))
newcmap2 = ListedColormap(newcolors[2:14])

reg = np.zeros((181,360))
R = np.zeros((181,360))
for i in range(181):
    for j in range(360):
        reg[i,j], B1 = optimize.curve_fit(f_1, preall, tasall[:,i,j])[0]
        
        
        y = tas[:,i,j]
        mean = np.mean(y)  # 1.y mean
        ss_tot = np.sum((y - mean) ** 2)  # 2.total sum of squares
        ss_res = np.sum((y - f_1(pre, reg[i,j],B1)) ** 2)  # 3.residual sum of squares
        R[i,j] = 1138* ((ss_tot-ss_res) / ss_res) 
        

np.save("/Users/fuzhenghang/Documents/大四上/热浪/中间数据/june_amipsprd_Reg_cesm10-z200.npy",reg)
np.save("/Users/fuzhenghang/Documents/大四上/热浪/中间数据/june_amipsprd_R_cesm10-z200.npy",R)

reg = np.load("/Users/fuzhenghang/Documents/大四上/热浪/中间数据/june_amipsprd_Reg_cesm10-z200.npy")
R = np.load("/Users/fuzhenghang/Documents/大四上/热浪/中间数据/june_amipsprd_R_cesm10-z200.npy")
print(reg)
proj = ccrs.PlateCarree()  #中国为左
leftlon, rightlon, lowerlat, upperlat =(10,360,0,65)
lon_formatter = cticker.LongitudeFormatter()
lat_formatter = cticker.LatitudeFormatter()
fig = plt.figure(figsize=(10,6),dpi=600)

levels = [np.arange(-0.012,0.012001,0.002)]
cm = [newcmap2,cmaps.MPL_BrBG,cmaps.BlueDarkRed18]#cmaps.GMT_no_green
ax=[]
x1 = [0]
yy = [1]
dx = 0.8
dy = 0.4
loc = [[0.65, 1.02, 0.015, 0.35]]
for i in range(1):
    ax.append(fig.add_axes([x1[i],yy[i],dx,dy],projection = proj))

ti = ['CESM2-CAM6: H200 Ensemble Spread']
lab=[np.arange(-0.012,0.012001,0.006)]
#la = ['dagpm/(W)']
for i in range(1):
    ax[i].set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax[i].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.6,color='k')
    
    shp_path1 = r'/Users/fuzhenghang/Documents/python/tibetan/tibetan.shp'
    reader = Reader(shp_path1)
    tpfeat=cfeature.ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), edgecolor='grey', facecolor='none')
    ax[i].add_feature(tpfeat, linewidth=1.2)
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

