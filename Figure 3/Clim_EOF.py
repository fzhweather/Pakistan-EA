#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 12:20:35 2023

@author: fuzhenghang
"""
# In[0]

from eofs.standard import Eof
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import cartopy.mpl.ticker as cticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr  
import numpy as np
import cmaps
import pandas as pd
import scipy
from matplotlib.colors import ListedColormap
cmap=plt.get_cmap(cmaps.MPL_BrBG)
newcolors=cmap(np.linspace(0, 8, 256))
newcmap1 = ListedColormap(newcolors[4:29])

mpl.rcParams["font.family"] = 'cm' 
mpl.rcParams["mathtext.fontset"] = 'cm'  # 数学文字字体
mpl.rcParams["font.size"] = 8
mpl.rcParams["axes.linewidth"] = 0.8
plt.rcParams['ytick.direction'] = 'out'

d1 = xr.open_dataset(r'/Users/fuzhenghang/Documents/ERA5/cru/cru_1901-20221deg.nc',use_cftime=True)
lon = d1.variables['lon'][:]
lat = d1.variables['lat'][:]
print(lat)
time = d1['time'][:]
pre6 = d1['pre'][(time.dt.month==6)][:] #mm/day
pre7 = d1['pre'][(time.dt.month==7)][:]
pre8 = d1['pre'][(time.dt.month==8)][:]
pre9 = d1['pre'][(time.dt.month==9)][:]


pre = [pre6,pre7,pre8,pre9]
cli_pre = np.zeros((4,181,360))
std_pre = np.zeros((4,181,360))
for i in range(4):
    cli_pre[i] = np.mean(pre[i],axis=0)
    std_pre[i] = np.std(pre[i],axis=0)


d1 = xr.open_dataset(r'/Users/fuzhenghang/Documents/ERA5/cru/cru_1901-20221deg.nc',use_cftime=True)
lon = d1.variables['lon'][:]
lat = d1.variables['lat'][:]
time = d1['time'][:]
pre6 = d1['pre'][(time.dt.year>=1901)&(time.dt.year<=2022)&(time.dt.month==6)][:]/30 #mm/day
pre7 = d1['pre'][(time.dt.year>=1901)&(time.dt.year<=2022)&(time.dt.month==7)][:]/31
pre8 = d1['pre'][(time.dt.year>=1901)&(time.dt.year<=2022)&(time.dt.month==8)][:]/31
pre9 = d1['pre'][(time.dt.year>=1901)&(time.dt.year<=2022)&(time.dt.month==9)][:]/30

pre6 = (pre6 - np.mean(pre6,axis=0))/np.std(pre6,axis=0)
pre7 = (pre7 - np.mean(pre7,axis=0))/np.std(pre7,axis=0)
pre8 = (pre8 - np.mean(pre8,axis=0))/np.std(pre8,axis=0)
pre9 = (pre9 - np.mean(pre9,axis=0))/np.std(pre9,axis=0)

pre = [pre6[:,95:125,60:90],pre7[:,95:125,60:90],pre8[:,95:125,60:90],pre9[:,95:125,60:90]]
pb,pa=scipy.signal.butter(3,2/11,'highpass')
for k in range(4):
   for i in range(30):
       for j in range(30):
           pre[k][:,i,j]=scipy.signal.filtfilt(pb,pa,pre[k][:,i,j])
           
        
#preall = d1['pre'][(time.dt.month<=9)&(time.dt.month>=6)][:]
#preall = preall.groupby('time.year').mean(dim='time')
#preall = (preall - np.mean(preall,axis=0))/np.std(preall,axis=0)
coslat = np.cos(np.deg2rad(np.array(lat[95:125])))
wgts = np.sqrt(coslat)[..., np.newaxis] 

pcs = np.zeros((2,4,122))
pattern = np.zeros((2,4,181,360))
var=np.zeros((2,4))
for i in range(4):
    eof = Eof(np.array(pre[i]), weights = wgts)   #进行eof分解
    u = eof.eofs(eofscaling=0, neofs=2)  # 得到空间模态U eofscaling 对得到的场进行放缩 （1为除以特征值平方根，2为乘以特征值平方根，默认为0不处理） neofs决定输出的空间模态场个数
    PC = eof.pcs(pcscaling=1, npcs=2)  # 同上 npcs决定输出的时间序列个数
    s = eof.varianceFraction(neigs=2)   # 得到前neig个模态的方差贡献
    pcs[0,i]=np.array(PC[:,0])
    pcs[1,i]=np.array(PC[:,1])
    pattern[0,i,95:125,60:90] = u[0]
    pattern[1,i,95:125,60:90] = u[1]
    var[:,i]=s# In[1]
proj = ccrs.PlateCarree(central_longitude=180)  
leftlon, rightlon, lowerlat, upperlat = (60,90,5,35)
lon_formatter = cticker.LongitudeFormatter()
lat_formatter = cticker.LatitudeFormatter()
fig = plt.figure(figsize=(12,7.5),dpi=1200)

x1 = [0,0.16,0.32,0,0.16,0.32,0.48,0,0.16,0.32,0.48]
yy = [1,1,1,0.73,0.73,0.73,0.73,0.46,0.46,0.46,0.46]
dx = 0.23
dy = 0.23
ax = []
level = [np.arange(-0.08,0.08001,0.008),np.arange(0,480.01,40)]
tick = [np.arange(-0.08,0.081,0.04),np.arange(0,500,120)]
loc = [[0.515, 1.03, 0.01, 0.18],[0.515, 0.76, 0.01, 0.18]]
co = [newcmap1,cmaps.WhiteBlue,cmaps.precip4_11lev]#cmaps.precip3_16lev_r
title = ['EOF1-Jun','EOF1-July','EOF1-Aug','clim.-Jun','clim.-Jul','clim.-Aug']
tit2 = ['a','b','c','d','e','f']
for i in range(6):
    ax.append(fig.add_axes([x1[i],yy[i],dx,dy],projection = proj))

for i in range(6):
    ax[i].set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax[i].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.6,color='black')
    ax[i].add_feature(cfeature.BORDERS.with_scale('50m'),zorder=1,color='black',lw=0.35)
    gl=ax[i].gridlines(draw_labels=True, linewidth=0.3, color='k', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator(np.arange(-180,181,10))
    gl.ylocator = mticker.FixedLocator(np.arange(0,80,10))
    gl.ypadding=50
    gl.xpadding=50
    gl.top_labels    = False    
    gl.right_labels  = False
    ax[i].text(-119,36,title[i])
    ax[i].text(-122,36,tit2[i],fontweight='bold',fontsize = 11)
    if i not in [0,3]:
        gl.left_labels  = False
    if i <3 :
        gl.bottom_labels  = False
        ax[i].text(-99,36,'{:.2f}%'.format(var[0,i]*100))
        cb=ax[i].contourf(lon,lat,pattern[0,i], levels=level[0],cmap=co[0] ,transform=ccrs.PlateCarree(),extend='both',zorder=0)

        
    elif i<=7:
        cb=ax[i].contourf(lon,lat,cli_pre[i-3], levels=level[1],cmap=co[2] ,transform=ccrs.PlateCarree(),extend='both',zorder=0)
        
    
    if i == 3:
        cbar=plt.colorbar(cb,cax=fig.add_axes(loc[1]),orientation='vertical',ticks=tick[1],aspect=20,shrink=0.2,pad=0.06)#方向
        cbar.ax.tick_params(length=1.8,width=0.4,pad=1.5)
        cbar.set_label('mm/month',labelpad=1)
    if i == 1:
        cbar=plt.colorbar(cb,cax=fig.add_axes(loc[0]),orientation='vertical',ticks=tick[0],aspect=20,shrink=0.2,pad=0.06)#方向
        cbar.ax.tick_params(length=1.8,width=0.4,pad=1.5)
        cbar.set_label('%',labelpad=1)
    
    

df1 = pd.DataFrame({'June':pcs[0,0],'July':pcs[0,1],'August':pcs[0,2],'September':pcs[0,3]})
corr1=df1.corr()
print(corr1)
df1 = pd.DataFrame({'June':pcs[1,0],'July':pcs[1,1],'August':pcs[1,2],'September':pcs[1,3]})
corr1=df1.corr()
print(corr1)