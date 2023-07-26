#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 09:56:52 2023

@author: fuzhenghang
"""

# In[0]
import spreadconvec_temp as SC
import mean_cesmconvec_temp as SSTC
#import spread_cesm_z200 as SZ
#import mean_reg_zg as SSTZ


import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import cartopy.mpl.ticker as cticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr  
import numpy as np
import cmaps
from cartopy.io.shapereader import Reader

# In[1]

mpl.rcParams["mathtext.fontset"] = 'cm'  # 数学文字字体
mpl.rcParams["font.size"] = 8
mpl.rcParams["axes.linewidth"] = 0.9
plt.rcParams['ytick.direction'] = 'out'


d1 = xr.open_dataset('/Users/fuzhenghang/Documents/ERA5/temp_geo_5monthly9_1979_2022.nc')
lon = d1.variables['longitude'][:]
lat = d1.variables['latitude'][:]

proj = ccrs.PlateCarree(central_longitude=180)
fig = plt.figure(figsize=(9,5),dpi=1000)
ax=[]
x1 = [0.076,0.463,0,0,0]
yy = [0.72,0.72,0.35,0.00]
dx = [0.38,0.38,0.92,0.92,0.92]
dy = [0.36,0.36,0.3,0.3,0.3]
loc = [[0.84, 0.74, 0.012, 0.32],[0.84, 0.53, 0.012, 0.25],[0.84, 0.36, 0.012, 0.28],[0.84, 0.012, 0.012, 0.28],[0.335, 0.028, 0.0085, 0.32],[0.903, 0.028, 0.0085, 0.32]]
data = [SSTC.corr1,SC.corr1]
datap = [SSTC.corrp1,SC.corrp1]
level = [SSTC.levels,SC.levels]
#regd = [SSTZ.reg,SZ.reg]
lev =[np.arange(-2.7,2.71,0.3),np.arange(-1.8,1.8001,0.2)]
ti = [np.arange(-2.4,2.4001,1.2),np.arange(-1.6,1.6001,0.8)]
for i in range(2):
    ax.append(fig.add_axes([x1[i],yy[i],dx[i],dy[i]],projection = proj))
tit1 = ['a','b','C','D']
tit2 = ['Ensemble mean','Ensemble spread','H200: SST-forced pattern','H200: Atmospheric internal pattern']
cr = ['0.13','0.01']
for i in range(2):
    if i <= 1:
        leftlon, rightlon, lowerlat, upperlat = (60,150,5,55)
        gl=ax[i].gridlines(draw_labels=True, linewidth=0.35, color='k', alpha=0.5, linestyle='--',zorder=8)
        gl.xlocator = mticker.FixedLocator(np.arange(0,181,20))
        gl.ylocator = mticker.FixedLocator(np.arange(10,80,20))
        ax[i].add_feature(cfeature.OCEAN.with_scale('50m'),zorder=7,color='w',lw=0)
        ax[i].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.6,color='k',zorder=2)
        ax[i].text(56-180,56,tit1[i],fontsize=12,fontweight='bold')
        ax[i].text(62-180,56,tit2[i],fontsize=10)
        ax[i].text(135-180,56,cr[i],fontsize=10)
    else:
        leftlon, rightlon, lowerlat, upperlat = (1,359,-10,70)
        gl=ax[i].gridlines(draw_labels=True, linewidth=0.35, color='k', alpha=0.5, linestyle='--',zorder=8)
        gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
        gl.ylocator = mticker.FixedLocator(np.arange(-10,80,20))
        ax[i].add_feature(cfeature.OCEAN.with_scale('50m'),zorder=1,color='w',lw=0)
        ax[i].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.3,color='k',zorder=2)
        ax[i].text(-188,72,tit1[i],fontsize=12,fontweight='bold')
        ax[i].text(-176,72,tit2[i],fontsize=10)
        
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax[i].set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    
    
    
    gl.top_labels    = False    
    gl.right_labels  = False
    gl.ypadding=50
    gl.xpadding=50
    
    if i in [1]:
        gl.left_labels    = False 
    if i in [2]:
        gl.bottom_labels    = False 
    if i <= 1:
        cb=ax[i].contourf(SSTC.cycle_lon,SSTC.lats,data[i], levels=np.arange(-1,1.001,0.1),cmap=cmaps.hotcold_18lev ,transform=ccrs.PlateCarree(),extend='both',zorder=0)
        for l1 in range(95,145,2):
            for l2 in range(60,150,2):
                if datap[i][l1,l2]<0.05 and data[i][l1,l2]<0:
                    ax[i].text(l2-180,l1-90,'.',fontsize=8,fontweight = 'bold',color = 'navy')
                if datap[i][l1,l2]<0.05 and data[i][l1,l2]>0:
                    ax[i].text(l2-180,l1-90,'.',fontsize=8,fontweight = 'bold',color = 'darkred')
        poly1 = plt.Polygon(SC.xy1,edgecolor='red',linestyle='--',fc="none", lw=1.3, alpha=1,transform=ccrs.PlateCarree(),zorder=7)
        ax[i].add_patch(poly1)
        poly2 = plt.Polygon(SC.xy2,edgecolor='b',linestyle='--',fc="none", lw=1.3, alpha=1,transform=ccrs.PlateCarree(),zorder=7)
        ax[i].add_patch(poly2)
    if i == 0:
        position1=fig.add_axes(loc[i])#位置[左,下,长度,宽度]
        cbar=plt.colorbar(cb,cax=position1,orientation='vertical',ticks=np.arange(-1,01.001,0.5),
                         aspect=20,shrink=0.2,pad=0.06)
        cbar.ax.tick_params(length=1.8,width=0.4,pad=1.5)
    """
    if i > 1:
        cb=ax[i].contourf(SSTC.lons,SSTC.lats,regd[i-2], levels=lev[i-2],cmap=cmaps.ncaccept ,transform=ccrs.PlateCarree(),extend='both',zorder=1)
        poly1 = plt.Polygon(SC.xy1,edgecolor='red',linestyle='--',fc="none", lw=1, alpha=1,transform=ccrs.PlateCarree(),zorder=7)
        ax[i].add_patch(poly1)
        poly2 = plt.Polygon(SC.xy2,edgecolor='b',linestyle='--',fc="none", lw=1, alpha=1,transform=ccrs.PlateCarree(),zorder=7)
        ax[i].add_patch(poly2)
        position1=fig.add_axes(loc[i])#位置[左,下,长度,宽度]
        cbar=plt.colorbar(cb,cax=position1,orientation='vertical',ticks=ti[i-2],
                         aspect=20,shrink=0.2,pad=0.06)
        cbar.ax.tick_params(length=1.8,width=0.4,pad=1.5)
        ax[i].text(181,-13,'gpm')
    for l1 in range(-10,70,5):
        for l2 in range(0,360,5):
            if i == 3:
                if SZ.R[90+l1,l2]>3.926 and regd[i-2][90+l1,l2]>0:
                    ax[i].text(l2-180,l1,'·',color ='darkred',zorder=1,fontweight='bold',fontsize=9)
                if SZ.R[90+l1,l2]>3.926 and regd[i-2][90+l1,l2]<0:
                    ax[i].text(l2-180,l1,'·',color ='navy',zorder=1,fontweight='bold',fontsize=9)
            if i == 2:
                if SSTZ.R[90+l1,l2]>4.130 and regd[i-2][90+l1,l2]>0:
                    ax[i].text(l2-180,l1,'·',color ='darkred',zorder=1,fontweight='bold',fontsize=9)
                if SSTZ.R[90+l1,l2]>4.130 and regd[i-2][90+l1,l2]<0:
                    ax[i].text(l2-180,l1,'·',color ='navy',zorder=1,fontweight='bold',fontsize=9)
                    """


    