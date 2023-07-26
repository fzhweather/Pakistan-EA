#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 10:07:07 2023

@author: fuzhenghang
"""
# In[0]
import MCA as M
import MCA78bkl as MB

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
from scipy.stats import pearsonr


# In[1]
mpl.rcParams["mathtext.fontset"] = 'cm'  # 数学文字字体
mpl.rcParams["font.size"] = 8
mpl.rcParams["axes.linewidth"] = 0.9
plt.rcParams['ytick.direction'] = 'out'


d1 = xr.open_dataset('/Users/fuzhenghang/Documents/ERA5/temp_geo_5monthly9_1979_2022.nc')
lon = d1.variables['longitude'][:]
lat = d1.variables['latitude'][:]

proj = ccrs.PlateCarree()  #中国为左
fig = plt.figure(figsize=(12,8),dpi=1000)
ax=[]
x1 = [0,0,0.472,0.472]
yy = [0.77,0.51,0.77,0.51]
dx = 0.45
dy = 0.22
loc = [[0.46, 0.77, 0.0085, 0.22],[0.93, 0.77, 0.0085, 0.22],[0.535, 0, 0.18, 0.012],[0.742, 0, 0.18, 0.012]]
tit = ['OLR (ERA5) & Heatwave (ERA5)','OLR (NOAA) & Heatwave (BEST)']
for i in range(4):
    if i not in [1,3]:
        if i <=3:
            ax.append(fig.add_axes([x1[i],yy[i],dx,dy],projection = proj))
        else:
            ax.append(fig.add_axes([x1[i],yy[i],0.265,0.265],projection = proj))
    else:
        ax.append(fig.add_axes([x1[i],yy[i],dx,dy]))


for i in range(4):
    if i not in [1,3]:
        leftlon, rightlon, lowerlat, upperlat = (60,140,20,45)
        leftlon, rightlon, lowerlat, upperlat = (60,140,20,45)
        
        lon_formatter = cticker.LongitudeFormatter()
        lat_formatter = cticker.LatitudeFormatter()
        ax[i].set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
        ax[i].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k',zorder=2)
        ax[i].add_feature(cfeature.OCEAN.with_scale('50m'),zorder=1,color='w',lw=0)
        
        gl=ax[i].gridlines(draw_labels=True, linewidth=0.35, color='k', alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator(np.arange(60,181,20))
        gl.ylocator = mticker.FixedLocator(np.arange(10,80,10))
        gl.top_labels    = False    
        gl.right_labels  = False
        if i == 2:
            gl.left_labels  = False
        gl.ypadding=50
        gl.xpadding=50
        shp_path1 = r'/Users/fuzhenghang/Documents/python/Pakistan/tibetan/tibetan.shp'
        reader = Reader(shp_path1)
        tpfeat=cfeature.ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), edgecolor='grey', facecolor='none',zorder=2)
        ax[i].add_feature(tpfeat, linewidth=0.8,linestyle='--')
        ax[i].text(100,46,tit[i//2],fontsize=10,ha='center')
        if i == 0:
            CS=ax[i].contour(M.lo[i],M.la[i],-1*M.pattern[0],[-0.1+0.02*k for k in range(11)], cmap = cmaps.MPL_BrBG_r ,linewidths = 1.8,transform=ccrs.PlateCarree(),extend='both',zorder=1)
            ax[i].clabel(CS, inline=1, fontsize=7,colors = 'k')
            cb1=ax[i].contourf(M.lo[i+1],M.la[i+1],-1*M.pattern[i+1], levels=np.arange(-0.064,0.064001,0.008),cmap = M.newcmap3 ,transform=ccrs.PlateCarree(),extend='both',zorder=0)
            poly = plt.Polygon(M.xy,edgecolor='k',linestyle='--',fc="none", lw=2, alpha=0.8,transform=ccrs.PlateCarree(),zorder=7)
            ax[i].add_patch(poly)
            position=fig.add_axes(loc[1])
            cbar1=plt.colorbar(cb1,cax=position,orientation='vertical',ticks = M.ti[0],
                             aspect=20,shrink=0.2,pad=0.06)
            ax[i].text(57.8,46.5,'a',fontsize=15,fontweight='bold')
        elif i == 2:
            CS=ax[i].contour(MB.lo[i-2],MB.la[i-2],-1*MB.pattern[0],[-0.1+0.02*k for k in range(11)], cmap = cmaps.MPL_BrBG_r ,linewidths = 1.8,transform=ccrs.PlateCarree(),extend='both',zorder=1)
            ax[i].clabel(CS, inline=1, fontsize=7,colors = 'k')
            cb1=ax[i].contourf(MB.lo[i-1],MB.la[i-1],-1*MB.pattern[i-1], levels=np.arange(-0.064,0.064001,0.008),cmap = M.newcmap3 ,transform=ccrs.PlateCarree(),extend='both',zorder=0)
            poly = plt.Polygon(M.xy,edgecolor='k',linestyle='--',fc="none", lw=2, alpha=0.8,transform=ccrs.PlateCarree(),zorder=7)
            ax[i].add_patch(poly)
            position=fig.add_axes(loc[1])
            cbar1=plt.colorbar(cb1,cax=position,orientation='vertical',ticks = M.ti[0],
                             aspect=20,shrink=0.2,pad=0.06)
            ax[i].text(57.8,46.5,'c',fontsize=15,fontweight='bold')
        
        
    else:
        x = [1979+i for i in range(44)]
        ax[i].set_xlim(1978.5,2022.5)
        ax[i].set_ylim(-3.2,4.2)
        ax[i].set_ylabel('Normalized Value',labelpad=1)
        ax[i].set_yticks([-3,-2,-1,0,1,2,3,4])
        ax[i].tick_params(length=2,width=0.4,pad=1.5)
        ax[i].axhline(y=0,  linestyle='-',linewidth = 0.35,color='black',alpha=1,zorder=0)
        if i == 1:
            ax[i].text(1977,4.55,'b',fontsize=15,fontweight='bold')
            ax[i].yaxis.tick_left()
            ax[i].yaxis.set_label_position("left")
            y1 = np.zeros((46))-0.6
            y2 = y1+1.2
            x1 = [1978+i for i in range(46)]
            ax[i].fill_between(x1, y1, y2, color='gray',alpha = 0.2,linewidths=0)
            ax[i].text(1980,3.3,M.lab[0],fontsize=9)
            ax[i].plot(x,-1*M.y1,'-o',color='royalblue',markersize=4,lw = 2,label='OLR')
            ax[i].plot(x,-1*M.y2,'-*',color='tomato',markersize=6,lw = 2,label='Heatwave')
            ax[i].legend(frameon=False,loc='lower right',ncol=2)
            #ax[i].text(2012.5,2.5,'Year 2022: 3.89→',fontsize=9,c='r')
            ax[i].set_ylabel('Year',labelpad=1)
            ax[i].annotate('Year 2022: 3.89', xy=(2022, 3.89), xytext=(2014,2.6), arrowprops=dict(arrowstyle="->", color="r", hatch='*',))
        if i == 3:
            ax[i].text(1977,4.55,'d',fontsize=15,fontweight='bold')
            x = [1979+i for i in range(41)]
            ax[i].yaxis.tick_right()
            ax[i].yaxis.set_label_position("right")
            y1 = np.zeros((46))-0.6
            y2 = y1+1.2
            x1 = [1978+i for i in range(46)]
            ax[i].fill_between(x1, y1, y2, color='gray',alpha = 0.2,linewidths=0)
            ax[i].text(1980,3.3,MB.lab[0],fontsize=9)
            ax[i].plot(x,-1*MB.y1,'-o',color='royalblue',markersize=4,lw = 2,label='OLR')
            ax[i].plot(x,-1*MB.y2,'-*',color='tomato',markersize=6,lw = 2,label='Heatwave')
            #ax[i].text(2012.5,2.5,'Year 2022: 3.89→',fontsize=9,c='r')
            ax[i].text(2011.5,3.3,'OLR Corr.: 0.93**',fontweight='bold',c='royalblue')
            ax[i].text(1996.5,3.3,'Heatwave Corr.: 0.94**',fontweight='bold',c='tomato')
            
  
        
        
        