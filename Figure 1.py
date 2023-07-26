#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 10:07:07 2023

@author: fuzhenghang
"""
# In[0]
import CRU_corr as CC

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
mpl.rcParams["axes.linewidth"] = 0.8
plt.rcParams['ytick.direction'] = 'out'


d1 = xr.open_dataset('/Users/fuzhenghang/Documents/ERA5/temp_geo_5monthly9_1979_2022.nc')
lon = d1.variables['longitude'][:]
lat = d1.variables['latitude'][:]

proj = ccrs.PlateCarree()  #中国为左
fig = plt.figure(figsize=(10,4),dpi=1000)
ax=[]
x1 = [0,0,0,0,0.49,0.49,0.49,0.7,0.7,0.7]
yy = [0.77,0.38,0.24,0,0.67,0.35,0.04,0.67,0.35,0.04]
dx = [0.45,0.45]
dy = [0.6,0.4]
loc = [[0.46, 0.87, 0.0085, 0.4],[0.46, 0.24, 0.0085, 0.22],[0.535, 0, 0.18, 0.012],[0.742, 0, 0.18, 0.012]]

for i in range(2):
    if i == 0:
        ax.append(fig.add_axes([x1[i],yy[i],dx[i],dy[i]],projection = proj))
    else:
        ax.append(fig.add_axes([x1[i],yy[i],dx[i],dy[i]]))


for i in range(2):
    if i not in [1,3]:
        if i == 0:
            leftlon, rightlon, lowerlat, upperlat = (60,140,15,45)
            data = CC.corr
            datap = CC.corrp
        
        
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
        gl.ypadding=50
        gl.xpadding=50
        if i == 0:
            shp_path1 = r'/Users/fuzhenghang/Documents/python/Pakistan/tibetan/tibetan.shp'
            reader = Reader(shp_path1)
            tpfeat=cfeature.ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), edgecolor='grey', facecolor='none',zorder=2)
            ax[i].add_feature(tpfeat, linestyle='--',linewidth=0.8)
            cb=ax[i].contourf(CC.cycle_lon,lat,data, levels = np.arange(-0.7,0.7001,0.1),cmap = cmaps.hotcold_18lev ,transform=ccrs.PlateCarree(),extend='both',zorder=0)
            poly1 = plt.Polygon(CC.xy1,edgecolor='r',linestyle='--',fc="none", lw=2, alpha=1,transform=ccrs.PlateCarree(),zorder=7)
            poly2 = plt.Polygon(CC.xy2,edgecolor='b',linestyle='--',fc="none", lw=2, alpha=1,transform=ccrs.PlateCarree(),zorder=7)
            ax[i].add_patch(poly1)
            ax[i].add_patch(poly2)
            for l1 in range(46,70,2):
                for l2 in range(60,140,2):
                    if datap[l1,l2]<0.05 and data[l1,l2]<0:
                        ax[i].text(l2,90-l1,'.',fontsize=15,fontweight = 'bold',color = 'navy')
                    if datap[l1,l2]<0.05 and data[l1,l2]>0:
                        ax[i].text(l2,90-l1,'.',fontsize=15,fontweight = 'bold',color = 'darkred')
            position=fig.add_axes(loc[0])
            cbar=plt.colorbar(cb,cax=position,orientation='vertical',ticks=np.arange(-0.6,0.6001,0.3),
                             aspect=20,shrink=0.2,pad=0.06)
            cbar.set_label('Corr.',labelpad=1)
            ax[i].text(56,46,'a',fontsize=12,fontweight='bold')
        
                
    else:
        ax[0].text(56,10,'b',fontsize=12,fontweight='bold')
        x = [1979+i for i in range(44)]
        ax[i].yaxis.tick_right()
        ax[i].yaxis.set_label_position("right")
        ax[i].set_xlim(1978.5,2022.5)
        ax[i].set_ylim(-3,4.2)
        ax[i].set_ylabel('Normalized Value',labelpad=1)
        ax[i].set_yticks([-3,-2,-1,0,1,2,3,4])
        ax[i].tick_params(length=2,width=0.4,pad=1.5)
        ax[i].axhline(y=0,  linestyle='-',linewidth = 0.35,color='black',alpha=1,zorder=0)
        if i == 1:
            ax[i].bar(x,CC.pren,color='lightskyblue',lw=2,label='Precipitation')
            ax[i].plot(x,CC.hwn,'-o',color='tomato',lw=2,ms=4,label='Heatwave')
            ax[i].axhline(y=1,  linestyle='dashed',linewidth = 0.35,color='black',alpha=1,zorder=0)
            ax[i].axhline(y=-1,  linestyle='dashed',linewidth = 0.35,color='black',alpha=1,zorder=0)
            ax[i].axvline(x=2013,  linestyle='-',linewidth = 2,color='black',alpha=0.2,zorder=0)
            const1,p1 = pearsonr(CC.pre, CC.hw)
            ax[i].text(1980,3.3,'correlation = %.2f**'%const1,color='k',fontsize=8)
            ax[i].legend(frameon=False,loc='lower left',ncol=3)
            ax[i].text(2013,-2.7,'2013',ha='center')
            #ax[i].text(2012.5,2.5,'Year 2022: 3.99→',fontsize=9,c='r')
            ax[i].annotate('Year 2022: 3.99', xy=(2022, 3.99), xytext=(2013,2.6), arrowprops=dict(arrowstyle="->", color="r", hatch='*',))
        