#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 19:29:21 2023

@author: fuzhenghang
"""

# In[0]
import response as R
import response6 as R6
import responset as Rt
import lbmdiv as D
import hovmoller as H
import hovmoller6 as H6
import hovmollert as Ht


import netCDF4 as nc
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
from cartopy.util import add_cyclic_point
mpl.rcParams["mathtext.fontset"] = 'cm'  # 数学文字字体
mpl.rcParams["font.size"] = 8
mpl.rcParams["axes.linewidth"] = 0.9
plt.rcParams['ytick.direction'] = 'out'

la = 114
lo = 67
lonw=12
law=10

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

lon = np.empty(4)
lat = np.empty(4)
lon[0],lat[0] = 105, 27  # lower left (ll)
lon[1],lat[1] = 123, 27  # lower right (lr)
lon[2],lat[2] = 123, 33  # upper right (ur)
lon[3],lat[3] = 105, 33  # upper left (ul)
x, y =  lon, lat
xy2 = list(zip(x,y))

d2 = nc.Dataset(r'/Users/fuzhenghang/Documents/大四下/毕业论文/LBM/pre78r.nc')
time = d2['time'][:]
lon = d2['lon'][:]
lat = d2['lat'][:]

proj = ccrs.PlateCarree(central_longitude=180)
fig = plt.figure(figsize=(12,7),dpi=1000)
ax=[]
x1 = [0,0.32,0.64,0,0.32,0.64,0,0.32,0.64,0,0.32,0.64]
yy = np.array([0.79,0.79,0.79,0.58,0.58,0.58,0.37,0.37,0.37,0.05,0.05,0.05])-0.02
dx = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]
dy = [0.18,0.18,0.18,0.18,0.18,0.18,0.18,0.18,0.18,0.28,0.28,0.28]
loc = [[0.84, 0.74, 0.012, 0.32],[0.84, 0.53, 0.012, 0.25],[0.84, 0.36, 0.012, 0.28],[0.84, 0.012, 0.012, 0.28],[0.335, 0.028, 0.0085, 0.32],[0.903, 0.028, 0.0085, 0.32]]
data1 = [R.z[-1,10],R6.z[-1,10],Rt.z[-1,10]]
data2 = [R.z[-1,6],R6.z[-1,6],Rt.z[-1,6]]
data3 = [R.u[-1,10,::,::],R6.u[-1,10,::,::],Rt.u[-1,10,::,::],R.u[-1,6,::,::],R6.u[-1,6,::,::],Rt.u[-1,6,::,::]]
data4 = [R.v[-1,10,::,::],R6.v[-1,10,::,::],Rt.v[-1,10,::,::],R.v[-1,6,::,::],R6.v[-1,6,::,::],Rt.v[-1,6,::,::]]
tt = ['a','b','c','d','e','f','g','h','i','j','k','l']
for i in range(3):
    for j in range(64):
        for k in range(128):
            if (data3[i][j,k]**2 + data4[i][j,k]**2)<=0.01:
                data3[i][j,k]=np.nan
                data4[i][j,k]=np.nan
for i in range(3,6):
    for j in range(64):
        for k in range(128):
            if (data3[i][j,k]**2 + data4[i][j,k]**2)<=0.01:
                data3[i][j,k]=np.nan
                data4[i][j,k]=np.nan


for i in range(9):
    ax.append(fig.add_axes([x1[i],yy[i],dx[i],dy[i]],projection = proj))
proj1 = ccrs.PlateCarree()
for i in range(9,12):
    ax.append(fig.add_axes([x1[i],yy[i],dx[i],dy[i]],projection = proj1))
cm =cmaps.ncaccept
cm = cmaps.pengsa
for i in range(9):
    leftlon, rightlon, lowerlat, upperlat = (0,180,0,60)
    gl=ax[i].gridlines(draw_labels=True, linewidth=0.35, color='k', alpha=0.5, linestyle='--',zorder=8)
    gl.xlocator = mticker.FixedLocator(np.arange(0,181,30))
    gl.ylocator = mticker.FixedLocator(np.arange(0,80,20))
    #ax[i].add_feature(cfeature.OCEAN.with_scale('50m'),zorder=7,color='w',lw=0)
    ax[i].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.3,color='k',zorder=2)
    shp_path1 = r'/Users/fuzhenghang/Documents/python/tibetan/tibetan.shp'
    reader = Reader(shp_path1)
    tpfeat=cfeature.ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), edgecolor='grey', facecolor='none',zorder=2)
    ax[i].add_feature(tpfeat, linewidth=0.3)
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax[i].set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    gl.top_labels    = False    
    gl.right_labels  = False
    gl.ypadding=50
    gl.xpadding=50
    poly2 = plt.Polygon(xy2,edgecolor='b',linestyle='--',fc="none", lw=1, alpha=1,transform=ccrs.PlateCarree(),zorder=7)
    ax[i].add_patch(poly2)
    if i not in [0,3,6]:
        gl.left_labels  = False
    if i not in [6,7,8]:
        gl.bottom_labels  = False
    if i <= 2:
        ax[i].text(-188,62,tt[i],fontsize=12,fontweight='bold')
        cb1=ax[i].contourf(lon,lat,data1[i], levels=np.arange(-27,27.01,3),cmap=cm ,transform=ccrs.PlateCarree(),extend='both',zorder=0)
        cq = ax[i].quiver(lon[::2],lat[::2],data3[i][::2,::2],data4[i][::2,::2],color='#555555',
                              transform=ccrs.PlateCarree(),scale=50,width=0.004)   
        if i== 0:
            ax[i].quiverkey(cq, X=0.87, Y = 1.06, U=3 ,angle = 0,label='3 m/s',labelpos='E', color = '#555555',labelcolor = 'k')
            position2=fig.add_axes([0.945, 0.79, 0.009, 0.15])
            cbar1=plt.colorbar(cb1,cax=position2,orientation='vertical',ticks=[-24,-12,0,12,24],
                             aspect=20,shrink=0.2,pad=0.06)
            ax[i].text(386,0,'gpm')
    elif i <= 5:
        ax[i].text(-188,62,tt[i],fontsize=12,fontweight='bold')
        cb1=ax[i].contourf(lon,lat,data2[i-3], levels=np.arange(-18,18.01,2),cmap=cm ,transform=ccrs.PlateCarree(),extend='both',zorder=0)
        cq = ax[i].quiver(lon[::2],lat[::2],data3[i][::2,::2],data4[i][::2,::2],color='#555555',
                              transform=ccrs.PlateCarree(),scale=70,width=0.004)
        if i == 3:
            ax[i].quiverkey(cq, X=0.87, Y = 1.06, U=3 ,angle = 0,label='5 m/s',labelpos='E', color = '#555555',labelcolor = 'k')
            position2=fig.add_axes([0.945, 0.58, 0.009, 0.15])
            cbar1=plt.colorbar(cb1,cax=position2,orientation='vertical',ticks=[-16,-8,0,8,16],
                             aspect=20,shrink=0.2,pad=0.06)
            ax[i].text(386,0,'gpm')
    elif i <= 8:
        ax[i].text(-188,62,tt[i],fontsize=12,fontweight='bold')
        lons = D.lons
        out0, cycle_lon =add_cyclic_point(D.out[i-6,:,:], coord=lons)
        #out1, cycle_lon =add_cyclic_point(D.zz[i-6], coord=lons)
        cb1=ax[i].contourf(D.cycle_lon,D.lats,out0, levels=D.levels[1],cmap=cm ,transform=ccrs.PlateCarree(),extend='both',zorder=0)
        #ax[i].contour(D.cycle_lon,D.lats,out1,[-30,-25,-20,-15,15,20,25,30],linewidths=1,alpha=0.75,colors=['k'],zorder=1)
        if i !=7:
            CS=ax[i].contour(D.cycle_lon,D.lat3,D.ut0,[15,20,25],linewidths=0.75,alpha=0.75,colors=['magenta'],zorder=1)
        else:
            CS=ax[i].contour(D.cycle_lon,D.lat3,D.ut01,[15,20,25],linewidths=0.75,alpha=0.75,colors=['magenta'],zorder=1)
        cq = ax[i].quiver(D.lons[::2],D.lats[::2],D.d2u[i-6,::2,::2],D.d2v[i-6,::2,::2],color='#555555',
                          transform=ccrs.PlateCarree(),scale=15,width=0.004,alpha=1)   
        if i == 6:
            ax[i].quiverkey(cq, X=0.87, Y = 1.06, U=1 ,angle = 0,label='1 m/s',alpha=1,labelpos='E', color = '#555555',labelcolor = '#555555')
            position2=fig.add_axes([0.945, 0.37, 0.009, 0.15])
            cbar1=plt.colorbar(cb1,cax=position2,orientation='vertical',ticks=[-16,-8,0,8,16],
                             aspect=20,shrink=0.2,pad=0.06)
            ax[i].text(386,0,'10$^{-11}$/s')
data5 = [H.fx1[:,::3],H6.fx1[:,::3],Ht.fx1[:,::3]]
data6 = [H.fy1[:,::3],H6.fy1[:,::3],Ht.fy1[:,::3]]
data7 = [H.za1,H6.za1,Ht.za1]
for i in range(3):
    for j in range(25):
        for k in range(128//3):
            if (data5[i][j,k]**2 + data6[i][j,k]**2)<=0.01:
                data5[i][j,k]=np.nan
                data6[i][j,k]=np.nan
level = [4+4*i for i in range(25)]
for i in range(9,12):
    ax[i].text(-8,-1,tt[i],fontsize=12,fontweight='bold')
    ax[i].set_xticks([0,30,60,90,120,150])
    ax[i].set_yticks([100,80,60,40,20])
    ax[i].set_yticklabels(['25 Day','20 Day','15 Day','10 Day','5 Day'])
    ax[i].invert_yaxis()
    cq=ax[i].quiver(H.lon[::3],level,data5[i-9],data6[i-9],scale=24,width=0.004,color = '#555555')
    #cq=ax[i].quiver(lon[::],level,winds[2][:,::4],winds[3][:,::4]*(-100),color='gray',scale=120,width=0.0048,linewidth=0.3)
   
    c = ax[i].contourf(H.x[0],level ,data7[i-9],levels=np.arange(-16,16.001,2), extend = 'both',zorder=0, cmap=cmaps.ncaccept)
    ax[i].xaxis.set_major_formatter(cticker.LongitudeFormatter())  
    ax[i].set_xlim(1,180)
    if i != 9:
        ax[i].set_yticklabels([])
    if i== 9:
        ax[i].quiverkey(cq, X=0.87, Y = 1.06, U = 1,angle = 0,label='1 m$^2$/s$^2$',labelpos='E', color = '#555555')
        position2=fig.add_axes([0.945, 0.07, 0.009, 0.215])
        cbar1=plt.colorbar(c,cax=position2,orientation='vertical',ticks=[-16,-8,0,8,16],
                         aspect=20,shrink=0.2,pad=0.06)
        ax[i].text(565,90,'gpm')






