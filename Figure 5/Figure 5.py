#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:43:04 2023

@author: fuzhenghang
"""

# In[0]
import hw_pre_diff as HP
import cir_diff as C
import vertical_diff as V

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
from matplotlib.colors import ListedColormap
cmap=plt.get_cmap(cmaps.GMT_red2green_r)
newcolors=cmap(np.linspace(0, 1, 256))
newcmap1 = ListedColormap(newcolors[20:235:10])

mpl.rcParams["mathtext.fontset"] = 'cm'  # 数学文字字体
mpl.rcParams["font.size"] = 8
mpl.rcParams["axes.linewidth"] = 0.9
plt.rcParams['ytick.direction'] = 'out'


d1 = xr.open_dataset('/Users/fuzhenghang/Documents/ERA5/temp_geo_5monthly9_1979_2022.nc')
lon = d1.variables['longitude'][:]
lat = d1.variables['latitude'][:]

proj = ccrs.PlateCarree()  #中国为左
fig = plt.figure(figsize=(12,7),dpi=1000)
ax=[]
x1 = [0,0,0.33,0.33,0,0.33]
yy = [0.72,0.43,0.72,0.43,0,0]
dx = [0.33,0.33,0.66,0.66,0.33,0.66]
dy = [0.25,0.25,0.25,0.25,0.375,0.375]
loc = [[0.335, 0.735, 0.0085, 0.22],[0.335, 0.445, 0.0085, 0.22],[0.903, 0.735,0.0085, 0.22],[0.903, 0.448, 0.0085, 0.22],[0.335, 0.028, 0.0085, 0.32],[0.903, 0.028, 0.0085, 0.32]]

for i in range(6):
    if i != 4:
        ax.append(fig.add_axes([x1[i],yy[i],dx[i],dy[i]],projection = proj))
    else: 
        ax.append(fig.add_axes([x1[i],yy[i],dx[i],dy[i]]))
label = ['a','b']
sc = ['mm/month','day/year','gpm','Pa/s','K','$10^{-11}/s$']
label2 = ['c','d']
#cofi = cmaps.pengsa
#cofi = plt.cm.bwr
cofi = cmaps.ncaccept
wc = '#555555'
ic = '#3A3A3A'
bc = 'navy'
rc = 'darkred'
fs = 12

tik = [[-60,-30,0,30,60],[-0.04,-0.02,0,0.02,0.04]]
for i in range(6):
    if i <= 1:
        leftlon, rightlon, lowerlat, upperlat = (60,150,5,45)
        shp_path1 = r'/Users/fuzhenghang/Documents/python/Pakistan/tibetan/tibetan.shp'
        reader = Reader(shp_path1)
        tpfeat=cfeature.ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), edgecolor='grey', facecolor='none',zorder=2)
        ax[i].add_feature(tpfeat, linewidth=0.8, linestyle='--')
    elif i in [2,3]:
        leftlon, rightlon, lowerlat, upperlat = (30,160,20,60)
    elif i == 5:
        leftlon, rightlon, lowerlat, upperlat = (30,160,0,60)
    if i != 4:
        
        lon_formatter = cticker.LongitudeFormatter()
        lat_formatter = cticker.LatitudeFormatter()
        ax[i].set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
        ax[i].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.4,color='k',zorder=2)
        ax[i].add_feature(cfeature.OCEAN.with_scale('50m'),zorder=1,color='w',lw=0)
        
        gl=ax[i].gridlines(draw_labels=True, linewidth=0.35, color='k', alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator(np.arange(0,181,20))
        gl.ylocator = mticker.FixedLocator(np.arange(10,80,10))
        gl.top_labels    = False    
        gl.right_labels  = False
        gl.ypadding=50
        gl.xpadding=50
    
    if i <= 1:
        gl.top_labels    = False    
        gl.right_labels  = False
        gl.bottom_labels  = True
        ax[i].text(53,46,label[i],fontsize=14,fontweight='bold')
        if  i ==0:
            gl.bottom_labels  = False
            cb1=ax[i].contourf(HP.lon,HP.lat,HP.data[i], levels=HP.levels[i],cmap=HP.cm[i] ,transform=ccrs.PlateCarree(),extend='both',zorder=0)
            for ii in range(95,135,2):
                for jj in range(60,150,2):
                    if HP.datap[i][ii,jj]<=0.05 and HP.data[i][ii,jj]>0:
                        ax[i].text(jj,ii-90,'·',color ='darkgreen',zorder=1,fontweight='bold',fontsize=fs)
                    if HP.datap[i][ii,jj]<=0.05 and HP.data[i][ii,jj]<0:
                        ax[i].text(jj,ii-90,'·',color ='saddlebrown',zorder=1,fontweight='bold',fontsize=fs)
        else:
            ax[i].text(53,-1,'e',fontsize=14,fontweight='bold')
            cb1=ax[i].contourf(lon,lat,HP.data[i], levels=HP.levels[i],cmap=HP.cm[i] ,transform=ccrs.PlateCarree(),extend='both',zorder=0)
            for ii in range(45,86,2):
                for jj in range(60,150,2):
                    if HP.datap[i][ii,jj]<=0.05 and HP.data[i][ii,jj]<0:
                        ax[i].text(jj,90-ii,'.',color =bc,zorder=1,fontweight='bold',fontsize=fs)
                    if HP.datap[i][ii,jj]<=0.05 and HP.data[i][ii,jj]>0:
                        ax[i].text(jj,90-ii,'.',color =rc,zorder=1,fontweight='bold',fontsize=fs)
        position2=fig.add_axes(loc[i])#位置[左,下,长度,宽度]
        cbar1=plt.colorbar(cb1,cax=position2,orientation='vertical',ticks=HP.ti[i],
                         aspect=20,shrink=0.2,pad=0.06)#方向 
        cbar1.ax.tick_params(length=1.8,width=0.4,pad=1.5)
        ax[i].text(151,5,sc[i],fontsize=8)
         
    if i in [2,3]:
        ax[i].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.6,color='k',zorder=2)
        ax[i].text(23,61,label2[i-2],fontsize=14,fontweight='bold')
        if  i == 2:
            gl.bottom_labels  = False
            cb1=ax[i].contourf(lon,lat,C.data[i-2], levels=C.levels[i-2],cmap=cofi,transform=ccrs.PlateCarree(),extend='both',zorder=1)#cmap=C.cm[i-2]
            for ii in range(31,69,2):
                for jj in range(30,160,2):
                    if C.datap[i-2][ii,jj]<=0.05 and C.data[i-2][ii,jj]<0:
                        ax[i].text(jj,90.5-ii,'.',color =bc,zorder=1,fontweight='bold',fontsize=fs)
                    if C.datap[i-2][ii,jj]<=0.05 and C.data[i-2][ii,jj]>0:
                        ax[i].text(jj,90.5-ii,'.',color =rc,zorder=1,fontweight='bold',fontsize=fs)
            cq = ax[i].quiver(C.lon[::3],C.lat[::3],C.winds[(i-2)*2][::3,::3],C.winds[(i-2)*2+1][::3,::3],color=wc,
                              transform=ccrs.PlateCarree(),scale=230,width=0.0028,zorder=2)
            ax[i].quiverkey(cq, X=0.9, Y = 1.046, U=C.uu[i-2] ,angle = 0,label=C.labe[i-2],labelpos='E', color = wc,labelcolor = 'k')
        if i == 3:
            #cb1=ax[i].pcolormesh(lon,lat,C.data[i-1], vmin=-0.05,vmax=0.05,cmap=cofi,transform=ccrs.PlateCarree(),zorder=1,alpha = 1)
            cb1=ax[i].contourf(lon,lat,C.data[i-1], levels = np.arange(-0.045,0.04501,0.005),extend='both',cmap=cofi,transform=ccrs.PlateCarree(),zorder=1,alpha = 1)
            for ii in range(30,69,2):
                for jj in range(30,160,2):
                    if C.datap[i-1][ii,jj]<=0.05 and C.data[i-1][ii,jj]<0:
                        ax[i].text(jj,90.5-ii,'.',color =bc,ha='center',va='center',zorder=2,fontweight='bold',fontsize=fs)
                    if C.datap[i-1][ii,jj]<=0.05 and C.data[i-1][ii,jj]>0:
                        ax[i].text(jj,90.5-ii,'.',color =rc,ha='center',va='center',zorder=2,fontweight='bold',fontsize=fs)
            cb=ax[i].contour(lon,lat,C.data[i-2], levels=[-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30],colors = ic ,transform=ccrs.PlateCarree(),linewidths=0.8,zorder=2)
            ax[i].clabel(cb, inline=1, fontsize=6.5)
        cbar=plt.colorbar(cb1,cax=fig.add_axes(loc[i]),orientation='vertical',ticks=tik[i-2],
                         aspect=20,shrink=0.2,pad=0.06)#方向
        ax[i].text(161.5,20,sc[i],fontsize=8)

        #ax[i].quiver(C.lon[::3],C.lat[::3],C.wind[(i-2)*2][::3,::3],C.wind[(i-2)*2+1][::3,::3],color='gray',
                                            #transform=ccrs.PlateCarree(),scale=C.sca[i-2],width=0.003,edgecolor='w',linewidth=0.18)
        #cb1=ax[i].contour(lon,lat,C.data[i-1], [-0.02,-0.01,0.01,0.02],colors = 'k',transform=ccrs.PlateCarree(),zorder=1,alpha = 1)
   
    if i == 5:
        ax[i].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.6,color='k',zorder=2)
        ax[i].text(23,61.5,'f',fontsize=14,fontweight='bold')
        cb=ax[i].contourf(lon,V.lat,V.drws, levels=np.arange(-36,36+0.00001,6.0000),cmap=cofi,transform=ccrs.PlateCarree(),extend='both',zorder=1)
        for k in range(30,89,2):
            for j in range(30,160,2):
                if V.prws[k,j]<=0.05 and V.drws[k,j]<0:
                    ax[i].text(j,90.5-k,'.',color =bc,ha='center',va='center',zorder=1,fontweight='bold',fontsize=fs)
                if V.prws[k,j]<=0.05 and V.drws[k,j]>0:
                    ax[i].text(j,90.5-k,'.',color =rc,ha='center',va='center',zorder=1,fontweight='bold',fontsize=fs)
        cq=ax[i].quiver(V.lon[::4],V.lat[::4],V.wind[0][::4,::4],V.wind[1][::4,::4],scale=35,color=wc,width=0.0036,zorder=3)
        #cq=ax[i].quiver(lon[::4],lat[::4],winds[0][::4,::4],winds[1][::4,::4],color='gray',scale=30,width=0.003,linewidth=0.3)
        ax[i].quiverkey(cq, X=0.9, Y = 1.046, U = 2,angle = 0,label='2 m/s',labelpos='E', color = wc,labelcolor = 'k')
        CS=ax[i].contour(V.lon,V.lat,V.u850c,[10,20,30],linewidths=1,alpha=1,colors=[ic],zorder=1)
        ax[i].clabel(CS, inline=1, fontsize=6.5)
        cbar=plt.colorbar(cb,fig.add_axes(loc[i]),orientation='vertical',ticks=[-36,-18,0,18,36],
                        aspect=20,shrink=0.2,pad=0.06)#方向 
        cbar.ax.tick_params(length=1.8,width=0.4,pad=1.5)
        ax[i].text(161.5,0,sc[i],fontsize=8)


    if i == 4:
        ax[i].set_yscale('symlog')
        ax[i].set_yticks([1000, 500,300, 200, 100])
        ax[i].set_yticklabels(['1000','500','300','200','100'])
        ax[i].invert_yaxis()
        ax[i].set_ylabel('hPa',fontsize=9)
        c = ax[i].contourf(V.x[i-4],V.level ,V.data[i-3],levels=np.arange(-2,2.001,0.2), extend = 'both',zorder=1, cmap=cofi)
        for k in range(25):
            for j in range(40,140,2):
                if V.pz_ver[0,k,j]<=0.05 and V.data[i-3][k,j]<0:
                    ax[i].text(j,V.h[k],'.',color =bc,ha='center',va='center',zorder=1,fontweight='bold',fontsize=fs)
                if V.pz_ver[0,k,j]<=0.05 and V.data[i-3][k,j]>0:
                    ax[i].text(j,V.h[k],'.',color =rc,ha='center',va='center',zorder=1,fontweight='bold',fontsize=fs)
        ax[i].xaxis.set_major_formatter(cticker.LongitudeFormatter())  
        ax[i].fill_between(V.lon, V.oro_ver1, 1000, where=V.oro_ver1 < 1000, facecolor='darkgrey',zorder=12)
        #CS=ax[i].contour(V.lon,V.level,V.theta1,[310+5*j for j in range(15)],linewidths=0.92,alpha=0.4,colors=[ic],zorder=1)
        #ax[i].clabel(CS, inline=1, fontsize=6.5)
        ax[i].set_xlim(40,140)
        cbar=plt.colorbar(c,fig.add_axes(loc[i]),orientation='vertical',ticks=[-2,-1,0,1,2],
                        aspect=20,shrink=0.2,pad=0.06)#方向 
        cbar.ax.tick_params(length=1.8,width=0.4,pad=1.5)
        cq=ax[i].quiver(V.lon[::4],V.level,V.wind[2][:,::4],V.wind[3][:,::4]*(-100),color=wc,scale=120,width=0.0048,zorder=10)
        #cq=ax[i].quiver(V.lon[::4],V.level,V.winds[2][:,::4],V.winds[3][:,::4]*(-100),color='gray',scale=120,width=0.0048,linewidth=0.3)
        ax[i].quiverkey(cq, X=0.87, Y = 1.046, U = 10,angle = 0,label='10m/s',labelpos='E', color = wc,labelcolor = 'k')
        ax[i].text(142.3,1000,sc[i],fontsize=8)
        

         
    