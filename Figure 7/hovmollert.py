#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 19:32:29 2023

@author: fuzhenghang
"""


import xarray as xr
import numpy as np
import netCDF4 as nc
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
mpl.rcParams["font.size"] = 7
mpl.rcParams["axes.linewidth"] = 1

d1 = xr.open_dataset(r'/Users/fuzhenghang/Documents/大四下/毕业论文/LBM/era40.clim.t42.nc')
#print(d1.variables.keys())
#print(np.array(d1['levels']))
z_tmp = d1['z'][6:8,10,]
z_tmp = np.mean(z_tmp,axis=0)

u_tmp = d1['u'][6:8,10,]
u_tmp = np.mean(u_tmp,axis=0)

v_tmp = d1['v'][6:8,10,]
v_tmp = np.mean(v_tmp,axis=0)

lat = d1['latitude']
lon = d1['longitude']
d2 = nc.Dataset(r'/Users/fuzhenghang/Documents/大四下/毕业论文/LBM/pretr.nc')
za = d2['z'][:,10,]
print(za.shape)
za = np.array(za[:,::-1,:])

#print(za[-1,15,45])
#print(lat)
a=6371393 #地球半径
omega=7.292e-5 #自转角速度
levs = 200/1000
#print(lon)
dlon=(np.gradient(lon)*np.pi/180.0).reshape((1,-1))
dlat=(np.gradient(lat)*np.pi/180.0).reshape((-1,1))
coslat = (np.cos(np.array(lat)*np.pi/180)).reshape((-1,1))
sinlat = (np.sin(np.array(lat)*np.pi/180)).reshape((-1,1))

#计算科氏力
f=np.array(2*omega*np.sin(lat*np.pi/180.0)).reshape((-1,1)) 
#计算|U|
d1 = nc.Dataset(r'/Users/fuzhenghang/Documents/大四下/毕业论文/LBM/era40.clim.t42.nc')
#print(d1.variables.keys())
#print(np.array(d1['levels']))
z_tmp = d1['z'][6:8,10,]
z_tmp = np.mean(z_tmp,axis=0)

u_tmp = d1['u'][6:8,10,]
u_tmp = np.mean(u_tmp,axis=0)

v_tmp = d1['v'][6:8,10,]
v_tmp = np.mean(v_tmp,axis=0)
wind = np.sqrt(u_tmp**2+v_tmp**2)
print(wind[48,64+25])
#计算括号外的参数，a^2可以从括号内提出


c=(levs)*coslat/(2*a*a*wind)
#print(c)
#Φ`
#print(lon)
#print(lat)
#print(za)
#Ψ`
streamf = 9.8*za/f

#计算各个部件，难度在于二阶导，变量的名字应该可以很容易看出我是在计算哪部分
dzdlon = np.gradient(streamf,axis = 1)/dlon
ddzdlonlon = np.gradient(dzdlon,axis = 1)/dlon
dzdlat = np.gradient(streamf,axis = 0)/dlat
ddzdlatlat = np.gradient(dzdlat,axis = 0)/dlat
ddzdlatlon = np.gradient(dzdlat,axis = 1)/dlon
#这是X,Y分量共有的部分
x_tmp = dzdlon*dzdlon-streamf*ddzdlonlon
y_tmp = dzdlon*dzdlat-streamf*ddzdlatlon
#计算两个分量
fx = c * ((u_tmp/coslat/coslat)*x_tmp+v_tmp*y_tmp/coslat)
fy = c * ((u_tmp/coslat)*y_tmp+v_tmp*x_tmp)
print(fx[-1,40,30])
#print(lat)
za1 = za[:25]
fx = fx[:25]
fy = fy[:25]
ss = 42
ee = 47
za1 = np.mean(za1[:,ss:ee,:],axis=1)#17:46 39:20
fx1 = np.mean(fx[:,ss:ee,:],axis=1)
fy1 = np.mean(fy[:,ss:ee,:],axis=1)
print(za1.shape)
fig = plt.figure(figsize=(10,10),dpi=800)

ax=[]
x1 = [0,0,0.45,0,0.45]
yy = [10.95,0.56,0.56,0,0]
dx = [0.45,0.4,0.15,0.4,0.15]
dy = 1
xla=['Longitude','Latitude']
x=[lon,lat]
level = [4+4*i for i in range(25)]
#loc = [[0.52, 1.13, 0.015, 0.35],[1.12, 1.13, 0.015, 0.35],[0.52, 0.65, 0.015, 0.35],[1.12, 0.65, 0.015, 0.35]]
proj = ccrs.PlateCarree()  #中国为左
ax.append(fig.add_axes([x1[0],yy[0],dx[0],dy],projection = proj))
for i in range(1):
    ax[i].set_xticks([0,30,60,90,120,150])
    ax[i].set_yticks([100,80,60,40,20])
    ax[i].set_yticklabels(['25 Day','20 Day','15 Day','10 Day','5 Day'])
    ax[i].invert_yaxis()
    cq=ax[i].quiver(lon[::2],level,fx1[:,::2],fy1[:,::2],scale=18,width=0.0025,edgecolor='w',linewidth=0.2)
    #cq=ax[i].quiver(lon[::],level,winds[2][:,::4],winds[3][:,::4]*(-100),color='gray',scale=120,width=0.0048,linewidth=0.3)
    ax[i].quiverkey(cq, X=0.87, Y = 1.046, U = 1,angle = 0,label='1 $m^2/s^2$',labelpos='E', color = 'k',labelcolor = 'k')
    
   
    c = ax[i].contourf(x[i],level ,za1,levels=np.arange(-16,16.001,2), extend = 'both',zorder=0, cmap=plt.cm.bwr)
    ax[i].xaxis.set_major_formatter(cticker.LongitudeFormatter())  
    ax[i].set_xlim(1,180)
    