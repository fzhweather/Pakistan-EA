#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:11:32 2023

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
    elif p[i]<-th and h[i]<-th:
        low.append(i)


d1 = xr.open_dataset(r'/Users/fuzhenghang/Documents/ERA5/uvw_5monthly9_1979_2022.nc',use_cftime=True)
lon = d1.variables['longitude'][:]
lat = d1.variables['latitude'][:]
time = d1['time'][:]
u200 = d1['u'][(time.dt.month>=7)&(time.dt.month<=8)][:,0,14,:,:]
v200 = d1['v'][(time.dt.month>=7)&(time.dt.month<=8)][:,0,14,:,:]
u200 = u200.groupby('time.year').mean(dim='time')
v200 = v200.groupby('time.year').mean(dim='time')
scipy.signal.detrend(u200, axis=0, type='linear', bp=0, overwrite_data=True)
scipy.signal.detrend(v200, axis=0, type='linear', bp=0, overwrite_data=True)

u850 = d1['u'][(time.dt.month>=7)&(time.dt.month<=8)][:,0,30,:,:] #850hPa u
u850 = u850.groupby('time.year').mean(dim='time')
v850 = d1['v'][(time.dt.month>=7)&(time.dt.month<=8)][:,0,30,:,:] #850hPa v
v850 = v850.groupby('time.year').mean(dim='time')
scipy.signal.detrend(u850, axis=0, type='linear', bp=0, overwrite_data=True)
scipy.signal.detrend(v850, axis=0, type='linear', bp=0, overwrite_data=True)

d2 = xr.open_dataset(r'/Users/fuzhenghang/Documents/ERA5/temp_geo_5monthly9_1979_2022.nc',use_cftime=True)

z200 = d2['z'][:,0,14,:,:][(time.dt.month>=7)&(time.dt.month<=8)]/98.064
z200 = z200.groupby('time.year').mean(dim='time')
z500 = d2['z'][:,0,21,:,:][(time.dt.month>=7)&(time.dt.month<=8)]/98.064
z500 = z500.groupby('time.year').mean(dim='time')
scipy.signal.detrend(z200, axis=0, type='linear', bp=0, overwrite_data=True)
scipy.signal.detrend(z500, axis=0, type='linear', bp=0, overwrite_data=True)

d3 = xr.open_dataset(r'/Users/fuzhenghang/Documents/ERA5/precip_5month9_1979_2022.nc',use_cftime=True)
time = d3['time'][:]
pre = d3['tp'][:,0,:,:][(time.dt.month>=7)&(time.dt.month<=8)]*1000
pre = pre.groupby('time.year').mean(dim='time')
scipy.signal.detrend(pre, axis=0, type='linear', bp=0, overwrite_data=True)

d4 = xr.open_dataset(r'/Users/fuzhenghang/Documents/ERA5/uvw_5monthly9_1979_2022.nc',use_cftime=True)
time = d4['time'][:]
t200 = d4['w'][(time.dt.month>=7)&(time.dt.month<=8)][:,0,21,:,:] #200-700hPa温度
t200 = t200.groupby('time.year').mean(dim='time')
t=scipy.signal.detrend(t200, axis=0, type='linear', bp=0, overwrite_data=True)
print(d4.variables)


hw = np.load("/Users/fuzhenghang/Documents/python/Pakistan/npydata/R_hw_days_1979-2022_monthly_anomaly.npy")
for i in range(44):
    hw[i]=sum(hw[i*3+1:i*3+3])/2
scipy.signal.detrend(hw[0:44], axis=0, type='linear', bp=0, overwrite_data=True)


u200h = np.zeros((len(high),181,360))
u200l = np.zeros((len(low),181,360))
v200h = np.zeros((len(high),181,360))
v200l = np.zeros((len(low),181,360))
u850h = np.zeros((len(high),181,360))
u850l = np.zeros((len(low),181,360))
v850h = np.zeros((len(high),181,360))
v850l = np.zeros((len(low),181,360))
z200h = np.zeros((len(high),181,360))
z200l = np.zeros((len(low),181,360))
z500h = np.zeros((len(high),181,360))
z500l = np.zeros((len(low),181,360))
th = np.zeros((len(high),181,360))
tl = np.zeros((len(low),181,360))

r=0
print(u200h[0].shape)
for i in high:
    u200h[r] = u200[i]
    v200h[r] = v200[i]
    u850h[r] = u850[i]
    v850h[r] = v850[i]
    z200h[r] = z200[i]
    z500h[r] = z500[i]
    th[r] = t[i]
    r+=1
r=0
for i in low:
    u200l[r] = u200[i]
    v200l[r] = v200[i]
    u850l[r] = u850[i]
    v850l[r] = v850[i]
    z200l[r] = z200[i]
    z500l[r] = z500[i]
    tl[r] = t[i]
    r+=1

_,pu200 = ttest_ind(u200h,u200l,equal_var=False)
_,pv200 = ttest_ind(v200h,v200l,equal_var=False)
_,pu850 = ttest_ind(u850h,u850l,equal_var=False)
_,pv850 = ttest_ind(v850h,v850l,equal_var=False)
_,pz200 = ttest_ind(z200h,z200l,equal_var=False)
_,pz500 = ttest_ind(z500h,z500l,equal_var=False)
_,pt = ttest_ind(th,tl,equal_var=False)


u200h = np.mean(u200h,axis=0)
v200h = np.mean(v200h,axis=0)
u850h = np.mean(u850h,axis=0)
v850h = np.mean(v850h,axis=0)
z200h = np.mean(z200h,axis=0)
z500h = np.mean(z500h,axis=0)
th = np.mean(th,axis=0)
u200l = np.mean(u200l,axis=0)
v200l = np.mean(v200l,axis=0)
u850l = np.mean(u850l,axis=0)
v850l = np.mean(v850l,axis=0)
z200l = np.mean(z200l,axis=0)
z500l = np.mean(z500l,axis=0)
tl = np.mean(tl,axis=0)
# In[1]

proj = ccrs.PlateCarree()  #中国为左
fig = plt.figure(figsize=(12,6),dpi=1000)
ax=[]
x1 = [0,0,0]
yy = [0.99,0.76,0.53]
dx = 0.32
dy = 0.21
loc = [[0.326, 1.02, 0.007, 0.15],[0.326, 0.79, 0.007, 0.15],[0.326, 0.56, 0.007, 0.15]]
cm = [newcmap2,newcmap2,newcmap3]
for i in range(3):
    ax.append(fig.add_axes([x1[i],yy[i],dx,dy],projection = proj))
data = [10*(z200h-z200l),10*(z500h-z500l),th-tl]
wind = [u200h-u200l,v200h-v200l,u850h-u850l,v850h-v850l]
print(wind[1].shape)
datap=[pz200,pz500,pt]
u200s = np.zeros((181,360))
v200s = np.zeros((181,360))
u850s = np.zeros((181,360))
v850s = np.zeros((181,360))
for ii in range(181):
    for jj in range(360):
        if pu200[ii,jj]<=0.05 or pv200[ii,jj]<=0.05:
            u200s[ii,jj]=wind[0][ii,jj]
            v200s[ii,jj]=wind[1][ii,jj]
            wind[0][ii,jj]=np.nan
            wind[1][ii,jj]=np.nan
        else:
            u200s[ii,jj]=np.nan
            v200s[ii,jj]=np.nan
        if pu850[ii,jj]<=0.05 or pv850[ii,jj]<=0.05:
            u850s[ii,jj]=wind[2][ii,jj]
            v850s[ii,jj]=wind[3][ii,jj]
            wind[2][ii,jj]=np.nan
            wind[3][ii,jj]=np.nan
        else:
            u850s[ii,jj]=np.nan
            v850s[ii,jj]=np.nan
        #if abs(data[2][ii,jj])<0.01:
            #data[2][ii,jj] = np.nan
winds = [u200s,v200s,u850s,v850s]
sca=[150,50]
labe=['8 m/s','3 m/s']
uu=[8,3]
"""
i=1
ax.append(fig.add_axes([x1[i],yy[i],dx,dy]))
"""
tit = ['(a) Diff JA H200 & UV200','(b) Diff JA H500 & UV850','(c) Diff 850-200-hPa Temp','(d) Diff JA H500 & UV850']
ti = [10*np.arange(-6,6.00001,3),10*np.arange(-2,2.00001,1),np.arange(-2,2.00001,1)]
for i in range(3):
    leftlon, rightlon, lowerlat, upperlat = (30,160,20,60)
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax[i].set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax[i].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.3,color='k',zorder=2)
    
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
    if  i <=1:
        gl.bottom_labels  = False
    
    levels = [10*np.arange(-6,6.01,1),10*np.arange(-3,3.01,0.5),np.arange(-2.1,2.101,0.3)]
    cb1=ax[i].contourf(lon,lat,data[i], levels=levels[i],cmap=cm[i] ,transform=ccrs.PlateCarree(),extend='both',zorder=0)
    position2=fig.add_axes(loc[i])#位置[左,下,长度,宽度]
    cbar1=plt.colorbar(cb1,cax=position2,orientation='vertical',ticks=ti[i],
                     aspect=20,shrink=0.2,pad=0.06)#方向 
    cbar1.ax.tick_params(length=1.8,width=0.4,pad=1.5)
    
    
    ax[i].text(30,61,tit[i],fontweight='bold',fontsize=6)
    if i<3:
        for ii in range(30,71,2):
            for jj in range(30,160,2):
                if datap[i][ii,jj]<=0.05:
                    ax[i].text(jj,90-ii,'.')
    if i<2:    
        cq = ax[i].quiver(lon[::3],lat[::3],wind[(i)*2][::3,::3],wind[(i)*2+1][::3,::3],color='gray',
                          transform=ccrs.PlateCarree(),scale=sca[i],width=0.003,edgecolor='w',linewidth=0.18)   
        ax[i].quiverkey(cq, X=0.87, Y = 1.046, U=uu[i] ,angle = 0,label=labe[i],labelpos='E', color = 'k',labelcolor = 'k')
        ax[i].quiver(lon[::3],lat[::3],winds[(i)*2][::3,::3],winds[(i)*2+1][::3,::3],color='k',
                          transform=ccrs.PlateCarree(),scale=sca[i],width=0.003,edgecolor='w',linewidth=0.18)
d1 = xr.open_dataset(r'/Users/fuzhenghang/Documents/ERA5/uvw_5monthly9_1979_2022.nc',use_cftime=True)                
time = d1['time'][:]
u850 = d1['u'][:,0,14,:,:][(time.dt.month>=7)&(time.dt.month<=8)]
u850 = u850.groupby('time.year').mean(dim='time')
u850c = np.mean(u850,axis=0)
CS=ax[2].contour(lon,lat,u850c,[5,15,25],linewidths=0.92,alpha=0.4,colors=['k'])
ax[2].clabel(CS, inline=1, fontsize=6.5)
plt.show()







