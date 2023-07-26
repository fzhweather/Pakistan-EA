#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:17:01 2023

@author: fuzhenghang
"""


# In[0]
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr  
import numpy as np
from scipy.stats import pearsonr
import scipy
from sklearn.linear_model import LinearRegression


mpl.rcParams["font.family"] = 'cm' 
mpl.rcParams["mathtext.fontset"] = 'cm'  # 数学文字字体
mpl.rcParams["font.size"] = 8
mpl.rcParams["axes.linewidth"] = 1
plt.rcParams['ytick.direction'] = 'out'

#easterly index & temp grad
f_z = xr.open_dataset('/Users/fuzhenghang/Documents/ERA5/temp_geo_5monthly9_1979_2022.nc')
f_z1 = xr.open_dataset('/Users/fuzhenghang/Documents/ERA5/uvw_5monthly9_1979_2022.nc')

z_ver1 = f_z1['u'].loc[f_z1.time.dt.month.isin([7,8])].loc['1979-01-01':'2022-12-01',1,200,33:27,105:122]
z_ver1 = np.array(z_ver1).mean((1)).reshape((44,2,18)).mean((1,2))
scipy.signal.detrend(z_ver1, axis=0, type='linear', bp=0, overwrite_data=True)


z_ver2 = f_z['t'].loc[f_z.time.dt.month.isin([7,8])].loc['1979-01-01':'2022-12-01',1,200:850,34:26,104:123]
z_ver2 = np.array(z_ver2).mean((1)).reshape((44,2,9,20)).mean((1))
scipy.signal.detrend(z_ver2, axis=0, type='linear', bp=0, overwrite_data=True)
tempgd = np.zeros((44,7,18))
for i in range(18):
    for j in range(7):
        tempgd[:,j,i]=(z_ver2[:,j,i]-z_ver2[:,j+2,i])/(2*110940)      
tgd = np.mean(tempgd,axis=(1,2))


hw = np.load("/Users/fuzhenghang/Documents/大四上/热浪/中间数据/R_hw_days_1979-2022_monthly_anomaly.npy")
for i in range(44):
    hw[i]=sum(hw[i*3+1:i*3+3])/2
scipy.signal.detrend(hw[0:44], axis=0, type='linear', bp=0, overwrite_data=True)
hwindex = hw[0:44,57:64,105:122]
hwindex = np.mean(hwindex,axis=(1,2))


z_ver3 = f_z['t'].loc[f_z.time.dt.month.isin([7,8])].loc['1979-01-01':'2022-12-01',1,1000,34:26,104:123]
z_ver3 = np.array(z_ver3).reshape((44,2,9,20)).mean((1,2,3))
scipy.signal.detrend(z_ver3, axis=0, type='linear', bp=0, overwrite_data=True)

d2 = xr.open_dataset(r'/Users/fuzhenghang/Documents/ERA5/monthly_t2m_1979_2022.nc',use_cftime=True)
time = d2['time'][:]
t2m = d2['t2m'][:][(time.dt.month>=7)&(time.dt.month<=8)]
t2m = t2m.groupby('time.year').mean(dim='time')
t2m = np.array(t2m)
t2m = t2m[:,57:64,105:123]
t2m = np.mean(t2m,axis=(1,2))
#t2m=t2m-np.mean(t2m)
scipy.signal.detrend(t2m, axis=0, type='linear', bp=0, overwrite_data=True)

const1,p1 = pearsonr(z_ver1, t2m)
const2,p2 = pearsonr(tgd, t2m)
print(const1,const2)
const1,p1 = pearsonr(z_ver1, hwindex)
const2,p2 = pearsonr(tgd, hwindex)
print(const1,const2)
const1,p1 = pearsonr(z_ver1, z_ver3)
const2,p2 = pearsonr(tgd, z_ver3)
#print(const1,const2)


# In[1]
z_ver3 = f_z1['u'].loc[f_z1.time.dt.month.isin([7,8])].loc['1979-01-01':'2022-12-01',1,200,33:24,67:78]
z_ver3 = np.array(z_ver3).mean((1)).reshape((44,2,12)).mean((1,2))
scipy.signal.detrend(z_ver3, axis=0, type='linear', bp=0, overwrite_data=True)

z_ver4 = f_z['t'].loc[f_z.time.dt.month.isin([7,8])].loc['1979-01-01':'2022-12-01',1,200:850,34:23,66:79]
z_ver4 = np.array(z_ver4).mean((1)).reshape((44,2,12,14)).mean((1))
scipy.signal.detrend(z_ver4, axis=0, type='linear', bp=0, overwrite_data=True)
tempgd1 = np.zeros((44,7,12))
for i in range(12):
    for j in range(7):
        tempgd1[:,j,i]=(z_ver4[:,j,i]-z_ver4[:,j+2,i])/(2*110940)      
tgd1 = np.mean(tempgd1,axis=(1,2))


d3 = xr.open_dataset(r'/Users/fuzhenghang/Documents/ERA5/cru/cru_1901-20221deg.nc',use_cftime=True)
time = d3['time'][:]
pre = d3['pre'][(time.dt.month>=7)&(time.dt.month<=8)&(time.dt.year>=1979)&(time.dt.year<=2022)][:,:,:]
pre = pre.groupby('time.year').mean(dim='time')
pre = np.array(pre)
prep = pre[:,114:124,67:79]
prep = np.nanmean(prep,axis=(1,2))
scipy.signal.detrend(prep, axis=0, type='linear', bp=0, overwrite_data=True)


const1,p1 = pearsonr(z_ver3, prep)
const2,p2 = pearsonr(tgd1, prep)
print(const1,const2)
# In[2]
fig = plt.figure(figsize=(8,8),dpi=1000)
ax=[]
x1 = [0,0.4,0,0.4]
yy = [1,1,0.62,0.62]
dx = 0.32
dy = 0.3
xl = ['YRV-t2m, K','YRV-t2m, K','PNWI-precip., mm/month','PNWI-precip., mm/month']
yl = ['YRV-zonal wind, m/s','YRV-dT/dy, ×10$^6$K/m','PNWI-zonal wind, m/s','PNWI-dT/dy, ×10$^6$K/m']
for i in range(4):
    ax.append(fig.add_axes([x1[i],yy[i],dx,dy]))
tit=['a      ','b      ','c     ','d      ']
x = [t2m,t2m,prep,prep]
y = [z_ver1,1000000*tgd,z_ver3,1000000*tgd1]
co = ['tomato','tomato','royalblue','royalblue']
cor = ['r','r','b','b']
words = ['r = -0.87**','r = 0.89**','r = -0.73**','r = 0.76**']
eq = ['y = -4.60x','y = 0.74x','y = -1.32x','y = 0.28x']
xloc = [0.7,-1.6,35,-75]
xt = [-2.9,-2.9,-132,-132]
yt = [9.45,2.1,6.3,2.1]
for i in range(4):
    ax[i].text(xt[i],yt[i],tit[i],fontweight='bold',fontsize=12)
    ax[i].text(xloc[i],yt[i]/1.5,words[i],fontsize=9)
    ax[i].text(xloc[i],yt[i]/2,eq[i],fontsize=9)
    ax[i].scatter(x[i],y[i],s=18,color=co[i],alpha=0.66,linewidth=0)
    regressor = LinearRegression()
    regressor = regressor.fit(np.reshape(x[i],(-1, 1)),np.reshape(y[i],(-1, 1)))
    print(regressor.coef_)
    ax[i].plot(np.reshape(x[i],(-1,1)), regressor.predict(np.reshape(x[i],(-1,1))),color=cor[i],linewidth=1.5)
    ax[i].grid('--',linewidth=0.3,alpha=0.5)
    ax[i].set_xlabel(xl[i])
    ax[i].set_ylabel(yl[i])
    ax[i].set_xlim(-2.2,2.2)

ax[0].set_ylim(-9,9)
ax[0].set_yticks([-9,-6,-3,0,3,6,9])
ax[1].set_ylim(-2,2)
ax[1].set_yticks([-2,-1,0,1,2])
ax[2].set_ylim(-6,6)
ax[2].set_yticks([-6,-3,0,3,6])
ax[3].set_ylim(-2,2)
ax[3].set_yticks([-2,-1,0,1,2])
ax[2].set_xlim(-100,100)
ax[3].set_xlim(-100,100)
    







