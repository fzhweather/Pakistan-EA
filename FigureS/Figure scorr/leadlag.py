#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 17:02:27 2023

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
import netCDF4 as nc
import dateutil.parser  #引入这个库来将日期字符串转成统一的datatime时间格式
from cartopy.util import add_cyclic_point
from cartopy.io.shapereader import Reader
from scipy.stats import pearsonr
import pandas as pd
import scipy
import seaborn as sns
sns.reset_orig()
mpl.rcParams["font.family"] = 'cm'  
mpl.rcParams["mathtext.fontset"] = 'cm' 
mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.linewidth"] = 1
plt.rcParams['hatch.color'] = 'k' 
pb,pa=scipy.signal.butter(3,2/11,'highpass')

d1 = xr.open_dataset(r'/Users/fuzhenghang/Documents/ERA5/sst/ersst5.190101_202302.nc',use_cftime=True)
print(d1.variables.keys())
lon = d1.variables['lon'][:]
lat = d1.variables['lat'][:]
time = d1['time'][:]

sst = d1['sst'][(time.dt.year>=1901)&(time.dt.year<=2021)][:,0,:,:]
sst = np.array(sst)
for i in range(89):
    for j in range(180):
        for m in range(12):
            sst[m::12,i,j]=scipy.signal.filtfilt(pb,pa,sst[m::12,i,j])

nino = np.zeros((1452))


for i in range(1452):
    co2=0
    for l in range(10):
        for m in range(50):#5N-5S, 170W-120W
            if not np.isnan(sst[i,85+l,m+190]):
                nino[i]+=sst[i,85+l,m+190]
                co2+=1
    nino[i]=nino[i]/co2
nino34 = np.zeros((1452))    
for i in range(2,1450):
    nino34[i]=np.mean(nino[i-2:i+3],axis=0)
nino34[0] = np.mean(nino[0:3],axis=0)
nino34[1] = np.mean(nino[0:4],axis=0)
nino34[1450] = np.mean(nino[1448:1451],axis=0)
nino34[1451] = np.mean(nino[1448:],axis=0)
#现在有1901年1月-10月开始到2021年12月的

d2 = xr.open_dataset("/Users/fuzhenghang/Documents/ERA5/amip/ts.ensmean.amip.1880-2014_1deg.nc")
#print(d2.variables.keys())
lons = d2['lon']
lats = d2['lat']
time1 = d2['time'][:]   
tas = d2['TS'][(time1.dt.month>=7)&(time1.dt.month<=8)&(time1.dt.year>=1901)&(time1.dt.year<=2020)]
tas = tas.groupby('time.year').mean(dim='time')

pre = np.zeros((114))
hw = np.zeros((114))
d1 = xr.open_dataset(r'/Users/fuzhenghang/Documents/ERA5/amip/olr.ensmean.amip.1880-2014_1deg.nc',use_cftime=True)

time = d1['time'][:]
tp = d1['FLNT'][(time.dt.month>=7)&(time.dt.month<=8)&(time.dt.year>=1901)&(time.dt.year<=2020)][:,:,:]
tp = -tp.groupby('time.year').mean(dim='time')



la = 114
lo = 67
lonw=12
law=10

for i in range(law):
    for j in range(lonw):
        for k in range(114):
            pre[k]+=tp[k,i+la,j+lo]
           
for i in range(8):
    for j in range(18):
        for k in range(114):   
            hw[k]+=tas[k,i+116,j+105]
pre = scipy.signal.filtfilt(pb,pa,pre)
hw = scipy.signal.filtfilt(pb,pa,hw)
pre = (pre-np.mean(pre))/np.std(pre)
hw = (hw-np.mean(hw))/np.std(hw)
index = (pre*hw-np.mean(pre*hw))/np.std(pre*hw)

hwobs = np.zeros((44))
preobs = np.zeros((122))
hw_raw= np.load("/Users/fuzhenghang/Documents/大四上/热浪/中间数据/R_hw_days_1979-2022_monthly.npy")
days_map = np.zeros((44,181,360))

for i in range(44):
    days_map[i,:,:]=np.sum(hw_raw[i*3+1:i*3+3,:,:],axis=0)

d1 = xr.open_dataset(r'/Users/fuzhenghang/Documents/ERA5/cru/cru_1901-20221deg.nc',use_cftime=True)
long = d1.variables['lon'][:]
lati = d1.variables['lat'][:]
time = d1['time'][:]
tp = d1['pre'][(time.dt.month>=7)&(time.dt.month<=8)][:,:,:]

tp = tp.groupby('time.year').mean(dim='time')
print(tp.shape)
preobs = tp[:,114:124,67:79]
hwsobs = days_map[:,57:64,105:123]
preobs = np.nanmean(preobs,axis=(1,2))
hwobs = np.nanmean(days_map,axis=(1,2))
scipy.signal.detrend(preobs, axis=0, type='linear', bp=0, overwrite_data=True)
scipy.signal.detrend(hwobs, axis=0, type='linear', bp=0, overwrite_data=True)



rnino=np.zeros((4,36))
pnino=np.zeros((4,36))
#计算nino
for i in range(36):
    rnino[2,i],pnino[2,i] = pearsonr(nino34[924+i:1345+i:12], pre[78:])
    rnino[3,i],pnino[3,i] = pearsonr(nino34[924+i:1345+i:12], hw[78:])
    rnino[0,i],pnino[0,i] = pearsonr(nino34[924+i:1345+i:12], preobs[78:114])
    rnino[1,i],pnino[1,i] = pearsonr(nino34[924+i:1345+i:12], hwobs[:36])

# In[1]
fig = plt.figure(figsize=(12,6),dpi=1000)
ax=[]
x1 = [0,0.35,0.54,0.81,0,0.27,0.54,0.81]
yy = [1,1,1,1,0.5,0.54,0.54,0.54]
dx = 0.3
dy = 0.5

corr = [rnino]
sigp = [pnino]

for i in range(1):
    ax.append(fig.add_axes([x1[i],yy[i],dx,dy]))
#title = ['Observations','CESM2-CAM6']
#title2 = ['a','b']
color1 = ['royalblue','tomato','magenta','limegreen','r']
label1=['OLR','Heatwaves','SWIO']
x = [-18+i for i in range(36)]

for i in range(1):
    ax[i].yaxis.tick_left()
    ax[i].yaxis.set_label_position("left") 
    ax[i].set_xlim(-18,17)
    ax[i].set_xticks([-18,-12,-6,0,6,12])
    ax[i].axvspan(0.5, 2.5, alpha=0.1, color='k',lw=0)
    ax[i].tick_params(length=4,width=0.7,pad=1.5)
    ax[i].axhline(y=0,  linestyle='-',linewidth = 1,color='black',alpha=1,zorder=1)
    #ax[i].fill_between(x, 0.6, -0.6, facecolor='k', alpha=0.12,zorder=0)
    #ax[1].axhline(y=-1,  linestyle='--',linewidth = 0.35,color='black',alpha=1,zorder=0)
    ax[i].grid(linestyle='--',lw = 0.6,zorder=0)
    #ax[i].text(-17.5,0.53,title[i],fontsize=12)
    #ax[i].text(-20.5,0.53,title2[i],fontweight='bold',fontsize=14)
    ax[i].set_ylim(-0.72,0.5)
    if i in [0,4]:
        ax[i].set_ylabel('Corr.',labelpad=1)
    if i not in [0,4]:
        ax[i].set_yticklabels([])
    if i <=1:
        #ax[i].axvline(x=i-1,linestyle='-',linewidth = 2.5,color='black',alpha=0.23,zorder=0)
        ax[i].set_xticklabels(['Jan(-1)','July(-1)','Jan(0)','Jul(0)','Jan(1)','Jul(1)'],fontsize=9)
        for j in range(2):
            ax[i].plot(x,corr[0][i*2+j],'-',color=color1[j],markersize=4,alpha=1,lw = 2,label=label1[j],zorder=4)
            #ax[i].set_xticklabels([])
        
            for k in range(36):
                if sigp[0][i*2+j,k]<0.05 and sigp[0][i*2+j,k]>0.01:
                    ax[i].scatter(x[k],corr[0][i*2+j,k],color=color1[j],s=20,alpha=1,zorder=5)
                elif sigp[0][i*2+j,k]<=0.01:
                    ax[i].scatter(x[k],corr[0][i*2+j,k],color=color1[j],s=30,marker='*',alpha=1,zorder=5)
   
 
ax[0].legend(frameon=False,loc='best',ncol=1,fontsize=11)   

pb,pa=scipy.signal.butter(3,2/11,'highpass')
d1 = xr.open_dataset(r'/Users/fuzhenghang/Documents/ERA5/sst/ersst5.190101_202302.nc',use_cftime=True)
lon = d1.variables['lon'][:]
lat = d1.variables['lat'][:]
time = d1['time'][:]
sst = d1['sst'][(time.dt.year>=1901)&(time.dt.year<=2021)][:,0,:,:]
sst = np.array(sst)

for i in range(89):
    for j in range(180):
        for m in range(12):
            sst[m::12,i,j]=scipy.signal.filtfilt(pb,pa,sst[m::12,i,j])

nino = np.zeros((1452))

def par(ye,x1,x2,x3):
   r_ab=pearsonr(x1,x2)[0]
   r_ac=pearsonr(x1,x3)[0]
   r_bc=pearsonr(x2,x3)[0]
   r_ab_c=(r_ab-r_ac*r_bc)/(((1-r_ac**2)**0.5)*((1-r_bc**2)**0.5))
   t = r_ab_c*(ye-1-2)**0.5/((1-r_ab_c**2)**0.5)
   return r_ab_c,t
for i in range(1452):
    co2=0
    for l in range(5):
        for m in range(25):#5N-5S, 170W-120W
            if not np.isnan(sst[i,42+l,m+5+90]):
                nino[i]+=sst[i,42+l,m+5+90]
                co2+=1
    nino[i]=nino[i]/co2
nino34 = np.zeros((1452))    
for i in range(2,1450):
    nino34[i]=np.mean(nino[i-2:i+3],axis=0)
nino34[0] = np.mean(nino[0:3],axis=0)
nino34[1] = np.mean(nino[0:4],axis=0)
nino34[1450] = np.mean(nino[1448:1451],axis=0)
nino34[1451] = np.mean(nino[1448:],axis=0)
# In[1]
nino7 = nino34[6::12]
nino8 = nino34[7::12]
nindex = (nino7+nino8)/2  
aa=79
const1,p1 = pearsonr(hw[aa:],pre[aa:])
print(const1,p1)  
const1,p1 = pearsonr(hw[aa:],nindex[aa:114])
print(const1,p1)  
const1,p1 = pearsonr(pre[aa:],nindex[aa:114])
print(const1,p1)    
const1,p1 = par(114,hw[aa:],pre[aa:],nindex[aa:114])
print(const1,p1)      

const1,p1 = par(114,hw[aa:],nindex[aa:114],pre[aa:])
print(const1,p1)      
