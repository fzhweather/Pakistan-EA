#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 16:20:37 2023

@author: fuzhenghang
"""

# In[0]
import xarray as xr
import numpy as np
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
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.linewidth"] = 1.2
la1 = 34
la2 = 26
lo1 = 60
lo2 = 80


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


fwd = xr.open_dataset('/Users/fuzhenghang/Documents/ERA5/uvw_5monthly9_1979_2022.nc')
lats = fwd['latitude']
lons = fwd['longitude']
u_300 = fwd['u'].loc[fwd.time.dt.month.isin([7,8])].loc['1979-01-01':'2022-12-01',1,200,:,:]
u_300 = np.array(u_300).reshape((44,2,181,360)).mean((1))
v_300 = fwd['v'].loc[fwd.time.dt.month.isin([7,8])].loc['1979-01-01':'2022-12-01',1,200,:,:]
v_300 = np.array(v_300).reshape((44,2,181,360)).mean((1))

# The standard interface requires that latitude and longitude be the leading
# dimensions of the input wind components, and that wind components must be
# either 2D or 3D arrays. The data read in is 3D and has latitude and
# longitude as the last dimensions. The bundled tools can make the process of
# re-shaping the data a lot easier to manage.
uwnd, uwnd_info = prep_data(u_300, 'tyx')
vwnd, vwnd_info = prep_data(v_300, 'tyx')

# It is also required that the latitude dimension is north-to-south. Again the
# bundled tools make this easy.
lats, uwnd, vwnd = order_latdim(lats, uwnd, vwnd)

# Create a VectorWind instance to handle the computation of streamfunction and
# velocity potential.
w = VectorWind(uwnd, vwnd)

# Compute the streamfunction and velocity potential. Also use the bundled
# tools to re-shape the outputs to the 4D shape of the wind components as they
# were read off files.
sf, vp = w.sfvp()
sf = recover_data(sf, uwnd_info)
vp = recover_data(vp, uwnd_info)

vpcli = vp.mean(0)
vp = vp - vpcli

u300chi, v300chi = w.irrotationalcomponent()

u300chi = recover_data(u300chi, uwnd_info)
v300chi = recover_data(v300chi, uwnd_info)

u300cli = u300chi.mean(0)
v300cli = v300chi.mean(0)
u300chi = u300chi-u300cli
v300chi = v300chi-v300cli
#print(v300chi.shape)
divwd = np.zeros((2,44,181,360))
divwd[0] = u300chi
divwd[1] = v300chi

scipy.signal.detrend(vp, axis=0, type='linear', bp=0, overwrite_data=True)
scipy.signal.detrend(divwd[0], axis=0, type='linear', bp=0, overwrite_data=True)
scipy.signal.detrend(divwd[1], axis=0, type='linear', bp=0, overwrite_data=True)   

divu, divv = w.irrotationalcomponent()
div = w.divergence()
avrt = w.absolutevorticity()
avrt_zonal, avrt_meridional = w.gradient(avrt)
rbws2 = -(divu*avrt_zonal+divv*avrt_meridional)
rbws1 = -avrt*div
RWS = rbws1+rbws2

# In[1]
f_z = xr.open_dataset('/Users/fuzhenghang/Documents/ERA5/temp_geo_5monthly9_1979_2022.nc')
f_z1 = xr.open_dataset('/Users/fuzhenghang/Documents/ERA5/uvw_5monthly9_1979_2022.nc')
#print(f_z1.variables.keys())

z_ver1 = f_z['t'].loc[f_z.time.dt.month.isin([7,8])].loc['1979-01-01':'2022-12-01',1,100:1000,la1:la2,:]
z_ver1 = np.array(z_ver1).mean((2)).reshape((44,2,-1,360)).mean((1))
print(z_ver1.shape)
theta = np.zeros((44,27,360))
p = [100,  125,
        150,  175,  200,  225,  250,  300,  350,  400,  450,  500,  550,  600,
        650,  700,  750,  775,  800,  825,  850,  875,  900,  925,  950,  975,
       1000]
for y in range(44):
   for i in range(27):
          for k in range(360):
              theta[y,i,k]=ww(z_ver1[y,i,k],p[i])
    
z_ver1 = z_ver1 - z_ver1.mean((0))
scipy.signal.detrend(z_ver1, axis=0, type='linear', bp=0, overwrite_data=True)

# In[2]

z_ver2 = f_z['t'].loc[f_z.time.dt.month.isin([7,8])].loc['1979-01-01':'2022-12-01',1,100:1000,:,lo1:lo2]    
z_ver2 = np.array(z_ver2).mean((3)).reshape((44,2,-1,181)).mean((1))

thetala = np.zeros((44,27,181))
p = [100,  125,
        150,  175,  200,  225,  250,  300,  350,  400,  450,  500,  550,  600,
        650,  700,  750,  775,  800,  825,  850,  875,  900,  925,  950,  975,
       1000]
for y in range(44):
   for i in range(27):
          for k in range(181):
              thetala[y,i,k]=ww(z_ver2[y,i,k],p[i])
z_ver2 = z_ver2 - z_ver2.mean((0))
scipy.signal.detrend(z_ver2, axis=0, type='linear', bp=0, overwrite_data=True)
z_ver3 = f_z1['u'].loc[f_z1.time.dt.month.isin([7,8])].loc['1979-01-01':'2022-12-01',1,100:1000,la1:la2,:]
z_ver3 = np.array(z_ver3).mean((2)).reshape((44,2,-1,360)).mean((1))
z_ver3 = z_ver3 - z_ver3.mean((0))
scipy.signal.detrend(z_ver3, axis=0, type='linear', bp=0, overwrite_data=True)

z_ver4 = f_z1['v'].loc[f_z1.time.dt.month.isin([7,8])].loc['1979-01-01':'2022-12-01',1,100:1000,la1:la2,:]
z_ver4 = np.array(z_ver4).mean((2)).reshape((44,2,-1,360)).mean((1))
z_ver4 = z_ver4 - z_ver4.mean((0))
scipy.signal.detrend(z_ver4, axis=0, type='linear', bp=0, overwrite_data=True)

z_ver5 = f_z1['w'].loc[f_z1.time.dt.month.isin([7,8])].loc['1979-01-01':'2022-12-01',1,100:1000,la1:la2,:]
z_ver5 = np.array(z_ver5).mean((2)).reshape((44,2,-1,360)).mean((1))
z_ver5 = z_ver5 - z_ver5.mean((0))
scipy.signal.detrend(z_ver5, axis=0, type='linear', bp=0, overwrite_data=True)

z_ver6 = f_z1['u'].loc[f_z1.time.dt.month.isin([7,8])].loc['1979-01-01':'2022-12-01',1,100:1000,:,lo1:lo2]    
z_ver6 = np.array(z_ver6).mean((3)).reshape((44,2,-1,181)).mean((1))
z_ver6 = z_ver6 - z_ver6.mean((0))
scipy.signal.detrend(z_ver6, axis=0, type='linear', bp=0, overwrite_data=True)

z_ver7 = f_z1['v'].loc[f_z1.time.dt.month.isin([7,8])].loc['1979-01-01':'2022-12-01',1,100:1000,:,lo1:lo2]    
z_ver7 = np.array(z_ver7).mean((3)).reshape((44,2,-1,181)).mean((1))
z_ver7 = z_ver7 - z_ver7.mean((0))
scipy.signal.detrend(z_ver7, axis=0, type='linear', bp=0, overwrite_data=True)

z_ver8 = f_z1['w'].loc[f_z1.time.dt.month.isin([7,8])].loc['1979-01-01':'2022-12-01',1,100:1000,:,lo1:lo2]    
z_ver8 = np.array(z_ver8).mean((3)).reshape((44,2,-1,181)).mean((1))
z_ver8 = z_ver8 - z_ver8.mean((0))
scipy.signal.detrend(z_ver8, axis=0, type='linear', bp=0, overwrite_data=True)

z_ver = [z_ver1,z_ver3,z_ver4,z_ver5,z_ver2,z_ver6,z_ver7,z_ver8]
for i in range(8):
    print(z_ver[i].shape)
z_verh = np.zeros((4,len(high),27,360))
z_verl = np.zeros((4,len(low),27,360))

z_verh1 = np.zeros((4,len(high),27,181))
z_verl1 = np.zeros((4,len(low),27,181))
        
vph = np.zeros((len(high),181,360))
vpl = np.zeros((len(low),181,360))
div0h = np.zeros((len(high),181,360))
div0l = np.zeros((len(low),181,360))
div1h = np.zeros((len(high),181,360))
div1l = np.zeros((len(low),181,360))


rwsh = np.zeros((len(high),181,360))
rwsl = np.zeros((len(low),181,360))

r=0
for i in high:
    vph[r] = vp[i]
    div0h[r] = divwd[0,i]
    div1h[r] = divwd[1,i]
    rwsh[r] = RWS[:,:,i]
    for j in range(4):
        z_verh[j,r] = z_ver[j][i]
        z_verh1[j,r] = z_ver[j+4][i]
    r+=1
r=0
for i in low:
    vpl[r] = vp[i]
    div0l[r] = divwd[0,i]
    div1l[r] = divwd[1,i]
    rwsl[r] = RWS[:,:,i]
    for j in range(4):
        z_verl[j,r] = z_ver[j][i]
        z_verl1[j,r] = z_ver[j+4][i]
    r+=1

_,prws = ttest_ind(rwsh,rwsl,equal_var=False)
_,pvp = ttest_ind(vph,vpl,equal_var=False)
_,pdiv0 = ttest_ind(div0h,div0l,equal_var=False)
_,pdiv1 = ttest_ind(div1h,div1l,equal_var=False)
pz_ver = np.zeros((4,27,360))
pz_ver1 = np.zeros((4,27,181))
for j in range(4):
    _,pz_ver[j] = ttest_ind(z_verh[j], z_verl[j],equal_var=False)
    _,pz_ver1[j] = ttest_ind(z_verh1[j], z_verl1[j],equal_var=False)

rwsh = np.mean(rwsh,axis=0)
vph = np.mean(vph,axis=0)
div0h = np.mean(div0h,axis=0)
div1h = np.mean(div1h,axis=0)
z_verh = np.mean(z_verh,axis=1)
z_verh1 = np.mean(z_verh1,axis=1)

rwsl = np.mean(rwsl,axis=0)
vpl = np.mean(vpl,axis=0)
div0l = np.mean(div0l,axis=0)  
div1l = np.mean(div1l,axis=0) 
z_verl = np.mean(z_verl,axis=1)
z_verl1 = np.mean(z_verl1,axis=1)


# In[3]

year = np.arange(1979,2022.1,1)
level = f_z['level'].loc[100:1000]
lat = f_z['latitude']
lon = f_z['longitude']

data = [vph-vpl,z_verh[0]-z_verl[0],z_verh1[0]-z_verl1[0]]
wind = [div0h-div0l,div1h-div1l,z_verh[1]-z_verl[1],z_verh[3]-z_verl[3],z_verh1[2]-z_verl1[2],z_verh1[3]-z_verl1[3]]

drws = 1e11*(rwsh-rwsl)

datap=[pvp,pz_ver[0],pz_ver1[0]]

u200s = np.zeros((181,360))
v200s = np.zeros((181,360))
u850s = np.zeros((27,360))
v850s = np.zeros((27,360))
u850s1 = np.zeros((27,181))
v850s1 = np.zeros((27,181))
drwss = np.zeros((181,360))
for ii in range(181):
    for jj in range(360):
        if prws[ii,jj]>=0.05:
            drwss[ii,jj]=np.nan
        else:
            drwss[ii,jj]=drws[ii,jj]
        if pdiv0[ii,jj]>=0.05 and pdiv1[ii,jj]>=0.05:
            u200s[ii,jj]=wind[0][ii,jj]
            v200s[ii,jj]=wind[1][ii,jj]
            wind[0][ii,jj]=np.nan
            wind[1][ii,jj]=np.nan
        else:
            u200s[ii,jj]=np.nan
            v200s[ii,jj]=np.nan
for ii in range(27):
    for jj in range(360):        
        if pz_ver[1,ii,jj]>=0.05 and pz_ver[3,ii,jj]>=0.05:
            u850s[ii,jj]=wind[2][ii,jj]
            v850s[ii,jj]=wind[3][ii,jj]
            wind[2][ii,jj]=np.nan
            wind[3][ii,jj]=np.nan
        else:
            u850s[ii,jj]=np.nan
            v850s[ii,jj]=np.nan
    for jj in range(181): 
        if pz_ver1[2,ii,jj]>=0.05 and pz_ver1[3,ii,jj]>=0.05:
            u850s1[ii,jj]=wind[4][ii,jj]
            v850s1[ii,jj]=wind[5][ii,jj]
            wind[4][ii,jj]=np.nan
            wind[5][ii,jj]=np.nan
        else:
            u850s1[ii,jj]=np.nan
            v850s1[ii,jj]=np.nan
winds = [u200s,v200s,u850s,v850s,u850s1,v850s1]

print(wind[0].shape)       
fig = plt.figure(figsize=(10,6),dpi=600)

ax=[]
x1 = [0,0,0.45,0,0.45]
yy = [1,0.56,0.56,0,0]
dx = [0.6,0.4,0.15,0.4,0.15]
dy = 0.4
xla=['Longitude','Latitude']
x=[lon,lat]
#loc = [[0.52, 1.13, 0.015, 0.35],[1.12, 1.13, 0.015, 0.35],[0.52, 0.65, 0.015, 0.35],[1.12, 0.65, 0.015, 0.35]]
proj = ccrs.PlateCarree()  #中国为左
ax.append(fig.add_axes([x1[0],yy[0],dx[0],dy],projection = proj))
for i in range(1,3):
    ax.append(fig.add_axes([x1[i],yy[i],dx[i],dy]))
for i in range(1,3):
    ax[i].set_yscale('symlog')
    ax[i].set_yticks([1000, 500,300, 200, 100])
    ax[i].set_yticklabels(['1000','500','300','200','100'])
    ax[i].invert_yaxis()
    if i == 1:
        cq=ax[i].quiver(lon[::4],level,wind[2][:,::4],wind[3][:,::4]*(-100),scale=120,width=0.0048,edgecolor='w',linewidth=0.3)
        cq=ax[i].quiver(lon[::4],level,winds[2][:,::4],winds[3][:,::4]*(-100),color='gray',scale=120,width=0.0048,linewidth=0.3)
        ax[i].quiverkey(cq, X=0.87, Y = 1.046, U = 10,angle = 0,label='10m/s',labelpos='E', color = 'k',labelcolor = 'k')
        ax[i].set_ylabel('Level (hPa)',fontsize=10)
    if i == 2:
        cq=ax[i].quiver(lat[::4],level,wind[4][:,::4],wind[5][:,::4]*(-100),scale=20,width=0.012,edgecolor='w',linewidth=0.3)
        cq=ax[i].quiver(lat[::4],level,winds[4][:,::4],winds[5][:,::4]*(-100),color='gray',scale=20,width=0.012,linewidth=0.3)
        ax[i].quiverkey(cq, X=0.8, Y = 1.046, U = 3,angle = 0,label='3m/s',labelpos='E', color = 'k',labelcolor = 'k')
    ax[i].set_xlabel(xla[i-1],fontsize=10)
    c = ax[i].contourf(x[i-1],level ,data[i],levels=np.arange(-2,2.001,0.2), extend = 'both',zorder=0, cmap=plt.cm.bwr)
    print(data[1].shape)


ax[1].xaxis.set_major_formatter(cticker.LongitudeFormatter())  
ax[2].xaxis.set_major_formatter(cticker.LatitudeFormatter()) 
ax[0].set_title('(a) 200hPa Rossby wave source & div. wind',loc='left',fontsize=10)
ax[1].set_title('(b) lev-lon',loc='left',fontsize=10)
ax[2].set_title('(c) lev-lat',loc='left',fontsize=10)
ax[1].set_xlim(40,140)
ax[2].set_xlim(20,50)
ax[2].set_yticklabels([])
cbar=plt.colorbar(c,cax=fig.add_axes([0.61, 0.62, 0.012, 0.3]),orientation='vertical',ticks=[-2,-1,0,1,2],
                 aspect=20,shrink=0.2,pad=0.06)#方向 
cbar.set_label('K',labelpad=1)
#地形填充
f_oro = xr.open_dataset('/Users/fuzhenghang/Documents/大四上/热浪/ERA5/geo.nc')
oro_ver1 = np.array(f_oro['z'].loc[:,la1:la2,:]).mean((1))[0]/9.8
oro_ver1 = 1013*(1-6.5/288000*oro_ver1)**5.255
ax[1].fill_between(lon, oro_ver1, 1000, where=oro_ver1 < 1000, facecolor='darkgrey',zorder=5)
oro_ver2 = np.array(f_oro['z'].loc[:,:,lo1:lo2]).mean((2))[0]/9.8
oro_ver2 = 1013*(1-6.5/288000*oro_ver2)**5.255
lat_oro = f_oro.latitude 
ax[2].fill_between(lat, oro_ver2, 1000, where=oro_ver2 < 1000, facecolor='darkgrey',zorder=5)  

i=0
leftlon, rightlon, lowerlat, upperlat = (0,180,0,60)
lon_formatter = cticker.LongitudeFormatter()
lat_formatter = cticker.LatitudeFormatter()
ax[i].set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
ax[i].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')
shp_path1 = r'/Users/fuzhenghang/Documents/python/tibetan/tibetan.shp'
reader = Reader(shp_path1)
tpfeat=cfeature.ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), edgecolor='grey', facecolor='none')
ax[i].add_feature(tpfeat, linewidth=0.4,edgecolor='k')
gl=ax[i].gridlines(draw_labels=True, linewidth=0, color='k', alpha=0.5, linestyle='--')
gl.xlocator = mticker.FixedLocator(np.arange(0,181,30))
gl.ylocator = mticker.FixedLocator(np.arange(0,80,20))
gl.ypadding=15
gl.xpadding=15
gl.top_labels    = False    
gl.right_labels  = False
gl.bottom_labels  = True
levels = np.arange(-40,40+0.00001,8.0000)
cb=ax[i].contourf(lon,lat,drws, levels=levels,cmap=newcmap2,transform=ccrs.PlateCarree(),extend='both',zorder=0)
cq=ax[i].quiver(lon[::4],lat[::4],wind[0][::4,::4],wind[1][::4,::4],scale=30,width=0.003,edgecolor='w',linewidth=0.3)
#cq=ax[i].quiver(lon[::4],lat[::4],winds[0][::4,::4],winds[1][::4,::4],color='gray',scale=30,width=0.003,linewidth=0.3)
ax[i].quiverkey(cq, X=0.9, Y = 1.046, U = 1,angle = 0,label='1m/s',labelpos='E', color = 'k',labelcolor = 'k')
position1=fig.add_axes([0.61, 1.05, 0.012, 0.3])#位置[左,下,长度,宽度]
cbar=plt.colorbar(cb,cax=position1,orientation='vertical',ticks=[-32,-16,0,16,32],
                aspect=20,shrink=0.2,pad=0.06)#方向 
cbar.set_label('x$10^-$$^1$$^1/s$',labelpad=1)
cbar.ax.tick_params(length=1.8,width=0.4,pad=1.5)


theta1 = np.mean(theta,axis=0)
CS=ax[1].contour(lon,level,theta1,[310+5*i for i in range(15)],linewidths=0.92,alpha=0.4,colors=['k'])
ax[1].clabel(CS, inline=1, fontsize=6.5)

d1 = xr.open_dataset(r'/Users/fuzhenghang/Documents/ERA5/uvw_5monthly9_1979_2022.nc',use_cftime=True)                
time = d1['time'][:]
u850 = d1['u'][:,0,14,:,:][(time.dt.month>=7)&(time.dt.month<=8)]
u850 = u850.groupby('time.year').mean(dim='time')
u850c = np.mean(u850,axis=0)
CS=ax[0].contour(lon,lat,u850c,[10,20,30],linewidths=1,alpha=0.4,colors=['k'],zorder=1)
ax[0].clabel(CS, inline=1, fontsize=6.5)
#theta2 = np.mean(thetala,axis=0)
#CS=ax[2].contour(lat,level,theta2,[310+5*i for i in range(15)],linewidths=0.92,alpha=0.4,colors=['k'])
#ax[2].clabel(CS, inline=1, fontsize=6.5)


h = [100,  125,150,  175,  200,  225,  250,  300,  350,  400,  450,  500,  550,  600,650,  700,  750,  775,  800,  825,  850,  875,  900,  925,  950,  975,
       1000]
"""
for i in range(27):
    for j in range(20,51):
        if pz_ver1[0,i,j]>=0.05:
            ax[2].text(90-j,h[i],'.',color='slategray',fontsize=12)
    for j in range(40,140):
        if pz_ver[0,i,j]>=0.05:
            ax[1].text(j,h[i],'.',color='slategray',fontsize=12)

for i in range(30,91,2):
    for j in range(0,180),2:
        if pvp[i,j]>=0.05:
            ax[0].text(j,90-i,'.',color='slategray',fontsize=12)

"""



