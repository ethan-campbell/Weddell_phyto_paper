import xarray as xr
import numpy as np
import pandas as pd
import scipy.interpolate as spin
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

### read netCDF of MLD data
data = xr.open_mfdataset('/Users/Ethan/Documents/Research/Git/Miscellaneous/connor_izumi_data/'
                         'connor_izumi_jamstec_argo_ml_*.nc',
                         concat_dim='time')
lons = data['LONGITUDE'].values
lons[lons < 0] += 360
data['LONGITUDE'] = lons
data['time'] = [datetime(2017,9,4) + 0.5 * (datetime(2017,9,13) - datetime(2017,9,4)),
                datetime(2017,9,14) + 0.5 * (datetime(2017,9,23) - datetime(2017,9,14)),
                datetime(2017,9,24) + 0.5 * (datetime(2017,10,3) - datetime(2017,9,24))]
mld = data['MLD']

# interp missing data
lon_grid, lat_grid = np.meshgrid(mld['LONGITUDE'],mld['LATITUDE'])
for t in range(len(mld['time'])):
    original_shape = mld.isel(time=t).values.shape

    # transform to 1D
    x = lon_grid.ravel()
    y = lat_grid.ravel()
    z = mld.isel(time=t).values.copy().ravel()

    # delete NaNs
    x_clean = x[~np.isnan(z)]
    y_clean = y[~np.isnan(z)]
    z_clean = z[~np.isnan(z)]

    # interpolate data
    z_previously_nan = spin.griddata((x_clean,y_clean),z_clean,(x[np.isnan(z)],y[np.isnan(z)]),method='linear')

    # replace NaNs with interpolated data
    z[np.isnan(z)] = z_previously_nan

    # transform back to 2D
    z_2D = np.reshape(z,original_shape)

    # put back into xarray
    mld.load()
    mld[t] = z_2D

# read Excel sheet
cruise_data = pd.read_excel('/Users/Ethan/Documents/Research/Git/Miscellaneous/connor_izumi_cruise_track.xlsx'
                            ,header=4)
cruise_data = cruise_data.iloc[range(0,len(cruise_data)+1,2),:]

# calculate MLDs
for i in range(len(cruise_data)):
    # if np.isnan(cruise_data['Mixed Layer Depth (m)'].iloc[i]):
    dt = cruise_data['Date_Time'].iloc[i]
    lon = cruise_data['Longitude (º W)'].iloc[i]
    if lon < 0:
        lon += 360
    lat = cruise_data['Latitude (º N)'].iloc[i]
    mld_lookup = float(mld.sel(time=dt,LONGITUDE=lon,LATITUDE=lat,method='nearest'))
    cruise_data['Mixed Layer Depth (m)'].iloc[i] = mld_lookup



### interpolate wind data
data = xr.open_mfdataset('/Users/Ethan/Documents/Research/Git/Miscellaneous/connor_izumi_data/'
                         'CCMP_Wind_Analysis_*_V02.0_L3.0_RSS.nc',
                         concat_dim='time')
uwnd = data['uwnd']
vwnd = data['vwnd']
wind = (uwnd ** 2 + vwnd ** 2) ** 0.5

cruise_data['Wind Speed CCMP'] = cruise_data['Wind Speed (m/s)']

# add columns for wind speeds
for j in range(0,61):
    cruise_data.loc[:,'Wind Speed T-' + str(j)] = cruise_data.loc[:,'Wind Speed CCMP']

# calculate wind speeds
for i in range(len(cruise_data)):
    # if np.isnan(cruise_data['Mixed Layer Depth (m)'].iloc[i]):
    lon = cruise_data['Longitude (º W)'].iloc[i]
    if lon < 0:
        lon += 360
    lat = cruise_data['Latitude (º N)'].iloc[i]
    for j in range(0,61):
        print('T-{0} from '.format(j) + str(cruise_data['Date_Time'].iloc[i]))
        dt = cruise_data['Date_Time'].iloc[i] - timedelta(days=j)
        windspeed_lookup = float(wind.sel(time=dt,longitude=lon,latitude=lat,method='nearest'))
        cruise_data['Wind Speed T-' + str(j)].iloc[i] = windspeed_lookup

# delete column
cruise_data.drop(labels='Wind Speed CCMP')

# export to Excel
cruise_data.to_excel('/Users/Ethan/Desktop/connor_izumi_cruise_track_updated.xlsx')
