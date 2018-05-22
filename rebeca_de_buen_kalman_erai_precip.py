from numpy import *
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import load_product as ldp
import download_product as dlp

# instructions:
# (1) run script with download = True, analyze = False; check that file downloaded successfully
# (2) run script with download = False, analyze = True
download = False
analyze = True

# data locations
data_dir = '/Users/Ethan/Documents/Research/Git/Miscellaneous/rebeca_de_buen_kalman_data/'
stations_filename = 'ecobici_stations.csv'
erai_filename_old = 'mexico_city_erai_precip_unprocessed.nc'
erai_filename_new = 'mexico_city_erai_precip_processed.nc'
output_filename = 'erai_precip_mexico_city.csv'

# calculate average station location
stations = pd.read_csv(data_dir + stations_filename)
stations['ecobici_lon'] = stations['ecobici_lon'] + 360  # convert to °E (0-360)
mean_lat = stations['ecobici_lat'].mean()
mean_lon = stations['ecobici_lon'].mean()

# submit MARS request for ERA-Interim reanalysis fields
if download:
    # download ERAI data (note: coordinates in format 'N/W/S/E'; Mexico City at ~19°N, ~99°W)
    dlp.ecmwf(date_range='2016-01-01/to/2017-12-31',area='22/-101/17/-97',output_filename=data_dir + erai_filename_old,
              type='fc',step='3/6/9/12',time='00/12',params=['tp'])
    # process ERAI data
    ldp.load_ecmwf(data_dir,erai_filename_old,export_to_dir=data_dir,export_filename=erai_filename_new,verbose=True)

if analyze:
    # load ERAI data, do necessary conversions, extract time series of interest
    erai_daily = ldp.load_ecmwf(data_dir,erai_filename_new)
    erai_daily['time'] -= timedelta64(6,'h')  # convert UTC to local winter (non-DST) time zone, UTC-6
    assert erai_daily['tp'].units == 'm/s', 'Error: precip units must be m/s.'
    erai_daily['tp'] = erai_daily['tp'] * 60 * 60 * 1000  # convert precip units from m/s to mm/hr
    mc_precip = erai_daily['tp'].sel(lats=mean_lat,lons=mean_lon,method='nearest')
    mc_precip.load()  # load Dask array into memory
    mc_precip[mc_precip < 0] = 0  # zero out values from floating-point error
    mc_precip_series = mc_precip.to_pandas()  # flatten from xarray to Pandas Series

    # make quick plot of time series
    plt.figure(figsize=(10,5))
    plt.plot(mc_precip_series,c='darkblue')
    plt.ylim([0,plt.ylim()[1]])
    plt.xlim([mc_precip_series.index.min(),mc_precip_series.index.max()])
    plt.gcf().autofmt_xdate()
    plt.ylabel('Average precipitation rate (mm/hr)')
    plt.title('ERA-Interim 3-hourly deaccumulated precipitation for Mexico City ({0:.01f}°N, {1:.01f}°W)'
              .format(float(mc_precip['lats']),-1*(float(mc_precip['lons'])-360)))
    plt.savefig(data_dir + 'erai_precip_mexico_city.pdf')
    plt.close()

    # export data
    mc_precip_series.index.name = 'Datetime (UTC-6)'
    mc_precip_series.name = 'Average precipitation rate (mm/hr)'
    mc_precip_series.to_csv(data_dir + output_filename,index=True,header=True)