# -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta
from numpy import *
import pandas as pd
import xarray as xr

import geo_tools as gt

def load_ecmwf(data_dir,filename,datetime_range=None,lat_range=None,lon_range=None,
               export_to_dir=None,export_filename=None,export_chunks=True,verbose=False,super_verbose=False):
    """ Opens ERA-Interim or ERA-40 reanalysis data files downloaded in netCDF format with a custom grid.

    Secondary use:
        Run this routine on newly downloaded files to calculate derived quantities and de-accumulate forecasts. Use
        argument <<export_to_dir>> to export new version, then manually delete original.

    Args:
        data_dir: directory of data file
        filename: filename including extension
        datetime_range: None or [Datetime0,Datetime1] or [Datestring0,Datestring1] to subset fields
            note: to slice with open right end, e.g., use [Datetime0,None]
            note: selection is generous, so ['2016-1-1','2016-1-1'] will include all hours on January 1, 2016
            note: example of Datestring: '2016-1-1-h12'
        lat_range: None or [lat_N,lat_S] to subset fields (!!! - descending order - !!!)
        lon_range: None or [lon_W,lon_E] to subset fields
        export_to_dir: None or directory to export new netCDF containing derived quantities, modified variables
            note: in this case, will not return Dataset, and will ignore datetime_range when export_chunks is True
            note: this must be a *different* directory than data_dir!
        export_filename: None or new filename to use when exporting (including extension)
        export_chunks: True or False (True for first call on a large file; False when calling on chunks of that file)
        verbose: True or False
        super_verbose: True or False (print every processing time step)

    Returns:
        all_data: xarray Dataset with coordinates (time,lats,lons); examples of accessing/slicing follow:
            all_data.loc[dict(time='2016-1-1')]                            to extract without slicing
            all_data.sel(lats=slice(-60,-70))                              to slice all variables
            all_data['skt'].values                                         to convert to eager NumPy array
            all_data['skt'][0,:,:]                                         to slice data using indices (t=0)
            all_data['skt'].loc['2016-1-1':'2016-2-1',-60:-70,0:10]        to slice data using values
            all_data['skt']['latitude']                                    to get view of 1-D coordinate
            all_data['skt']['time']                                        NumPy Datetime coordinate
            all_data['skt']['doy']                                         fractional day-of-year coordinate
            pd.to_datetime(all_data['skt']['time'].values)                 useable Datetime version of the above
            all_data['skt'].attrs['units']
            all_data['skt'].attrs['long_name']

    Note: as shown above, 'doy' (fractional day-of-year) is included as a secondary coordinate with dimension 'time'.

    The following derived quantities are calculated here:
        'curlt': wind stress curl using 'iews' and 'inss'
        'div': divergence of 10-m wind using 'u10' and 'v10'
        'div_ice': estimated sea-ice divergence given 30% turning to left and 2% scaling of u10 and v10
        'q2m': 2-m specific humidity from 'msl' and 'd2m'
        'si10': 10-m wind speed from 'u10' and 'v10' (evaluated lazily using Dask only if export_to_dir is None)

    Note: this is an implementation of ldp.load_ecmwf_deprecated() that uses xarray to more efficiently access files.

    Saved data files:
        'erai_monthly_mean_weddell.nc':    ERA-Interim, Jan 1979 - Dec 2017, monthly mean of daily mean
                                           grid: 0.75° x 0.75°, area: 40.0°S 90.0°W 90.0°S 90.0°E
                                     vars: msl - Mean sea level pressure (Pa) –> (hPa)
                                           sp - Surface pressure (Pa) –> (hPa)
                                           sst - Sea surface temperature (K) –> (°C)
                                           skt - Skin temperature (K) –> (°C)
                                           t2m - Temperature at 2 meters (K) –> (°C)
                                           u10, v10 - U, V wind components at 10 m (m/s)
                                           si10 - wind speed at 10 m (m/s)
        'erai_monthly_mean_weddell_forecast.nc':
                                           ERA-Interim, Jan 1979 - Dec 2017, monthly mean of daily mean
                                           grid: 0.75° x 0.75°, area: 40.0°S 90.0°W 90.0°S 90.0°E
                                     vars: iews, inss - U, V instantaneous turbulent surface stress (N/m^2)
        'erai_daily_weddell.nc':           ERA-Interim, 1979-01-01 - 2017-12-31, daily, step=0 (analysis),
                                             times 0:00, 6:00, 12:00, 18:00
                                           grid: 0.75° x 0.75°, area: 40.0°S 90.0°W 90.0°S 90.0°E
                                     vars: msl - Mean sea level pressure (Pa) –> (hPa)
                                           sst - Sea surface temperature (K) –> (°C)
                                           skt - Skin temperature (K) –> (°C)
                                           t2m - Temperature at 2 meters (K) –> (°C)
                                           u10, v10 - U, V wind components at 10 m (m/s)
                                           d2m - Dewpoint temperature at 2 meters (K)
                                           q2m - Specific humidity at 2 meters, calculated here from msl and 2d (kg/kg)
        'erai_daily_weddell_forecast.nc':  ERA-Interim, 1979-01-01 - 2017-12-31, daily, steps = 6, 12 (forecast)
                                             times 0:00 and 12:00
                                           grid: 0.75° x 0.75°, area: 40.0°S 90.0°W 90.0°S 90.0°E
                                     vars: iews, inss - U, V instantaneous turbulent surface stress (N/m^2)
                                           tp - Total precipitation (m) –> Precipitation rate (m/s)
                                           sf - Snowfall (m water equivalent) –> Snowfall rate (m/s)
                                           e - Evaporation (m) –> Evaporation rate (m/s), positive for evap to atmos
                                           sshf - Surface sensible heat flux (J/m^2) -> (W/m^2)
                                           slhf - Surface latent heat flux (J/m^2) -> (W/m^2)
                                           ssr - Surface net solar radiation (shortwave) (J/m^2) -> (W/m^2)
                                           str - Surface net thermal radiation (longwave) (J/m^2) -> (W/m^2)
                                           strd - Surface thermal radiation (longwave) downwards (J/m^2) –> (W/m^2)
    """

    # export mode may require splitting numerical processing into chunks
    max_chunk = 0.5  # in GB, maximum file size to attempt to process in memory
    if export_to_dir is not None and export_chunks:
        file_size = os.path.getsize(data_dir + filename)/10e8  # file size in GB
        if file_size > max_chunk:
            num_chunks = int(ceil(file_size/max_chunk))
            all_data = xr.open_dataset(data_dir + filename)
            num_times = all_data.dims['time']
            times_per_chunk = int(ceil(num_times/num_chunks))
            all_times = all_data['time'].values
            all_data.close()

            # process and save data in chunks
            slice_start_indices = arange(0,num_times,times_per_chunk)
            just_filename,just_extension = os.path.splitext(filename)
            for chunk_counter, start_idx in enumerate(slice_start_indices):
                end_idx = start_idx + times_per_chunk - 1
                if end_idx >= len(all_times): end_idx = -1  # for final chunk, use last time index
                dt_range = [str(all_times[start_idx]),str(all_times[end_idx])]
                if verbose: print('>> Processing chunk {0} of {1} from {2} to {3}'
                                  ''.format(chunk_counter+1,len(slice_start_indices),*dt_range))
                load_ecmwf(data_dir,filename,datetime_range=dt_range,lat_range=lat_range,lon_range=lon_range,
                           export_filename='{0}_chunk_{1:03d}{2}'.format(just_filename,chunk_counter+1,just_extension),
                           export_to_dir=export_to_dir,export_chunks=False,verbose=verbose)

            # open all chunks and concatenate as Dataset
            if verbose: print('>> Opening all chunks of {0}'.format(filename))
            all_data = xr.open_mfdataset(export_to_dir + '{0}_chunk_*{1}'.format(just_filename,just_extension),
                                         concat_dim='time',autoclose=True,chunks={'time':100})
            bypass_normal_open = True
        else:
            bypass_normal_open = False
    else:
        bypass_normal_open = False

    if not bypass_normal_open:
        if verbose: print('>> Opening {0}'.format(filename))
        all_data = xr.open_dataset(data_dir+filename,autoclose=True,chunks={'time':100})   # O(100 MB) per chunk

    if 'longitude' in all_data and 'latitude' in all_data:
        all_data = all_data.rename({'latitude':'lats','longitude':'lons'})

    if datetime_range is not None:
        all_data = all_data.sel(time=slice(datetime_range[0],datetime_range[1]))
    if lat_range is not None:
        all_data = all_data.sel(lats=slice(lat_range[0],lat_range[1]))
    if lon_range is not None:
        all_data = all_data.sel(lons=slice(lon_range[0],lon_range[1]))

    for var in all_data.data_vars:
        if verbose: print('>>>> Examining variable {0}'.format(var))

        if all_data[var].attrs['units'] == 'Pa':
            orig_name = all_data[var].attrs['long_name']
            all_data[var] /= 100.0
            all_data[var].attrs = {'units':'hPa','long_name':orig_name}
        elif all_data[var].attrs['units'] == 'K' and var != 'd2m':
            orig_name = all_data[var].attrs['long_name']
            all_data[var] -= 273.15
            all_data[var].attrs = {'units':'°C','long_name':orig_name}

        # de-accumulate forecast fields (hours 0 and 12), if not already
        if var in ['tp','e','sf','sshf','slhf','ssr','str','strd'] and 'deaccumulated' not in all_data[var].attrs:
            orig_name = all_data[var].attrs['long_name']
            orig_units = all_data[var].attrs['units']
            time_index = pd.to_datetime(all_data[var].time.values)

            if time_index[0].hour == 0 or time_index[0].hour == 12:
                all_data[var][dict(time=0)] /= 2
                first_step = 1
            else:
                first_step = 0
            if time_index[-1].hour == 6 or time_index[-1].hour == 18:
                last_step = len(time_index) - 1
            else:
                last_step = len(time_index)

            all_data[var].load() # load Dask array into memory (which means reasonably small chunks are necessary!)
            all_data[var][first_step+1:last_step:2] -= all_data[var][first_step:last_step:2].values

            seconds_in_6_hours = 6 * 60 * 60
            all_data[var] /= seconds_in_6_hours

            if var == 'e': all_data[var] *= -1

            all_data[var].attrs['long_name'] = orig_name
            all_data[var].attrs['units'] = orig_units

            if   all_data[var].attrs['units'] == 'm':       all_data[var].attrs['units'] = 'm/s'
            elif all_data[var].attrs['units'] == 'J m**-2': all_data[var].attrs['units'] = 'W/m^2'

            all_data[var].attrs['deaccumulated'] = 'True'

    # calculate 2-m specific humidity from surface pressure and dewpoint temperature, if available
    # uses Equations 7.4 and 7.5 on p. 92 of ECMWF IFS Documentation, Ch. 7:
    #   https://www.ecmwf.int/sites/default/files/elibrary/2015/9211-part-iv-physical-processes.pdf
    if 'q2m' not in all_data and 'd2m' in all_data and 'msl' in all_data:
        if verbose: print('>>>> Calculating 2-m specific humidity')

        # constants for Teten's formula for saturation water vapor pressure over water [not ice] (Eq. 7.5)
        # origin: Buck (1981)
        a1 = 611.21 # Pa
        a3 = 17.502 # unitless
        a4 = 32.19  # K
        T_0 = 273.16 # K

        # saturation water vapor pressure; units: Pa
        e_sat_at_Td = a1 * exp(a3 * (all_data['d2m'] - T_0) / (all_data['d2m'] - a4))

        # saturation specific humidity at dewpoint temperature (Eq. 7.4)
        # note conversion of surface pressure from hPa back to Pa
        R_dry_over_R_vap = 0.621981  # gas constant for dry air over gas constant for water vapor, p. 110
        q_sat_at_Td = R_dry_over_R_vap * e_sat_at_Td / (100*all_data['msl'] - (e_sat_at_Td*(1.0 - R_dry_over_R_vap)))

        all_data['q2m'] = q_sat_at_Td
        all_data['q2m'].attrs['units'] = 'kg/kg'
        all_data['q2m'].attrs['long_name'] = 'Specific humidity at 2 m'

    # calculate 10-m wind speed from u, v
    # note: this evaluates lazily using Dask, so expect processing hangs upon computation (instead of load)
    # note: included only if not exporting to a new netCDF file (don't want to take up unnecessary space)
    if 'si10' not in all_data and 'u10' in all_data and 'v10' in all_data and export_to_dir is None:
        if verbose: print('>>>> Calculating 10-m wind speed')
        all_data['si10'] = (all_data['u10']**2 + all_data['v10']**2)**0.5
        all_data['si10'].attrs['units'] = 'm/s'
        all_data['si10'].attrs['long_name'] = '10 metre wind speed'

    # calculate estimated sea-ice drift velocity using 10-m wind u, v
    # assume 2% drift velocity scaling and turning angle of 30° to left of winds
    #  (Wang et al. 2014, Scientific Reports, "Cyclone-induced rapid creation of extreme Antarctic sea ice conditions")
    # note: these vectors are used to compute estimated sea-ice divergence, and are deleted afterwards
    if 'ui10' not in all_data and 'vi10' not in all_data and 'u10' in all_data and 'v10' in all_data:
        if verbose: print('>>>> Calculating estimated sea-ice drift velocity')
        scaling = 0.02
        turning_angle = 30.0   # positive for counter-clockwise (to left of wind)
        turning_angle_radians = turning_angle / 180.0 * pi
        transform = cos(turning_angle_radians) + sin(turning_angle_radians) * 1j
        rotated_u_v = (all_data['u10'] + all_data['v10'] * 1j) * transform
        all_data['ui10'] = rotated_u_v.real * scaling
        all_data['ui10'].attrs['units'] = 'm/s'
        all_data['ui10'].attrs['long_name'] = 'Estimated sea-ice drift velocity, eastward component'
        all_data['vi10'] = rotated_u_v.imag * scaling
        all_data['vi10'].attrs['units'] = 'm/s'
        all_data['vi10'].attrs['long_name'] = 'Estimated sea-ice drift velocity, northward component'

    # value-added calculations
    def field_derivs(for_dx,for_dy):
        data_shape = for_dy.shape
        ddx = zeros(data_shape)
        ddy = zeros(data_shape)
        lat_spacing = gt.distance_between_two_coors(for_dy['lats'][0],for_dy['lons'][0],
                                                    for_dy['lats'][1],for_dy['lons'][0]) * -1
        lon_spacing = array([gt.distance_between_two_coors(for_dy['lats'][lat_idx],for_dy['lons'][0],
                                                           for_dy['lats'][lat_idx],for_dy['lons'][1])
                             for lat_idx in range(len(for_dy['lats']))])
        nonzero_mask = lon_spacing > 0   # to deal with poles (90 and -90), where dx is zero
        for dt_idx in range(len(for_dy['time'])):
            if super_verbose: print('>>>>>> time {0} of {1}'.format(dt_idx+1,len(for_dy['time'])))
            ddy[dt_idx,:,:] = gradient(for_dy[dt_idx,:,:],lat_spacing,axis=0)
            ddx[dt_idx,nonzero_mask,:] \
                = gradient(for_dx[dt_idx,nonzero_mask,:],1.0,axis=1) / lon_spacing[nonzero_mask,None]
            ddx[dt_idx,~nonzero_mask,:] = NaN   # to deal with poles (90 and -90), where dx is zero
        return ddx, ddy

    # calculate wind stress curl (d(tau_y)/dx - d(tau_x)/dy)
    if 'curlt' not in all_data and 'iews' in all_data and 'inss' in all_data:
        if verbose: print('>>>> Calculating wind stress curl')

        ddx, ddy = field_derivs(for_dx=all_data['inss'],for_dy=all_data['iews'])
        all_data['curlt'] = all_data['iews'].copy()
        all_data['curlt'].values = (ddx - ddy) * 10**7
        all_data['curlt'].attrs['units'] = r'10$^{-7}$ N m$^{-3}$'
        all_data['curlt'].attrs['long_name'] = 'Wind stress curl'

    # calculate 10-m wind divergence (d(u10)/dx + d(v10)/dy)
    if 'div' not in all_data and 'u10' in all_data and 'v10' in all_data:
        if verbose: print('>>>> Calculating 10-m wind divergence')

        ddx,ddy = field_derivs(for_dx=all_data['u10'],for_dy=all_data['v10'])
        all_data['div'] = all_data['u10'].copy()
        all_data['div'].values = (ddx + ddy) * 10**5
        all_data['div'].attrs['units'] = r'10$^{-5}$ s$^{-1}$'
        all_data['div'].attrs['long_name'] = '10-m wind divergence'

    # calculate estimated sea-ice divergence (d(ui10)/dx + d(vi10)/dy)
    if 'div_ice' not in all_data and 'ui10' in all_data and 'vi10' in all_data:
        if verbose: print('>>>> Calculating estimated sea-ice divergence')

        ddx,ddy = field_derivs(for_dx=all_data['ui10'],for_dy=all_data['vi10'])
        all_data['div_ice'] = all_data['ui10'].copy()
        all_data['div_ice'].values = (ddx + ddy) * 10**5
        all_data['div_ice'].attrs['units'] = r'10$^{-5}$ s$^{-1}$'
        all_data['div_ice'].attrs['long_name'] = 'Estimated sea-ice divergence'

        # delete sea-ice drift data
        del all_data['ui10']
        del all_data['vi10']

    # add day-of-year as a secondary coordinate with dimension 'time'
    if 'doy' not in all_data.coords:
        datetime_index = pd.to_datetime(all_data['time'].values)
        doy_index = datetime_index.dayofyear + datetime_index.hour / 24. + datetime_index.minute / 60.
        all_data.coords['doy'] = ('time',doy_index)

    if export_to_dir is not None:
        # set encoding only if exporting to a new netCDF file here (!)
        # remember to do this if exporting to a new netCDF file elsewhere...
        #
        # changing encoding (scale factor and offset) is necessary because the original netCDF file's encoding
        #   results in truncation/loss of precision when applied to the processed variables here (some of which
        #   where divided by large numbers, for instance)
        # these formulae for optimal scale factors and offsets are from:
        #   http://james.hiebert.name/blog/work/2015/04/18/NetCDF-Scale-Factors/
        for var in all_data.data_vars:
            n_bits = 16  # because int16
            var_max = asscalar(all_data[var].max().values)  # .values necessary because of lazy Dask evaluation
            var_min = asscalar(all_data[var].min().values)
            all_data[var].encoding['dtype'] = 'int16'
            all_data[var].encoding['scale_factor'] = (var_max - var_min) / ((2**n_bits) - 1)
            all_data[var].encoding['add_offset'] = var_min + (2**(n_bits - 1) * all_data[var].encoding['scale_factor'])
            all_data[var].encoding['_FillValue'] = -9999

        if export_filename is None: new_filename = filename
        else:                       new_filename = export_filename
        all_data.to_netcdf(export_to_dir + new_filename)
        all_data.close()
    else:
        return all_data