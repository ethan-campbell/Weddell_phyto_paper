# -*- coding: utf-8 -*-

# external imports
from numpy import *
from scipy import stats
from sklearn.neighbors import KernelDensity
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle

# internal imports
import load_product as ldp
import geo_tools as gt
import time_tools as tt

# root directory for data files
data_dir = '/Users/Ethan/Documents/Research/2016-08 - UW/Data/'

# sub-directories for data files
argo_gdac_dir = data_dir + 'Argo/'
soccom_dir = argo_gdac_dir + 'SOCCOM/'
era_custom_dir = data_dir + 'Reanalysis/ECMWF_Weddell_processed/'

# sub-directories for existing serialized ("pickled") processed data
figure_pickle_dir = data_dir + 'Processed_pickle_archives/'
argo_index_pickle_dir = argo_gdac_dir + 'Argo_index_pickles/'

# output directory
output_dir = '/Users/Ethan/Documents/Research/2016-08 - UW/Results/2019_10_10_storm_impact/'


######### parameters #########

use_float_prof_pairs_pickle = True
use_storm_pickle = True  # requires serialized data created by <<use_float_prof_pairs_pickle>>
verbose = True

float_lon_bounds = [-60,30]
float_lat_bounds = [-90,-60]
float_toi = [datetime(2002,1,1),datetime(2017,12,31)]  # end date should be last day of ERA-Interim fields
min_cast_min_depth = 20.0   # require cast to start above 20 m
min_cast_max_depth = 200.0  # require cast to end below 200 m
reject_mld = 500.0          # reject casts with MLD deeper than 500 m
prof_min_interval = timedelta(days=2)
prof_max_interval = timedelta(days=11)
prof_max_distance = 100     # km

storm_pres_threshold = 960  # hPa
storm_ws_threshold = 20     # m/s

# box side length in (deg lon, deg lat) ... equivalent to ~100 km search radius
storm_search_box_dim = (3,2)


######### aggregate float data #########

if not use_float_prof_pairs_pickle:
    float_counter = 0
    float_prof_pairs = []
    argo_gdac_index = pickle.load(open(argo_index_pickle_dir + 'argo_gdac_index.pickle','rb'))
    argo_soccom_index = pickle.load(open(argo_index_pickle_dir + 'argo_soccom_index.pickle','rb'))
    toi_int = [tt.convert_datetime_to_14(float_toi[0]),tt.convert_datetime_to_14(float_toi[1])]
    for wmoid in argo_gdac_index['wmoids']:
        this_float_meta = ldp.argo_gdac_float_meta(argo_gdac_index['local_prof_index'],wmoid)
        toi_match = logical_and(this_float_meta['prof_datetimes'] >= toi_int[0],
                                this_float_meta['prof_datetimes'] <= toi_int[1])
        lon_match = logical_and(this_float_meta['prof_lons'] >= float_lon_bounds[0],
                                this_float_meta['prof_lons'] <= float_lon_bounds[1])
        lat_match = logical_and(this_float_meta['prof_lats'] >= float_lat_bounds[0],
                                this_float_meta['prof_lats'] <= float_lat_bounds[1])
        prof_match = logical_and(logical_and(toi_match,lon_match),lat_match)
        prof_nums_match = array(this_float_meta['prof_nums'])[prof_match]
        if sum(prof_match) < 2: continue
        if verbose: print(wmoid)
        float_data = ldp.argo_float_data(wmoid,argo_gdac_dir,argo_gdac_index,argo_soccom_index,
                                         prof_nums=prof_nums_match,compute_extras=False)

        # from all profiles with good data, calculate MLD and ML averages
        processed_profs = []
        for pidx in range(len(float_data['profiles'])):
            if float_data['profiles'][pidx]['psal']['depth'][0] > min_cast_min_depth \
                    or float_data['profiles'][pidx]['psal']['depth'][-1] < min_cast_max_depth:
                if verbose:
                    print('>>> rejected: ',float_data['profiles'][pidx]['psal']['depth'][0],
                          float_data['profiles'][pidx]['psal']['depth'][-1])
                continue

            prof_datetime \
                = tt.convert_tuple_to_datetime(tt.convert_18_to_tuple(float_data['profiles'][pidx]['datetime']))

            prof_position_flag = float_data['profiles'][pidx]['position_flag']  # 1 is good, 2 is under ice, 9 is bad
            if prof_position_flag == 9: continue

            mld = gt.mld(float_data['profiles'][pidx],bottom_return='NaN',verbose_warn=verbose)
            if mld is None: continue
            if isnan(mld): continue
            if mld > reject_mld: continue
            if verbose: print(mld,prof_datetime)

            ptmp_ml = gt.vert_prof_eval(float_data['profiles'][pidx],'ptmp',(0.0,mld),z_coor='depth',extrap='nearest')
            psal_ml = gt.vert_prof_eval(float_data['profiles'][pidx],'psal',(0.0,mld),z_coor='depth',extrap='nearest')
            if ptmp_ml is None or psal_ml is None: continue
            if isnan(ptmp_ml) or isnan(psal_ml): continue

            processed_prof = dict()
            processed_prof['datetime'] = prof_datetime
            processed_prof['doy'] = prof_datetime.timetuple().tm_yday
            processed_prof['lat'] = float_data['profiles'][pidx]['lat']
            processed_prof['lon'] = float_data['profiles'][pidx]['lon']
            processed_prof['position_flag'] = prof_position_flag
            processed_prof['mld'] = mld
            processed_prof['ptmp_ml'] = ptmp_ml
            processed_prof['psal_ml'] = psal_ml
            processed_profs.append(processed_prof)

        # pair up profiles matching criteria
        if len(processed_profs) < 2: continue
        this_float_prof_pairs = []
        for pidx in range(len(processed_profs) - 1):  # to second-to-last profile
            # time interval check
            prof_interval = processed_profs[pidx + 1]['datetime'] - processed_profs[pidx]['datetime']
            if prof_interval > prof_max_interval or prof_interval < prof_min_interval: continue

            # distance check
            prof_distance = gt.distance_between_two_coors(processed_profs[pidx]['lat'],processed_profs[pidx]['lon'],
                                                          processed_profs[pidx + 1]['lat'],
                                                          processed_profs[pidx + 1]['lon'])
            prof_distance = prof_distance / 1000  # meters to km

            # determine under-ice status (if either profile assumed under ice, count pair as under ice)
            if processed_profs[pidx]['position_flag'] == 2 or processed_profs[pidx + 1]['position_flag'] == 2:
                under_ice = True   # under ice
            else:
                under_ice = False  # not under ice

            # calculate average profile location crudely, by averaging lat/lons
            avg_lat = mean([processed_profs[pidx]['lat'],processed_profs[pidx + 1]['lat']])
            avg_lon = mean([processed_profs[pidx]['lon'],processed_profs[pidx + 1]['lon']])

            # calculate average DOY (important note: this can exceed 365 if profiles span December to January)
            if processed_profs[pidx + 1]['doy'] - processed_profs[pidx]['doy'] > 0:
                avg_doy = mean([processed_profs[pidx]['doy'],processed_profs[pidx + 1]['doy']])
            else:
                avg_doy = mean([processed_profs[pidx]['doy'],365 + processed_profs[pidx + 1]['doy']])

            # save data
            prof_pair = dict()
            prof_pair['doy'] = avg_doy
            prof_pair['lat'] = avg_lat
            prof_pair['lon'] = avg_lon
            prof_pair['under_ice'] = under_ice
            prof_pair['datetime'] = (processed_profs[pidx]['datetime'],processed_profs[pidx + 1]['datetime'])
            prof_pair['mld'] = (processed_profs[pidx]['mld'],processed_profs[pidx + 1]['mld'])
            prof_pair['ptmp_ml'] = (processed_profs[pidx]['ptmp_ml'],processed_profs[pidx + 1]['ptmp_ml'])
            prof_pair['psal_ml'] = (processed_profs[pidx]['psal_ml'],processed_profs[pidx + 1]['psal_ml'])
            this_float_prof_pairs.append(prof_pair)
        float_prof_pairs.extend(this_float_prof_pairs)
        float_counter += 1
    pickle.dump(float_prof_pairs,open(output_dir + 'float_prof_pairs','wb'))
    text_file = open(current_results_dir + 'float_prof_pair_info.txt','w')
    text_file.write('{0} profile pairs found from {1} floats'.format(len(float_prof_pairs),float_counter))
    text_file.close()
else:
    float_prof_pairs = pickle.load(open(output_dir + 'float_prof_pairs','rb'))


######### co-locate with storms #########

if use_storm_pickle:
    float_prof_pair_storms = pickle.load(open(output_dir + 'float_prof_pair_storms','rb'))
else:
    print('Loading daily ERA-Interim fields...')
    erai_daily = ldp.load_ecmwf(era_custom_dir,'erai_daily_weddell.nc')
    print('... fields loaded.')

    datetimes = erai_daily['msl']['time']
    datetimes = pd.Series(datetimes).dt.to_pydatetime()   # conversion from NumPy datetime64 to native Datetimes

    lons = erai_daily['lons'].values
    lats = erai_daily['lats'].values
    lon_grid, lat_grid = meshgrid(erai_daily['lons'],erai_daily['lats'])

    for prof_pair_idx,prof_pair in enumerate(float_prof_pairs):

        if prof_pair_idx == 19:
            pause_here = True

        if verbose: print('>>> prof pair {0} of {1}'.format(prof_pair_idx,len(float_prof_pairs)))
        datetimes_between = logical_and(datetimes > prof_pair['datetime'][0],datetimes < prof_pair['datetime'][1])
        datetime_mask = where(datetimes_between)[0]
        doi = datetimes[datetimes_between]

        search_box = [prof_pair['lon'] - 0.5 * storm_search_box_dim[0],prof_pair['lon'] + 0.5 * storm_search_box_dim[0],
                      prof_pair['lat'] - 0.5 * storm_search_box_dim[1],prof_pair['lat'] + 0.5 * storm_search_box_dim[1]]
        lon_mask = where(logical_and(lons >= search_box[0],lons <= search_box[1]))[0]
        lat_mask = where(logical_and(lats >= search_box[2],lats <= search_box[3]))[0]
        
        search_lon_grid = lon_grid[:,lon_mask][lat_mask,:]
        search_lat_grid = lat_grid[:,lon_mask][lat_mask,:]

        search_fields_msl = erai_daily['msl'][datetime_mask][:,lat_mask,:][:,:,lon_mask].values
        search_fields_U10 = erai_daily['si10'][datetime_mask][:,lat_mask,:][:,:,lon_mask].values
        search_field_lats = search_lat_grid.flatten()
        search_field_lons = search_lon_grid.flatten()

        # for each time step, search for storms
        prof_pair['periods_with_storms'] = 0  # number of 6-hourly snapshots with a storm identified using thresholds
        prof_pair['all_min_pres'] = []        # pressure minima for 6-hourly snapshots
        prof_pair['all_mean_pres'] = []       # average pressures for 6-hourly snapshots
        prof_pair['all_max_ws'] = []          # wind speed maxima for 6-hourly snapshots
        prof_pair['all_mean_ws'] = []         # average wind speeds for 6-hourly snapshots
        for t in range(len(datetime_mask)):
            search_field_msl_flat = search_fields_msl[t,:,:].flatten()
            min_pres_index = nanargmin(search_field_msl_flat)
            min_pres = search_field_msl_flat[min_pres_index]
            min_pres_lat = search_field_lats[min_pres_index]
            min_pres_lon = search_field_lons[min_pres_index]
            mean_pres = mean(search_field_msl_flat)
            
            search_field_U10_flat = search_fields_U10[t,:,:].flatten()
            max_ws_index = nanargmax(search_field_U10_flat)
            max_ws = search_field_U10_flat[max_ws_index]
            mean_ws = mean(search_field_U10_flat)
            
            ##### old xarray implementation (way too slow; deprecates some code above if implemented):
            #
            # search_field_msl = erai_daily['msl'].sel(lats=slice(search_box[3],search_box[2]),
            #                                          lons=slice(search_box[0],search_box[1]),time=dt)
            # search_field_U10 = erai_daily['si10'].sel(lats=slice(search_box[3],search_box[2]),
            #                                           lons=slice(search_box[0],search_box[1]),time=dt)
            #
            # min_pres = float(search_field_msl.min(axis=(0,1)).values)
            # min_pres_lat = search_field_msl.where(search_field_msl == min_pres,drop=True).lats.values[0]
            # min_pres_lon = search_field_msl.where(search_field_msl == min_pres,drop=True).lons.values[0]
            #
            # max_ws = float(search_field_U10.max(axis=(0,1)).values)
            # max_ws_lat = search_field_U10.where(search_field_U10 == max_ws,drop=True).lats.values[0]
            # max_ws_lon = search_field_U10.where(search_field_U10 == max_ws,drop=True).lons.values[0]
            
            if min_pres < storm_pres_threshold and max_ws > storm_ws_threshold:
                prof_pair['periods_with_storms'] += 1
            prof_pair['all_min_pres'].append(min_pres)
            prof_pair['all_mean_pres'].append(mean_pres)
            prof_pair['all_max_ws'].append(max_ws)
            prof_pair['all_mean_ws'].append(mean_ws)
        if verbose: print(prof_pair['periods_with_storms'],prof_pair['all_min_pres'],prof_pair['all_max_ws'])

    float_prof_pair_storms = float_prof_pairs
    pickle.dump(float_prof_pair_storms,open(output_dir + 'float_prof_pair_storms','wb'))

mld_change = []
ptmp_ml_change = []
psal_ml_change = []
under_ice = []
doys = []
num_periods_with_storms = []
min_min_pres = []
mean_mean_pres = []
max_max_ws = []
mean_mean_ws = []
mean_highest_max_ws = []
for prof_pair in float_prof_pair_storms:
    mld_change.append(prof_pair['mld'][1] - prof_pair['mld'][0])
    ptmp_ml_change.append(prof_pair['ptmp_ml'][1] - prof_pair['ptmp_ml'][0])
    psal_ml_change.append(prof_pair['psal_ml'][1] - prof_pair['psal_ml'][0])
    under_ice.append(prof_pair['under_ice'])
    doys.append(prof_pair['doy'])
    num_periods_with_storms.append(prof_pair['periods_with_storms'])
    min_min_pres.append(min(prof_pair['all_min_pres']))
    mean_mean_pres.append(mean(prof_pair['all_min_pres']))
    max_max_ws.append(max(prof_pair['all_max_ws']))
    mean_mean_ws.append(max(prof_pair['all_mean_ws']))
    mean_highest_max_ws.append(mean(sort(prof_pair['all_max_ws'])[-8:]))
        
mld_change = array(mld_change); ptmp_ml_change = array(ptmp_ml_change); psal_ml_change = array(psal_ml_change)
under_ice = array(under_ice); doys = array(doys); num_periods_with_storms = array(num_periods_with_storms)
min_min_pres = array(min_min_pres); mean_mean_pres = array(mean_mean_pres)
max_max_ws = array(max_max_ws); mean_mean_ws = array(mean_mean_ws); mean_highest_max_ws = array(mean_highest_max_ws)


######### some exploratory plots #########

plt.figure()
plt.scatter(min_min_pres[under_ice],mld_change[under_ice],c='k',s=2)
plt.xlabel('Minimum pressure between float profile pairs (hPa)')
plt.ylabel('MLD change (m)')
plt.savefig(output_dir + 'pres_min_vs_mld_change.pdf')

plt.figure()
plt.scatter(mean_mean_pres[under_ice],mld_change[under_ice],c='k',s=2)
plt.xlabel('Average pressure between float profile pairs (hPa)')
plt.ylabel('MLD change (m)')
plt.savefig(output_dir + 'pres_mean_vs_mld_change.pdf')

plt.figure()
plt.scatter(max_max_ws[under_ice],mld_change[under_ice],c='k',s=2)
plt.xlabel('Maximum wind speed between float profile pairs (m/s)')
plt.ylabel('MLD change (m)')
plt.savefig(output_dir + 'ws_max_vs_mld_change.pdf')

plt.figure()
plt.scatter(mean_mean_ws[under_ice],mld_change[under_ice],c='k',s=2)
plt.xlabel('Average wind speed between float profile pairs (m/s)')
plt.ylabel('MLD change (m)')
plt.savefig(output_dir + 'ws_mean_vs_mld_change.pdf')

plt.figure()
plt.scatter(mean_highest_max_ws[under_ice],mld_change[under_ice],c='k',s=2)
plt.xlabel('Average of highest 48 hours of wind speeds between float profile pairs (m/s)')
plt.ylabel('MLD change (m)')
plt.savefig(output_dir + 'ws_highest_vs_mld_change.pdf')

plt.figure()
mld_axis = arange(-50,50,1)
kde_without_storms \
    = KernelDensity(kernel='gaussian',bandwidth=5.0)\
    .fit(mld_change[logical_and(under_ice,num_periods_with_storms == 0)].reshape(-1,1))
log_dens_without_storms = kde_without_storms.score_samples(mld_axis.reshape(-1,1))
kde_with_storms \
    = KernelDensity(kernel='gaussian',bandwidth=5.0)\
    .fit(mld_change[logical_and(under_ice,num_periods_with_storms > 0)].reshape(-1,1))
log_dens_with_storms = kde_with_storms.score_samples(mld_axis.reshape(-1,1))
plt.fill_between(mld_axis,exp(log_dens_without_storms),color='k',alpha=0.5,label='Zero storms',zorder=2)
plt.fill_between(mld_axis,exp(log_dens_with_storms),color='r',alpha=0.5,label='One or more storms',zorder=3)
old_ylim = plt.ylim()
plt.plot([0,0],[0,old_ylim[1]],'k--',zorder=1)
plt.ylim([0,old_ylim[1]])
plt.legend(frameon=False)
plt.title('Storms between profile pairs identified\nusing thresholds: ≥ 20 m/s and ≤ 960 hPa')
plt.xlabel('MLD change (m)')
plt.ylabel('Kernel density estimate')
plt.savefig(output_dir + 'mld_change_storm_vs_no_storm.pdf')

plt.figure()
mld_axis = arange(-50,50,1)
kde_0 \
    = KernelDensity(kernel='gaussian',bandwidth=5.0)\
    .fit(mld_change[logical_and(under_ice,mean_highest_max_ws < 15)].reshape(-1,1))
log_dens_0 = kde_0.score_samples(mld_axis.reshape(-1,1))
kde_1 \
    = KernelDensity(kernel='gaussian',bandwidth=5.0)\
    .fit(mld_change[logical_and(under_ice,mean_highest_max_ws >= 15)].reshape(-1,1))
log_dens_1 = kde_1.score_samples(mld_axis.reshape(-1,1))
kde_2 \
    = KernelDensity(kernel='gaussian',bandwidth=5.0)\
    .fit(mld_change[logical_and(under_ice,mean_highest_max_ws >= 18)].reshape(-1,1))
log_dens_2 = kde_2.score_samples(mld_axis.reshape(-1,1))
plt.fill_between(mld_axis,exp(log_dens_0),color='k',alpha=0.5,label='< 15 m/s',zorder=2)
plt.fill_between(mld_axis,exp(log_dens_1),color='orange',alpha=0.4,label='≥ 15 m/s',zorder=3)
plt.fill_between(mld_axis,exp(log_dens_2),color='red',alpha=0.3,label='≥ 18 m/s',zorder=4)
old_ylim = plt.ylim()
plt.plot([0,0],[0,old_ylim[1]],'k--',zorder=1)
plt.ylim([0,old_ylim[1]])
plt.legend(frameon=False)
plt.title('Metric: average of highest 48 hours of wind speeds')
plt.xlabel('MLD change (m)')
plt.ylabel('Kernel density estimate')
plt.savefig(output_dir + 'mld_change_by_ws.pdf')