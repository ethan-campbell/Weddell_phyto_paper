# -*- coding: utf-8 -*-

# external imports
from numpy import *
import pandas as pd
import pandas.plotting._converter as pandacnv   # FIXME: only necessary due to Pandas 0.21.0 bug with Datetime plotting
pandacnv.register()                             # FIXME: only necessary due to Pandas 0.21.0 bug with Datetime plotting
from scipy import signal
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt

# internal imports
import load_product as ldp

# data directories
data_dir = '/Users/Ethan/Documents/Research/2016-08 - UW/Data/Argo/Drift/'
output_dir = '/Users/Ethan/Documents/Research/Git/Weddell_polynya/Temporary results/'

######### load APEX data #########

wmoid = 5904471
drift_data = ldp.argo_drift_data(data_dir,wmoid,status='R',bad_pres_range=(500,1500))
drift_pres = pd.Series(index=drift_data['datetime'],data=drift_data['pres'])
drift_temp = pd.Series(index=drift_data['datetime'],data=drift_data['temp'])
t = (drift_temp.index - drift_temp.index[0]).astype('timedelta64[m]') / (24 * 60)  # units: days
y = drift_temp.values.copy()

######### analyze APEX data #########

f = 60 * 60 * 24 * (2 * 7.2921e-5 * sin(2 * pi * 65 / 360) / (2 * pi))  # f at 65°

# Lomb-Scargle method
ls_freq = linspace(0.01,3.0,10000)
Pxx_den = LombScargle(t,y,normalization='psd').power(ls_freq)
Pxx_den_6hr = LombScargle(t[::6],y[::6],normalization='psd').power(ls_freq)

# plot
plt.figure(figsize=(6,8))
plt.subplot(2,1,1)
plt.semilogy(ls_freq,signal.medfilt(Pxx_den,kernel_size=501),'k',label='Hourly')
plt.semilogy(ls_freq,signal.medfilt(Pxx_den_6hr,kernel_size=501),'b',label='6-hourly subsampled')
plt.xlim([0.1,2.5])
plt.ylim([5e-7,8e-3])
old_ylim = plt.ylim()
plt.plot([f,f],old_ylim,'k--',label=r'$f$ at 65°S')
plt.ylim(old_ylim)
plt.legend(frameon=False)
plt.xlabel('Frequency (cpd)')
plt.ylabel(r'Power (°C$^2$/cpd)')
plt.title('Lomb-Scargle method')

# Welch's method
welch_freq,Pxx_den = signal.welch(y,24,nperseg=256*4,detrend='linear',average='median',scaling='density')
welch_freq_6hr,Pxx_den_6hr = signal.welch(y[::6],24/6,nperseg=256,detrend='linear',average='median',scaling='density')

# plot
plt.subplot(2,1,2)
plt.semilogy(welch_freq,Pxx_den,'k',label='Hourly')
plt.semilogy(welch_freq_6hr,Pxx_den_6hr,'b',label='6-hourly subsampled')
plt.xlim([0.1,2.5])
plt.ylim([5e-7,8e-3])
old_ylim = plt.ylim()
plt.plot([f,f],old_ylim,'k--',label=r'$f$ at 65°S')
plt.ylim(old_ylim)
plt.legend(frameon=False)
plt.xlabel('Frequency (cpd)')
plt.ylabel(r'Power (°C$^2$/cpd)')
plt.title('Welch\'s method')
plt.tight_layout()

plt.savefig(output_dir + '5904471_APEX_park_depth_temp_psd.pdf')


######### load Navis data #########

ken_data = pd.read_excel(data_dir + '5904673 (0571) - from Ken Johnson.xlsx')

t = ((pd.to_datetime(ken_data['Date']) - pd.to_datetime(ken_data['Date'])[0]).values.astype('timedelta64[m]')
     / timedelta64(1,'m')) / (24 * 60)
y = ken_data['Temp'].values

t = t[~isnan(y)]
y = y[~isnan(y)]

######### analyze Navis data #########

f = 60 * 60 * 24 * (2 * 7.2921e-5 * sin(2 * pi * 60 / 360) / (2 * pi))  # f at 60°

# Lomb-Scargle method
ls_freq = linspace(0.01,3.0,10000)
Pxx_den = LombScargle(t,y,normalization='psd').power(ls_freq)

# plot
plt.figure(figsize=(6,8))
plt.subplot(2,1,1)
plt.semilogy(ls_freq,signal.medfilt(Pxx_den,kernel_size=501),'b',label='6-hourly')
plt.xlim([0.1,2.5])
plt.ylim([1e-5,8e-3])
old_ylim = plt.ylim()
plt.plot([f,f],old_ylim,'k--',label=r'$f$ at 60°S')
plt.ylim(old_ylim)
plt.legend(frameon=False)
plt.xlabel('Frequency (cpd)')
plt.ylabel(r'Power (°C$^2$/cpd)')
plt.title('Lomb-Scargle method')

# Welch's method
welch_freq,Pxx_den = signal.welch(y,24/6,nperseg=256/4,detrend='linear',average='median',scaling='density')

# plot
plt.subplot(2,1,2)
plt.semilogy(welch_freq,Pxx_den,'b',label='6-hourly')
plt.xlim([0.1,2.5])
plt.ylim([1e-5,8e-3])
old_ylim = plt.ylim()
plt.plot([f,f],old_ylim,'k--',label=r'$f$ at 60°S')
plt.ylim(old_ylim)
plt.legend(frameon=False)
plt.xlabel('Frequency (cpd)')
plt.ylabel(r'Power (°C$^2$/cpd)')
plt.title('Welch\'s method')
plt.tight_layout()

plt.savefig(output_dir + '5904673_Navis_park_depth_temp_psd.pdf')