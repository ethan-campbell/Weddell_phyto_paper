# -*- coding: utf-8 -*-

# external imports
from numpy import *
import pandas as pd
import pandas.plotting._converter as pandacnv   # FIXME: only necessary due to Pandas 0.21.0 bug with Datetime plotting
pandacnv.register()                             # FIXME: only necessary due to Pandas 0.21.0 bug with Datetime plotting
from scipy import signal
import matplotlib.pyplot as plt

# internal imports
import load_product as ldp

# data directories
data_dir = '/Users/Ethan/Documents/Research/2016-08 - UW/Data/Argo/Drift/'
output_dir = '/Users/Ethan/Documents/Research/Git/Weddell_polynya/Temporary results/'

# load data
wmoid = 5904471
drift_data = ldp.argo_drift_data(data_dir,wmoid,status='R',bad_pres_range=(500,1500))
drift_pres = pd.Series(index=drift_data['datetime'],data=drift_data['pres'])
drift_temp = pd.Series(index=drift_data['datetime'],data=drift_data['temp'])
t = (drift_temp.index - drift_temp.index[0]).astype('timedelta64[m]') / (24 * 60)  # units: days
y = drift_temp.values.copy()

######### analysis #########

f = 60 * 60 * 24 * (2 * 7.2921e-5 * sin(2 * pi * 65 / 360) / (2 * pi))  # f at 65°

# Lomb-Scargle method
freq = linspace(0.1,2.25,100000)
pgram = signal.lombscargle(t,y,freq)
pgram_6hr = signal.lombscargle(t[::6],y[::6],freq)

# plot
plt.figure(figsize=(6,8))
plt.subplot(2,1,1)
plt.semilogy(freq,signal.medfilt(pgram,kernel_size=1001),'k',label='Hourly')
plt.semilogy(freq,signal.medfilt(pgram_6hr,kernel_size=1001),'b',label='6-hourly subsampled')
plt.xlim([0.1,2.25])
old_ylim = plt.ylim()
plt.plot([f,f],old_ylim,'k--',label=r'$f$')
plt.ylim(old_ylim)
# plt.text(f,10**-4,r'$f$',fontsize=18)
plt.legend(frameon=False)
plt.xlabel('Frequency (cpd)')
plt.title('Lomb-Scargle method')

# Welch's method
welch_freq,Pxx_den = signal.welch(y,24,nperseg=256 * 4,detrend='linear',average='median',scaling='density')
plt.semilogy(welch_freq,Pxx_den,'k',label='Hourly')
welch_freq,Pxx_den = signal.welch(y[::6],4,nperseg=256,detrend='linear',average='median',scaling='density')
plt.semilogy(welch_freq,Pxx_den,'b',label='6-hourly subsampled')

# plot
plt.subplot(2,1,2)
plt.xlim([0.1,2.25])
old_ylim = plt.ylim()
plt.plot([f,f],old_ylim,'k--',label=r'$f$')
plt.ylim(old_ylim)
plt.legend(frameon=False)
plt.xlabel('Frequency (cpd)')
plt.title('Welch\'s method')
plt.tight_layout()

plt.savefig(output_dir + '5904471_park_depth_temp_psd.pdf')