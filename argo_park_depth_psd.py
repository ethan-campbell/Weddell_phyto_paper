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
ls_freq = linspace(0.01,12.0,10000)
Pxx_den = LombScargle(t,y,normalization='psd').power(ls_freq)
ls_freq_6hr = linspace(0.01,2.5,10000)
Pxx_den_6hr = LombScargle(t[::6],y[::6],normalization='psd').power(ls_freq_6hr)

# plot
plt.figure(figsize=(6,8))
plt.subplot(2,1,1)
plt.semilogy(ls_freq,signal.medfilt(Pxx_den,kernel_size=251),'k',label='Hourly')
plt.semilogy(ls_freq_6hr,signal.medfilt(Pxx_den_6hr,kernel_size=251),'b',label='6-hourly subsampled')
plt.xlim([0.1,2.5])
plt.ylim([5e-7,8e-3])
old_ylim = plt.ylim()
plt.plot([f,f],old_ylim,'k--',label=r'$f$ at 65°S')
plt.ylim(old_ylim)
plt.legend(frameon=False)
plt.xlabel('Frequency (cpd)')
plt.ylabel(r'Power (°C$^2$/cpd)')
plt.title('5904471 (APEX) - Lomb-Scargle method')

# Welch's method
welch_freq,Pxx_den \
    = signal.welch(y,24,nperseg=256,window='hanning',detrend='linear',average='median',scaling='density')
welch_freq_6hr,Pxx_den_6hr \
    = signal.welch(y[::6],24/6,nperseg=256/4,window='hanning',detrend='linear',average='median',scaling='density')

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
plt.title('5904471 (APEX) - Welch\'s method')
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
ls_freq = linspace(0.01,2.0,10000)
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
plt.title('5904673 (Navis) - Lomb-Scargle method')

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
plt.title('5904673 (Navis) - Welch\'s method')
plt.tight_layout()

plt.savefig(output_dir + '5904673_Navis_park_depth_temp_psd.pdf')


######### tests #########

run_tests = False
if run_tests:
    t = arange(0,1526,1/24)
    y_test = 10 * sin(2*pi*1.8*t)
    y_test_noisy = y_test + (2*random.rand(len(t)) - 1)

    # test 1
    plt.figure()
    ls_freq = linspace(0.01,12.0,10000)
    Pxx_den = LombScargle(t,y_test_noisy,normalization='psd').power(ls_freq)
    plt.plot(ls_freq,signal.medfilt(Pxx_den,kernel_size=21),'r',label='L-S')
    welch_freq,Pxx_den \
        = signal.welch(y_test_noisy,24,nperseg=256,window='hanning',detrend='linear',average='median',scaling='density')
    plt.plot(welch_freq,Pxx_den,'g',label='Welch')
    plt.legend(frameon=False)

    # test 2
    plt.figure()
    ls_freq = linspace(0.00001,12.0,10000)
    ls_Pxx_den = LombScargle(t,y_test_noisy,normalization='psd').power(ls_freq)
    plt.plot(ls_freq,signal.medfilt(ls_Pxx_den,kernel_size=21),'r',label='L-S')
    welch_freq,welch_Pxx_den \
        = signal.welch(y_test_noisy,24,nperseg=256*4,window='hanning',detrend='linear',average='median',scaling='density')
    plt.plot(welch_freq,welch_Pxx_den,'g',label='Welch')
    plt.legend(frameon=False)
    print(trapz(ls_Pxx_den,ls_freq))
    print(trapz(welch_Pxx_den,welch_freq))
    print(std(y_test_noisy)**2)

    # test 3
    def classical_periodogram(t, y):
        N = len(t)
        frequency = fft.fftfreq(N, t[1] - t[0])
        y_fft = fft.fft(y)
        positive = (frequency > 0)
        return frequency[positive], (1. / N) * abs(y_fft[positive]) ** 2

    y_rand = 10*random.randn(len(t))

    plt.figure()
    ls_freq = linspace(0.00001,12.0,100000)
    ls_Pxx_den = LombScargle(t,y_rand,normalization='psd').power(ls_freq)
    plt.plot(ls_freq,signal.medfilt(ls_Pxx_den,kernel_size=21),'r',label='L-S')
    welch_freq, welch_Pxx_den \
        = signal.welch(y_rand,24,nperseg=256*4,window='hanning',detrend='linear',average='median',scaling='density')
    plt.plot(welch_freq,welch_Pxx_den,'g',label='Welch')
    four_freq, four_Pxx_den = classical_periodogram(t,y_rand)
    plt.plot(four_freq,four_Pxx_den,'b',label='Classical Fourier periodogram')
    plt.legend(frameon=False)

    print(trapz(ls_Pxx_den,ls_freq))
    print(trapz(welch_Pxx_den,welch_freq))
    print(trapz(four_Pxx_den,four_freq))
    print(std(10*random.randn(len(t)))**2)