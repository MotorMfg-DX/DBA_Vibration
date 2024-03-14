import numpy as np
from nptdms import TdmsFile
from nptdms import tdms
import pandas as pd
import matplotlib.pyplot as plt

gain = 0.00152587890625     # Gain (Setting)
wf_increment = 0.0001953125     # Time Resolution (Setting)
num_sample = 5120       # Number of Samplings

### read tdms file ###
filename = '★00400750.tdms'
tdms_file = TdmsFile.read(filename)
df_actwave = tdms_file.as_dataframe(time_index=False)
df_actwave.columns = ['CW', 'CCW']

# Convert signal to V
df_actwave.insert(0, 'Time', df_actwave.index * wf_increment)
df_actwave['CW'] = df_actwave['CW'] * gain
df_actwave['CCW'] = df_actwave['CCW'] * gain

### Show vibration graphs ###
plt.figure()
plt.subplots_adjust(wspace=0.4, hspace=0.6)

plt.subplot(2, 1, 1)
plt.plot(df_actwave['Time'], df_actwave['CW'])
plt.title('CW')
plt.xlabel('Time [sec]')
plt.ylabel('Output [V]')
# plt.ylim(-2, 2)

plt.subplot(2, 1, 2)
plt.plot(df_actwave['Time'], df_actwave['CCW'])
plt.title('CCW')
plt.xlabel('Time [sec]')
plt.ylabel('Output [V]')
# plt.ylim(-2, 2)

### FFT ###
# 窓関数を用いない。0パディングしてそのままの波形が連続であると仮定する。
# 設備で使用している窓関数(Hanning)が狙っている周波数特性に対して適切かどうか不明なため。
# 対称の周波数特性を整理した上で窓関数の吟味を行う。
# 今回はFFT→OAと計算していくので、周波数領域における振幅値が重要になる。

padding_num_sample = 8192       # Number of samling for FFT = 2^13
padding_total_T = wf_increment * num_sample * (padding_num_sample / num_sample)        # new measuring time domain after 0 padding [s]
fft_fs = padding_num_sample / padding_total_T     # Sampling rate after 0 padding [Hz]
padding_dt = 1 / fft_fs     # Time resolution after 0 padding [s]

# 0 Padding
df_0padding = pd.DataFrame(index=range(padding_num_sample - num_sample), columns=['Time', 'CW', 'CCW'])
df_0padding.fillna(0, inplace=True)
df_0padding_wave = df_actwave.copy()
df_0padding_wave = pd.concat([df_0padding_wave, df_0padding], ignore_index=True)

ar_0padding_T = np.arange(0, padding_total_T, padding_dt)
df_0padding_T = pd.DataFrame(data=ar_0padding_T, columns=['Time'])
df_0padding_wave = df_0padding_wave.drop('Time', axis=1)
df_0padding_wave = pd.concat([df_0padding_T, df_0padding_wave], axis=1)

# Amplitude correction factor
acf_CW = (df_actwave['CW'].sum() / num_sample) / (df_0padding_wave['CW'].sum() / padding_num_sample)
acf_CCW = (df_actwave['CCW'].sum() / num_sample) / (df_0padding_wave['CCW'].sum() / padding_num_sample)
df_0padding_wave['CW'] = df_0padding_wave['CW'] * acf_CW
df_0padding_wave['CCW'] = df_0padding_wave['CCW'] * acf_CCW

# FFT
df_fft = pd.DataFrame(columns=['freq', 'CW', 'CCW', 'rCW', 'rCCW'])     # rCW, rCCWはFFT実部の波形
df_fft['CW'] = np.fft.fft(df_0padding_wave['CW'], axis=0)
df_fft['CCW'] = np.fft.fft(df_0padding_wave['CCW'], axis=0)
df_fft['freq'] = np.fft.fftfreq(padding_num_sample, d=padding_dt)
df_fft['rCW'] = df_fft['CW'].abs() / (padding_num_sample / 2)
df_fft['rCCW'] = df_fft['CCW'].abs() / (padding_num_sample / 2)

### POA ###


### Show graphs ###
plt.figure()
plt.subplots_adjust(wspace=0.4, hspace=0.6)

plt.subplot(2, 1, 1)
plt.plot(df_fft.iloc[1: int(padding_num_sample / 2), [0]], df_fft.iloc[1: int(padding_num_sample / 2), [3]])
plt.yscale('log')
plt.title('CW')
plt.xlabel('f [Hz]')
plt.ylabel('PSD [(m/s^2)^2/Hz]')

plt.subplot(2, 1, 2)
plt.plot(df_fft.iloc[1: int(padding_num_sample / 2), [0]], df_fft.iloc[1: int(padding_num_sample / 2), [4]])
plt.yscale('log')
plt.title('CCW')
plt.xlabel('f [Hz]')
plt.ylabel('PSD [(m/s^2)^2/Hz]')

plt.show()
