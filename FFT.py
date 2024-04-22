# OK/NGでの処理の分岐はできていない
# peakの検出
# コンター図の作成

import numpy as np
from nptdms import TdmsFile
from nptdms import tdms
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from scipy.signal import find_peaks
import os

path = os.getcwd()      # Current directory
gain = 0.00152587890625     # Gain (Setting)
wf_increment = 0.0001953125     # Time Resolution (Setting)
num_sample = 5120       # Number of Samplings
padding_num_sample = 8192       # Number of samling for FFT = 2^13
padding_total_T = wf_increment * num_sample * (padding_num_sample / num_sample)        # new measuring time domain after 0 padding [s]
fft_fs = padding_num_sample / padding_total_T     # Sampling rate after 0 padding [Hz]
padding_dt = 1 / fft_fs     # Time resolution after 0 padding [s]
cnt = 0    # File count
f_ave = False       # Average flag True: Average ON
ylim_act = [-2.0, 2.0]      # y axis limit for actwave
ylim_fft = [10**-9, 10**-1]     # y axis limit for FFT

def read_tdms_file(filename):
    """ tdmsファイルを読み込んで波形を取得する"""
    """ 引数: """
    """ filename: str, tdmsファイル(拡張子込で指定) """
    """ 戻り値 """
    """ df_actwave[CW, CCW]: dataframe """
    tdms_file = TdmsFile.read(filename)
    df_actwave = tdms_file.as_dataframe(time_index=False)
    df_actwave.columns = ['CW', 'CCW']
    df_actwave.insert(0, 'Time', df_actwave.index * wf_increment)
    df_actwave['CW'] = df_actwave['CW'] * gain
    df_actwave['CCW'] = df_actwave['CCW'] * gain
    filename = str.rstrip('.tdms')

    return df_actwave

def plot_actwave(df_actwave, filename, ylim=None, ext='png'):
    """ actwave波形を描画して保存する """
    """ 引数: """
    """ df_actwave: dataframe, actwave波形 """
    """ filename: str, 保存ファイル名(拡張子なし) """
    """ ylim: list, y軸の範囲(min, max) """
    """ ext: str, 保存ファイルの拡張子 """
    plt.figure(filename + '_actwave')
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plt.subplot(2, 1, 1)
    plt.plot(df_actwave['Time'], df_actwave['CW'])
    plt.title('CW')
    plt.xlabel('Time [sec]')
    plt.ylabel('Output [V]')
    plt.xlim(0, 1)
    if ylim != None:
        plt.ylim(ylim)
        y_step = (np.amax(ylim) - np.amin(ylim)) / 4
        plt.yticks(np.arange(np.amin(ylim), np.amax(ylim) + y_step, y_step))
    plt.subplot(2, 1, 2)
    plt.plot(df_actwave['Time'], df_actwave['CCW'])
    plt.title('CCW')
    plt.xlabel('Time [sec]')
    plt.ylabel('Output [V]')
    plt.xlim(0, 1)
    if ylim != None:
        plt.ylim(ylim)
        y_step = (np.amax(ylim) - np.amin(ylim)) / 4
        plt.yticks(np.arange(np.amin(ylim), np.amax(ylim) + y_step, y_step))
    plt.savefig(filename + '_actwave.' + ext)

def zero_padding(df_actwave, padding_num_sample, padding_total_T, padding_dt):
    """ 0パディングを行う """
    """ 埋める数値はそれぞれの波形の平均値 """
    """ 引数: """
    """ df_actwave: dataframe, actwave波形 """
    """ padding_num_sample: int, 0パディング後のサンプル数 """
    """ padding_total_T: float, 0パディング後の総時間 """
    """ padding_dt: float, 0パディング後の時間分解能 """
    """ 戻り値 """
    """ df_0padding_wave: dataframe, 0パディング後の波形 """
    df_0padding = pd.DataFrame(index=range(padding_num_sample - num_sample), columns=['Time', 'CW', 'CCW'])
    df_0padding.fillna({'Time': 0, 'CW':df_actwave['CW'].mean(), 'CCW':df_actwave['CCW'].mean()}, inplace=True)
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

    return df_0padding_wave

def culc_FFT(df_0padding_wave, padding_num_sample, padding_dt):
    """ FFTを計算する """
    """ 計算した結果はPSDに変換する """
    """ 引数: """
    """ df_0padding_wave: dataframe, 0パディング後の波形 """
    """ padding_num_sample: int, 0パディング後のサンプル数 """
    """ padding_dt: float, 0パディング後の時間分解能 """
    """ 戻り値 """
    """ df_fft: dataframe, FFT結果 """
    df_fft = pd.DataFrame(columns=['freq', 'CW', 'CCW', 'CW_PSD', 'CCW_PSD'])     # CW_PSD, CCW_PSDはPSD
    df_fft['CW'] = np.fft.fft(df_0padding_wave['CW'], axis=0)
    df_fft['CCW'] = np.fft.fft(df_0padding_wave['CCW'], axis=0)
    df_fft['freq'] = np.fft.fftfreq(padding_num_sample, d=padding_dt)
    df_fft['CW_PSD'] = (df_fft['CW'].abs() / (padding_num_sample / 2))**2
    df_fft['CCW_PSD'] = (df_fft['CCW'].abs() / (padding_num_sample / 2))**2
    return df_fft

def culc_POA(df_fft):
    """ POAを計算する """
    """ 引数: """
    """ df_fft: dataframe, FFT結果 """
    """ 戻り値 """
    """ POA_CW: float, CWのPOA """
    """ POA_CCW: float, CCWのPOA """
    POA_CW = 0
    POA_CCW = 0
    l_freq = []
    l_POA_CW = []
    l_POA_CCW = []
    j = 0       # Index

    # 300Hzから1500HzまでのPOAを計算
    for i in df_fft['freq']:
        if 300 <= i <= 1500:
            POA_CW += df_fft.iat[j, 3]
            POA_CCW += df_fft.iat[j, 4]
            l_freq.append(i)
            l_POA_CW.append(POA_CW)
            l_POA_CCW.append(POA_CCW)
        j += 1
    df_POA = pd.DataFrame(data={'freq': l_freq, 'CW': l_POA_CW, 'CCW': l_POA_CCW},
        columns=['freq', 'CW', 'CCW'])

    return POA_CW, POA_CCW

def plot_fft(df_fft, POA_CW, POA_CCW, filename, ext='png', ylim=None):
    """ FFT結果を描画して保存する """
    """ 引数: """
    """ df_fft: dataframe, FFT結果 """
    """ POA_CW: float, CWのPOA """
    """ POA_CCW: float, CCWのPOA """
    """ filename: str, 保存ファイル名(拡張子なし) """
    """ ext: str, 保存ファイルの拡張子 """
    """ ylim: list, y軸の範囲(min, max) """
    plt.figure(filename + '_PSDwave')
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plt.subplot(2, 1, 1)
    plt.plot(df_fft.iloc[1: int(padding_num_sample / 2), [0]], df_fft.iloc[1: int(padding_num_sample / 2), [3]])
    plt.yscale('log')
    plt.title('CW')
    plt.xlabel('f [Hz]')
    plt.ylabel('PSD [V^2/Hz]')
    plt.xlim(0,2500)
    if ylim != None:
        plt.ylim(ylim)
    plt.text(2000, 10**-2, format(POA_CW, '.4f'))
    plt.subplot(2, 1, 2)
    plt.plot(df_fft.iloc[1: int(padding_num_sample / 2), [0]], df_fft.iloc[1: int(padding_num_sample / 2), [4]])
    plt.yscale('log')
    plt.title('CCW')
    plt.xlabel('f [Hz]')
    plt.ylabel('PSD [V^2/Hz]')
    plt.xlim(0,2500)
    if ylim != None:
        plt.ylim(ylim)
    plt.text(2000, 10**-2, format(POA_CCW, '.4f'))
    plt.savefig(filename + '_PSDwave.' + ext)

def find_peak(df_fft, filename, PSD_height=10**-4, discrete_distance=3, sample_size=2**12):
    """ FFT結果からピークを検出する """
    """ 引数: """
    """ df_fft: dataframe, FFT結果 """
    """ filename: str, 保存ファイル名(拡張子なし) """
    """ PSD_height: float, ピークの高さ閾値 """
    """ discrete_distance: int, ピーク間の最小距離 """
    """ sample_size: int, サンプル数 """
    """ 戻り値 """
    """ df_peak: dataframe, ピーク情報 """
    df_CW_peak = pd.DataFrame(columns=['CW_freq', 'CW_PSD'])
    df_CCW_peak = pd.DataFrame(columns=['CCW_freq', 'CCW_PSD'])

    # ピークの取得
    CW_peak_index, _ = find_peaks(df_fft['CW_PSD'], height=PSD_height, distance=discrete_distance)
    CCW_peak_index, _ = find_peaks(df_fft['CCW_PSD'], height=PSD_height, distance=discrete_distance)
    CW_peak_index = CW_peak_index[CW_peak_index < sample_size / 2]      # Sampling theorem
    CCW_peak_index = CCW_peak_index[CCW_peak_index < sample_size / 2]      # Sampling theorem

    # 1つのデータフレームにまとめる
    df_CW_peak['CW_freq'] = df_fft.iloc[CW_peak_index, 0]
    df_CW_peak['CW_PSD'] = df_fft.iloc[CW_peak_index, 3]
    df_CCW_peak['CCW_freq'] = df_fft.iloc[CCW_peak_index, 0]
    df_CCW_peak['CCW_PSD'] = df_fft.iloc[CCW_peak_index, 4]
    df_peak = pd.concat([df_CW_peak, df_CCW_peak], axis=1, sort=False)

    return df_peak

### FFT ###
# Tkinterでフォルダを指定する。
# フォルダ内のすべてのtdmsファイルを読み込む。
# FFTは窓関数を用いない。0パディングしてそのままの波形が連続であると仮定する。
# 実波形に原因不明のオフセットが発生しているため、0パディングではなく波形の平均値で埋める。
# 設備で使用している窓関数(Hanning)が狙っている周波数特性に対して適切かどうか不明なため。
# 対称の周波数特性を整理した上で窓関数の吟味を行う。
# 今回はFFT→OAと計算していくので、周波数領域における振幅値が重要になる。

root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory(title='Select Folder')
df_actwave_ave = pd.DataFrame(columns=['Time', 'CW', 'CCW'])     # Average actwave
df_FFT_ave = pd.DataFrame(columns=['freq', 'CW_PSD', 'CCW_PSD'])     # Average FFT

for filename in os.listdir(folder_path):
    if filename.endswith('.tdms'):
        # FFT
        cnt += 1
        df_actwave = read_tdms_file(folder_path + '/' + filename)       # Read tdms file
        filename = filename.rstrip('.tdms')     # Remove extension
        df_0padding_wave = zero_padding(df_actwave, padding_num_sample, padding_total_T, padding_dt)        # 0 padding
        df_fft = culc_FFT(df_0padding_wave, padding_num_sample, padding_dt)     # FFT
        POA_CW, POA_CCW = culc_POA(df_fft)      # Culcrate POA

        # Average
        if f_ave == True:
            if cnt == 1:
                df_actwave_ave = df_actwave.copy()
                df_FFT_ave = df_fft.copy()
            else:
                df_actwave_ave['CW'] = df_actwave_ave['CW'] + df_actwave['CW']
                df_actwave_ave['CCW'] = df_actwave_ave['CCW'] + df_actwave['CCW']
                df_FFT_ave['CW_PSD'] = df_FFT_ave['CW_PSD'] + df_fft['CW_PSD']
                df_FFT_ave['CCW_PSD'] = df_FFT_ave['CCW_PSD'] + df_fft['CCW_PSD']
                
        # Plot
        plot_actwave(df_actwave, filename, ylim=ylim_act)        # Plot actwave
        plot_fft(df_fft, POA_CW, POA_CCW, filename, ylim=ylim_fft)     # Plot FFT
        plt.show()

if f_ave == True:
    df_actwave_ave['CW'] = df_actwave_ave['CW'] / cnt
    df_actwave_ave['CCW'] = df_actwave_ave['CCW'] / cnt
    df_FFT_ave['CW_PSD'] = df_FFT_ave['CW_PSD'] / cnt
    df_FFT_ave['CCW_PSD'] = df_FFT_ave['CCW_PSD'] / cnt
    POA_CW_ave, POA_CCW_ave = culc_POA(df_FFT_ave)      # Culcrate POA

    # ピーク値に関してcsvファイルに書き込む
    # csvファイルが存在する場合は追記をし、存在しない場合は新規作成する
    df_peak = find_peak(df_FFT_ave, 'Average_peak')        # Find peak
    if os.path.isfile(path + '/Average_peak.csv'):
        df_existing = pd.read_csv(path + '/Average_peak.csv')
        df_combined = pd.concat([df_existing, df_peak], ignore_index=True, axis=1)      # Combine existing csv and new peak dataframe
        df_combined.to_csv(path + '/Average_peak.csv', index=False)
    else:
        df_peak.to_csv(path + '/Average_peak.csv', index=False)     # Create new peak csv

    # Plot
    plot_actwave(df_actwave_ave, 'Average', ylim=ylim_act)        # Plot average actwave
    plot_fft(df_FFT_ave, POA_CW_ave, POA_CCW_ave, 'Average', ylim=ylim_fft)        # Plot average FFT
    plt.show()
