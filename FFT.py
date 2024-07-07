# カラーバーの範囲
# コンター図のz軸範囲

import numpy as np
from nptdms import TdmsFile
from nptdms import tdms
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import filedialog
from scipy.signal import find_peaks
import scipy.interpolate
import os
import re
import sys

# 固定値
gain = 0.00152587890625     # Gain (Setting)
wf_increment = 0.0001953125     # Time Resolution (Setting)
num_sample = 5120       # Number of Samplings
padding_num_sample = 8192       # Number of samling for FFT = 2^13
padding_total_T = wf_increment * num_sample * (padding_num_sample / num_sample)        # new measuring time domain after 0 padding [s]
fft_fs = padding_num_sample / padding_total_T     # Sampling rate after 0 padding [Hz]
padding_dt = 1 / fft_fs     # Time resolution after 0 padding [s]
cnt = 0    # File count of average
contour_cnt = 0     # File count of contour

# 上下限値用の変数
ylim_act = [-2.0, 2.0]      # y axis limit for actwave
ylim_fft = [10**-9, 10**-1]     # y axis limit for FFT
color_max = 10**-4      # Color bar max value
color_min = 10**-7      # Color bar min value

# 機能のフラグ管理
f_ave = False       # Average flag True: Average ON
f_peak = False      # Peak flag True: Peak detect ON
f_fftresult_csv = False      # FFT result csv flag True: FFT result csv ON
f_plot_graph = False     # Plot graph flag True: Plot graph ON
f_contour = True        # Contour flag True: Making Contour Map ON
f_contour_csv = False       # Contour csv flag True: Contour csv ON, when it is True, f_contour must be True
f_3D_Contour = True     # 3D Contour flag True: 3D Contour ON, when it is True, f_contour must be True
# f_interpolation = False     # Interpolation flag True: Interpolation ON in contour map
# y_interpolation = 100       # y axis interpolation interval in contour map

def read_tdms_file(filename):
    """ tdmsファイルを読み込んで波形を取得する

    引数
    -------
    filename: str
        tdmsファイル(拡張子込で指定)
    
    戻り値
    -------
    df_actwave[Time, CW, CCW]: dataframe """

    tdms_file = TdmsFile.read(filename)
    df_actwave = tdms_file.as_dataframe(time_index=False)
    df_actwave.columns = ['CW', 'CCW']
    df_actwave.insert(0, 'Time', df_actwave.index * wf_increment)
    df_actwave['CW'] = df_actwave['CW'] * gain
    df_actwave['CCW'] = df_actwave['CCW'] * gain
    filename = str.rstrip('.tdms')

    return df_actwave

def plot_actwave(df_actwave, filename, ylim=None, ext='png'):
    """ 時間軸波形を描画して保存する

    引数
    -------
    df_actwave[Time, CW, CCW]: dataframe
        時間軸波形
    filename: str
        保存ファイル名(拡張子なし)
    ylim: list
        y軸の範囲(min, max)
    ext: str
        保存ファイルの拡張子 """

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
    """ 実波形の後ろに定数補正追加する(0 padding)
    設備各stに存在しているオフセットを補正するため埋める数値はそれぞれの波形の平均値としている
    また、振幅補正を行うため、0 padding後の波形の総和が0 padding前の波形の総和と等しくなるように調整する(ACF)

    引数
    -------
    df_actwave[Time, CW, CCW]: dataframe
        時間軸波形
    padding_num_sample: int
        0パディング後のサンプル数
    padding_total_T: float
        0パディング後の総時間
    padding_dt: float
        0パディング後の時間分解能
    
    戻り値
    -------
    df_0padding_wave[Time, CW, CCW]: dataframe, 0パディング後の波形 """

    df_0padding = pd.DataFrame(index=range(padding_num_sample - num_sample), columns=['Time', 'CW', 'CCW'])
    df_0padding.fillna({'Time': 0, 'CW':df_actwave['CW'].mean(), 'CCW':df_actwave['CCW'].mean()}, inplace=True)
    df_0padding_wave = df_actwave.copy()
    df_0padding_wave = pd.concat([df_0padding_wave, df_0padding], ignore_index=True)
    ar_0padding_T = np.arange(0, padding_total_T, padding_dt)       # Time array
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
    """ FFTを計算する
    計算した結果はPSDに変換する

    引数
    -------
    df_0padding_wave[Time, CW, CCW]: dataframe
        0パディング後の波形
    padding_num_sample: int
        0パディング後のサンプル数
    padding_dt: float
        0パディング後の時間分解能

    戻り値
    -------
    df_fft[freq, CW, CCW, CW_PSD, CCW_PSD]: dataframe
        FFT結果を格納するdataframe
            freq: 周波数
            CW: CWのFFT結果
            CCW: CCWのFFT結果
            CW_PSD: CWのPSD
            CCW_PSD: CCWのPSD """

    df_fft = pd.DataFrame(columns=['freq', 'CW', 'CCW', 'CW_PSD', 'CCW_PSD'])     # CW_PSD, CCW_PSDはPSD
    df_fft['CW'] = np.fft.fft(df_0padding_wave['CW'], axis=0)
    df_fft['CCW'] = np.fft.fft(df_0padding_wave['CCW'], axis=0)
    df_fft['freq'] = np.fft.fftfreq(padding_num_sample, d=padding_dt)
    df_fft['CW_PSD'] = (df_fft['CW'].abs() / (padding_num_sample / 2))**2
    df_fft['CCW_PSD'] = (df_fft['CCW'].abs() / (padding_num_sample / 2))**2

    # ナイキスト周波数までのデータを抽出
    df_fft = df_fft.iloc[0: int(padding_num_sample / 2), :]

    return df_fft

def culc_POA(df_fft):
    """ POAを計算する
    POAは300Hzから1500HzまでのPSDの総和

    引数
    -------
    df_fft[freq, CW, CCW, CW_PSD, CCW_PSD]: dataframe
        FFT結果

    戻り値
    -------
    POA_CW: float
        CWのPOA
    POA_CCW: float
        CCWのPOA """

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
    """ FFT結果を描画して保存する

    引数
    -------
    df_fft[freq, CW, CCW, CW_PSD, CCW_PSD]: dataframe
        FFT結果
    POA_CW: float
        CWのPOA
    POA_CCW: float
        CCWのPOA
    filename: str
        保存ファイル名(拡張子なし)
    ext: str
        保存ファイルの拡張子
    ylim: list
        y軸の範囲(min, max) """
    
    plt.figure(filename + '_PSDwave')
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plt.subplot(2, 1, 1)
    plt.plot(df_fft.iloc[1:, [0]], df_fft.iloc[1:, [3]])        # 0Hzは除外
    plt.yscale('log')
    plt.title('CW')
    plt.xlabel('f [Hz]')
    plt.ylabel('PSD [V^2/Hz]')
    plt.xlim(0,2500)
    if ylim != None:
        plt.ylim(ylim)
    plt.text(2000, 10**-2, format(POA_CW, '.4f'))
    plt.subplot(2, 1, 2)
    plt.plot(df_fft.iloc[1:, [0]], df_fft.iloc[1:, [4]])        # 0Hzは除外
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
    """ FFT結果からピークを検出する

    引数
    -------
    df_fft: dataframe[freq, CW, CCW, CW_PSD, CCW_PSD]
        FFT結果
    filename: str
        保存ファイル名(拡張子なし)
    PSD_height: float
        ピークの高さ閾値
        PSD絶対値で閾値以上をピークとしている
    discrete_distance: int
        ピーク間の最小距離
        ピークとピークの間に必要なサンプル数
    sample_size: int
        サンプル数
    
    戻り値
    -------
    df_peak: dataframe[freq, CW_freq, CW_PSD, CCW_freq, CCW_PSD]
        ピーク情報 """

    # ピーク情報を格納するデータフレームを作成
    df_peak = pd.DataFrame(columns=['freq'])
    df_peak['freq'] = df_fft['freq']
    df_peak.set_index('freq', inplace=True)
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
    df_CW_peak.set_index('CW_freq', inplace=True)
    df_CCW_peak['CCW_freq'] = df_fft.iloc[CCW_peak_index, 0]
    df_CCW_peak['CCW_PSD'] = df_fft.iloc[CCW_peak_index, 4]
    df_CCW_peak.set_index('CCW_freq', inplace=True)
    df_peak = pd.concat([df_peak, df_CW_peak], axis=1)
    df_peak = pd.concat([df_peak, df_CCW_peak], axis=1)

    return df_peak

class ContourMap:
    """ コンター図を作成する 
        x軸: 周波数、y軸: 回転数、z軸: PSD

        引数
        -------
        df_fft: dataframe[freq, CW, CCW, CW_PSD, CCW_PSD]
            FFT結果
            初期値は最高回転数のデータを使い、高回転数順でz_append()を用いてデータを追加すること
        filename: str
            保存ファイル名(拡張子なし) """

    def __init__(self, df_fft, filename):
        self.x = np.array(df_fft['freq'])
        self.y = self.extract_rpm(filename)
        self.z_CW = np.array(df_fft['CW_PSD'])
        self.z_CCW = np.array(df_fft['CCW_PSD'])

    def extract_rpm(self, s):
        """ ファイル名から回転数を抽出する
            抜き出す回転数の条件は"rpm"で始まる数字
            例: ***rpm1000***.tdmsの場合、1000を抽出する

        引数
        -------
        s: str
            ファイル名
        
        戻り値
        -------
        rpm: array[(int)]
            回転数 """

        rpm = re.search(r'rpm(\d+)', s)
        if rpm != None:
            rpm = int(rpm.group(1))       # 回転数の抽出とint型へのキャスト
            rpm = np.array([rpm])       # numpy配列への変換
        else:
            print('"rpm" is not found in the filename')
        
        return rpm

    def rpm_append(self, filename):
        """ filenameに含まれる回転数を抽出し、self.rpmの末尾に追加する
            self.yは高回転数から低回転数の順に接続されることを前提としている

        引数
        -------
        filename: str
            保存ファイル名(拡張子なし) """

        new_rpm = self.extract_rpm(filename)
        self.y = np.append(self.y, new_rpm)

    def z_append(self, df_fft):
        """ df_fftからデータを抽出し、self.zの末尾に追加する

        引数
        -------
        df_fft: dataframe[freq, CW, CCW, CW_PSD, CCW_PSD]
            FFT結果 """

        self.z_CW = np.vstack([self.z_CW, df_fft['CW_PSD'].to_numpy()])
        self.z_CCW = np.vstack([self.z_CCW, df_fft['CCW_PSD'].to_numpy()])

    def plot_2Dcontour(self, ext='png', ylim=None, color_max=color_max, color_min=color_min):
        """ 2次元コンター図を描画して保存する
        2次元コンター図はCWとCCWのコンター図を並べて表示する
        出力されるコンター図は等高線パターン(contourf())と格子点パターン(pcolormesh())の2つ
        カラーバーの範囲は対数スケールで、color_maxとcolor_minで指定する
        カラーバーの範囲を逸脱した値はclip(上下限値に変更)される

        引数
        -------
        ext: str
            保存ファイルの拡張子
        ylim: list(float, float)
            z軸の範囲(min, max)
        color_max: float
            カラーバーの最大値
        color_min: float
            カラーバーの最小値 """

        plt.figure('ContourMap')
        plt.subplots_adjust(wspace=0.4, hspace=0.6)
        X, Y = np.meshgrid(self.x, self.y)        # 2次元メッシュグリッドの作成

        # カラーマップの作成
        # BoundaryNorm()にするかLogNorm()にするかは検討中 → LogNorm()に決定
        cmap = plt.get_cmap('coolwarm')
        norm = colors.LogNorm(vmin=color_min, vmax=color_max, clip=True)

        # contour()での描画
        plt.subplot(2, 1, 1)        # CWのコンター図設定

        # ylimが指定されている場合はylimを利用してz軸の範囲を指定する
        if ylim != None:
            plt.contourf(X, Y, self.z_CW, cmap=cmap, norm=norm, levels=np.linspace(color_min, color_max, 100))

        else:
            plt.contourf(X, Y, self.z_CW, cmap=cmap, norm=norm)
        
        plt.colorbar(label='PSD [V^2/Hz]')
        plt.title('CW_Contour')
        plt.xlabel('f [Hz]')
        plt.ylabel('N [rpm]')
        plt.xlim(0, 2500)

        plt.subplot(2, 1, 2)        # CCWのコンター図設定
        
        # ylimが指定されている場合はylimを利用してz軸の範囲を指定する
        if ylim != None:
            plt.contourf(X, Y, self.z_CCW, cmap=cmap, norm=norm, levels=np.linspace(color_min, color_max, 100))

        else:
            plt.contourf(X, Y, self.z_CCW, cmap=cmap, norm=norm,)

        plt.colorbar(label='PSD [V^2/Hz]')
        plt.title('CCW_Contour')
        plt.xlabel('f [Hz]')
        plt.ylabel('N [rpm]')
        plt.xlim(0, 2500)

        plt.savefig('ContourMap.' + ext)
        plt.show()

        # pcolormesh()での描画
        plt.figure('ContourMap_pcolormesh')
        plt.subplots_adjust(wspace=0.4, hspace=0.6)
        
        plt.subplot(2, 1, 1)        # CWのコンター図設定
        plt.pcolormesh(X, Y, self.z_CW, cmap=cmap, norm=norm)
        plt.colorbar(label='PSD [V^2/Hz]')
        plt.title('CW_pcolormesh')
        plt.xlabel('f [Hz]')
        plt.ylabel('N [rpm]')
        plt.xlim(0, 2500)

        plt.subplot(2, 1, 2)        # CCWのコンター図設定
        plt.pcolormesh(X, Y, self.z_CCW, cmap=cmap, norm=norm)
        plt.colorbar(label='PSD [V^2/Hz]')
        plt.title('CCW_pcolormesh')
        plt.xlabel('f [Hz]')
        plt.ylabel('N [rpm]')
        plt.xlim(0, 2500)

        plt.savefig('ContourMap_pcolormesh.' + ext)
        plt.show()

    # def plot_2D_Contour_with_interpolation(self, dy=39, ext='png', ylim=None,kernel='thin_plate'):
    #     """ 使用禁止。 スプライン補間はデータ数に対して計算量が膨大になるため。
    #      FFT結果に対して補間を行い、コンター図を描画して保存する 
    #      補間方法はRBF補間(Radial Basis Function Interpolation)を利用する

    #     引数
    #     -------
    #     dy: int
    #         y軸の補間間隔 
    #     ext: str
    #         保存ファイルの拡張子 
    #     ylim: list(float, float)
    #         z軸の範囲(min, max) """
        
    #     # 既知データの処理
    #     # 第1引数は2次元配列ではなければならない
    #     x_obs_grid, y_obs_grid = np.meshgrid(self.x, self.y)       # 2次元メッシュグリッドの作成

    #     # Rbf()でもメモリエラー(45GB)
    #     # rbf_CW = scipy.interpolate.Rbf(x_obs_grid.ravel(), y_obs_grid.ravel(), self.z_CW.ravel(), function=kernel)       # CWの補間
    #     # rbf_CCW = scipy.interpolate.Rbf(x_obs_grid.ravel(), y_obs_grid.ravel(), self.z_CCW.ravel(), function=kernel)       # CCWの補間

    #     # CW, CCWの補間をRBFInterpolator()で行ったところ、メモリエラーが発生した
    #     # rbf_CW = scipy.interpolate.RBFInterpolator(np.vstack((x_obs_grid.ravel(), y_obs_grid.ravel())).T, self.z_CW.ravel(), kernel=kernel)       # CWの補間
    #     # rbf_CCW = scipy.interpolate.RBFInterpolator(np.vstack((x_obs_grid.ravel(), y_obs_grid.ravel())).T, self.z_CCW.ravel(), kernel=kernel)       # CCWの補間

    #     # scipy.interpolate.griddataの場合、self.x, self.yの配列の大きさが等しくないといけないので使用しない。
    #     # CW_grid = scipy.interpolate.griddata((self.x, self.y), self.z_CW, (x_grid, y_grid), method=kernel)       # CWの補間
    #     # CCW_grid = scipy.interpolate.griddata((self.x, self.y), self.z_CCW, (x_grid, y_grid), method=kernel)       # CCWの補間

    #     # グラフプロット
    #     plt.figure('ContourMap w/Interpolation')
    #     plt.subplots_adjust(wspace=0.4, hspace=0.6)

    #     plt.subplot(2, 1, 1)        # CWのコンター図設定

    #     # ylimが指定されている場合はylimを利用してz軸の範囲を指定する
    #     if ylim != None:
    #         plt.pcolormesh(x_new_grid, y_new_grid, z_CW_interpolation, cmap='rainbow', norm=colors.LogNorm(vmin=ylim[0], vmax=ylim[1]), shading='gouraud')
    #     else:
    #         plt.pcolormesh(x_new_grid, y_new_grid, z_CW_interpolation, cmap='rainbow', norm=colors.LogNorm(), shading='gouraud')
        
    #     plt.colorbar(label='PSD [V^2/Hz]')
    #     plt.title('CW')
    #     plt.xlabel('f [Hz]')
    #     plt.ylabel('N [rpm]')
    #     plt.xlim(0, 2500)

    #     plt.subplot(2, 1, 2)        # CCWのコンター図設定
        
    #     # ylimが指定されている場合はylimを利用してz軸の範囲を指定する
    #     # colorbarの範囲は対数スケール
    #     if ylim != None:
    #         plt.pcolormesh(x_new_grid, y_new_grid, z_CCW_interpolation, cmap='rainbow', norm=colors.LogNorm(vmin=ylim[0], vmax=ylim[1]), shading='gouraud')
    #     else:
    #         plt.pcolormesh(x_new_grid, y_new_grid, z_CCW_interpolation, cmap='rainbow', norm=colors.LogNorm(), shading='gouraud')
        
    #     plt.colorbar(label='PSD [V^2/Hz]')
    #     plt.title('CCW')
    #     plt.xlabel('f [Hz]')
    #     plt.ylabel('N [rpm]')
    #     plt.xlim(0, 2500)

    #     plt.savefig('ContourMap_wInterpolation.' + ext)
    #     plt.show()

    #     # # 既知データの処理
    #     # # メッシュグリッドでx, y座標系を作成し、[x, y]の要素を持つ2次元配列を作成する
    #     # # zはCWとCCW両方を1次元配列で作成する
    #     # x_obs_grid, y_obs_grid = np.meshgrid(self.x, self.y)       # 2次元メッシュグリッドの作成
    #     # x_obs_grid = x_obs_grid.ravel()
    #     # y_obs_grid = y_obs_grid.ravel()
    #     # xy_obs_grid = np.stack([x_obs_grid, y_obs_grid], axis=1)       # (x, y)座標の配列
    #     # z_CW = self.z_CW.ravel()
    #     # z_CCW = self.z_CCW.ravel()

    #     # # 補間
    #     # # x座標は補間せずそのままのデータを利用する
    #     # # y座標は最小値と最大値の間をdy個で補間する
    #     # # z座標は薄板スプライン補間を利用する
    #     # y_min, y_max = np.amin(self.y), np.amax(self.y)
    #     # y_new_grid = np.linspace(y_min, y_max, dy)       # 補間後のy座標
    #     # x_new_grid, y_new_grid = np.meshgrid(self.x, y_new_grid)       # 2次元メッシュグリッドの作成
    #     # x_new_grid = x_new_grid.ravel()
    #     # y_new_grid = y_new_grid.ravel()
    #     # xy_new_grid = np.stack([x_new_grid, y_new_grid], axis=1)       # (x, y)座標の配列
    #     # z_CW_interpolation = scipy.interpolate.RBFInterpolator(xy_obs_grid, z_CW, kernel=kernel).__call__(xy_new_grid)       # CWの補間
    #     # z_CCW_interpolation = scipy.interpolate.RBFInterpolator(xy_obs_grid, z_CCW, kernel=kernel).__call__(xy_new_grid)       # CCWの補間

    #     # # グラフプロット
    #     # plt.figure('ContourMap w/Interpolation')
    #     # plt.subplots_adjust(wspace=0.4, hspace=0.6)

    #     # plt.subplot(2, 1, 1)        # CWのコンター図設定

    #     # # ylimが指定されている場合はylimを利用してz軸の範囲を指定する
    #     # if ylim != None:
    #     #     plt.pcolormesh(x_new_grid, y_new_grid, z_CW_interpolation, cmap='rainbow', norm=colors.LogNorm(vmin=ylim[0], vmax=ylim[1]), shading='gouraud')
    #     # else:
    #     #     plt.pcolormesh(x_new_grid, y_new_grid, z_CW_interpolation, cmap='rainbow', norm=colors.LogNorm(), shading='gouraud')
        
    #     # plt.colorbar(label='PSD [V^2/Hz]')
    #     # plt.title('CW')
    #     # plt.xlabel('f [Hz]')
    #     # plt.ylabel('N [rpm]')
    #     # plt.xlim(0, 2500)

    #     # plt.subplot(2, 1, 2)        # CCWのコンター図設定
        
    #     # # ylimが指定されている場合はylimを利用してz軸の範囲を指定する
    #     # # colorbarの範囲は対数スケール
    #     # if ylim != None:
    #     #     plt.pcolormesh(x_new_grid, y_new_grid, z_CCW_interpolation, cmap='rainbow', norm=colors.LogNorm(vmin=ylim[0], vmax=ylim[1]), shading='gouraud')
    #     # else:
    #     #     plt.pcolormesh(x_new_grid, y_new_grid, z_CCW_interpolation, cmap='rainbow', norm=colors.LogNorm(), shading='gouraud')
        
    #     # plt.colorbar(label='PSD [V^2/Hz]')
    #     # plt.title('CCW')
    #     # plt.xlabel('f [Hz]')
    #     # plt.ylabel('N [rpm]')
    #     # plt.xlim(0, 2500)

    #     # plt.savefig('ContourMap_wInterpolation.' + ext)
    #     # plt.show()

    def plot_3Dcontour(self, ylim=None, color_max=color_max, color_min=color_min):
        """ 3次元コンター図を描画する
        3次元コンター図はアトラクティブな処理が可能

        引数
        -------
        ylim: list(float, float)
            z軸の範囲(min, max) 
        color_max: float
            カラーバーの最大値
        color_min: float
            カラーバーの最小値 """

        X, Y = np.meshgrid(self.x, self.y)        # 2次元メッシュグリッドの作成

        fig = plt.figure('ContourMap 3D')

        # CWのコンター図
        ax = fig.add_subplot(121, projection='3d')
        norm = colors.LogNorm(vmin=color_min, vmax=color_max)
        surf = ax.plot_surface(X, Y, self.z_CW, cmap='coolwarm', norm=norm, linewidth=0, antialiased=False)

        ax.set_title('CW')
        ax.set_xlabel('f [Hz]')
        ax.set_ylabel('N [rpm]')
        ax.set_zlabel('PSD [V^2/Hz]')
        ax.set_xlim(0, 2500)
        ax.set_zscale('log')

        # CCWのコンター図
        ax = fig.add_subplot(122, projection='3d')
        norm = colors.LogNorm(vmin=color_min, vmax=color_max)
        surf = ax.plot_surface(X, Y, self.z_CCW, cmap='coolwarm', norm=norm, linewidth=0, antialiased=False)
        fig.colorbar(surf)

        ax.set_title('CCW')
        ax.set_xlabel('f [Hz]')
        ax.set_ylabel('N [rpm]')
        ax.set_zlabel('PSD [V^2/Hz]')
        ax.set_xlim(0, 2500)
        ax.set_zscale('log')

#        ax.view_init(elev=30, azim=30)

        plt.show()

    def contour_toCSV(self, filename='ContourMap'):
        """ コンター図のデータをCSVファイルに書き込む

        引数
        -------
        filename: str
            保存ファイル名(拡張子なし) """

        df_csv = pd.DataFrame({
            'x': pd.Series(self.x),
            'y': pd.Series(self.y),
        })

        # 座標のcsvを出力
        df_csv.to_csv(filename + '_axis.csv', index=False)       # 行インデックスを書き出さない

        # PSDのcsvを出力
        np.savetxt(filename + '_CW.csv', self.z_CW, delimiter=',')
        np.savetxt(filename + '_CCW.csv', self.z_CCW, delimiter=',')

def natural_keys(text):
    """ 文字列を自然数でソートするためのキーを生成する

    引数
    -------
    text: str
        ソート対象の文字列

    戻り値
    -------
    key: list(int, str)
    ソートキー """

    def atoi(text):
        return int(text) if text.isdigit() else text

    return [atoi(c) for c in re.split('(\d+)', text)]

def dump_fft2csv(actwave, fft, filename, index=False):
    """ FFT結果をCSVファイルに書き込む
    書き出すcsvは2種類で、1つは実波形、もう1つはFFT結果
    実波形はTime, CW, CCWの3列、FFT結果はfreq, CW, CCW, CW_PSD, CCW_PSDの5列

    引数
    -------
    actwave: dataframe
        実波形のデータフレーム
    fft: dataframe
        FFT結果を格納しているデータフレーム
    filename: str
        保存ファイル名(拡張子なし) """

    actwave.to_csv(filename + '_actwave.csv', index=index)       # 実波形の書き出し
    fft.to_csv(filename + '_fft.csv', index=index)       # FFT結果の書き出し

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
path = os.getcwd()      # Current directoryroot = tk.Tk()

for filename in sorted(os.listdir(folder_path),key=natural_keys, reverse=True if f_contour else False):     # Sort files by natural number and read them in order

    # ファイルがtdms形式の場合のみ処理を行う
    if filename.endswith('.tdms'):
        # FFT
        cnt += 1
        df_actwave = read_tdms_file(folder_path + '/' + filename)       # Read tdms file
        filename = filename.rstrip('.tdms')     # Remove extension
        df_0padding_wave = zero_padding(df_actwave, padding_num_sample, padding_total_T, padding_dt)        # 0 padding
        df_fft = culc_FFT(df_0padding_wave, padding_num_sample, padding_dt)     # FFT
        POA_CW, POA_CCW = culc_POA(df_fft)      # Culcrate POA

        # CSVファイルに書き出す
        if f_fftresult_csv == True:
            dump_fft2csv(df_actwave, df_fft, filename, index=False)        # Dump to csv

        # 複数の実波形信号にAverage処理を行う
        # 複数の実は軽信号をFFTした結果のAverage処理も行う
        if f_ave == True:

            # 1つ目のファイルに対する処理
            if cnt == 1:
                df_actwave_ave = df_actwave.copy()
                df_FFT_ave = df_fft.copy()
            
            # 2つ目以降のファイルに対する処理
            else:
                df_actwave_ave['CW'] = df_actwave_ave['CW'] + df_actwave['CW']
                df_actwave_ave['CCW'] = df_actwave_ave['CCW'] + df_actwave['CCW']
                df_FFT_ave['CW_PSD'] = df_FFT_ave['CW_PSD'] + df_fft['CW_PSD']
                df_FFT_ave['CCW_PSD'] = df_FFT_ave['CCW_PSD'] + df_fft['CCW_PSD']
                
        # 1つ1つの実波形・FFT結果のPlot処理
        if f_plot_graph == True:
            plot_actwave(df_actwave, filename, ylim=ylim_act)        # Plot actwave
            plot_fft(df_fft, POA_CW, POA_CCW, filename, ylim=ylim_fft)     # Plot FFT
            plt.show()

        # Contour図を作成する処理
        if f_contour == True:
            contour_cnt += 1

            # 1つ目のファイルに対する処理
            if contour_cnt == 1:
                contour_map = ContourMap(df_fft, filename)      # Making ContourMap class

            # 2つ目以降のファイルに対する処理
            else:
                contour_map.rpm_append(filename)        # Get rpm from filename
                contour_map.z_append(df_fft)        # Append fft result data

        # FFT波形のPeak値を取得する処理
        # ピーク値に関してcsvファイルに書き込む
        # csvファイルが存在する場合は追記をし、存在しない場合は新規作成する
        if f_peak == True:
            df_peak = find_peak(df_fft, filename)        # Find peak

            # csvファイルが存在する場合は追記をし、存在しない場合は新規作成する
            if os.path.isfile(path + '/peak.csv'):
                df_existing = pd.read_csv(path + '/peak.csv')
                df_combined = pd.concat([df_existing, df_peak], axis=1)      # Combine existing csv and new peak dataframe

            else:
                df_peak.to_csv(path + '/peak.csv')     # Create new peak csv

    # tdmsファイル以外の場合
    else:
        print('This folder does NOT have tdms files.')
        sys.exit()

# 最終的な実波形とFFT結果のAverage処理
if f_ave == True:
    df_actwave_ave['CW'] = df_actwave_ave['CW'] / cnt
    df_actwave_ave['CCW'] = df_actwave_ave['CCW'] / cnt
    df_FFT_ave['CW_PSD'] = df_FFT_ave['CW_PSD'] / cnt
    df_FFT_ave['CCW_PSD'] = df_FFT_ave['CCW_PSD'] / cnt
    POA_CW_ave, POA_CCW_ave = culc_POA(df_FFT_ave)      # Culcrate POA

    # Average処理波形でPeak検波を行う場合
    if f_peak == True:

        # ピーク値に関してcsvファイルに書き込む
        # csvファイルが存在する場合は追記をし、存在しない場合は新規作成する
        df_peak = find_peak(df_fft, filename)        # Find peak

        # csvファイルが存在する場合は追記をし、存在しない場合は新規作成する
        if os.path.isfile(path + '/peak.csv'):
            df_existing = pd.read_csv(path + '/peak.csv')
            df_combined = pd.concat([df_existing, df_peak], axis=1)      # Combine existing csv and new peak dataframe

        else:
            df_peak.to_csv(path + '/peak.csv')     # Create new peak csv

    # グラフをPlotする処理
    if f_plot_graph == True:
        plot_actwave(df_actwave_ave, 'Average', ylim=ylim_act)        # Plot average actwave
        plot_fft(df_FFT_ave, POA_CW_ave, POA_CCW_ave, 'Average', ylim=ylim_fft)        # Plot average FFT
        plt.show()

# Contour図を作図する処理
if f_contour == True:
    contour_map.plot_2Dcontour(ylim=ylim_fft)        # Plot 2D contour
    # contour_map.plot_2D_Contour_with_interpolation(ylim=ylim_fft)

    # csvファイルへの書き込み
    if f_contour_csv == True:
        contour_map.contour_toCSV()        # Save contour data to csv

    # 3Dコンター図を作図する処理
    # 作図されるグラフはインタラクティブな処理が可能
    if f_3D_Contour == True:
        contour_map.plot_3Dcontour()        # Plot 3D contour

# f_contourがFalseであるにも関わらず、f_contour_csvまたはf_3D_contourがTrueの場合
else:
    if f_contour_csv == True:
        print('Contour CSV is not created. Please set f_contour to True.')
    
    if f_3D_Contour == True:
        print('3D Contour is not created. Please set f_contour to True.')
