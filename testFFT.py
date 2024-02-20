import numpy as np
import matplotlib.pyplot as plt

### Quantization ###
f_s = 4096       # sampling rate [Hz]
t_fin = 1        # measuring time [s]
dt = 1/f_s      # sampling period [s]
N = int(f_s * t_fin)       # number of sampling = 2^12 per sec

### Input signal ###
f0 = 440        # frequency [Hz]
t = np.arange(0, t_fin, dt)      # measuring time [s]
y = np.sin(2*np.pi*f0*t)        # input signal [V]

# ### Graph of input signal ###
# plt.plot(t, y)
# plt.xlim(0, 10*10**-3)
# plt.show()

# ### FFT ###
# y_fft = np.fft.fft(y)
# freq = np.fft.fftfreq(N, d=dt)
# Amp = abs(y_fft/(N/2))

# ### Graph of FFT result ###
# plt.plot(freq[1:int(N/2)], Amp[1:int(N/2)])
# plt.xscale('log')
# plt.show()

### 0 Padding FFT ###
new_N = 8192        # new number of sampling 2^13
t_fin = t_fin * (new_N / N)     # new time domain after 0 padding [s]
f_s = new_N / t_fin     # new sampling rate [Hz]
dt = 1 / f_s

pad = np.zeros(new_N - N)
data_pad = np.insert(y, N, pad)
acf = (sum(np.abs(y)) / len(y)) / (sum(np.abs(data_pad)) / len(data_pad))       # amplitude correction factor

new_t = np.arange(0, t_fin, dt)
new_y = data_pad * acf

### Graph of input signal ###
plt.plot(new_t, new_y)
plt.show()

### FFT ###
y_fft = np.fft.fft(new_y)
freq = np.fft.fftfreq(new_N, d=dt)
Amp = abs(y_fft/(new_N/2))

### Graph of FFT result ###
plt.plot(freq[1:int(new_N/2)], Amp[1:int(new_N/2)])
plt.xscale('log')
plt.show()