import numpy as np
from nptdms import TdmsFile
from nptdms import tdms
import pandas as pd
import matplotlib.pyplot as plt

gain = 0.00152587890625     # Gain (Setting)
wf_increment = 0.0001953125     # Time Resolution (Setting)
num_sample = 5120       # Number of Samplings

# read tdms file
filename = '00100243.tdms'
tdms_file = TdmsFile.read(filename)
df_actwave = tdms_file.as_dataframe(time_index=False)
df_actwave.columns = ['CW', 'CCW']

# Convert signal to V
df_actwave.insert(0, 'Time', df_actwave.index * wf_increment)
df_actwave['CW'] = df_actwave['CW'] * gain
df_actwave['CCW'] = df_actwave['CCW'] * gain

# Show vibration graphs
plt.figure()
plt.subplots_adjust(wspace=0.4, hspace=0.6)

plt.subplot(2, 1, 1)
plt.plot(df_actwave['Time'], df_actwave['CW'])
plt.title('CW')
plt.xlabel('T [sec]')
plt.ylabel('V [V]')
plt.ylim(-0.3, 0.3)

plt.subplot(2, 1, 2)
plt.plot(df_actwave['Time'], df_actwave['CCW'])
plt.title('CCW')
plt.xlabel('T [sec]')
plt.ylabel('V [V]')
plt.ylim(-0.3, 0.3)

plt.show()

# FFT


print(df_actwave)
