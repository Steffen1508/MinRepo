import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp_signal


df = pd.read_csv("MinRepo\Data&AI\Lektion6\DailyDelhiClimateTrain.csv")
print(df.columns.tolist())  # find din signalkolonne

df = df.sort_values("date").reset_index(drop=True)

signal = df["meantemp"].to_numpy()
fs = 1 / 86400  # daglige målinger → 1 sample per dag

t = np.arange(len(signal)) / fs




# 1. Define the FIR filter kernel h(n)
M = 15
h = np.ones(M) / M  # A moving average filter of length 15

# 2. Compute the frequency response
# w is the array of frequencies, h_freq is the complex response
w, h_freq = sp_signal.freqz(h, worN=1024)

# 3. Calculate magnitude in dB (adding small epsilon to avoid log(0))
magnitude_db = 20 * np.log10(np.abs(h_freq) + 1e-12)

# 4. Normalize frequency axis to range [0, 1] (where 1 is the Nyquist frequency)
normalized_freq = w / np.pi



filtered_signal = np.convolve(signal, h, mode='same')

plt.plot(t, signal, alpha=0.4, label="Råsignal")
plt.plot(t, filtered_signal, label="FIR filtreret")
plt.xlabel("Tid [s]")
plt.ylabel("meantemp")
plt.legend()
plt.show()