# TODO
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

"""
from engine import run_sandbox
run_sandbox()
"""

#Opgave 1/2

fs = 1000
T = 10
t = np.linspace(0, T, fs*T)

x = np.sin(2*np.pi*10*t)


mask = t <= 2
plt.plot(t[mask], x[mask])
plt.xlabel("Tid (s)")
plt.ylabel("Amplitude")
plt.title("Motorvibration - første 2 sekunder")
plt.grid(True)
#plt.show()

#Opgave 3

fft_x = np.fft.fft(x)

fft_freq = np.fft.fftfreq(len(fft_x),d=1/fs)


mask = fft_freq >= 0
plt.plot(fft_freq[mask], np.abs(fft_x[mask]))
plt.title("FFT analyse")
plt.xlabel("Frekvens (Hz)")
plt.ylabel("Magnitude")
#plt.show()

#Opave 4

støj = np.random.normal(0,0.5,len(t))

x_støj = np.sin(2*np.pi*10*t) + støj

fft_x = np.fft.fft(x_støj)

fft_freq = np.fft.fftfreq(len(fft_x),d=1/fs)


plt.plot(fft_freq[mask], np.abs(fft_x[mask]))
plt.title("FFT analyse med støj")
plt.xlabel("Frekvens (Hz)")
plt.ylabel("Magnitude")
plt.show()