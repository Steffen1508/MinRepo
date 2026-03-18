
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import ipywidgets as widgets
from IPython.display import display, clear_output

def make_signal(fs, T, f_rot, a_rot, f_struct, a_struct, noise_std,
                fault_on, fault_t0, f_fault, a_fault, noise_seed):
    t = np.arange(0, T, 1/fs)

    x = a_rot*np.sin(2*np.pi*f_rot*t) + a_struct*np.sin(2*np.pi*f_struct*t)

    if fault_on:
        idx = t >= fault_t0
        x[idx] += a_fault*np.sin(2*np.pi*f_fault*t[idx])

    if noise_std > 0:
        rng = np.random.default_rng(noise_seed)
        x += noise_std * rng.standard_normal(len(t))

    return t, x


def dft(x, fs):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1/fs)
    return f, np.abs(X)


def plot_all(t, x, fs, nperseg):
    f, mag = dft(x, fs)

    f_stft, tau, Z = stft(x, fs, nperseg=nperseg)

    plt.figure()
    plt.plot(t[:2000], x[:2000])
    plt.title("Time signal (first 2 s)")
    plt.show()

    plt.figure()
    plt.plot(f, mag)
    plt.xlim(0,200)
    plt.title("DFT magnitude")
    plt.show()

    plt.figure()
    plt.pcolormesh(tau, f_stft, np.abs(Z))
    plt.ylim(0,200)
    plt.title("STFT Spectrogram")
    plt.show()


def run_sandbox():

    fs = widgets.IntSlider(value=1000,min=200,max=3000,step=100,description="fs")
    T = widgets.IntSlider(value=10,min=2,max=20,step=1,description="T")

    f_rot = widgets.FloatSlider(value=10,min=1,max=50,step=0.5,description="f_rot")
    f_struct = widgets.FloatSlider(value=40,min=1,max=200,step=1,description="f_struct")

    noise = widgets.FloatSlider(value=0.2,min=0,max=1,step=0.05,description="noise")
    seed = widgets.IntSlider(value=0,min=0,max=1000,description="seed")

    fault_on = widgets.Checkbox(value=True,description="fault")
    fault_t0 = widgets.FloatSlider(value=5,min=0,max=10,step=0.5,description="fault_t0")
    f_fault = widgets.FloatSlider(value=90,min=10,max=200,step=1,description="f_fault")

    stft_win = widgets.IntSlider(value=256,min=32,max=1024,step=32,description="stft_win")

    out = widgets.Output()

    def update(change=None):
        with out:
            clear_output(wait=True)
            t,x = make_signal(
                fs.value,T.value,
                f_rot.value,0.7,
                f_struct.value,0.3,
                noise.value,
                fault_on.value,
                fault_t0.value,
                f_fault.value,
                0.6,
                seed.value
            )
            plot_all(t,x,fs.value,stft_win.value)

    for w in [fs,T,f_rot,f_struct,noise,seed,fault_on,fault_t0,f_fault,stft_win]:
        w.observe(update,"value")

    ui = widgets.VBox([
        fs,T,f_rot,f_struct,noise,seed,
        fault_on,fault_t0,f_fault,stft_win
    ])

    display(ui,out)
    update()
