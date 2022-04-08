import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import numpy as np
import wave
import sys



def find_fourier_helper(signal, frame_rate, start, end):
    
    signal = signal[start:end]

    timesteps = np.linspace(0, len(signal) / frame_rate, num=len(signal))
    timestep = timesteps[1] - timesteps[0]

    samples = timesteps.shape[0]

    yf = fft(signal)
    xf = fftfreq(samples, timestep)[:samples//2]

    return yf, xf, samples



def animate_fourier(filename, time_window):

    spf = wave.open(filename, "r")
    signal = spf.readframes(-1)
    signal = np.frombuffer(signal, "int16")
    frame_rate = spf.getframerate()

    # If Stereo
    if spf.getnchannels() == 2:
        print("Just mono files")
        sys.exit(0)
    
    signal_length = signal.size
    num_windows = signal_length // time_window

    for window_id in range(num_windows):
        yf, xf, samples = find_fourier_helper(signal, frame_rate, window_id*time_window, (window_id + 1)*time_window)
        plt.cla()
        plt.xlim(0, 2000)
        plt.ylim(0, 20000)
        plt.grid()
        plt.plot(xf, 2.0/samples * np.abs(yf[0:samples//2]))
        plt.pause(0.1)


plt.figure(1)
plt.xlim(0, 3000)
plt.grid()

animate_fourier("high_test.wav", 2500)
animate_fourier("low_test.wav", 2500)

plt.show()