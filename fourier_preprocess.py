import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import numpy as np
import wave
import sys
import csv



def find_fourier_helper(signal, frame_rate, start, end):
    
    signal = signal[start:end]

    timesteps = np.linspace(0, len(signal) / frame_rate, num=len(signal))
    timestep = timesteps[1] - timesteps[0]

    samples = timesteps.shape[0]

    yf = fft(signal)
    xf = fftfreq(samples, timestep)[:samples//2]

    return yf, xf, samples



def run_fourier(audio_filename, time_window, shift_size, pause_time = 0.1, animate = False, write_to_csv = False, csv_filename = "testcsv.csv"):

    spf = wave.open(audio_filename, "r")
    signal = spf.readframes(-1)
    signal = np.frombuffer(signal, "int16")
    frame_rate = spf.getframerate()

    # If Stereo
    if spf.getnchannels() == 2:
        print("Just mono files")
        sys.exit(0)

    signal_length = signal.size
    current_loc = 0

    if write_to_csv:
        f_csv = open(csv_filename, 'w')
        f_csv.truncate()
        csv_writer = csv.writer(f_csv)

    while current_loc + time_window + 1 < signal_length:

        yf, xf, samples = find_fourier_helper(signal, frame_rate, current_loc, current_loc + time_window)
        yf = yf[0:samples//2]

        y_vals = 2.0/samples * np.abs(yf[0:samples//2])

        if animate:
            plt.cla()
            plt.plot(xf, y_vals)
            plot_helper(pause_time, audio_filename)

        if write_to_csv: csv_writer.writerow(y_vals)

        current_loc += shift_size



def plot_helper(pause_time, audio_filename):

    audio_filename = audio_filename[:-4]

    plt.xlim(0, 5000)
    plt.ylim(0, 14000)
    plt.grid()
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title("Fast Fourier Transform for " + audio_filename)

    plt.pause(pause_time)


pause_time = 0.001

plt.figure(1)
run_fourier("Advay-OAMF.wav", 2500, 1000, pause_time, animate=True, write_to_csv=True, csv_filename="sample_audio_1.csv")
plt.show()