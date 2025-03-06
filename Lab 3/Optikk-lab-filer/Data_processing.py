import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sc
from scipy import signal  
import math 

# =================== Variables ===================
filename = "Pulse_test_2.txt"

# =================== Import text file ===================
data = np.loadtxt(filename)
data = data - np.mean(data,axis=0)
#Split into channels for Red, Green and Blue
Red = data[:,0]
Green = data[:,1]
Blue = data[:,2]
#Create a time axis
time = np.arange(len(data))

#Filtrere data??


# Regne ut puls med FFT
def fft(data_kanal):
    Nfft = len(data_kanal)
    data_fft = np.fft.fft(data_kanal, Nfft)
    return data_fft

Green_fft = fft(Green)
Blue_fft = fft(Blue)
Red_fft = fft(Red)

# Regne ut puls med autocorrelasjon
def autocorrelasjon(data_kanal):
    data_autocorr = sc.correlate(data_kanal, data_kanal, mode = 'full')
    data_autocorr = data_autocorr / np.max(data_autocorr)
    return data_autocorr

def find_puls_auto(data_kanal):
    rxy = autocorrelasjon(data_kanal)
    peaks_indices = sc.find_peaks(rxy,height = 0.4, threshold=None)
    print(peaks_indices)

find_puls_auto(Green)

Green_autocorr = autocorrelasjon(Green)
Blue_autocorr = autocorrelasjon(Blue)
Red_autocorr = autocorrelasjon(Red)


#Plot raw data
""" plt.plot(Red, "r")
plt.plot(Green, "g")
plt.plot(Blue, "b")
plt.show() """

#Plot FFT
plt.plot(Blue_fft, "b")
plt.plot(Red_fft, "r")
plt.plot(Green_fft, "g")
plt.show()

#Plot Autocorrelasjon
plt.plot(Blue_autocorr, "b")
plt.plot(Red_autocorr, "r")
plt.plot(Green_autocorr, "g")
plt.show()