import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sc


# =================== Variables ===================
filename = "Pulse_test_2.txt"
fs = 30 #funnet ved å ta lengden på data og dele på tiden spilt inn, også mulig å lese av i terminalen etter kjørt roi.py filen

# =================== Import text file ===================
data = np.loadtxt(filename)
data = data - np.mean(data,axis=0)
#Split into channels for Red, Green and Blue
Red = data[:,0]
Green = data[:,1]
Blue = data[:,2]
#Create a time axis
time = np.arange(len(data))
print(f"lengden på data i grønt_signal (samples): {len(Green)}")


# =================== Filtrere data med båndpass Butterworth filter ===================
""" def digitalt_filter(N, low_freq, high_freq, data_2B_filtered): #N er orden på filteret 
    nyquist = fs / 2
    low = low_freq / nyquist  #Normaliserer frekvensen
    high = high_freq / nyquist  
    filter_coefficients = sc.butter(N, [low, high], "bandpass") #python tupple with coeffs
    filtrert_data = sc.lfilter(filter_coefficients[0], filter_coefficients[1], data_2B_filtered)
    return filtrert_data #er et array

Red_filtrert = digitalt_filter(4, 20, , Red)
Green_filtrert = digitalt_filter(4, 20, 100, Green)
Blue_filtrert = digitalt_filter(4, 20, 100, Blue)
 """



def digitalt_filter(high_freq, data_2B_filtered): 
    N = 4 #N er orden på filteret 
    nyquist = fs / 2  #Normaliserer frekvensen
    high = high_freq / nyquist  
    filter_coefficients = sc.butter(N, high, "low") #python tupple with coeffs
    filtrert_data = sc.lfilter(filter_coefficients[0], filter_coefficients[1], data_2B_filtered)
    return filtrert_data #er et array


# =================== Regne ut puls med FFT ===================
def fft(data_kanal):
    Nfft = len(data_kanal)
    data_fft = np.fft.fft(data_kanal, Nfft)
    return np.abs(data_fft)


# =================== Regne ut puls med autocorrelasjon ===================
def autocorrelasjon(data_kanal):
    data_autocorr = sc.correlate(data_kanal, data_kanal, mode = 'full')
    data_autocorr = data_autocorr / np.max(data_autocorr)
    return data_autocorr

def find_puls_auto(data_kanal):
    rxy = autocorrelasjon(data_kanal)
    peaks_indices = sc.find_peaks(rxy,height = 0.4, threshold=None)
    print(peaks_indices)  #denne på jobbes litt mer med, den finner ikke puls nå uten å gjøre mer matematikk



#Plot raw data
def plot_rådata():
    plt.plot(Red, "r")
    plt.plot(Green, "g")
    plt.plot(Blue, "b")
    plt.show()

#Plot av filtrert data
def plot_filtrert_data():
    Red_filtrert = digitalt_filter(3, Red)
    Green_filtrert = digitalt_filter(3, Green)
    Blue_filtrert = digitalt_filter(3, Blue)

    plt.plot(Red_filtrert, "r")
    plt.plot(Green_filtrert, "g")
    plt.plot(Blue_filtrert, "b")
    plt.show()

#Plot FFT
def plot_FFT(data_kanal):
    signallengde_frames = len(data_kanal)
    frames = np.arange(0, len(data_kanal))
    signallengde_tid = signallengde_frames/30 #fps
    frekvens_resolution = 1/signallengde_tid # 1/sek = Hz
    freqs = frekvens_resolution*frames #datapunkter i frames*Hz
    bpm = freqs*60 #konverterer fra frekvens til bmp

    
    plt.plot(bpm, fft(Red), "r")
    plt.plot(bpm, fft(Green), "g")
    plt.plot(bpm, fft(Blue), "b")
    plt.show()


#Plot Autocorrelasjon
def plot_autocorr(data_kanal): #her kan man ta inn rå eller filtrert data
    Green_autocorr = autocorrelasjon(data_kanal)
    Blue_autocorr = autocorrelasjon(data_kanal)
    Red_autocorr = autocorrelasjon(data_kanal)

    plt.plot(Blue_autocorr, "b")
    plt.plot(Red_autocorr, "r")
    plt.plot(Green_autocorr, "g")
    plt.show()

plot_autocorr(digitalt_filter(3, Green))