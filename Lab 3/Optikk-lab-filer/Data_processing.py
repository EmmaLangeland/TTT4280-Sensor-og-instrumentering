import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sc


# =================== Variables ===================
filename = "Pulse_test_2.txt"
fs = 30 #funnet ved å ta lengden på data og dele på tiden spilt inn, også mulig å lese av i terminalen etter kjørt roi.py filen

# =================== Import text file ===================
def file_to_data(filnavn):
    data = np.loadtxt(filnavn)
    data = data - np.mean(data,axis=0)
    #Split into channels for Red, Green and Blue
    Red = data[:,0]
    Green = data[:,1]
    Blue = data[:,2]
    #Create a time axis
    #time = np.arange(len(data))
    data_kanaler = [Red, Green, Blue]
    return data_kanaler


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

def find_puls_auto(data_kanal): #denne på jobbes litt mer med, den finner ikke puls nå uten å gjøre mer matematikk
    rxy = autocorrelasjon(data_kanal)
    peaks_indices = sc.find_peaks(rxy,height = 0.4, threshold=None)
    pulse = 1
    return pulse




def generate_pulse_vec(filnavn_liste, color_index): #data_kanaler[0] gir Rød, 0 = rød, 1 = grønn, 2 = blå
    puls_vec = []
    for filnavn in filnavn_liste: #5 ulike målinger
        data_kanaler = file_to_data(filnavn)
        puls_vec.append(find_puls_auto(data_kanaler[color_index])) 
    return puls_vec




def standardavvik(puls_vec):
    std = np.std(puls_vec, ddof=1)
    return std


def varians(pulse_vec):
    var = np.var(pulse_vec)
    return var


#Plotter PSD og regner ut SNR

""" Effekttetthetsspektrum = (abs(X_f)**2) #PSD
PSD_log = 10*np.log10(Effekttetthetsspektrum)
PSD_normalisert = PSD_log - np.max(PSD_log) #normalisering

signal_sum = 0
noise_sum = 0
 

for i in range(len(Effekttetthetsspektrum)):
    if  frekvenser[np.argmax(Effekttetthetsspektrum)]-10 < frekvenser[i] < frekvenser[np.argmax(Effekttetthetsspektrum)]+10:
        #print(f"frekvenskomponent: {Effekttetthetsspektrum[i]}")
        signal_sum += Effekttetthetsspektrum[i]
    else:
        noise_sum += Effekttetthetsspektrum[i]

SNR = 10*np.log10(np.abs(signal_sum / noise_sum))
print(f"SNR for en enkelt kanal {SNR} i dB") """


r,g,b = file_to_data(filename)

#Plot raw data
def plot_rådata():
    plt.plot(r, "r")
    plt.plot(g, "g")
    plt.plot(b, "b")
    plt.show()

#Plot av filtrert data
def plot_filtrert_data():
    Red_filtrert = digitalt_filter(3, r)
    Green_filtrert = digitalt_filter(3, g)
    Blue_filtrert = digitalt_filter(3, b)

    plt.plot(Red_filtrert, "r")
    plt.plot(Green_filtrert, "g")
    plt.plot(Blue_filtrert, "b")
    plt.show()

#Plot FFT
def plot_FFT(data_kanal):
    signallengde_frames = len(data_kanal)
    frames = np.arange(0, signallengde_frames)
    signallengde_tid = signallengde_frames/30 #fps
    frekvens_resolution = 1/signallengde_tid # 1/sek = Hz
    freqs = frekvens_resolution*frames #datapunkter i frames*Hz
    bpm = freqs*60 #konverterer fra frekvens til bmp
    print(f"dette er bpm: {bpm}")
    
    
    plt.plot(bpm, fft(r), "r")
    plt.plot(bpm, fft(g), "g")
    plt.plot(bpm, fft(b), "b")
    plt.show()


#Plot Autocorrelasjon
def plot_autocorr(data_kanal1, data_kanal2, data_kanal3): #her kan man ta inn rå eller filtrert data
    Green_autocorr = autocorrelasjon(data_kanal1)
    Blue_autocorr = autocorrelasjon(data_kanal2)
    Red_autocorr = autocorrelasjon(data_kanal3)

    plt.plot(Blue_autocorr, "b")
    plt.plot(Red_autocorr, "r")
    plt.plot(Green_autocorr, "g")
    plt.show()

plot_autocorr(digitalt_filter(3,r), digitalt_filter(3, g),digitalt_filter(3, b))

plot_FFT(g)


filnavn_lst = ["maling1", "maling2", "m3",  "m4", "m5"]