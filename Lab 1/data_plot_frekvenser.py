import numpy as np
import matplotlib.pyplot as plt
from raspi_import import raspi_import

#--------------------------------------------------------------------------------
#hente inn data fra måling
#Sinus_100Hz, Sinus_500Hz, Sinus_1500Hz, Sinus_2000Hz, Sinus_1000Hz

sample_period, data = raspi_import('Sinus_2000Hz.bin')

f = 2000
k= 3.3/4096 #2^12 pga 12-bits ADC bit_signal = Signal/k
data = data[1:,:]*k 
data = data - np.mean(data, axis=0) #fjerner DC-komponenten til signalet

#------------------------------------------------------------------------
#hanning vindu med og uten padding 
def hanning(data, pad):
    window = np.hanning(len(data))  # Lager et Hanning-vindu med samme lengde som den data #erstatt med np.ones for å kun padde
    hanning_window = data * window
    hanning_window_padded = np.pad(hanning_window, (0,pad)) #zero padder bak
    return hanning_window_padded

#-------------------------------------------------------------------------
#Plot av fft

fs = 1/ sample_period
x_t = data[:,0] #x_t er dataen til en av ADC-ene

#hanning og evt padding
x_t_hann = hanning(x_t, 2**16) #kaller på hanning vindu funksjonen. juster antall pads etter ønske. 2**15 - len(x_t)
N_fft_hann= x_t_hann.shape[0]
X_f_hann = np.fft.fft(x_t_hann, N_fft_hann) #fft av x_t
X_f_hann = abs(X_f_hann)
X_f_hann = X_f_hann[:int(N_fft_hann/2)] #kutter spekteret i to for å kun vise det reelle signalet
X_f_db_hann= 10*np.log10(X_f_hann) 
X_f_db_normalisert_hann = X_f_db_hann - np.max(X_f_db_hann)


frekvenser_hann = np.linspace(0,fs/2,int(N_fft_hann/2)) #for zero pad

plt.plot(frekvenser_hann, X_f_db_normalisert_hann, label=f'f = {f}Hz', color='C9')
plt.xscale('log')
plt.xlabel('Frekvenser i log skala [Hz]')
plt.ylabel('Relativ amplitude [dB]')
plt.title('Frekvensspektrum i dB skala')
plt.ylim(-100, 10)
plt.legend(loc='upper left')
plt.grid()
plt.show()

#-------------------------------------------------------------
#Plotter PSD og regner ut SNR

Effekttetthetsspektrum = (abs(X_f_hann)**2) #PSD
PSD_log = 10*np.log10(Effekttetthetsspektrum)
PSD_normalisert = PSD_log - np.max(PSD_log) #normalisering

signal_sum = 0
noise_sum = 0
 

for i in range(len(Effekttetthetsspektrum)):
    if  frekvenser_hann[np.argmax(Effekttetthetsspektrum)]-10 < frekvenser_hann[i] < frekvenser_hann[np.argmax(Effekttetthetsspektrum)]+10:
        #print(f"frekvenskomponent: {Effekttetthetsspektrum[i]}")
        signal_sum += Effekttetthetsspektrum[i]
    else:
        noise_sum += Effekttetthetsspektrum[i]

SNR = 10*np.log10(np.abs(signal_sum / noise_sum))
print(f"SNR for en enkelt kanal {SNR} i dB")


plt.plot(frekvenser_hann, PSD_normalisert)
plt.xscale('log')
plt.xlabel('Frekvenser i log skala [Hz]')
plt.ylabel('Relativ amplitude [db]')
plt.title('Effekttetthetsspektrum i dB skala')
plt.grid()
plt.show()
