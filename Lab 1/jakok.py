import numpy as np
import matplotlib.pyplot as plt
from raspi_import import raspi_import
import csv



C = 0.806*(10**-3) #conversion factor from ADC value to voltage
#N = 12 #number of bits
#V_ref = 3.3 #reference voltage

#V_conv = V_ref/(2**N - 1) #IKKE TENK PÅ DENNE, SKAL BLIR DET SAMME SOM C, ER FOR Å GJØRE OM DET DIGITALE SIGNALET TIL VOLT.


sample_period, data = raspi_import('1kHz.bin') #Navnet på filen dere har tatt signalet fra. 
#print(data.shape)
#print(sample_period)



fs = 1/sample_period

#print(data[:, 0])


frequencies = np.linspace(-fs/2, fs/2, data.shape[0])


data = data - np.mean(data, axis=0) #fjerner DC-komponenten til signalet

data_voltage = data*C #konverterer til volt fra binære tall

x = data_voltage[:, 1] #henter ut data fra kanal n-1. Dataene kommer i kolonner, en kolonne per ADC man har. Denne henter ut kolonne 1.


X_f = np.fft.fftshift(np.fft.fft(x)) #Fouriertransformasjon av signalet, med fft shift for å få spektere til å passefrekvensene fra -fs/2 til fs/2

periodogram = np.abs(X_f)**2 #Effekttetthetsspektrum

#The exponent 2 could be multiplied by 10 to get 20 which is a normal format of the dB.
periodogram_db = 10*np.log10(periodogram) #Effekttetthetsspektrum i desibel
normalized_periodogram_db = periodogram_db - np.max(periodogram_db) #Normaliserer spekteret

t = np.linspace(0, data.shape[0]*sample_period, data.shape[0]) #Tidsvektor lengden av signalet ganger sampleperioden.


plt.plot(t,x)
plt.title('Tidsdomene')
plt.xlabel('Tid [s]')
plt.ylabel('Amplitude [V]')
plt.xlim(0.0001, 0.005)
plt.show()


plt.plot(frequencies,periodogram)
plt.title('Periodogram/effekttetthetsspektrum')
plt.xlabel('Frekvens [Hz]')
plt.ylabel('Effekt [V^2]')
plt.xlim(-1200,1200)
plt.show()

plt.plot(frequencies,normalized_periodogram_db)
plt.title('Normalisert effekttetthetsspektrum i desibel')
plt.xlabel('Frekvens [Hz]')
plt.xlim(-1200,1200)
plt.ylabel('Effekt [dB]')
plt.show()