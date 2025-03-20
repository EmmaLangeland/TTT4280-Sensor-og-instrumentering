import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sc 
import csv

#import pandas as pd

G = -10 #20dB 20*log(10) = 20

R_2 = 100000 #100kOhm

R_1 = - R_2/G

f_L = 3.5
f_H =2.8e3 #Hz

C_1 = (1/(2*np.pi*R_1*f_L)) *1e6 #microfarad
C_2 = (1/(2*np.pi*R_2*f_H)) *1e12 #picofarad


print("R_1: ", R_1)
print("R_2: ", R_2)

print("C_1: ", C_1)
print("C_2: ", C_2)



# Les inn kolonner (tilpass navnene hvis nødvendig)
frekvens = data['Frekvens (Hz)']         # eller f.eks 'Frequency'
amplitude = data['Amplitude (V)']        # eller 'Amplitude (dB)', alt etter hva du har

# Hvis amplituden er i volt og du ønsker dB:
amplitude_db = 20 * np.log10(amplitude)

# Plot amplitude Bode-plot
plt.figure(figsize=(10,6))
plt.semilogx(frekvens, amplitude_db, label="Amplitude (dB)")
plt.axvline(3.5, color='red', linestyle='--', label='fL = 3.5 Hz')
plt.axvline(2800, color='green', linestyle='--', label='fH = 2.8 kHz')
plt.xlabel("Frekvens [Hz]")
plt.ylabel("Amplitude [dB]")
plt.title("Bode-plot av båndpassfilter")
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()



header = []
data = []
filename = 'network .csv'

#Henter data fra csvfil
with open(filename) as csvfile:
    csvreader = csv.reader(csvfile)
    #Leser første linje i csv-fila (den med navn til kanalene)
    header = next(csvreader)
    for datapoint in csvreader:
        values = [float(value) for value in datapoint]
        data.append(values)


#Legger inn data fra hver kanal i hver sin liste
time = [(p[0]) for p in data]
ch1 = [(p[1]) for p in data]
ch2 = [(p[2]) for p in data]


plt.xscale("log")
plt.ylabel("Demping [dB]")
plt.xlabel("Frekvens [Hz]")
plt.title("Network Analyse")
#plt.plot(time, ch1, label = '')

plt.axhline(y=-1.99, linestyle='--', color='black', label='-2dB')
plt.plot(time, ch2, label = '|H($f$)|', color= 'orange')


plt.xlim([1, 200000])
plt.ylim([-50, 10])

plt.plot(11.47, -5, 'gx', label= '(11.47Hz , -5dB)') # Intersection point



""" plt.axvline(456,linestyle='--', color='red')
plt.axvline(3655,linestyle='--', color='red')
 """

""" 
plt.plot(3655, -3, 'go', label= '(3655Hz , -3dB)') # Intersection point """

plt.grid()
plt.legend()
plt.show()