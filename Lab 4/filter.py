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


# ---- Lese CSV ----
header = []
data = []
filename = 'bpfilter1.csv'  # Bruk faktiske filnavn her

with open(filename) as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    for datapoint in csvreader:
        values = [float(value) for value in datapoint]
        data.append(values)

# ---- Legg inn data i separate lister ----
frekvens = [p[0] for p in data]      # Frekvens [Hz]
amplitude = [p[1] for p in data]     # Demping [dB] (kanal 2 i vårt tilfelle)

# ---- Plot ----
plt.figure(figsize=(10,6))
plt.xscale("log")
plt.ylabel("Demping [dB]")
plt.xlabel("Frekvens [Hz]")
plt.title("Bode-plot av båndpassfilter 1")

# Selve måledataen
plt.plot(frekvens, amplitude, label='|H(f)|', color='orange')

# Horisontal linje ved -3 dB
plt.axhline(y=17, linestyle='--', color='black', label='-3 dB fra 20dB forsterkning')

# Vertikale knekkfrekvenser (lav og høy)
plt.axvline(x=3.5, linestyle='--', color='blue', label='fL = 3.5 Hz')
plt.axvline(x=2800, linestyle='--', color='green', label='fH = 2.8 kHz')

# Marker evt. -3 dB punktene manuelt hvis kjent:
# plt.plot(11.5, -3, 'ro', label='Knekkpunkt 1')
# plt.plot(2600, -3, 'ro', label='Knekkpunkt 2')

plt.xlim([1, 5000])
plt.ylim([-5, 25])
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()


# ---- Lese CSV ----
header = []
data = []
filename = 'bpfilter2.csv'  # Bruk faktiske filnavn her

with open(filename) as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    for datapoint in csvreader:
        values = [float(value) for value in datapoint]
        data.append(values)

# ---- Legg inn data i separate lister ----
frekvens = [p[0] for p in data]      # Frekvens [Hz]
amplitude = [p[1] for p in data]     # Demping [dB] (kanal 2 i i vårt tilfelle)

# ---- Plot ----
plt.figure(figsize=(10,6))
plt.xscale("log")
plt.ylabel("Demping [dB]")
plt.xlabel("Frekvens [Hz]")
plt.title("Bode-plot av båndpassfilter 2")

# Selve måledataen
plt.plot(frekvens, amplitude, label='|H(f)|', color='orange')

# Horisontal linje ved -3 dB
plt.axhline(y=17, linestyle='--', color='black', label='-3 dB fra 20dB forsterkning')

# Vertikale knekkfrekvenser (lav og høy)
plt.axvline(x=3.5, linestyle='--', color='blue', label='fL = 3.5 Hz')
plt.axvline(x=2800, linestyle='--', color='green', label='fH = 2.8 kHz')

# Marker evt. -3 dB punktene manuelt hvis kjent:
# plt.plot(11.5, -3, 'ro', label='Knekkpunkt 1')
# plt.plot(2600, -3, 'ro', label='Knekkpunkt 2')

plt.xlim([1, 5000])
plt.ylim([-5, 25])
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
