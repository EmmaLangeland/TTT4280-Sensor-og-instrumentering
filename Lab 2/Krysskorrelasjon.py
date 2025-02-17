import numpy as np
import matplotlib.pyplot as plt
#from sympy import *
from raspi_import import raspi_import
import scipy.signal as sc


#hente inn data fra måling
sample_period, data = raspi_import('1000Hz_sinus.bin')

#juster data
k= 3.3/4096 #2^12 pga 12-bits ADC bit_signal = Signal/k
data = data[5000:,:]*k 
data = data - np.mean(data, axis=0) #fjerner DC-komponenten til signalet

fs = 1/sample_period

mic_1 = data[:,0] #ADC0
mic_2 = data[:,2] #ADC2
mic_3 = data[:,3] #ADC3

#Slower
#r_xy = np.correlate(x,y, mode='full')
#r_xy_abs = np.abs(r_xy)

#Faster
def krysskorrelasjon(x,y): #sett inn data fra ønskede mikrofoner
    r_xy = sc.correlate(x,y, mode='full')
    return r_xy

def delay(x, r_xy): #tar inn parametere x:ett av signalene, r_xy:krysskorrelsjonen
    r_xy_abs = np.abs(r_xy)
    l_max = np.argmax(r_xy_abs) - (len(x) - 1)
    delta_t = l_max/fs
    print(delta_t)
    return delta_t

#teori
t = np.linspace(-5,5, 1000)*np.pi*2
r_11 = krysskorrelasjon(np.sinc(t+2),np.sinc(t+1))

plt.plot(np.sinc(t))
plt.plot(np.sinc(t+1))
plt.plot(r_11)
plt.show()


#med seg selv
r_11 = krysskorrelasjon(mic_1,mic_3) 

#mellom mikrofonene
""" r_12 = krysskorrelasjon(mic_1,mic_2)
r_13 = krysskorrelasjon(mic_1,mic_3)
r_23 = krysskorrelasjon(mic_1,mic_1) """

print(f"lengden på data: {len(data)}")
print(f"lengden på krysskorrelasjeonen: {len(r_11)}")
print(f"delay mellom : {delay(mic_1,r_11)}")
print(f"spike ved sample nr: {np.argmax(r_11)}")
plt.plot(r_11)
plt.show()