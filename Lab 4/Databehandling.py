import numpy as np
import matplotlib.pyplot as plt
from raspi_import import raspi_import
import scipy.signal as sc

#********************************************************
#================= GENERAL FUNCTIONS ====================
#********************************************************

#============ Hente inn data fra måling ==================

def Hente_data(filnavn):
    sample_period, data = raspi_import(filnavn)

    k= 3.3/4096 #2^12 pga 12-bits ADC bit_signal = Signal/k
    data = data[5000:,:]*k 
    data = data - np.mean(data, axis=0) #fjerner DC-komponenten til signalet

    time_ax = np.linspace(0, (data.shape[0]*sample_period), data.shape[0])
    return data, time_ax


#============== Regn ut FFT-en til signalene ==============
#Kombiner signalene for å få dopler shift
#Send ut frekvensspekteret med doplershift

def Regn_ut_FFT(data):
    IF_I = data[:,0]
    IF_Q = data[:,1]
    #Kombiner I og Q kanalene
    IQ = IF_I + 1j*IF_Q
    
    Nfft = len(IQ)
    Freq_dopler = np.fft.fft(IQ, Nfft)
    Freq_dopler = Freq_dopler[:len(data)//2]
    return np.abs(Freq_dopler)




#======================= Plot Data =======================

def Plot_raw(data):
    plt.plot(data[:,0])
    plt.plot(data[:,2])

def Plot_FFT(data):
    plt.plot(data)

#********************************************************
#======================== MAIN ==========================
#********************************************************

def MAIN(filnavn):
    #Hente ut data:
    data, time_ax = Hente_data(filnavn)

    #Regne ut FFT:
    Freq_FFT = Regn_ut_FFT(data)

    #Plot DATA ------------------
    Plot_raw(data)
    Plot_FFT(Freq_FFT)
    plt.show()


#================ Kjør Programmet =================

MAIN("5ms_mot.bin")
