import numpy as np
import matplotlib.pyplot as plt
from raspi_import import raspi_import
import scipy.signal as sc
import csv

#********************************************************
#================= GENERAL FUNCTIONS ====================
#********************************************************

#============ Hente inn data fra måling ==================

def Hente_data(filnavn):
    sample_period, data = raspi_import(filnavn)

    k= 3.3/4096 #2^12 pga 12-bits ADC bit_signal = Signal/k
    data = data[100:,:]*k 
    data = data - np.mean(data, axis=0) #fjerner DC-komponenten til signalet

    time_ax = np.linspace(0, (data.shape[0]*sample_period), data.shape[0])
    return data, time_ax


# CSV
""" def Hente_data_CSV(filnavn):
    header = []
    data = []
    with open(filnavn) as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        for datapoint in csvreader:
            values = [float(value) for value in datapoint]
            data.append(values)
    return data, header """



#============== Regn ut FFT-en til signalene ==============
#Kombiner signalene for å få dopler shift
#Send ut frekvensspekteret med doplershift

def Regn_ut_FFT(data):
    IF_I = data[:,0]
    IF_Q = data[:,4]
    #Kombiner I og Q kanalene
    IQ = IF_I + 1j*IF_Q
    
    Nfft = len(IQ)
    Freq_dopler = np.fft.fft(IQ, Nfft)
    Freq_dopler = Freq_dopler[:len(data)//2]
    return np.abs(Freq_dopler)

#FFT til db

def FFT_DB(data_FFT):
    data_fft_db= 10*np.log10(data_FFT) 
    data_fft_db_norm = data_fft_db - np.max(data_fft_db)
    return data_fft_db, data_fft_db_norm



#======================= Plot Data =======================

def Plot_raw(data):
    plt.plot(data[:,0])
    plt.show()
    plt.plot(data[:,4])
    plt.show()

def Plot_FFT(data):
    plt.plot(data)

#********************************************************
#======================== MAIN ==========================
#********************************************************

def MAIN(filnavn):
    #Hente ut data:
    data, time_ax = Hente_data(filnavn)
    #data = Hente_data_CSV(filnavn)

    #Regne ut FFT:
    Freq_FFT = Regn_ut_FFT(data)

    #Plot DATA ------------------
    Plot_raw(data)
    #Plot_FFT(Freq_FFT)
    plt.show()


#================ Kjør Programmet =================

MAIN("fram_speed_5.bin")
