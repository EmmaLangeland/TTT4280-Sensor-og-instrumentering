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
    return data, time_ax, sample_period


#============== Regn ut FFT-en til signalene ==============
#Kombiner signalene for å få dopler shift
#Send ut frekvensspekteret med doplershift
#FFT uten pad
def Regn_ut_FFT(data, filnavn):
    data_1, time_ax, sample_period = Hente_data(filnavn)

    fs = 1/sample_period
    IF_I = data[:,0] #ADC0
    IF_Q = data[:,4] #ADC4

    #Kombiner I og Q kanalene
    IQ = IF_I + 1j*IF_Q
    
    Nfft = len(IQ)
    FFT_dopler = np.fft.fft(IQ, Nfft)
    FFT_dopler = np.abs(FFT_dopler)
    Dopler_shift = np.fft.fftshift(FFT_dopler) #1kHz sample rate

    freqs = np.fft.fftfreq(Nfft, 1/fs)
    freqs = np.fft.fftshift(freqs)
    #FFT_dopler = FFT_dopler[:len(data)//2] #Kun halve spekteret
    return Dopler_shift, freqs
#FFT med pad
def Regn_ut_FFT_pad(data, filnavn):
    data_1, time_ax, sample_period = Hente_data(filnavn)

    fs = 1/sample_period
    IF_I = data[0] #ADC0
    IF_Q = data[4] #ADC4

    #Kombiner I og Q kanalene
    IQ = IF_I + 1j*IF_Q
    
    Nfft = len(IQ)
    FFT_dopler = np.fft.fft(IQ, Nfft)
    FFT_dopler = np.abs(FFT_dopler)
    Dopler_shift = np.fft.fftshift(FFT_dopler) #1kHz sample rate

    freqs = np.fft.fftfreq(Nfft, 1/fs)
    freqs = np.fft.fftshift(freqs)
    #FFT_dopler = FFT_dopler[:len(data)//2] #Kun halve spekteret
    return Dopler_shift, freqs

#FFT til db

def FFT_DB(data_FFT):
    data_fft_db= 10*np.log10(data_FFT) 
    data_fft_db_norm = data_fft_db - np.max(data_fft_db)
    return data_fft_db, data_fft_db_norm

def Hanning(data, pad):
    IF_I = data[:,0] #ADC0
    IF_Q = data[:,4] #ADC4
    window_I = np.hanning(len(IF_I))  # Lager et Hanning-vindu med samme lengde som den data #erstatt med np.ones for å kun padde
    window_Q = np.hanning(len(IF_Q))
    hanning_window_I = IF_I * window_I
    hanning_window_Q = IF_Q * window_Q
    hanning_window_padded_I = np.pad(hanning_window_I, (0,pad)) #zero padder bak
    hanning_window_padded_Q = np.pad(hanning_window_Q, (0,pad))
    hanning_window_padded = [hanning_window_padded_I,1,1,1, hanning_window_padded_Q] #Extra 1s to have the code work in FFTplot.
    return hanning_window_padded


def PSD(data, filnavn):
    X, freqs= Regn_ut_FFT(data, filnavn)
    Effekttetthetsspektrum = (abs(X)**2) #PSD
    PSD_log = 10*np.log10(Effekttetthetsspektrum)
    PSD_normalisert = PSD_log - np.max(PSD_log) #normalisering
    return Effekttetthetsspektrum, PSD_normalisert


def SNR(data, filnavn):

    signal_sum = 0
    noise_sum = 0
    Effekttetthetsspektrum, norm = PSD(data, filnavn)

    N_sum = 0
    N_noise = 0
    for i in range(len(Effekttetthetsspektrum)):
        if  np.argmax(Effekttetthetsspektrum)-50 < i < np.argmax(Effekttetthetsspektrum)+50:
            signal_sum += Effekttetthetsspektrum[i]
            N_sum += 1
        else:
            noise_sum += Effekttetthetsspektrum[i]
            N_noise += 1

    print(signal_sum)
    print(noise_sum)
    print(Effekttetthetsspektrum[0])
    print(Effekttetthetsspektrum[1])
    signal_sum_norm = signal_sum/N_sum
    noise_sum_norm = noise_sum / N_noise
    SNR = 10*np.log10(np.abs(signal_sum_norm / noise_sum_norm))

    return SNR

def radiell_hastighet(fft_dopler, freqs):
    index = np.argmax(fft_dopler)
    f_d = freqs[index]
    #print(f_d)
    #f_d = np.abs(f_d)
    f0 = 24*10**9 #Hz
    c = 3*10**8
    v_r = c*f_d/(2*f0)
    return v_r

def varians(hastigheter):
    return np.var(hastigheter)

def standardavvik(hastigheter):
    return np.std(hastigheter)


#======================= Plot Data =======================

def Plot_raw(data):
    plt.plot(data[:,0], label='ADC0')
    #plt.show()
    plt.plot(data[:,4], label='ADC4')
    plt.show()

def Plot_FFT(data, freqs):
    plt.plot(freqs, data)
    

#********************************************************
#======================== MAIN ==========================
#********************************************************

def MAIN(filnavn):
    #Hente ut data:
    data, time_ax, s = Hente_data(filnavn)

    #Plot raw data
    Plot_raw(data)
    plt.show()

    #pad data og legg til hanningvindu
    padded_data = Hanning(data, 2**17)

    #Regne ut FFT:
    #Uten padding
    FFT_dopler, freqs = Regn_ut_FFT(data, filnavn)
    #Med padding
    FFT_dopler_pad, freqs_pad = Regn_ut_FFT_pad(padded_data, filnavn)

    #Plot FFT
    #Uten padding
    Plot_FFT(FFT_dopler, freqs)
    plt.show()
    #Med padding
    Plot_FFT(FFT_dopler_pad, freqs_pad)
    plt.title("FFT med padding")
    plt.show()

    #FFT til db
    #Uten padding
    FFT_dopler_db, FFT_dopler_db_norm = FFT_DB(FFT_dopler)
    Plot_FFT(FFT_dopler_db, freqs)
    #Med padding
    FFT_dopler_db_pad, FFT_dopler_db_norm_pad = FFT_DB(FFT_dopler_pad)
    Plot_FFT(FFT_dopler_db_pad, freqs_pad)
    plt.title("FFT med padding log")
    plt.ylim(-10, 31)
    plt.show()

    #Regne ut SNR
    SNR_uten_pad = SNR(FFT_dopler, filnavn)
    SNR_pad = SNR(FFT_dopler, filnavn)
    print(f"SNR uten padding: {SNR_uten_pad} \n SNR med padding: {SNR_pad}")

    v = radiell_hastighet(FFT_dopler, freqs)
    v_pad = radiell_hastighet(FFT_dopler_pad, freqs)
    print(f"Hastighet uten padding: {v} \n Hastighet med padding: {v_pad}")



def FlereFiler(filnavn_lst):
    hastigheter = []
    for filnavn in filnavn_lst:
        data = Hente_data(filnavn)
        FFT_dopler, freqs = Regn_ut_FFT(data, filnavn)
        v = radiell_hastighet(FFT_dopler, freqs)
        hastigheter.append(v)
    print("hastigheter:", hastigheter)
    print("variansen av hastigheter:", varians(hastigheter))
    print("std av hastigheter:", standardavvik(hastigheter))



#================ Kjør Programmet =================
#Kjør funksjonen for kun 1 fil
MAIN("fram_speed_1.bin")

#Kjør funksjonen for flere filer
""" fram_fort = ["fram_fort_1.bin", "fram_fort_2.bin", "fram_fort_3.bin", "fram_fort_4.bin" ,"fram_fort_5.bin"] #Legg til filnavn
FlereFiler(fram_fort)
fram_speed = ["fram_speed_1.bin", "fram_speed_2.bin", "fram_speed_3.bin", "fram_speed_4.bin" ,"fram_speed_5.bin"] #Legg til filnavn
FlereFiler(fram_speed)
bak = ["bak_1.bin", "bak_2.bin", "bak_3.bin", "bak_4.bin" ,"bak_5.bin"] #Legg til filnavn   
FlereFiler(bak) """